import os
import sys
import numpy as np
import netCDF4
import argparse
import time

def reconstruct_dineof(eof_file, mean_file, mask_file, output_file, variable_name='chlor_a', chunk_size=10):
    """
    Reconstructs the DINEOF output from eof.nc and meandata.val.
    
    Args:
        eof_file (str): Path to eof.nc file.
        mean_file (str): Path to meandata.val file.
        mask_file (str): Path to the original mask file (NetCDF).
        output_file (str): Path to the output NetCDF file.
        variable_name (str): Name of the variable to reconstruct.
        chunk_size (int): Number of time steps to process at once.
    """
    print(f"Starting reconstruction from {eof_file}...", flush=True)
    
    # 1. Read Mean Value
    try:
        with open(mean_file, 'r') as f:
            mean_val = float(f.read().strip())
        print(f"Mean value read from {mean_file}: {mean_val}")
    except Exception as e:
        print(f"Error reading mean file: {e}")
        sys.exit(1)

    # 2. Open EOF file
    try:
        nc_eof = netCDF4.Dataset(eof_file, 'r')
        # Dimensions: Sigma(dim001), V(dim001, dim002), U(dim001, dim004, dim003)
        # dim001 = modes, dim002 = time, dim003 = lon, dim004 = lat
        
        U = nc_eof.variables['U'][:] # Shape: (modes, lat, lon)
        V = nc_eof.variables['V'][:] # Shape: (modes, time)
        Sigma = nc_eof.variables['Sigma'][:] # Shape: (modes,)
        
        n_modes = U.shape[0]
        n_lat = U.shape[1]
        n_lon = U.shape[2]
        n_time = V.shape[1]
        
        print(f"EOF Dimensions: Modes={n_modes}, Time={n_time}, Lat={n_lat}, Lon={n_lon}")
        
    except Exception as e:
        print(f"Error reading EOF file: {e}")
        sys.exit(1)

    # 3. Read Mask (to apply NaNs/FillValue back)
    try:
        nc_mask = netCDF4.Dataset(mask_file, 'r')
        if 'mask' in nc_mask.variables:
            mask = nc_mask.variables['mask'][:]
        else:
            # Fallback if mask variable name is different or using 2D mask logic from run_dineof.py
            # Assuming standard 'mask' variable for now
            print("Warning: 'mask' variable not found, looking for alternatives...")
            # Try to infer from dimensions
            sys.exit("Mask variable not found in mask file.")
            
        # Ensure mask is 2D (lat, lon)
        if mask.ndim == 3:
             mask = np.max(mask, axis=0) # Collapse time if present
        
        print(f"Mask loaded. Shape: {mask.shape}")
        
    except Exception as e:
        print(f"Error reading mask file: {e}")
        sys.exit(1)

    # 4. Pre-calculate Scaled Spatial Modes (U * Sigma)
    # U: (modes, lat, lon), Sigma: (modes,)
    # We want U_scaled[k, y, x] = U[k, y, x] * Sigma[k]
    print("Pre-calculating scaled spatial modes...", flush=True)
    U_scaled = U * Sigma[:, np.newaxis, np.newaxis]
    
    # Free up original U and Sigma to save memory
    del U
    del Sigma
    
    # 5. Create Output NetCDF
    print(f"Creating output file: {output_file}", flush=True)
    try:
        with netCDF4.Dataset(output_file, 'w', format='NETCDF4') as nc_out:
            # Copy dimensions from mask file (lat, lon) and define time
            nc_out.createDimension('time', n_time)
            nc_out.createDimension('lat', n_lat)
            nc_out.createDimension('lon', n_lon)
            
            # Helper to copy variable
            def copy_var(name, src_var):
                fill_val = getattr(src_var, '_FillValue', None)
                dst_var = nc_out.createVariable(name, src_var.datatype, src_var.dimensions, fill_value=fill_val)
                atts = src_var.__dict__.copy()
                if '_FillValue' in atts:
                    del atts['_FillValue']
                dst_var.setncatts(atts)
                dst_var[:] = src_var[:]

            # Copy lat/lon variables if they exist in mask file
            if 'lat' in nc_mask.variables:
                copy_var('lat', nc_mask.variables['lat'])
            
            if 'lon' in nc_mask.variables:
                copy_var('lon', nc_mask.variables['lon'])
                
            # Create time variable (copy from mask file if available, or create dummy)
            if 'time' in nc_mask.variables:
                copy_var('time', nc_mask.variables['time'])
            else:
                time_var = nc_out.createVariable('time', 'f4', ('time',))
                time_var[:] = np.arange(n_time)
                
            # Create the main variable
            out_var = nc_out.createVariable(variable_name, 'f4', ('time', 'lat', 'lon'), fill_value=-999.0, zlib=True)
            # out_var.missing_value = -999.0 # fill_value argument sets _FillValue

            
            # 6. Reconstruct in Chunks
            print(f"Reconstructing in chunks of {chunk_size} time steps...", flush=True)
            
            total_chunks = (n_time + chunk_size - 1) // chunk_size
            
            for i in range(total_chunks):
                start_t = i * chunk_size
                end_t = min((i + 1) * chunk_size, n_time)
                
                # V_chunk: (modes, chunk_time)
                V_chunk = V[:, start_t:end_t]
                
                # Reconstruction: X_chunk = sum_k (U_scaled[k] * V_chunk[k]) + Mean
                # U_scaled: (modes, lat, lon)
                # V_chunk: (modes, time_chunk)
                # Result: (time_chunk, lat, lon)
                
                # Using tensordot or einsum
                # 'k y x, k t -> t y x'
                X_chunk = np.einsum('kyx, kt -> tyx', U_scaled, V_chunk)
                
                # Add mean
                X_chunk += mean_val
                
                # Apply mask
                # Mask is (lat, lon). 0 means land/invalid in DINEOF usually? 
                # Let's check mask convention. 
                # In dineof.init: mask = 1 for water, 0 for land usually.
                # But let's verify with the user's file.
                # Assuming mask=1 is valid data.
                
                # Broadcast mask to chunk shape
                mask_broadcast = np.broadcast_to(mask, X_chunk.shape)
                
                # Apply fill value where mask is 0 (or invalid)
                # Note: DINEOF might use 0 for land.
                # Let's assume mask==0 is land.
                X_chunk[mask_broadcast == 0] = -999.0
                
                # Write to NetCDF
                out_var[start_t:end_t, :, :] = X_chunk
                
                print(f"Processed chunk {i+1}/{total_chunks} (Time {start_t}-{end_t})", flush=True)
                
        print("Reconstruction complete.")
        
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)
    finally:
        nc_eof.close()
        nc_mask.close()

def project_new_data(eof_file, mean_file, new_data_file, output_file, variable_name='chlor_a', chunk_size=10):
    """
    Projects new data onto the existing EOF basis and fills gaps.
    
    Args:
        eof_file (str): Path to eof.nc file.
        mean_file (str): Path to meandata.val file.
        new_data_file (str): Path to the new data NetCDF file.
        output_file (str): Path to the output NetCDF file.
        variable_name (str): Name of the variable.
        chunk_size (int): Number of time steps to process at once.
    """
    print(f"Starting projection of {new_data_file}...", flush=True)
    
    # 1. Read Mean Value
    try:
        with open(mean_file, 'r') as f:
            mean_val = float(f.read().strip())
    except Exception as e:
        sys.exit(f"Error reading mean file: {e}")

    # 2. Open EOF file
    try:
        nc_eof = netCDF4.Dataset(eof_file, 'r')
        U = nc_eof.variables['U'][:] # (modes, lat, lon)
        Sigma = nc_eof.variables['Sigma'][:] # (modes,)
        
        n_modes = U.shape[0]
        n_lat = U.shape[1]
        n_lon = U.shape[2]
        
        # Flatten U for projection: (modes, M) where M = lat*lon
        U_flat = U.reshape(n_modes, -1) 
        
        # Pre-calculate Basis = (U * Sigma).T -> (M, modes)
        # We want to solve x = Basis * v
        Basis = (U_flat * Sigma[:, np.newaxis]).T
        
    except Exception as e:
        sys.exit(f"Error reading EOF file: {e}")

    # 3. Open New Data
    try:
        nc_new = netCDF4.Dataset(new_data_file, 'r')
        if variable_name not in nc_new.variables:
             sys.exit(f"Variable {variable_name} not found in {new_data_file}")
             
        data_var = nc_new.variables[variable_name]
        n_time_new = data_var.shape[0]
        
        # Check spatial dimensions
        if data_var.shape[1:] != (n_lat, n_lon):
             sys.exit(f"Spatial dimensions mismatch. EOF: ({n_lat}, {n_lon}), New Data: {data_var.shape[1:]}")
             
    except Exception as e:
        sys.exit(f"Error reading new data file: {e}")

    # 4. Create Output NetCDF
    print(f"Creating output file: {output_file}", flush=True)
    try:
        with netCDF4.Dataset(output_file, 'w', format='NETCDF4') as nc_out:
            # Copy dimensions
            for dim_name, dim in nc_new.dimensions.items():
                nc_out.createDimension(dim_name, len(dim) if not dim.isunlimited() else None)
            
            # Copy variables (lat, lon, time)
            for var_name, var in nc_new.variables.items():
                if var_name != variable_name:
                    fill_val = getattr(var, '_FillValue', None)
                    out_v = nc_out.createVariable(var_name, var.datatype, var.dimensions, fill_value=fill_val)
                    atts = var.__dict__.copy()
                    if '_FillValue' in atts:
                        del atts['_FillValue']
                    out_v.setncatts(atts)
                    out_v[:] = var[:]
            
            # Create main variable
            out_var = nc_out.createVariable(variable_name, 'f4', ('time', 'lat', 'lon'), fill_value=-999.0, zlib=True)
            out_var.missing_value = -999.0
            
            # 5. Project and Reconstruct in Chunks
            print(f"Projecting in chunks of {chunk_size}...", flush=True)
            
            total_chunks = (n_time_new + chunk_size - 1) // chunk_size
            
            for i in range(total_chunks):
                start_t = i * chunk_size
                end_t = min((i + 1) * chunk_size, n_time_new)
                
                # Read chunk: (time, lat, lon)
                X_chunk = data_var[start_t:end_t, :, :]
                
                # Flatten spatial: (time, M)
                X_chunk_flat = X_chunk.reshape(X_chunk.shape[0], -1)
                
                # Subtract mean
                # Note: We assume missing values are masked or NaNs.
                # If masked array, fill with NaN for easier handling if needed, 
                # but we need to know where valid data is.
                
                if np.ma.is_masked(X_chunk_flat):
                    mask_chunk = X_chunk_flat.mask
                    data_chunk = X_chunk_flat.data
                else:
                    # Assume NaN or fill_value indicates missing
                    fill_val = getattr(data_var, '_FillValue', -999.0)
                    mask_chunk = (X_chunk_flat == fill_val) | np.isnan(X_chunk_flat)
                    data_chunk = X_chunk_flat
                
                # Subtract mean from valid data
                data_chunk = data_chunk - mean_val
                
                # Prepare output chunk
                X_rec_chunk = np.full(X_chunk_flat.shape, -999.0, dtype=np.float32)
                
                # Iterate over each time step in chunk to solve least squares
                # This could be parallelized, but let's keep it simple first.
                for t in range(X_chunk.shape[0]):
                    valid_idx = ~mask_chunk[t]
                    n_valid = np.sum(valid_idx)
                    
                    if n_valid == 0:
                        continue # No data to project
                    
                    # A = Basis[valid_idx, :] (N_valid, modes)
                    # b = data_chunk[t, valid_idx] (N_valid,)
                    
                    A = Basis[valid_idx, :]
                    b = data_chunk[t, valid_idx]
                    
                    # Solve A v = b for v
                    # v, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                    # Using normal equation for speed if N_valid >> modes: v = (A.T A)^-1 A.T b
                    # Or just lstsq
                    
                    try:
                         v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                         
                         # Reconstruct full field: x_rec = Basis * v + mean
                         x_rec = np.dot(Basis, v) + mean_val
                         X_rec_chunk[t, :] = x_rec
                    except Exception as err:
                        print(f"Error in projection at step {start_t+t}: {err}")
                
                # Reshape back to (time, lat, lon)
                X_rec_chunk_3d = X_rec_chunk.reshape(X_chunk.shape)
                
                # Apply land mask? 
                # Ideally we should apply the original static mask if we have it, 
                # but here we just fill gaps. The result covers the whole domain.
                # If we want to mask land, we should pass the mask file.
                # For now, we leave it as is (filled).
                
                out_var[start_t:end_t, :, :] = X_rec_chunk_3d
                print(f"Projected chunk {i+1}/{total_chunks}", flush=True)
                
        print("Projection complete.")
        
    except Exception as e:
        sys.exit(f"Error writing output file: {e}")
    finally:
        nc_eof.close()
        nc_new.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct DINEOF output or project new data.")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Reconstruct command
    rec_parser = subparsers.add_parser('reconstruct', help='Reconstruct from EOFs')
    rec_parser.add_argument("--eof", required=True, help="Path to eof.nc")
    rec_parser.add_argument("--mean", required=True, help="Path to meandata.val")
    rec_parser.add_argument("--mask", required=True, help="Path to original mask/input file")
    rec_parser.add_argument("--output", required=True, help="Path to output NetCDF file")
    rec_parser.add_argument("--var", default="chlor_a", help="Variable name")
    rec_parser.add_argument("--chunk", type=int, default=50, help="Time chunk size")
    
    # Project command
    proj_parser = subparsers.add_parser('project', help='Project new data')
    proj_parser.add_argument("--eof", required=True, help="Path to eof.nc")
    proj_parser.add_argument("--mean", required=True, help="Path to meandata.val")
    proj_parser.add_argument("--new", required=True, help="Path to new data NetCDF file")
    proj_parser.add_argument("--output", required=True, help="Path to output NetCDF file")
    proj_parser.add_argument("--var", default="chlor_a", help="Variable name")
    proj_parser.add_argument("--chunk", type=int, default=50, help="Time chunk size")

    args = parser.parse_args()
    
    if args.command == 'reconstruct':
        reconstruct_dineof(args.eof, args.mean, args.mask, args.output, args.var, args.chunk)
    elif args.command == 'project':
        project_new_data(args.eof, args.mean, args.new, args.output, args.var, args.chunk)
    else:
        parser.print_help()

