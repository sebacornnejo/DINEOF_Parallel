# DINEOF Parallel Version

This repository contains a modified, parallelized version of the **Data Interpolating Empirical Orthogonal Functions (DINEOF)** software.

**Original Repository:** [https://github.com/aida-alvera/DINEOF/tree/v2.0.0](https://github.com/aida-alvera/DINEOF/tree/v2.0.0)

This modification is designed to accelerate the interpolation process on multi-core systems by leveraging **OpenMP** for parallel loops and **OpenBLAS** for optimized matrix operations.

---

## 🛠️ Detailed Code Changes

The following modifications were made to the original v2.0.0 source code to enable parallel execution.

### 1. Matrix Multiplication Optimization (`ssvd_lancz.F`)
**File:** `DINEOF_Parallel/ssvd_lancz.F`
**Location:** Line ~217 (in `subroutine ssvd_lancz`)

| Original Code | Modified Code | Reason |
| :--- | :--- | :--- |
| `B = matmul(transpose(A),A)` | `call sgemm('T', 'N', n, n, m, 1.0, A, maxm, A, maxm, 0.0, B, n)` | **Performance:** The intrinsic `matmul` is often single-threaded. We replaced it with `sgemm` (Single Precision General Matrix Multiply) from the **BLAS** library, which is highly optimized and multi-threaded. This is the most computationally expensive step in the Lanczos solver. |

### 2. Parallel Reconstruction Loop (`dineof_utils.F90`)
**File:** `DINEOF_Parallel/dineof_utils.F90`
**Location:** Lines ~26-36 (in `subroutine valsvd`)

| Change | Code Added | Reason |
| :--- | :--- | :--- |
| **Added OpenMP Directive** | `!$OMP PARALLEL DO PRIVATE(VAL, K)` | **Parallelization:** This loop reconstructs missing values for every point. Since each point is independent, we distribute the work across multiple CPU cores. |
| **Added End Directive** | `!$OMP END PARALLEL DO` | Marks the end of the parallel region. |

### 3. Parallel Convergence Check (`dineof.F90`)
**File:** `DINEOF_Parallel/dineof.F90`
**Location:** Lines ~548-554 (Main program loop)

| Change | Code Added | Reason |
| :--- | :--- | :--- |
| **Added OpenMP Directive** | `!$OMP PARALLEL DO REDUCTION(+:VAL) PRIVATE(DIFF)` | **Parallelization & Reduction:** This loop calculates the difference between iterations to check for convergence. We use a `REDUCTION(+:VAL)` to safely sum the error across all threads without race conditions. |
| **Added End Directive** | `!$OMP END PARALLEL DO` | Marks the end of the parallel region. |

---

## 🚀 How to Run

This version is designed to run on **Linux**, **macOS**, and **Windows (via WSL)**.

### Prerequisites
1.  **Fortran Compiler:** `gfortran` (GNU Fortran)
2.  **OpenBLAS Library:** For optimized matrix math.
3.  **NetCDF Library:** For reading/writing `.nc` files.
4.  **Arpack Library:** For solving eigenvalue problems.

### 🐧 Linux / 🪟 Windows (WSL - Ubuntu)

1.  **Install Dependencies:**
    ```bash
    sudo apt-get update
    sudo apt-get install gfortran libopenblas-dev libnetcdf-dev libnetcdff-dev libarpack2-dev
    ```

2.  **Compile:**
    Navigate to the `DINEOF_Parallel` directory and run:
    ```bash
    make clean
    make
    ```

3.  **Run:**
    Use the provided wrapper script in the parent directory to run with all cores enabled:
    ```bash
    ./run_dineof_parallel.sh
    ```
    *This script automatically sets `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` to the maximum available cores.*

### 🍎 macOS (Apple Silicon/Intel)

1.  **Install Dependencies (via Homebrew):**
    ```bash
    brew install gcc openblas netcdf arpack
    ```

2.  **Configure Paths:**
    You may need to tell the compiler where Homebrew installed the libraries. Edit `DINEOF_Parallel/config.mk` if necessary, or ensure your environment variables are set:
    ```bash
    export LDFLAGS="-L/opt/homebrew/opt/openblas/lib -L/opt/homebrew/opt/netcdf/lib -L/opt/homebrew/opt/arpack/lib"
    export CPPFLAGS="-I/opt/homebrew/opt/openblas/include -I/opt/homebrew/opt/netcdf/include -I/opt/homebrew/opt/arpack/include"
    ```

3.  **Compile:**
    ```bash
    cd DINEOF_Parallel
    make clean
    make
    ```

4.  **Run:**
    ```bash
    ./run_dineof_parallel.sh
    ```

## Reconstruction and Projection

If the DINEOF process finishes but the output file is corrupted (e.g., due to configuration errors), or if you want to project new data onto the existing EOFs, use the `reconstruct_dineof.py` script.

### Reconstruction
To reconstruct the output from `eof.nc` and `meandata.val`:

```bash
/home/sebastian.cornejo/miniconda3/envs/dineof_env/bin/python3 reconstruct_dineof.py reconstruct \
    --eof DINEOF_Parallel/Output/eof.nc \
    --mean DINEOF_Parallel/Output/meandata.val \
    --mask DINEOF_input_v02_mask2d_1764175426.nc \
    --output DINEOF_Parallel/Output/dineof_reconstructed.nc \
    --chunk 100
```

### Projection (New Data)
To interpolate new data (e.g., new time steps) using the existing EOF basis:

```bash
/home/sebastian.cornejo/miniconda3/envs/dineof_env/bin/python3 reconstruct_dineof.py project \
    --eof DINEOF_Parallel/Output/eof.nc \
    --mean DINEOF_Parallel/Output/meandata.val \
    --new path/to/new_data.nc \
    --output path/to/projected_output.nc \
    --chunk 100
```

---

## ⚙️ Configuration

- **Input Parameters:** Controlled by `dineof.init`.
- **Output Directory:** Results are saved to `DINEOF_Parallel/Output/`.
- **Logs:** Execution progress is written to `DINEOF_Parallel/dineof_parallel.log`.

## ⚠️ Known Warnings

You may see messages like:
`sh: 1: getfattr: not found`
These are harmless system warnings related to file attributes on some Linux filesystems and **do not affect the calculation**.
