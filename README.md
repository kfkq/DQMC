# DQMC
Lightweight Determinant Quantum Monte Carlo (DQMC) Implementation

Philosophy: Lightweight enough to understand and start your own DQMC project.

## Requirements

To compile and run this project, you need the following dependencies installed on your system:

  * **CMake**: Version 3.10 or higher.
  * **A C++ Compiler**: Supporting the C++17 standard.
  * **MPI**: An MPI implementation like OpenMPI or MPICH.
  * **Armadillo**: A C++ linear algebra library. This project is configured to link it with MKL.
  * **Intel Math Kernel Library (MKL)**: A library of optimized math routines for scientific, engineering, and financial applications. The `MKLROOT` environment variable must be set to the MKL installation directory.
  * **HDF5**: A library for storing and managing large data.
  * **Python 3**: With the `numpy` and `scipy` libraries installed for the analysis scripts.

## Usage

### Compilation

```bash
cmake -S . -B build
cmake --build build
```

### Examples 
Run DQMC inside folder with `parameters.in`
```bash
cd examples
mpirun -np 42 ../build/dqmc
```
Final error estimation using jacknife analysis
```bash
python ../scripts/analysis.py
```

## TODO
- Global update through replica exchange
- Checkerboard implementation
- Optimization through delayed update and submatrix update (SciPost Phys. 18, 055 (2025))
- Canonical ensemble DQMC (Phys. Rev. E 107, 055302)
- Sign Problem mitigation with self-consistent constraint (Phys. Rev. B 99, 045108)