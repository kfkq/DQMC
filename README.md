# DQMC
Lightweight Determinant Quantum Monte Carlo (DQMC) Implementation

Philosophy: Lightweight enough to understand and start your own DQMC project.

## Requirements

To compile and run this project, you need the following dependencies installed on your system:

  * **CMake**: Version 3.10 or higher.
  * **A C++ Compiler**: Supporting the C++17 standard.
  * **MPI**: An MPI implementation like OpenMPI or MPICH.
  * **Armadillo**: A C++ linear algebra library.
  * **Intel Math Kernel Library (MKL)**
  * **HDF5**
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
- [x] Global update through replica exchange
- [ ] Checkerboard implementation
- [ ] Optimization through delayed update and submatrix update (SciPost Phys. 18, 055)
- [ ] Canonical ensemble DQMC (Phys. Rev. E 107, 055302)
- [ ] Sign Problem mitigation with self-consistent constraint (Phys. Rev. B 99, 045108)