# DQMC
Simple Determinant Quantum Monte Carlo (DQMC) Implementation

Philosophy: A simple codebase without over-engineering or excessive dependencies - easy to read and flexible enough to implement your own DQMC for research purposes.

Currently in heavy development and testing for new modern cpp implementation

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
mpirun -np 128 ../build/dqmc
```
Final error estimation using jacknife analysis
```bash
python ../scripts/analysis.py
```