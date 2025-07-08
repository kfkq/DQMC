# DQMC
Simple Determinant Quantum Monte Carlo (DQMC) Implementation

Philosophy: A simple codebase without over-engineering or excessive dependencies - easy to read and flexible enough to implement your own DQMC for research purposes.

How to use:

compilation

    ```bash
    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make
    cd ..
    ```

running on parameters.toml

    ```bash
    cd examples
    mpirun -np 128 ../build/dqmc_hubbard
    ```