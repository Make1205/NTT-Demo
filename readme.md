# 🚀 Kyber-Like NTT Demo: Performance Comparison (Naive vs Scalar vs AVX2)

This project demonstrates the Number Theoretic Transform (NTT) implementation using parameters similar to the Kyber Post-Quantum Cryptography scheme ($N=256, Q=3329, \zeta=17$). The primary goal is to provide a performance benchmark comparing polynomial multiplication across three implementation methods: Naive $O(N^2)$, standard $O(N \log N)$ NTT, and a simulated AVX2-accelerated NTT.

## 🎯 Features

* **Parameter Settings:** Uses core parameters relevant to Kyber, specifically $N=256$ and $Q=3329$.
* **Algorithms Implemented:**
    1.  **Naive $O(N^2)$ Multiplication:** Serves as the performance baseline.
    2.  **NTT Scalar C++ Version (Scalar):** Implements the $O(N \log N)$ polynomial multiplication using standard C++ for the NTT and Inverse NTT stages.
    3.  **NTT AVX2 Mixed Version (AVX2 Mix):** Accelerates the **Point-Wise Multiplication (BaseMul)** stage using AVX2 intrinsics to demonstrate SIMD throughput. The NTT/INTT stages rely on the C++ scalar implementation.
* **Benchmarking:** Uses the `__rdtsc()` intrinsic for precise CPU cycle counting performance measurement.

## 🛠️ Build and Run

The project uses CMake for building.

### Dependencies

* A C++17 compatible compiler (e.g., GCC, Clang, MSVC).
* An **x86-64 processor supporting AVX2 instructions** (required for the AVX2 benchmark).
* CMake (version $\ge 3.10$).

### Compilation Steps

1.  Clone the repository.
2.  Navigate to the project root directory.
3.  Build using CMake:

    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

### Running the Benchmark

Execute the generated test program `ntt_test`:

```bash
./ntt_test