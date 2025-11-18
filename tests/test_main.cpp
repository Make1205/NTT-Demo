#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include "ntt.h"

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

inline uint64_t cpucycles() { return __rdtsc(); }

using namespace KyberLab;

int main() {
    std::cout << "=== Kyber Performance: Naive vs NTT(Scalar) vs NTT(AVX2-Sim) ===\n";
    std::cout << "Parameters: N=256, q=3329\n\n";
    init_tables();

    // 准备数据
    std::vector<i16> a(N), b(N);
    for(int i=0; i<N; i++) { a[i] = rand() % Q; b[i] = rand() % Q; }

    // --- 1. Correctness Check (Naive vs Scalar NTT) ---
    std::cout << "1. Checking Correctness (Scalar NTT vs Naive)...\n";
    auto res_ntt = poly_multiply_cpp(a, b);
    auto res_naive = naive_multiply(a, b);
    bool pass = true;
    for(int i=0; i<N; i++) {
        i16 v1 = (res_ntt[i] < 0 ? res_ntt[i]+Q : res_ntt[i]);
        i16 v2 = (res_naive[i] < 0 ? res_naive[i]+Q : res_naive[i]);
        if(v1 != v2) { pass = false; break; }
    }
    if(pass) std::cout << "   [PASS] Scalar NTT is correct.\n\n";
    else { std::cout << "   [FAIL] Scalar NTT mismatch!\n"; return 1; }

    // --- 2. Performance Benchmarking ---
    const int ITER = 50000;
    uint64_t start, end;

    // A. Naive (由于太慢，只跑少量次数)
    // ------------------------------------------------
    start = cpucycles();
    for(int i=0; i<1000; i++) naive_multiply(a, b);
    end = cpucycles();
    double cyc_naive = (double)(end - start) / 1000.0;

    // B. NTT Scalar (C++)
    // ------------------------------------------------
    // Warmup
    for(int i=0; i<1000; i++) poly_multiply_cpp(a, b);
    
    start = cpucycles();
    for(int i=0; i<ITER; i++) poly_multiply_cpp(a, b);
    end = cpucycles();
    double cyc_cpp = (double)(end - start) / ITER;

    // C. NTT AVX2 (Simulation)
    // ------------------------------------------------
    // Warmup
    for(int i=0; i<1000; i++) poly_multiply_avx(a, b);

    start = cpucycles();
    for(int i=0; i<ITER; i++) poly_multiply_avx(a, b);
    end = cpucycles();
    double cyc_avx = (double)(end - start) / ITER;

    // --- 3. Report ---
    std::cout << "2. Speed Benchmark (Cycles)\n";
    std::cout << "---------------------------------------------------\n";
    std::cout << std::left << std::setw(20) << "Method" 
              << std::setw(15) << "Cycles" 
              << "Speedup (vs Naive)" << std::endl;
    std::cout << "---------------------------------------------------\n";
    
    std::cout << std::left << std::setw(20) << "Naive O(N^2)" 
              << std::setw(15) << (int)cyc_naive 
              << "1.00x" << std::endl;

    std::cout << std::left << std::setw(20) << "NTT (Scalar)" 
              << std::setw(15) << (int)cyc_cpp 
              << std::fixed << std::setprecision(2) << (cyc_naive / cyc_cpp) << "x" << std::endl;

    std::cout << std::left << std::setw(20) << "NTT (AVX2 Mix)*" 
              << std::setw(15) << (int)cyc_avx 
              << std::fixed << std::setprecision(2) << (cyc_naive / cyc_avx) << "x" << std::endl;
    
    std::cout << "---------------------------------------------------\n";
    std::cout << "* Note: AVX2 implementation in this demo accelerates\n";
    std::cout << "  the Point-Wise Multiplication stage to demonstrate throughput.\n";

    return 0;
}