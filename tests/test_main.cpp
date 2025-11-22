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
    std::cout << "=== Kyber Performance: Naive vs NTT(Scalar) vs NTT(AVX2) ===\n";
    std::cout << "Parameters: N=256, q=3329\n\n";
    init_tables();

    srand(12345);
    std::vector<i16> a(N), b(N);
    for(int i=0; i<N; i++) { 
        a[i] = rand() % Q; 
        b[i] = rand() % Q; 
    }

    // 1. 执行计算
    auto res_naive = naive_multiply(a, b);
    auto res_scalar = poly_multiply_cpp(a, b);
    auto res_avx    = poly_multiply_avx(a, b);

    // 2. 正确性检查
    std::cout << "1. Checking Correctness...\n";
    
    // Check Scalar vs Naive
    bool pass_scalar = true;
    for(int i=0; i<N; i++) {
        if(res_scalar[i] != res_naive[i]) { pass_scalar = false; break; }
    }
    std::cout << "   Scalar NTT vs Naive: " << (pass_scalar ? "[PASS]" : "[FAIL]") << "\n";

    // Check AVX vs Naive (最重要的检查)
    bool pass_avx = true;
    for(int i=0; i<N; i++) {
        if(res_avx[i] != res_naive[i]) { 
            std::cout << "   Mismatch at " << i << ": AVX=" << res_avx[i] << " Naive=" << res_naive[i] << "\n";
            pass_avx = false; 
            break; 
        }
    }
    std::cout << "   AVX2 NTT   vs Naive: " << (pass_avx ? "[PASS]" : "[FAIL]") << "\n\n";

    if (!pass_scalar || !pass_avx) return 1;

    // 3. Performance Benchmarking
    const int ITER = 50000;
    uint64_t start, end;

    // A. Naive (跑少一点，太慢)
    start = cpucycles();
    for(int i=0; i<500; i++) naive_multiply(a, b);
    end = cpucycles();
    double cyc_naive = (double)(end - start) / 500.0;

    // B. NTT Scalar
    start = cpucycles();
    for(int i=0; i<ITER; i++) poly_multiply_cpp(a, b);
    end = cpucycles();
    double cyc_scalar = (double)(end - start) / ITER;

    // C. NTT AVX2
    start = cpucycles();
    for(int i=0; i<ITER; i++) poly_multiply_avx(a, b);
    end = cpucycles();
    double cyc_avx = (double)(end - start) / ITER;

    // Report
    std::cout << "2. Speed Benchmark (Cycles)\n";
    std::cout << "---------------------------------------------------\n";
    std::cout << std::left << std::setw(20) << "Method" << std::setw(15) << "Cycles" << "Speedup" << std::endl;
    std::cout << "---------------------------------------------------\n";
    std::cout << std::left << std::setw(20) << "Naive O(N^2)" << std::setw(15) << (int)cyc_naive << "1.00x" << std::endl;
    std::cout << std::left << std::setw(20) << "NTT (Scalar)" << std::setw(15) << (int)cyc_scalar << std::fixed << std::setprecision(2) << (cyc_naive / cyc_scalar) << "x" << std::endl;
    std::cout << std::left << std::setw(20) << "NTT (AVX2)" << std::setw(15) << (int)cyc_avx << std::fixed << std::setprecision(2) << (cyc_naive / cyc_avx) << "x" << std::endl;
    std::cout << "---------------------------------------------------\n";

    return 0;
}