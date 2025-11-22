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

// 辅助打印函数
void print_poly(const std::string& label, const std::vector<i16>& p, int len = 16) {
    std::cout << std::left << std::setw(15) << label << ": [";
    for(int i=0; i<len; i++) std::cout << p[i] << (i==len-1?"":", ");
    std::cout << "...]\n";
}

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

    // 2. 正确性检查与DEBUG信息
    std::cout << "1. Checking Correctness...\n";
    
    // 打印前16个值进行对比
    print_poly("Naive", res_naive);
    print_poly("NTT(Scalar)", res_scalar);
    print_poly("NTT(AVX2)", res_avx);

    // Check Scalar vs Naive
    bool pass_scalar = true;
    for(int i=0; i<N; i++) {
        if(res_scalar[i] != res_naive[i]) { 
            std::cout << "   [FAIL] Scalar mismatch at " << i 
                      << ": Expected " << res_naive[i] << ", Got " << res_scalar[i] << "\n";
            pass_scalar = false; 
            break; 
        }
    }
    std::cout << "   Scalar NTT vs Naive: " << (pass_scalar ? "[PASS]" : "[FAIL]") << "\n";

    // Check AVX vs Naive
    bool pass_avx = true;
    for(int i=0; i<N; i++) {
        if(res_avx[i] != res_naive[i]) { 
            std::cout << "   [FAIL] AVX2 mismatch at " << i 
                      << ": Expected " << res_naive[i] << ", Got " << res_avx[i] << "\n";
            pass_avx = false; 
            break; 
        }
    }
    std::cout << "   AVX2 NTT   vs Naive: " << (pass_avx ? "[PASS]" : "[FAIL]") << "\n\n";

    if (!pass_scalar || !pass_avx) {
        std::cout << "!!! Correctness failed. Aborting benchmark.\n";
        return 1;
    }

    // 3. Performance Benchmarking
    const int ITER = 50000;
    uint64_t start, end;

    // A. Naive
    start = cpucycles();
    for(int i=0; i<100; i++) naive_multiply(a, b); // 跑少一点
    end = cpucycles();
    double cyc_naive = (double)(end - start) / 100.0;

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