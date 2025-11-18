#pragma once
#include <vector>
#include <cstdint>
#include <immintrin.h> // 核心头文件

using i16 = int16_t;
using i32 = int32_t;

namespace KyberLab {
    constexpr int N = 256;
    constexpr i16 Q = 3329;
    constexpr i16 ZETA = 17;

    void init_tables();

    // --- 纯 C++ 版本 ---
    std::vector<i16> poly_multiply_cpp(const std::vector<i16>& a, const std::vector<i16>& b);
    
    // --- AVX2 版本 ---
    // 为了内存对齐，AVX2 版本最好直接操作指针，但为了保持接口一致，我们内部处理
    std::vector<i16> poly_multiply_avx(const std::vector<i16>& a, const std::vector<i16>& b);

    // 朴素版本
    std::vector<i16> naive_multiply(const std::vector<i16>& a, const std::vector<i16>& b);
}