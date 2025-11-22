#include "ntt.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <immintrin.h>

namespace KyberLab {

    // =============================================================
    // 1. Constants & Tables
    // =============================================================
    // const int16_t Q = 3329;
    const int16_t QINV = -3327; 
    const int16_t MONT = 2285;  
    const int16_t R2 = 1353;    

    // AVX2 Constants
    const __m256i AVX_Q = _mm256_set1_epi16(Q);
    const __m256i AVX_QINV = _mm256_set1_epi16(QINV);
    const __m256i AVX_R2 = _mm256_set1_epi16(R2);
    const __m256i AVX_ZERO = _mm256_setzero_si256();

    // [Cite: ntt.c]
    const int16_t zetas[128] = {
      -1044,  -758,  -359, -1517,  1493,  1422,   287,   202,
       -171,   622,  1577,   182,   962, -1202, -1474,  1468,
        573, -1325,   264,   383,  -829,  1458, -1602,  -130,
       -681,  1017,   732,   608, -1542,   411,  -205, -1571,
       1223,   652,  -552,  1015, -1293,  1491,  -282, -1544,
        516,    -8,  -320,  -666, -1618, -1162,   126,  1469,
       -853,   -90,  -271,   830,   107, -1421,  -247,  -951,
       -398,   961, -1508,  -725,   448, -1065,   677, -1275,
      -1103,   430,   555,   843, -1251,   871,  1550,   105,
        422,   587,   177,  -235,  -291,  -460,  1574,  1653,
       -246,   778,  1159,  -147,  -777,  1483,  -602,  1119,
      -1590,   644,  -872,   349,   418,   329,  -156,   -75,
        817,  1097,   603,   610,  1322, -1285, -1465,   384,
      -1215,  -136,  1218, -1335,  -874,   220, -1187, -1659,
      -1185, -1530, -1278,   794, -1510,  -854,  -870,   478,
       -108,  -308,   996,   991,   958, -1460,  1522,  1628
    };

    void init_tables() {} 

    // =============================================================
    // 2. Core Arithmetic (Scalar & AVX)
    // =============================================================

    // Scalar Montgomery
    inline int16_t montgomery_reduce(int32_t a) {
        int16_t u = (int16_t)(a * (int32_t)QINV);
        int32_t t = (int32_t)((int64_t)u * Q);
        t = a - t;
        t >>= 16;
        return (int16_t)t;
    }

    inline int16_t barrett_reduce(int16_t a) {
        int16_t v = ((int32_t)a * 20159) >> 26;
        v = a - v * Q;
        return v;
    }

    inline int16_t fqmul(int16_t a, int16_t b) {
        return montgomery_reduce((int32_t)a * b);
    }

    // AVX2 Montgomery [a * b * R^-1 mod Q]
    inline __m256i fqmul_avx(__m256i a, __m256i b) {
        __m256i lo = _mm256_mullo_epi16(a, b);
        __m256i hi = _mm256_mulhi_epi16(a, b);
        __m256i t = _mm256_mullo_epi16(lo, AVX_QINV);
        __m256i t_times_q_hi = _mm256_mulhi_epi16(t, AVX_Q);
        return _mm256_sub_epi16(hi, t_times_q_hi);
    }

    // =============================================================
    // 3. Transform Logic (Scalar - Verified Correct)
    // =============================================================

    //
    void ntt(int16_t* r) {
        unsigned int len, start, j, k;
        int16_t t, zeta;
        k = 1;
        for(len = 128; len >= 2; len >>= 1) {
            for(start = 0; start < 256; start = j + len) {
                zeta = zetas[k++];
                for(j = start; j < start + len; j++) {
                    t = fqmul(zeta, r[j + len]);
                    r[j + len] = r[j] - t;
                    r[j] = r[j] + t;
                }
            }
        }
    }

    //
    void invntt(int16_t* r) {
        unsigned int start, len, j, k;
        int16_t t, zeta;
        const int16_t f = 1441; 
        k = 127;
        for(len = 2; len <= 128; len <<= 1) {
            for(start = 0; start < 256; start = j + len) {
                zeta = zetas[k--];
                for(j = start; j < start + len; j++) {
                    t = r[j];
                    r[j] = barrett_reduce(t + r[j + len]);
                    r[j + len] = r[j + len] - t;
                    r[j + len] = fqmul(zeta, r[j + len]);
                }
            }
        }
        for(j = 0; j < 256; j++)
            r[j] = fqmul(r[j], f);
    }

    // - Used only for scalar fallback or reference
    void basemul(int16_t r[2], const int16_t a[2], const int16_t b[2], int16_t zeta) {
        r[0]  = fqmul(a[1], b[1]);
        r[0]  = fqmul(r[0], zeta);
        r[0] += fqmul(a[0], b[0]);
        r[1]  = fqmul(a[0], b[1]);
        r[1] += fqmul(a[1], b[0]);
    }

    // =============================================================
    // 4. AVX2 Accelerators
    // =============================================================

    // 1. Input Conversion (O(N))
    void poly_tomont_avx(int16_t* r) {
        for(int i=0; i<256; i+=16) {
            __m256i a = _mm256_loadu_si256((__m256i*)&r[i]);
            a = fqmul_avx(a, AVX_R2);
            _mm256_storeu_si256((__m256i*)&r[i], a);
        }
    }

    // 2. Base Multiplication (O(N))
    void poly_basemul_avx(int16_t* r, const int16_t* a, const int16_t* b) {
        for(int i=0; i<256/16; i++) { 
            int16_t z0 = zetas[64 + i*4 + 0];
            int16_t z1 = zetas[64 + i*4 + 1];
            int16_t z2 = zetas[64 + i*4 + 2];
            int16_t z3 = zetas[64 + i*4 + 3];

            // Construct zeta vector for odd positions
            // Pair 7: -z3, Pair 6: z3, ..., Pair 0: z0
            __m256i vzeta = _mm256_set_epi16(
                0, -z3, 
                0,  z3, 
                0, -z2, 
                0,  z2, 
                0, -z1, 
                0,  z1, 
                0, -z0, 
                0,  z0
            );

            __m256i va = _mm256_loadu_si256((__m256i*)&a[i*16]);
            __m256i vb = _mm256_loadu_si256((__m256i*)&b[i*16]);

            __m256i vprod = fqmul_avx(va, vb);

            // Swap adjacent 16-bit values
            __m256i va_swap = _mm256_shufflelo_epi16(va, 0xB1);
            va_swap = _mm256_shufflehi_epi16(va_swap, 0xB1);
            
            __m256i vprod_cross = fqmul_avx(va_swap, vb);

            __m256i vprod_swap = _mm256_shufflelo_epi16(vprod, 0xB1);
            vprod_swap = _mm256_shufflehi_epi16(vprod_swap, 0xB1);
            
            __m256i term_zeta = fqmul_avx(vprod_swap, vzeta);
            __m256i r_even = _mm256_add_epi16(vprod, term_zeta);

            __m256i vprod_cross_swap = _mm256_shufflelo_epi16(vprod_cross, 0xB1);
            vprod_cross_swap = _mm256_shufflehi_epi16(vprod_cross_swap, 0xB1);
            
            __m256i r_odd = _mm256_add_epi16(vprod_cross, vprod_cross_swap);

            // Merge even/odd results
            __m256i res = _mm256_blend_epi16(r_even, r_odd, 0xAA);

            _mm256_storeu_si256((__m256i*)&r[i*16], res);
        }
    }

    // 3. Output Conversion & Scaling (O(N))
    // Applies TWO montgomery reductions to remove R^2 factor
    void poly_final_map_avx(int16_t* r) {
        const __m256i v_one = _mm256_set1_epi16(1); 

        for(int i=0; i<256; i+=16) {
            __m256i val = _mm256_loadu_si256((__m256i*)&r[i]);
            
            // Reduction 1: x * R^-1
            val = fqmul_avx(val, v_one);
            // Reduction 2: x * R^-2
            val = fqmul_avx(val, v_one);
            
            // Normalize to positive [0, Q)
            __m256i mask = _mm256_cmpgt_epi16(AVX_ZERO, val); 
            __m256i add_q = _mm256_and_si256(mask, AVX_Q);
            val = _mm256_add_epi16(val, add_q);
            
            _mm256_storeu_si256((__m256i*)&r[i], val);
        }
    }

    // =============================================================
    // 5. Wrappers
    // =============================================================

    std::vector<i16> poly_multiply_cpp(const std::vector<i16>& a, const std::vector<i16>& b) {
        std::vector<i16> ta = a;
        std::vector<i16> tb = b;
        std::vector<i16> res(N);

        // 1. To Mont
        for(int i=0; i<N; i++) { ta[i] = fqmul(ta[i], R2); tb[i] = fqmul(tb[i], R2); }
        
        // 2. NTT
        ntt(ta.data());
        ntt(tb.data());
        
        // 3. BaseMul
        for(int i = 0; i < N / 4; i++) {
            basemul(&res[4*i], &ta[4*i], &tb[4*i], zetas[64+i]);
            basemul(&res[4*i+2], &ta[4*i+2], &tb[4*i+2], -zetas[64+i]);
        }
        
        // 4. InvNTT (Output is val * 128^-1 * R^2)
        invntt(res.data());

        // 5. Final fix: Remove R^2
        for(int i=0; i<N; i++) {
            int16_t val = res[i];
            val = montgomery_reduce((int32_t)val); // Reduce 1
            val = montgomery_reduce((int32_t)val); // Reduce 2
            res[i] = (val < 0) ? val + Q : val;
        }
        return res;
    }

    // --- Fully Accelerated AVX2 Version ---
    std::vector<i16> poly_multiply_avx(const std::vector<i16>& a, const std::vector<i16>& b) {
        std::vector<i16> ta = a;
        std::vector<i16> tb = b;
        std::vector<i16> res(N);

        // 1. AVX Tomont
        poly_tomont_avx(ta.data());
        poly_tomont_avx(tb.data());

        // 2. Scalar NTT (Keep scalar for correctness)
        ntt(ta.data());
        ntt(tb.data());

        // 3. AVX BaseMul
        poly_basemul_avx(res.data(), ta.data(), tb.data());

        // 4. Scalar InvNTT
        invntt(res.data());

        // 5. AVX Final Map (Double Reduction)
        poly_final_map_avx(res.data());

        return res;
    }

    std::vector<i16> naive_multiply(const std::vector<i16>& a, const std::vector<i16>& b) {
        std::vector<i16> res(N, 0);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int idx = i + j;
                int32_t prod = (int32_t)a[i] * b[j];
                if (idx < N) {
                    res[idx] = (int16_t)(((int32_t)res[idx] + prod) % Q); 
                } else {
                    res[idx - N] = (int16_t)(((int32_t)res[idx - N] - prod) % Q);
                }
            }
        }
        for(int i=0; i<N; i++) {
            if(res[i] < 0) res[i] += Q;
        }
        return res;
    }
}