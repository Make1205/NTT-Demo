#include "ntt.h"
#include <vector>
#include <immintrin.h>
#include <iostream>
#include <iomanip>

namespace KyberLab {

    // =============================================================
    // 0. Configuration
    // =============================================================
    // [FINAL MODE] Performance benchmarking only
    static bool ENABLE_DEBUG = false; 

    void print_debug(const char* label, const int16_t* r) {
        if (!ENABLE_DEBUG) return;
        std::cout << std::left << std::setw(15) << label << ": [";
        for(int i=0; i<8; i++) std::cout << r[i] << ", ";
        std::cout << "...]\n";
    }

    // =============================================================
    // 1. Constants
    // =============================================================
    const int16_t QINV = -3327; 
    const int16_t R2 = 1353;    
    const int16_t F_FACTOR = 1441; 
    
    // AVX Constants
    const __m256i AVX_Q = _mm256_set1_epi16(3329);
    const __m256i AVX_QINV = _mm256_set1_epi16(-3327);
    const __m256i AVX_R2 = _mm256_set1_epi16(1353);
    const __m256i AVX_F = _mm256_set1_epi16(1441); 
    const __m256i AVX_BARRETT_MUL = _mm256_set1_epi16(20159); 

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
    // 2. High-Performance Math
    // =============================================================

    inline int16_t montgomery_reduce(int32_t a) {
        int16_t u = (int16_t)(a * (int32_t)QINV);
        int32_t t = (int32_t)((int64_t)u * 3329);
        t = a - t;
        t >>= 16;
        return (int16_t)t;
    }
    
    inline int16_t fqmul(int16_t a, int16_t b) {
        return montgomery_reduce((int32_t)a * b);
    }

    inline int16_t barrett_reduce(int16_t a) {
        int16_t v = ((int32_t)a * 20159) >> 26;
        v = a - v * 3329;
        return v;
    }

    inline __m256i montgomery_mul_avx(__m256i a, __m256i b) {
        __m256i t_low = _mm256_mullo_epi16(a, b);
        __m256i k = _mm256_mullo_epi16(t_low, AVX_QINV);
        __m256i m = _mm256_mulhi_epi16(k, AVX_Q);
        __m256i t_high = _mm256_mulhi_epi16(a, b);
        return _mm256_sub_epi16(t_high, m);
    }

    inline __m256i barrett_reduce_avx(__m256i a) {
        __m256i v = _mm256_mulhi_epi16(a, AVX_BARRETT_MUL);
        v = _mm256_srai_epi16(v, 10); 
        __m256i vq = _mm256_mullo_epi16(v, AVX_Q);
        return _mm256_sub_epi16(a, vq);
    }

    // =============================================================
    // 3. Robust AVX2 NTT
    // =============================================================

    inline void butterfly_avx(__m256i& a, __m256i& b, __m256i zeta) {
        __m256i t = montgomery_mul_avx(b, zeta);
        b = _mm256_sub_epi16(a, t);
        a = _mm256_add_epi16(a, t);
    }

    void ntt_avx(int16_t* r) {
        // --- Phase 1: AVX for Large Layers (128, 64, 32, 16) ---
        __m256i zeta = _mm256_set1_epi16(zetas[1]);
        for(int i=0; i<128; i+=16) {
            __m256i a = _mm256_loadu_si256((__m256i*)&r[i]);
            __m256i b = _mm256_loadu_si256((__m256i*)&r[i+128]);
            butterfly_avx(a, b, zeta);
            _mm256_storeu_si256((__m256i*)&r[i], a);
            _mm256_storeu_si256((__m256i*)&r[i+128], b);
        }

        int k = 2;
        for(int len=64; len>=16; len/=2) {
             for(int start=0; start<256; start+=2*len) {
                 zeta = _mm256_set1_epi16(zetas[k++]);
                 for(int i=start; i<start+len; i+=16) {
                    __m256i a = _mm256_loadu_si256((__m256i*)&r[i]);
                    __m256i b = _mm256_loadu_si256((__m256i*)&r[i+len]);
                    butterfly_avx(a, b, zeta);
                    _mm256_storeu_si256((__m256i*)&r[i], a);
                    _mm256_storeu_si256((__m256i*)&r[i+len], b);
                 }
             }
        }

        // --- Phase 2: Scalar for Small Layers (8, 4, 2) ---
        // Robust Fallback (Correctness Priority)
        for(int len = 8; len >= 2; len >>= 1) {
            for(int start = 0; start < 256; start += 2*len) {
                int16_t z = zetas[k++];
                for(int j = start; j < start + len; j++) {
                    int16_t t = fqmul(z, r[j + len]);
                    r[j + len] = r[j] - t;
                    r[j] = r[j] + t;
                }
            }
        }
    }

    // =============================================================
    // 4. Robust AVX2 InvNTT
    // =============================================================
    
    inline void inv_butterfly_avx(__m256i& a, __m256i& b, __m256i zeta) {
        __m256i sum = _mm256_add_epi16(a, b);
        __m256i diff = _mm256_sub_epi16(b, a); // b - a
        a = barrett_reduce_avx(sum);           
        b = montgomery_mul_avx(diff, zeta);
    }

    void invntt_avx(int16_t* r) {
        int k = 127;
        
        // --- Phase 1: Scalar for Small Layers (2, 4) ---
        for(int len = 2; len <= 4; len <<= 1) {
            for(int start = 0; start < 256; start += 2*len) {
                int16_t z = zetas[k--];
                for(int j = start; j < start + len; j++) {
                    int16_t t = r[j];
                    r[j] = barrett_reduce(t + r[j + len]);
                    r[j + len] = r[j + len] - t; // b - a
                    r[j + len] = fqmul(z, r[j + len]);
                }
            }
        }

        // --- Phase 2: AVX Layer 8 (Shuffle) ---
        for(int start = 0; start < 256; start += 16) {
             __m256i zeta = _mm256_set1_epi16(zetas[k--]);
             __m256i v = _mm256_loadu_si256((__m256i*)&r[start]);
             
             __m256i b_swapped = _mm256_permute2x128_si256(v, v, 0x01); 
             __m256i a = v; 
             
             __m256i sum = _mm256_add_epi16(a, b_swapped);
             __m256i diff = _mm256_sub_epi16(b_swapped, a); 
             
             a = barrett_reduce_avx(sum);          
             b_swapped = montgomery_mul_avx(diff, zeta); 
             
             __m256i final_v = _mm256_inserti128_si256(a, _mm256_castsi256_si128(b_swapped), 1);
             _mm256_storeu_si256((__m256i*)&r[start], final_v);
        }

        // --- Phase 3: AVX Large Layers (16, 32, 64, 128) ---
        for(int len = 16; len <= 128; len <<= 1) {
            for(int start = 0; start < 256; start += 2*len) {
                __m256i zeta = _mm256_set1_epi16(zetas[k--]);
                for(int j = start; j < start + len; j+=16) {
                    __m256i a = _mm256_loadu_si256((__m256i*)&r[j]);
                    __m256i b = _mm256_loadu_si256((__m256i*)&r[j+len]);
                    
                    inv_butterfly_avx(a, b, zeta);
                    
                    _mm256_storeu_si256((__m256i*)&r[j], a);
                    _mm256_storeu_si256((__m256i*)&r[j+len], b);
                }
            }
        }
        
        // Final Factor Scaling (AVX)
        const __m256i v_f = AVX_F;
        for(int i=0; i<256; i+=16) {
             __m256i a = _mm256_loadu_si256((__m256i*)&r[i]);
             a = montgomery_mul_avx(a, v_f);
             _mm256_storeu_si256((__m256i*)&r[i], a);
        }
    }

    // =============================================================
    // 5. Wrappers & BaseMul
    // =============================================================
    
    inline __m256i swap_adjacent_pairs(__m256i v) {
        v = _mm256_shufflelo_epi16(v, 0xB1);
        v = _mm256_shufflehi_epi16(v, 0xB1);
        return v;
    }

    void poly_basemul_avx(int16_t* r, const int16_t* a, const int16_t* b) {
        for(int i=0; i<256/16; i++) { 
            int16_t z0 = zetas[64 + i*4 + 0];
            int16_t z1 = zetas[64 + i*4 + 1];
            int16_t z2 = zetas[64 + i*4 + 2];
            int16_t z3 = zetas[64 + i*4 + 3];
            __m256i vzeta = _mm256_set_epi16(-z3, -z3, z3, z3, -z2, -z2, z2, z2, -z1, -z1, z1, z1, -z0, -z0, z0, z0);
            __m256i va = _mm256_loadu_si256((__m256i*)&a[i*16]);
            __m256i vb = _mm256_loadu_si256((__m256i*)&b[i*16]);
            __m256i va_swap = swap_adjacent_pairs(va);
            __m256i v_prod_rot = montgomery_mul_avx(va_swap, vb); 
            __m256i r1_vec = _mm256_add_epi16(v_prod_rot, swap_adjacent_pairs(v_prod_rot));
            __m256i v_prod = montgomery_mul_avx(va, vb);
            __m256i v_prod_swapped = swap_adjacent_pairs(v_prod);
            __m256i v_term_zeta = montgomery_mul_avx(v_prod_swapped, vzeta);
            __m256i r0_vec = _mm256_add_epi16(v_prod, v_term_zeta);
            __m256i res = _mm256_blend_epi16(r0_vec, r1_vec, 0xAA);
            _mm256_storeu_si256((__m256i*)&r[i*16], res);
        }
    }

    void poly_tomont_avx(int16_t* r) {
        for(int i=0; i<256; i+=16) {
            __m256i a = _mm256_loadu_si256((__m256i*)&r[i]);
            a = montgomery_mul_avx(a, AVX_R2);
            _mm256_storeu_si256((__m256i*)&r[i], a);
        }
    }

    void ntt(int16_t* r) {
        unsigned int len, start, j, k=1;
        int16_t t, zeta;
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
    void invntt(int16_t* r) {
        unsigned int start, len, j, k=127;
        int16_t t, zeta;
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
        for(j = 0; j < 256; j++) r[j] = fqmul(r[j], F_FACTOR); 
    }

    std::vector<i16> poly_multiply_cpp(const std::vector<i16>& a, const std::vector<i16>& b) {
        std::vector<i16> ta = a, tb = b, res(N);
        for(int i=0; i<N; i++) { ta[i] = fqmul(ta[i], R2); tb[i] = fqmul(tb[i], R2); }
        ntt(ta.data()); 
        if(ENABLE_DEBUG) print_debug("Scalar NTT", ta.data());
        
        ntt(tb.data());
        for(int i = 0; i < N / 4; i++) {
            int16_t r0, r1;
            int16_t z = zetas[64+i];
            r0  = fqmul(ta[4*i+1], tb[4*i+1]);
            r0  = fqmul(r0, z);
            r0 += fqmul(ta[4*i], tb[4*i]);
            r1  = fqmul(ta[4*i], tb[4*i+1]);
            r1 += fqmul(ta[4*i+1], tb[4*i]);
            res[4*i] = r0; res[4*i+1] = r1;
            z = -z;
            r0  = fqmul(ta[4*i+3], tb[4*i+3]);
            r0  = fqmul(r0, z);
            r0 += fqmul(ta[4*i+2], tb[4*i+2]);
            r1  = fqmul(ta[4*i+2], tb[4*i+3]);
            r1 += fqmul(ta[4*i+3], tb[4*i+2]);
            res[4*i+2] = r0; res[4*i+3] = r1;
        }
        if(ENABLE_DEBUG) print_debug("Scalar Base", res.data());

        invntt(res.data());
        if(ENABLE_DEBUG) print_debug("Scalar Inv", res.data());

        for(int i=0; i<N; i++) {
            int16_t val = res[i];
            val = montgomery_reduce((int32_t)val); 
            val = montgomery_reduce((int32_t)val); 
            res[i] = (val < 0) ? val + 3329 : val;
        }
        return res;
    }

    std::vector<i16> poly_multiply_avx(const std::vector<i16>& a, const std::vector<i16>& b) {
        if(ENABLE_DEBUG) std::cout << "\n--- AVX Debug Trace ---\n";

        std::vector<i16> ta = a, tb = b, res(N);
        poly_tomont_avx(ta.data());
        poly_tomont_avx(tb.data());
        
        ntt_avx(ta.data());
        if(ENABLE_DEBUG) print_debug("AVX NTT", ta.data());

        ntt_avx(tb.data());
        
        poly_basemul_avx(res.data(), ta.data(), tb.data()); // AVX
        if(ENABLE_DEBUG) print_debug("AVX Base", res.data());

        invntt_avx(res.data()); 
        if(ENABLE_DEBUG) print_debug("AVX Inv", res.data());
        
        // Final Map AVX
        const __m256i v_one = _mm256_set1_epi16(1); 
        for(int i=0; i<256; i+=16) {
            __m256i val = _mm256_loadu_si256((__m256i*)&res[i]);
            val = montgomery_mul_avx(val, v_one); 
            val = montgomery_mul_avx(val, v_one); 
            __m256i mask = _mm256_cmpgt_epi16(_mm256_setzero_si256(), val); 
            __m256i add_q = _mm256_and_si256(mask, AVX_Q);
            val = _mm256_add_epi16(val, add_q);
            _mm256_storeu_si256((__m256i*)&res[i], val);
        }
        return res;
    }

    std::vector<i16> naive_multiply(const std::vector<i16>& a, const std::vector<i16>& b) {
        std::vector<i16> res(N, 0);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int idx = i + j;
                int32_t prod = (int32_t)a[i] * b[j];
                if (idx < N) res[idx] = (int16_t)(((int32_t)res[idx] + prod) % 3329); 
                else res[idx - N] = (int16_t)(((int32_t)res[idx - N] - prod) % 3329);
            }
        }
        for(int i=0; i<N; i++) if(res[i] < 0) res[i] += 3329;
        return res;
    }
}