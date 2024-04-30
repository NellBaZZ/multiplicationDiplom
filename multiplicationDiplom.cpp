#include <immintrin.h>
#include <cstdint>
#include <iostream>


template<int part>
void processVectorHalf(const __m512i& a_vec, const __m512i& b_vec, uint64_t* result, size_t offset) {
    static_assert(part == 0 || part == 1, "Vector half must be 0 or 1");

    __m256i a_half = _mm512_extracti32x8_epi32(a_vec, part);
    __m256i b_half = _mm512_extracti32x8_epi32(b_vec, part);

    __m256i a_hi = _mm256_srli_epi32(a_half, 16);
    __m256i a_lo = _mm256_and_si256(a_half, _mm256_set1_epi32(0xFFFF));
    __m256i b_hi = _mm256_srli_epi32(b_half, 16);
    __m256i b_lo = _mm256_and_si256(b_half, _mm256_set1_epi32(0xFFFF));

    __m256i mul_hi = _mm256_mullo_epi32(a_hi, b_hi);
    __m256i mul_lo = _mm256_mullo_epi32(a_lo, b_lo);
    __m256i mul_cross1 = _mm256_mullo_epi32(a_hi, b_lo);
    __m256i mul_cross2 = _mm256_mullo_epi32(a_lo, b_hi);

    __m256i cross_sum = _mm256_add_epi32(mul_cross1, mul_cross2);
    __m256i cross_sum_shifted = _mm256_slli_epi32(cross_sum, 16);

    __m256i part_result = _mm256_add_epi32(mul_hi, cross_sum_shifted);
    part_result = _mm256_add_epi32(part_result, mul_lo);

    __m512i final_result = _mm512_cvtepu32_epi64(part_result);
    _mm512_storeu_si512((__m512i*)(result + offset), final_result);
}

// просотреть переносы для элементов вектора, а не всего вектора. Деление кнут д обязательно, не обязательно через вектора
void multiplyLongNumbers(const uint32_t* a, const uint32_t* b, uint64_t* result, size_t num_elements) {
    for (size_t i = 0; i < num_elements; i += 16) {
        __m512i a_vec = _mm512_loadu_si512((__m512i*)(a + i));
        __m512i b_vec = _mm512_loadu_si512((__m512i*)(b + i));
   
        processVectorHalf<0>(a_vec, b_vec, result, i);
        processVectorHalf<1>(a_vec, b_vec, result, i + 8);
        std::cout << "ResultProm[" << i << "] = " << result << std::endl;
    }
}


int main() {
    const int NUM_ELEMENTS = 16;
    uint32_t a[NUM_ELEMENTS], b[NUM_ELEMENTS];
    uint64_t result[NUM_ELEMENTS];
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        a[i] = i + 12345678;
        b[i] = i ;
    }

    multiplyLongNumbers(a, b, result, NUM_ELEMENTS);

    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        std::cout << "Result[" << i << "] = " << result[i] << std::endl;
    }

    return 0;
}