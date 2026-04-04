#include "foye_fastmath_fp32.hpp"

__m256 fy::simd::intrinsic::modf(__m256 x, __m256* intpart) noexcept
{
	const __m256i bits = _mm256_castps_si256(x);
	const __m256i sign_mask = _mm256_set1_epi32(0x80000000u);
	const __m256i abs_mask = _mm256_set1_epi32(0x7fffffffu);
	const __m256i frac_field_mask = _mm256_set1_epi32(0x007fffffu);
	const __m256i all_ones = _mm256_set1_epi32(-1);
	const __m256i zero_i = _mm256_setzero_si256();
	const __m256i c23 = _mm256_set1_epi32(23);
	const __m256i c127 = _mm256_set1_epi32(127);
	const __m256i c255 = _mm256_set1_epi32(255);

	__m256i sign = _mm256_and_si256(bits, sign_mask);
	__m256i abits = _mm256_and_si256(bits, abs_mask);
	__m256i exp = _mm256_and_si256(_mm256_srli_epi32(abits, 23), c255);
	__m256i j0 = _mm256_sub_epi32(exp, c127);

	__m256i is_lt_0 = _mm256_cmpgt_epi32(zero_i, j0);
	__m256i is_ge_23 = _mm256_or_si256(_mm256_cmpgt_epi32(j0, c23), _mm256_cmpeq_epi32(j0, c23));
	__m256i is_mid = _mm256_xor_si256(_mm256_or_si256(is_lt_0, is_ge_23), all_ones);

	__m256i frac_mask = _mm256_srlv_epi32(frac_field_mask, j0);
	__m256i masked_frac = _mm256_and_si256(bits, frac_mask);
	__m256i is_mid_integral = _mm256_and_si256(is_mid, _mm256_cmpeq_epi32(masked_frac, zero_i));
	__m256i is_mid_nonintegral = _mm256_and_si256(is_mid, _mm256_xor_si256(is_mid_integral, all_ones));

	__m256i int_bits_mid = _mm256_andnot_si256(frac_mask, bits);

	__m256i is_exp_ff = _mm256_cmpeq_epi32(exp, c255);
	__m256i frac_field = _mm256_and_si256(bits, frac_field_mask);
	__m256i is_nan = _mm256_and_si256(is_exp_ff, _mm256_xor_si256(_mm256_cmpeq_epi32(frac_field, zero_i), all_ones));

	__m256i int_bits = sign;
	int_bits = _mm256_blendv_epi8(int_bits, bits, is_ge_23);
	int_bits = _mm256_blendv_epi8(int_bits, bits, is_mid_integral);
	int_bits = _mm256_blendv_epi8(int_bits, int_bits_mid, is_mid_nonintegral);

	*intpart = _mm256_castsi256_ps(int_bits);

	__m128 x_lo = _mm256_castps256_ps128(x);
	__m128 x_hi = _mm256_extractf128_ps(x, 1);

	__m128i int_lo_i = _mm256_castsi256_si128(int_bits_mid);
	__m128i int_hi_i = _mm256_extractf128_si256(int_bits_mid, 1);

	__m128 int_lo = _mm_castsi128_ps(int_lo_i);
	__m128 int_hi = _mm_castsi128_ps(int_hi_i);

	__m128d x_lo_d0 = _mm_cvtps_pd(x_lo);
	__m128d x_lo_d1 = _mm_cvtps_pd(_mm_movehl_ps(x_lo, x_lo));
	__m128d x_hi_d0 = _mm_cvtps_pd(x_hi);
	__m128d x_hi_d1 = _mm_cvtps_pd(_mm_movehl_ps(x_hi, x_hi));

	__m128d int_lo_d0 = _mm_cvtps_pd(int_lo);
	__m128d int_lo_d1 = _mm_cvtps_pd(_mm_movehl_ps(int_lo, int_lo));
	__m128d int_hi_d0 = _mm_cvtps_pd(int_hi);
	__m128d int_hi_d1 = _mm_cvtps_pd(_mm_movehl_ps(int_hi, int_hi));

	__m128d frac_lo_d0 = _mm_fmsub_pd(x_lo_d0, _mm_set1_pd(1.0), int_lo_d0);
	__m128d frac_lo_d1 = _mm_fmsub_pd(x_lo_d1, _mm_set1_pd(1.0), int_lo_d1);
	__m128d frac_hi_d0 = _mm_fmsub_pd(x_hi_d0, _mm_set1_pd(1.0), int_hi_d0);
	__m128d frac_hi_d1 = _mm_fmsub_pd(x_hi_d1, _mm_set1_pd(1.0), int_hi_d1);

	__m128 frac_lo_f = _mm_movelh_ps(_mm_cvtpd_ps(frac_lo_d0), _mm_cvtpd_ps(frac_lo_d1));
	__m128 frac_hi_f = _mm_movelh_ps(_mm_cvtpd_ps(frac_hi_d0), _mm_cvtpd_ps(frac_hi_d1));
	__m256 frac_mid = _mm256_insertf128_ps(_mm256_castps128_ps256(frac_lo_f), frac_hi_f, 1);

	__m256 sign_zero = _mm256_castsi256_ps(sign);
	__m256 frac = x;
	__m256i need_sign_zero = _mm256_andnot_si256(is_nan, is_ge_23);
	frac = _mm256_blendv_ps(frac, sign_zero, _mm256_castsi256_ps(need_sign_zero));
	frac = _mm256_blendv_ps(frac, sign_zero, _mm256_castsi256_ps(is_mid_integral));
	frac = _mm256_blendv_ps(frac, frac_mid, _mm256_castsi256_ps(is_mid_nonintegral));

	return frac;
}