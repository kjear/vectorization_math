#include <foye_fastmath.hpp>

__m256 fy::simd::intrinsic::invcbrt(__m256 input) noexcept
{
	const __m256i sign_mask = _mm256_set1_epi32(0x80000000u);
	const __m256i exp_mask = _mm256_set1_epi32(0x7f800000u);
	const __m256i swap_perm = _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1);
	const __m256i magic_div3 = _mm256_set1_epi32(0xAAAAAAABu);

	const __m256i C1_vec = _mm256_set1_epi32(0x54AAAAAB);
	const __m256i C2_vec = _mm256_set1_epi32(0x58AAAAAB);

	const __m256 scale_factor = _mm256_set1_ps(16777216.0f);
	const __m256 one_ps = _mm256_set1_ps(1.0f);
	const __m256 two_ps = _mm256_set1_ps(2.0f);

	__m256i ix = _mm256_castps_si256(input);
	__m256i sign = _mm256_and_si256(ix, sign_mask);
	__m256i abs_ix = _mm256_andnot_si256(sign_mask, ix);

	__m256i exp_part = _mm256_and_si256(abs_ix, exp_mask);
	__m256i infnan_mask = _mm256_cmpeq_epi32(exp_part, exp_mask);

	__m256i zero_mask = _mm256_cmpeq_epi32(abs_ix, _mm256_setzero_si256());
	__m256i exp_zero = _mm256_cmpeq_epi32(exp_part, _mm256_setzero_si256());
	__m256i subnormal_mask = _mm256_andnot_si256(zero_mask, exp_zero);

	__m256 special_result = _mm256_div_ps(one_ps, input);
	__m256i special_mask = _mm256_or_si256(infnan_mask, zero_mask);

	__m256 scaled = _mm256_mul_ps(input, scale_factor);
	__m256i scaled_ix = _mm256_castps_si256(scaled);
	__m256i scaled_abs = _mm256_andnot_si256(sign_mask, scaled_ix);

	__m256i hx_for_div = _mm256_blendv_epi8(abs_ix, scaled_abs, subnormal_mask);

	__m256i prod_even = _mm256_mul_epu32(hx_for_div, magic_div3);
	__m256i q_even = _mm256_srli_epi64(prod_even, 33);

	__m256i odd_ix = _mm256_permutevar8x32_epi32(hx_for_div, swap_perm);
	__m256i prod_odd = _mm256_mul_epu32(odd_ix, magic_div3);
	__m256i q_odd = _mm256_srli_epi64(prod_odd, 33);
	__m256i q_odd_swapped = _mm256_permutevar8x32_epi32(q_odd, swap_perm);

	__m256i q = _mm256_or_si256(q_even, q_odd_swapped);

	__m256i offset = _mm256_blendv_epi8(C1_vec, C2_vec, subnormal_mask);
	__m256i y_bits = _mm256_or_si256(sign, _mm256_sub_epi32(offset, q));
	__m256 y = _mm256_castsi256_ps(y_bits);

	__m256 yy = _mm256_mul_ps(y, y);
	__m256 s = _mm256_mul_ps(_mm256_mul_ps(input, y), yy);
	__m256 num = _mm256_add_ps(two_ps, s);
	__m256 den = _mm256_fmadd_ps(two_ps, s, one_ps);
	y = _mm256_mul_ps(y, _mm256_div_ps(num, den));

	yy = _mm256_mul_ps(y, y);
	s = _mm256_mul_ps(_mm256_mul_ps(input, y), yy);
	num = _mm256_add_ps(two_ps, s);
	den = _mm256_fmadd_ps(two_ps, s, one_ps);
	y = _mm256_mul_ps(y, _mm256_div_ps(num, den));

	return _mm256_blendv_ps(y, special_result, _mm256_castsi256_ps(special_mask));
}