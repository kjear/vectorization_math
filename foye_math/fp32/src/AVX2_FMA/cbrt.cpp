#include "foye_fastmath_fp32.hpp"

static __m256 fast_div_nr1(__m256 a, __m256 b) noexcept
{
	const __m256 two = _mm256_set1_ps(2.0f);
	__m256 r = _mm256_rcp_ps(b);
	r = _mm256_mul_ps(r, _mm256_fnmadd_ps(b, r, two));
	return _mm256_mul_ps(a, r);
}

static __m256 cbrt_newton_step_rcp(__m256 x, __m256 t) noexcept
{
	const __m256 third = _mm256_set1_ps(1.0f / 3.0f);
	__m256 tt = _mm256_mul_ps(t, t);
	__m256 qx = fast_div_nr1(x, tt);
	return _mm256_fmadd_ps(_mm256_sub_ps(qx, t), third, t);
}

static __m256 cbrt_newton_step_div(__m256 x, __m256 t) noexcept
{
	const __m256 third = _mm256_set1_ps(1.0f / 3.0f);
	__m256 tt = _mm256_mul_ps(t, t);
	__m256 qx = _mm256_div_ps(x, tt);
	return _mm256_fmadd_ps(_mm256_sub_ps(qx, t), third, t);
}

__m256 fy::simd::intrinsic::cbrt(__m256 input) noexcept
{
	const __m256i sign_mask = _mm256_set1_epi32(0x80000000u);
	const __m256i exp_mask = _mm256_set1_epi32(0x7f800000u);
	const __m256i B1_vec = _mm256_set1_epi32(709958130);
	const __m256i B2_vec = _mm256_set1_epi32(642849266);
	const __m256i swap_perm = _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1);
	const __m256i magic = _mm256_set1_epi32(0xAAAAAAABu);
	const __m256  scale_factor = _mm256_set1_ps(16777216.0f);

	__m256i ix = _mm256_castps_si256(input);
	__m256i sign = _mm256_and_si256(ix, sign_mask);
	__m256i abs_ix = _mm256_andnot_si256(sign_mask, ix);

	__m256i exp_part = _mm256_and_si256(abs_ix, exp_mask);
	__m256i infnan_mask = _mm256_cmpeq_epi32(exp_part, exp_mask);
	__m256i zero_mask = _mm256_cmpeq_epi32(abs_ix, _mm256_setzero_si256());
	__m256i exp_zero = _mm256_cmpeq_epi32(exp_part, _mm256_setzero_si256());
	__m256i subnormal_mask = _mm256_andnot_si256(zero_mask, exp_zero);

	__m256 infnan_result = _mm256_add_ps(input, input);

	const __m256 pmask = _mm256_castsi256_ps(infnan_mask);
	__m256 special_result = _mm256_or_ps(_mm256_and_ps(pmask, infnan_result), _mm256_andnot_ps(pmask, input));
	__m256i special_mask = _mm256_or_si256(infnan_mask, zero_mask);

	__m256  scaled = _mm256_mul_ps(input, scale_factor);
	__m256i scaled_ix = _mm256_castps_si256(scaled);
	__m256i scaled_abs = _mm256_andnot_si256(sign_mask, scaled_ix);

	
	__m256i hx_for_div = _mm256_or_si256(
		_mm256_and_si256(subnormal_mask, scaled_abs), 
		_mm256_andnot_si256(subnormal_mask, abs_ix));

	__m256i prod_even = _mm256_mul_epu32(hx_for_div, magic);
	__m256i q_even = _mm256_srli_epi64(prod_even, 33);

	__m256i odd_ix = _mm256_permutevar8x32_epi32(hx_for_div, swap_perm);
	__m256i prod_odd = _mm256_mul_epu32(odd_ix, magic);
	__m256i q_odd = _mm256_srli_epi64(prod_odd, 33);
	__m256i q_odd_swapped = _mm256_permutevar8x32_epi32(q_odd, swap_perm);

	__m256i q = _mm256_or_si256(q_even, q_odd_swapped);

	__m256i offset = _mm256_or_si256(
		_mm256_and_si256(subnormal_mask, B2_vec), 
		_mm256_andnot_si256(subnormal_mask, B1_vec));

	__m256i t_bits = _mm256_or_si256(sign, _mm256_add_epi32(q, offset));
	__m256  t = _mm256_castsi256_ps(t_bits);

	t = cbrt_newton_step_rcp(input, t);
	t = cbrt_newton_step_rcp(input, t);
	t = cbrt_newton_step_div(input, t);

	const __m256 smask = _mm256_castsi256_ps(special_mask);
	return _mm256_or_ps(_mm256_and_ps(smask, special_result), _mm256_andnot_ps(smask, t));
}
