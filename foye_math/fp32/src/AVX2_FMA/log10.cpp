#include "foye_fastmath_fp32.hpp"

static __m256 log10_finalize_special(__m256 result, __m256 input, __m256i hx, __m256i ix) noexcept
{
	const __m256i sign_mask = _mm256_set1_epi32(0x80000000u);
	const __m256i exp_mask = _mm256_set1_epi32(0x7f800000u);
	const __m256i mant_mask = _mm256_set1_epi32(0x007fffffu);
	const __m256i qnan_bits = _mm256_set1_epi32(0x7fc00000u);
	const __m256i qnan_quiet = _mm256_set1_epi32(0x00400000u);
	const __m256i neg_inf_bits = _mm256_set1_epi32(0xff800000u);

	__m256i mask_zero = _mm256_cmpeq_epi32(ix, _mm256_setzero_si256());
	__m256i mask_sign = _mm256_cmpeq_epi32(_mm256_and_si256(hx, sign_mask), sign_mask);
	__m256i mask_exp_all_ones = _mm256_cmpeq_epi32(_mm256_and_si256(ix, exp_mask), exp_mask);
	__m256i mask_neg_nonzero = _mm256_andnot_si256(mask_zero, mask_sign);

	__m256i mant_zero = _mm256_cmpeq_epi32(_mm256_and_si256(ix, mant_mask), _mm256_setzero_si256());

	__m256i mask_inf = _mm256_and_si256(mask_exp_all_ones, mant_zero);
	__m256i mask_nan = _mm256_andnot_si256(mant_zero, mask_exp_all_ones);

	__m256i mask_pos_inf = _mm256_andnot_si256(mask_sign, mask_inf);
	__m256i mask_neg_invalid = _mm256_andnot_si256(mask_nan, mask_neg_nonzero);

	__m256 nan_quieted = _mm256_castsi256_ps(_mm256_or_si256(hx, qnan_quiet));

	result = _mm256_blendv_ps(result, input, _mm256_castsi256_ps(mask_pos_inf));
	result = _mm256_blendv_ps(result, _mm256_castsi256_ps(neg_inf_bits), _mm256_castsi256_ps(mask_zero));
	result = _mm256_blendv_ps(result, _mm256_castsi256_ps(qnan_bits), _mm256_castsi256_ps(mask_neg_invalid));
	result = _mm256_blendv_ps(result, nan_quieted, _mm256_castsi256_ps(mask_nan));

	return result;
}

static __m256 log10_core_normal(__m256 input, __m256i hx) noexcept
{
	const __m256 ivln10hi = _mm256_set1_ps(0x1.bcc0000000000p-2);
	const __m256 ivln10lo = _mm256_set1_ps(-0x1.09d5b20000000p-15);
	const __m256 log10_2hi = _mm256_set1_ps(0x1.3441000000000p-2);
	const __m256 log10_2lo = _mm256_set1_ps(0x1.a84fb60000000p-21);
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 zero = _mm256_setzero_ps();
	const __m256 half = _mm256_set1_ps(0.5f);
	const __m256 two = _mm256_set1_ps(2.0f);

	const __m256 Lp1 = _mm256_set1_ps(6.6666668653e-01f);
	const __m256 Lp2 = _mm256_set1_ps(4.0000000596e-01f);
	const __m256 Lp3 = _mm256_set1_ps(2.8571429849e-01f);
	const __m256 Lp4 = _mm256_set1_ps(2.2222198546e-01f);
	const __m256 Lp5 = _mm256_set1_ps(1.8183572590e-01f);
	const __m256 Lp6 = _mm256_set1_ps(1.5313838422e-01f);
	const __m256 Lp7 = _mm256_set1_ps(1.4798198640e-01f);

	const __m256i abs_mask = _mm256_set1_epi32(0x7fffffffu);
	const __m256i mant_mask = _mm256_set1_epi32(0x007fffffu);
	const __m256i hidden_bit = _mm256_set1_epi32(0x00800000u);
	const __m256i one_bits = _mm256_set1_epi32(0x3f800000u);
	const __m256i exp_bias = _mm256_set1_epi32(127);
	const __m256i round_const = _mm256_set1_epi32(0x004afb0d);
	const __m256i tiny_thr = _mm256_set1_epi32(0x38000000u);

	__m256i exp_part = _mm256_sub_epi32(_mm256_srli_epi32(hx, 23), exp_bias);

	__m256i mant = _mm256_and_si256(hx, mant_mask);
	__m256i i = _mm256_and_si256(_mm256_add_epi32(mant, round_const), hidden_bit);
	__m256i xbits = _mm256_or_si256(mant, _mm256_xor_si256(i, one_bits));
	__m256i k = _mm256_add_epi32(exp_part, _mm256_srli_epi32(i, 23));

	__m256 x = _mm256_castsi256_ps(xbits);
	__m256 y = _mm256_cvtepi32_ps(k);
	__m256 f = _mm256_sub_ps(x, one);

	__m256i af = _mm256_and_si256(_mm256_castps_si256(f), abs_mask);
	__m256 tiny_mask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(tiny_thr, af));

	__m256 hfsq = _mm256_mul_ps(half, _mm256_mul_ps(f, f));
	__m256 s = _mm256_div_ps(f, _mm256_add_ps(two, f));
	__m256 z = _mm256_mul_ps(s, s);

	__m256 R = _mm256_fmadd_ps(Lp7, z, Lp6);
	R = _mm256_fmadd_ps(R, z, Lp5);
	R = _mm256_fmadd_ps(R, z, Lp4);
	R = _mm256_fmadd_ps(R, z, Lp3);
	R = _mm256_fmadd_ps(R, z, Lp2);
	R = _mm256_fmadd_ps(R, z, Lp1);
	R = _mm256_mul_ps(R, z);

	__m256 t_full = _mm256_fmadd_ps(s, _mm256_add_ps(hfsq, R), _mm256_sub_ps(f, hfsq));
	__m256 t_tiny = _mm256_fnmadd_ps(_mm256_mul_ps(f, f), half, f);
	__m256 t = _mm256_blendv_ps(t_full, t_tiny, tiny_mask);

	__m256 r = _mm256_mul_ps(y, log10_2lo);
	r = _mm256_fmadd_ps(t, ivln10lo, r);
	r = _mm256_fmadd_ps(y, log10_2hi, r);
	r = _mm256_fmadd_ps(t, ivln10hi, r);

	__m256i mask_one = _mm256_cmpeq_epi32(hx, one_bits);
	return _mm256_blendv_ps(r, zero, _mm256_castsi256_ps(mask_one));
}

static __m256 log10_core_full(__m256 input, __m256i hx, __m256i ix) noexcept
{
	const __m256 two25 = _mm256_set1_ps(3.3554432000e+07f);
	const __m256 ivln10hi = _mm256_set1_ps(4.3432617188e-01f);
	const __m256 ivln10lo = _mm256_set1_ps(-3.1689971365e-05f);
	const __m256 log10_2hi = _mm256_set1_ps(3.0102920532e-01f);
	const __m256 log10_2lo = _mm256_set1_ps(7.9034151668e-07f);
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 zero = _mm256_setzero_ps();
	const __m256 half = _mm256_set1_ps(0.5f);
	const __m256 two = _mm256_set1_ps(2.0f);

	const __m256 Lp1 = _mm256_set1_ps(6.6666668653e-01f);
	const __m256 Lp2 = _mm256_set1_ps(4.0000000596e-01f);
	const __m256 Lp3 = _mm256_set1_ps(2.8571429849e-01f);
	const __m256 Lp4 = _mm256_set1_ps(2.2222198546e-01f);
	const __m256 Lp5 = _mm256_set1_ps(1.8183572590e-01f);
	const __m256 Lp6 = _mm256_set1_ps(1.5313838422e-01f);
	const __m256 Lp7 = _mm256_set1_ps(1.4798198640e-01f);

	const __m256i sign_mask = _mm256_set1_epi32(0x80000000u);
	const __m256i abs_mask = _mm256_set1_epi32(0x7fffffffu);
	const __m256i mant_mask = _mm256_set1_epi32(0x007fffffu);
	const __m256i hidden_bit = _mm256_set1_epi32(0x00800000u);
	const __m256i one_bits = _mm256_set1_epi32(0x3f800000u);
	const __m256i exp_bias = _mm256_set1_epi32(127);
	const __m256i sub_thresh = _mm256_set1_epi32(0x00800000u);
	const __m256i round_const = _mm256_set1_epi32(0x004afb0d);
	const __m256i minus_25 = _mm256_set1_epi32(-25);
	const __m256i tiny_thr = _mm256_set1_epi32(0x38000000u);

	__m256i mask_zero = _mm256_cmpeq_epi32(ix, _mm256_setzero_si256());
	__m256i mask_sign = _mm256_cmpeq_epi32(_mm256_and_si256(hx, sign_mask), sign_mask);

	__m256i mask_sub = _mm256_cmpgt_epi32(sub_thresh, ix);
	mask_sub = _mm256_andnot_si256(mask_zero, mask_sub);
	mask_sub = _mm256_andnot_si256(mask_sign, mask_sub);

	__m256 scaled = _mm256_blendv_ps(input, _mm256_mul_ps(input, two25), _mm256_castsi256_ps(mask_sub));
	__m256i k = _mm256_and_si256(mask_sub, minus_25);

	__m256i hx_scaled = _mm256_castps_si256(scaled);
	__m256i exp_part = _mm256_sub_epi32(_mm256_srli_epi32(hx_scaled, 23), exp_bias);
	k = _mm256_add_epi32(k, exp_part);

	__m256i mant = _mm256_and_si256(hx_scaled, mant_mask);
	__m256i i = _mm256_and_si256(_mm256_add_epi32(mant, round_const), hidden_bit);
	__m256i xbits = _mm256_or_si256(mant, _mm256_xor_si256(i, one_bits));
	k = _mm256_add_epi32(k, _mm256_srli_epi32(i, 23));

	__m256 x = _mm256_castsi256_ps(xbits);
	__m256 y = _mm256_cvtepi32_ps(k);
	__m256 f = _mm256_sub_ps(x, one);

	__m256i af = _mm256_and_si256(_mm256_castps_si256(f), abs_mask);
	__m256 tiny_mask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(tiny_thr, af));

	__m256 hfsq = _mm256_mul_ps(half, _mm256_mul_ps(f, f));
	__m256 s = _mm256_div_ps(f, _mm256_add_ps(two, f));
	__m256 z = _mm256_mul_ps(s, s);

	__m256 R = _mm256_fmadd_ps(Lp7, z, Lp6);
	R = _mm256_fmadd_ps(R, z, Lp5);
	R = _mm256_fmadd_ps(R, z, Lp4);
	R = _mm256_fmadd_ps(R, z, Lp3);
	R = _mm256_fmadd_ps(R, z, Lp2);
	R = _mm256_fmadd_ps(R, z, Lp1);
	R = _mm256_mul_ps(R, z);

	__m256 t_full = _mm256_fmadd_ps(s, _mm256_add_ps(hfsq, R), _mm256_sub_ps(f, hfsq));
	__m256 t_tiny = _mm256_fnmadd_ps(_mm256_mul_ps(f, f), half, f);
	__m256 t = _mm256_blendv_ps(t_full, t_tiny, tiny_mask);

	__m256 r = _mm256_mul_ps(y, log10_2lo);
	r = _mm256_fmadd_ps(t, ivln10lo, r);
	r = _mm256_fmadd_ps(y, log10_2hi, r);
	r = _mm256_fmadd_ps(t, ivln10hi, r);

	__m256i mask_one = _mm256_cmpeq_epi32(hx_scaled, one_bits);
	return _mm256_blendv_ps(r, zero, _mm256_castsi256_ps(mask_one));
}

__m256 fy::simd::intrinsic::log10(__m256 input) noexcept
{
	const __m256i sign_mask = _mm256_set1_epi32(0x80000000u);
	const __m256i abs_mask = _mm256_set1_epi32(0x7fffffffu);
	const __m256i exp_mask = _mm256_set1_epi32(0x7f800000u);
	const __m256i sub_thresh = _mm256_set1_epi32(0x00800000u);

	__m256i hx = _mm256_castps_si256(input);
	__m256i ix = _mm256_and_si256(hx, abs_mask);

	__m256i mask_zero = _mm256_cmpeq_epi32(ix, _mm256_setzero_si256());
	__m256i mask_sign = _mm256_cmpeq_epi32(_mm256_and_si256(hx, sign_mask), sign_mask);
	__m256i mask_exp_all_ones = _mm256_cmpeq_epi32(_mm256_and_si256(ix, exp_mask), exp_mask);
	__m256i mask_neg_nonzero = _mm256_andnot_si256(mask_zero, mask_sign);

	__m256i mask_special = _mm256_or_si256(mask_zero, mask_neg_nonzero);
	mask_special = _mm256_or_si256(mask_special, mask_exp_all_ones);

	__m256i mask_sub = _mm256_cmpgt_epi32(sub_thresh, ix);
	mask_sub = _mm256_andnot_si256(mask_zero, mask_sub);
	mask_sub = _mm256_andnot_si256(mask_sign, mask_sub);

	int sp = _mm256_movemask_ps(_mm256_castsi256_ps(mask_special));
	int sb = _mm256_movemask_ps(_mm256_castsi256_ps(mask_sub));

	if (sp == 0xFF)
	{
		return log10_finalize_special(_mm256_setzero_ps(), input, hx, ix);
	}

	if ((sp | sb) == 0)
	{
		return log10_core_normal(input, hx);
	}

	__m256 result = log10_core_full(input, hx, ix);

	if (sp == 0)
	{
		return result;
	}

	return log10_finalize_special(result, input, hx, ix);
}