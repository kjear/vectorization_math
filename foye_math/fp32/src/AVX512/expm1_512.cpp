#include "foye_fastmath_fp32.hpp"

static inline __m512 expm1_small_branch(__m512 input) noexcept
{
	const __m512 c2 = _mm512_set1_ps(0.5f);
	const __m512 c3 = _mm512_set1_ps(1.6666667163e-1f);
	const __m512 c4 = _mm512_set1_ps(4.1666667908e-2f);
	const __m512 c5 = _mm512_set1_ps(8.3333337670e-3f);

	__m512 x2 = _mm512_mul_ps(input, input);
	__m512 p = _mm512_fmadd_ps(c5, input, c4);
	p = _mm512_fmadd_ps(p, input, c3);
	p = _mm512_fmadd_ps(p, input, c2);

	return _mm512_fmadd_ps(x2, p, input);
}

static inline __m512 expm1_medium_branch(__m512 input) noexcept
{
	const __m512 one = _mm512_set1_ps(1.0f);
	const __m512 half = _mm512_set1_ps(0.5f);
	const __m512 three = _mm512_set1_ps(3.0f);
	const __m512 six = _mm512_set1_ps(6.0f);
	const __m512 q1 = _mm512_set1_ps(-3.3333212137e-2f);
	const __m512 q2 = _mm512_set1_ps(1.5807170421e-3f);

	__m512 hfx = _mm512_mul_ps(half, input);
	__m512 hxs = _mm512_mul_ps(input, hfx);
	__m512 r1 = _mm512_fmadd_ps(hxs, _mm512_fmadd_ps(hxs, q2, q1), one);
	__m512 t = _mm512_fnmadd_ps(r1, hfx, three);
	__m512 e = _mm512_mul_ps(
		hxs,
		_mm512_div_ps(
			_mm512_sub_ps(r1, t),
			_mm512_fnmadd_ps(input, t, six)));

	return _mm512_sub_ps(input, _mm512_fmsub_ps(input, e, hxs));
}

static inline __mmask16 expm1_special_mask(
	__m512 input,
	__m512i hx,
	__mmask16 mask_pos) noexcept
{
	const __m512 overflow_threshold_v = _mm512_set1_ps(88.72283935546875f);
	const __m512i i_inf = _mm512_set1_epi32(0x7f800000);
	const __m512i i_27ln2 = _mm512_set1_epi32(0x4195b844);

	__mmask16 mask_nan = _mm512_cmp_epi32_mask(hx, i_inf, _MM_CMPINT_GT);
	__mmask16 mask_inf = _mm512_cmp_epi32_mask(hx, i_inf, _MM_CMPINT_EQ);
	__mmask16 mask_huge = _mm512_cmp_epi32_mask(hx, i_27ln2, _MM_CMPINT_NLT);
	__mmask16 mask_negbig = mask_huge & ~mask_pos;
	__mmask16 mask_pos_over = _mm512_cmp_ps_mask(input, overflow_threshold_v, _CMP_GT_OQ);

	return mask_nan | mask_inf | mask_negbig | mask_pos_over;
}

static inline __m512 expm1_finalize_special(
	__m512 result,
	__m512 input,
	__m512i hx,
	__mmask16 mask_pos) noexcept
{
	const __m512 neg_one = _mm512_set1_ps(-1.0f);
	const __m512 infv = _mm512_set1_ps(std::numeric_limits<float>::infinity());
	const __m512 overflow_threshold_v = _mm512_set1_ps(88.72283935546875f);

	const __m512i i_inf = _mm512_set1_epi32(0x7f800000);
	const __m512i i_27ln2 = _mm512_set1_epi32(0x4195b844);
	const __m512i qnan_quiet_bit = _mm512_set1_epi32(0x00400000);

	__mmask16 mask_nan = _mm512_cmp_epi32_mask(hx, i_inf, _MM_CMPINT_GT);
	__mmask16 mask_inf = _mm512_cmp_epi32_mask(hx, i_inf, _MM_CMPINT_EQ);
	__mmask16 mask_posinf = mask_inf & mask_pos;
	__mmask16 mask_neginf = mask_inf & ~mask_pos;

	__mmask16 mask_huge = _mm512_cmp_epi32_mask(hx, i_27ln2, _MM_CMPINT_NLT);
	__mmask16 mask_negbig = mask_huge & ~mask_pos;
	__mmask16 mask_overflow = _mm512_cmp_ps_mask(input, overflow_threshold_v, _CMP_GT_OQ);

	__m512 nan_quieted = _mm512_castsi512_ps(
		_mm512_or_si512(_mm512_castps_si512(input), qnan_quiet_bit));

	result = _mm512_mask_blend_ps(mask_negbig, result, neg_one);
	result = _mm512_mask_blend_ps(mask_overflow, result, infv);
	result = _mm512_mask_blend_ps(mask_neginf, result, neg_one);
	result = _mm512_mask_blend_ps(mask_posinf, result, input);
	result = _mm512_mask_blend_ps(mask_nan, result, nan_quieted);

	return result;
}

static inline __m512 expm1_general_branch(__m512 input) noexcept
{
	const __m512 zero = _mm512_setzero_ps();
	const __m512 one = _mm512_set1_ps(1.0f);
	const __m512 half = _mm512_set1_ps(0.5f);
	const __m512 mhalf = _mm512_set1_ps(-0.5f);
	const __m512 two = _mm512_set1_ps(2.0f);
	const __m512 three = _mm512_set1_ps(3.0f);
	const __m512 six = _mm512_set1_ps(6.0f);
	const __m512 mquarter = _mm512_set1_ps(-0.25f);

	const __m512 q1 = _mm512_set1_ps(-3.3333212137e-2f);
	const __m512 q2 = _mm512_set1_ps(1.5807170421e-3f);

	const __m512 ln2_hi_v = _mm512_set1_ps(6.9313812256e-01f);
	const __m512 ln2_lo_v = _mm512_set1_ps(9.0580006145e-06f);
	const __m512 neg_ln2_lo_v = _mm512_set1_ps(-9.0580006145e-06f);
	const __m512 invln2_v = _mm512_set1_ps(1.4426950216e+00f);

	const __m512 twop127 = _mm512_set1_ps(0x1p127f);

	const __m512i sign_mask_i = _mm512_set1_epi32(0x80000000u);
	const __m512i abs_mask_i = _mm512_set1_epi32(0x7fffffffu);
	const __m512i i_zero = _mm512_setzero_si512();
	const __m512i i_one = _mm512_set1_epi32(1);
	const __m512i i_mone = _mm512_set1_epi32(-1);
	const __m512i i_23 = _mm512_set1_epi32(23);
	const __m512i i_56 = _mm512_set1_epi32(56);
	const __m512i i_127 = _mm512_set1_epi32(127);
	const __m512i i_128 = _mm512_set1_epi32(128);
	const __m512i i_bias = _mm512_set1_epi32(0x3f800000);
	const __m512i i_half_ln2 = _mm512_set1_epi32(0x3eb17218);
	const __m512i i_onehalf_ln2 = _mm512_set1_epi32(0x3F851592);
	const __m512i i_tiny = _mm512_set1_epi32(0x33000000);
	const __m512i i_1000000 = _mm512_set1_epi32(0x01000000);

	__m512i ix = _mm512_castps_si512(input);
	__m512i xsb = _mm512_and_si512(ix, sign_mask_i);
	__m512i hx = _mm512_and_si512(ix, abs_mask_i);
	__mmask16 mask_pos = _mm512_cmp_epi32_mask(xsb, i_zero, _MM_CMPINT_EQ);

	__mmask16 mask_big = _mm512_cmp_epi32_mask(hx, i_half_ln2, _MM_CMPINT_GT);
	__mmask16 mask_mid = _mm512_cmp_epi32_mask(i_onehalf_ln2, hx, _MM_CMPINT_GT);
	__mmask16 mask_bigmid = mask_big & mask_mid;
	__mmask16 mask_bignmid = mask_big & ~mask_mid;
	__mmask16 mask_tiny = _mm512_cmp_epi32_mask(i_tiny, hx, _MM_CMPINT_GT);

	__m512 hi1 = _mm512_mask_blend_ps(
		mask_pos,
		_mm512_add_ps(input, ln2_hi_v),
		_mm512_sub_ps(input, ln2_hi_v));

	__m512 lo1 = _mm512_mask_blend_ps(mask_pos, neg_ln2_lo_v, ln2_lo_v);
	__m512i k1i = _mm512_mask_blend_epi32(mask_pos, i_mone, i_one);

	__m512 adj = _mm512_mask_blend_ps(mask_pos, mhalf, half);
	__m512 fk2 = _mm512_fmadd_ps(invln2_v, input, adj);
	__m512i k2i = _mm512_cvttps_epi32(fk2);
	__m512 t2 = _mm512_cvtepi32_ps(k2i);
	__m512 hi2 = _mm512_fnmadd_ps(t2, ln2_hi_v, input);
	__m512 lo2 = _mm512_mul_ps(t2, ln2_lo_v);

	__m512 hi = input;
	hi = _mm512_mask_mov_ps(hi, mask_bignmid, hi2);
	hi = _mm512_mask_mov_ps(hi, mask_bigmid, hi1);

	__m512 lo = zero;
	lo = _mm512_mask_mov_ps(lo, mask_bignmid, lo2);
	lo = _mm512_mask_mov_ps(lo, mask_bigmid, lo1);

	__m512i k = i_zero;
	k = _mm512_mask_mov_epi32(k, mask_bignmid, k2i);
	k = _mm512_mask_mov_epi32(k, mask_bigmid, k1i);

	__m512 x = input;
	x = _mm512_mask_mov_ps(x, mask_big, _mm512_sub_ps(hi, lo));

	__m512 c = zero;
	c = _mm512_mask_mov_ps(
		c,
		mask_big,
		_mm512_sub_ps(_mm512_sub_ps(hi, x), lo));

	__m512 hfx = _mm512_mul_ps(half, x);
	__m512 hxs = _mm512_mul_ps(x, hfx);
	__m512 r1 = _mm512_fmadd_ps(hxs, _mm512_fmadd_ps(hxs, q2, q1), one);
	__m512 t = _mm512_fnmadd_ps(r1, hfx, three);
	__m512 e = _mm512_mul_ps(
		hxs,
		_mm512_div_ps(
			_mm512_sub_ps(r1, t),
			_mm512_fnmadd_ps(x, t, six)));

	__m512 res_k0 = _mm512_sub_ps(x, _mm512_fmsub_ps(x, e, hxs));

	__m512 emc = _mm512_sub_ps(e, c);
	__m512 e2 = _mm512_fmsub_ps(x, emc, c);
	e2 = _mm512_sub_ps(e2, hxs);

	__m512i twopk_bits = _mm512_add_epi32(i_bias, _mm512_slli_epi32(k, 23));
	__m512 twopk = _mm512_castsi512_ps(twopk_bits);

	__m512 xe2 = _mm512_sub_ps(x, e2);
	__m512 res_km1 = _mm512_fmsub_ps(half, xe2, half);

	__m512 res_k1a = _mm512_mul_ps(
		_mm512_set1_ps(-2.0f),
		_mm512_sub_ps(e2, _mm512_add_ps(x, half)));
	__m512 res_k1b = _mm512_fmadd_ps(two, _mm512_sub_ps(x, e2), one);
	__mmask16 mask_xlt_mquarter = _mm512_cmp_ps_mask(x, mquarter, _CMP_LT_OQ);
	__m512 res_k1 = _mm512_mask_blend_ps(mask_xlt_mquarter, res_k1b, res_k1a);

	__m512 y_big = _mm512_sub_ps(_mm512_add_ps(one, x), e2);
	__m512 res_bigk = _mm512_fmsub_ps(y_big, twopk, one);
	__mmask16 mask_k128 = _mm512_cmp_epi32_mask(k, i_128, _MM_CMPINT_EQ);
	__m512 y2 = _mm512_mul_ps(y_big, two);
	__m512 res_bigk_128 = _mm512_fmsub_ps(y2, twop127, one);
	res_bigk = _mm512_mask_mov_ps(res_bigk, mask_k128, res_bigk_128);

	__m512i shr = _mm512_srlv_epi32(i_1000000, k);
	__m512i t_lt23_bits = _mm512_sub_epi32(i_bias, shr);
	__m512 t_lt23 = _mm512_castsi512_ps(t_lt23_bits);
	__m512 res_lt23 = _mm512_mul_ps(
		_mm512_sub_ps(t_lt23, _mm512_sub_ps(e2, x)),
		twopk);

	__m512i t_ge23_bits = _mm512_slli_epi32(_mm512_sub_epi32(i_127, k), 23);
	__m512 t_ge23 = _mm512_castsi512_ps(t_ge23_bits);
	__m512 y_ge23 = _mm512_add_ps(
		_mm512_sub_ps(x, _mm512_add_ps(e2, t_ge23)),
		one);
	__m512 res_ge23 = _mm512_mul_ps(y_ge23, twopk);

	__mmask16 mask_k0 = _mm512_cmp_epi32_mask(k, i_zero, _MM_CMPINT_EQ);
	__mmask16 mask_km1 = _mm512_cmp_epi32_mask(k, i_mone, _MM_CMPINT_EQ);
	__mmask16 mask_k1 = _mm512_cmp_epi32_mask(k, i_one, _MM_CMPINT_EQ);
	__mmask16 mask_kle_n2 = _mm512_cmp_epi32_mask(i_mone, k, _MM_CMPINT_GT);
	__mmask16 mask_kgt56 = _mm512_cmp_epi32_mask(k, i_56, _MM_CMPINT_GT);
	__mmask16 mask_kbig = mask_kle_n2 | mask_kgt56;
	__mmask16 mask_kgt1 = _mm512_cmp_epi32_mask(k, i_one, _MM_CMPINT_GT);
	__mmask16 mask_klt23 = _mm512_cmp_epi32_mask(i_23, k, _MM_CMPINT_GT);
	__mmask16 mask_smallpos = mask_kgt1 & mask_klt23;

	__m512 res_nz = res_ge23;
	res_nz = _mm512_mask_mov_ps(res_nz, mask_smallpos, res_lt23);
	res_nz = _mm512_mask_mov_ps(res_nz, mask_kbig, res_bigk);
	res_nz = _mm512_mask_mov_ps(res_nz, mask_k1, res_k1);
	res_nz = _mm512_mask_mov_ps(res_nz, mask_km1, res_km1);

	__m512 result = res_nz;
	result = _mm512_mask_mov_ps(result, mask_k0, res_k0);
	result = _mm512_mask_mov_ps(result, mask_tiny, input);

	return expm1_finalize_special(result, input, hx, mask_pos);
}

__m512 fy::simd::intrinsic::expm1(__m512 input) noexcept
{
	const __m512 tiny_thr = _mm512_set1_ps(0x1p-25f);
	const __m512 small_thr = _mm512_set1_ps(1.0e-3f);
	const __m512 half_ln2 = _mm512_set1_ps(0.34657359028f);
	const __m512 overflow_threshold_v = _mm512_set1_ps(88.72283935546875f);
	const __m512 neg_27ln2_v = _mm512_set1_ps(-18.7149734497f);

	const __m512i abs_mask_i = _mm512_set1_epi32(0x7fffffff);
	__m512 ax = _mm512_castsi512_ps(
		_mm512_and_si512(_mm512_castps_si512(input), abs_mask_i));

	__mmask16 mask_tiny = _mm512_cmp_ps_mask(ax, tiny_thr, _CMP_LT_OQ);
	if (mask_tiny == 0xFFFF)
	{
		return input;
	}

	__mmask16 mask_small = _mm512_cmp_ps_mask(ax, small_thr, _CMP_LE_OQ);
	if (mask_small == 0xFFFF)
	{
		return expm1_small_branch(input);
	}

	__mmask16 mask_medium = _mm512_cmp_ps_mask(ax, half_ln2, _CMP_LE_OQ);
	if (mask_medium == 0xFFFF)
	{
		return expm1_medium_branch(input);
	}

	__mmask16 mask_all_over = _mm512_cmp_ps_mask(input, overflow_threshold_v, _CMP_GT_OQ);
	if (mask_all_over == 0xFFFF)
	{
		return _mm512_set1_ps(std::numeric_limits<float>::infinity());
	}

	__mmask16 mask_all_negbig = _mm512_cmp_ps_mask(input, neg_27ln2_v, _CMP_LE_OQ);
	if (mask_all_negbig == 0xFFFF)
	{
		return _mm512_set1_ps(-1.0f);
	}

	return expm1_general_branch(input);
}