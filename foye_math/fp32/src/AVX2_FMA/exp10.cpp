#include "foye_fastmath_fp32.hpp"

alignas(64) constexpr float exp2_tab16[16] = {
	1.0000000000000000000f, 1.0442737824274138403f,
	1.0905077326652576592f, 1.1387886347566916537f,
	1.1892071150027210667f, 1.2418578120734840486f,
	1.2968395546510096659f, 1.3542555469368926513f,
	1.4142135623730950488f, 1.4768261459394993114f,
	1.5422108254079408236f, 1.6104903319492543082f,
	1.6817928305074290861f, 1.7562521603732994831f,
	1.8340080864093424635f, 1.9152065613971472939f
};

static inline __m256 exp2_tab16_lookup(__m256i j) noexcept
{
	const __m256 tab_lo = _mm256_setr_ps(
		exp2_tab16[0], exp2_tab16[1], exp2_tab16[2], exp2_tab16[3],
		exp2_tab16[4], exp2_tab16[5], exp2_tab16[6], exp2_tab16[7]
	);

	const __m256 tab_hi = _mm256_setr_ps(
		exp2_tab16[8], exp2_tab16[9], exp2_tab16[10], exp2_tab16[11],
		exp2_tab16[12], exp2_tab16[13], exp2_tab16[14], exp2_tab16[15]
	);

	const __m256i seven = _mm256_set1_epi32(7);
	const __m256i j_lo = _mm256_and_si256(j, seven);
	const __m256i hi_mask_i = _mm256_cmpgt_epi32(j, seven);

	const __m256 lo_vals = _mm256_permutevar8x32_ps(tab_lo, j_lo);
	const __m256 hi_vals = _mm256_permutevar8x32_ps(tab_hi, j_lo);

	return _mm256_blendv_ps(lo_vals, hi_vals, _mm256_castsi256_ps(hi_mask_i));
}

static inline __m256 exp10_core_eval(__m256 input, __m256i* out_k) noexcept
{
	const __m256 one = _mm256_set1_ps(1.0f);

	const __m256 L = _mm256_set1_ps(53.150848388671875f);

	const __m256 C_hi = _mm256_set1_ps(0.0188143756240606308f);
	const __m256 C_lo = _mm256_set1_ps(-8.950618061473711e-10f);

	const __m256 log2_10_hi = _mm256_set1_ps(3.3219280242919921875f);
	const __m256 log2_10_lo = _mm256_set1_ps(7.059537015918825e-08f);

	const __m256 c5 = _mm256_set1_ps(1.33335581e-03f);
	const __m256 c4 = _mm256_set1_ps(9.61812911e-03f);
	const __m256 c3 = _mm256_set1_ps(5.55041087e-02f);
	const __m256 c2 = _mm256_set1_ps(2.40226507e-01f);
	const __m256 c1 = _mm256_set1_ps(6.93147182e-01f);

	__m256 md_rounded = _mm256_round_ps(
		_mm256_mul_ps(input, L),
		_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
	);

	__m256i m = _mm256_cvtps_epi32(md_rounded);
	__m256 mf = _mm256_cvtepi32_ps(m);

	__m256 r = _mm256_fnmadd_ps(mf, C_hi, input);
	r = _mm256_fnmadd_ps(mf, C_lo, r);

	__m256 z = _mm256_fmadd_ps(r, log2_10_hi, _mm256_mul_ps(r, log2_10_lo));

	__m256i j = _mm256_and_si256(m, _mm256_set1_epi32(15));
	__m256i k = _mm256_srai_epi32(m, 4);

	__m256 t = exp2_tab16_lookup(j);

	__m256 p = c5;
	p = _mm256_fmadd_ps(p, z, c4);
	p = _mm256_fmadd_ps(p, z, c3);
	p = _mm256_fmadd_ps(p, z, c2);
	p = _mm256_fmadd_ps(p, z, c1);
	p = _mm256_fmadd_ps(p, z, one);

	*out_k = k;
	return _mm256_mul_ps(t, p);
}

static inline __m256 exp10_scale_result_fast(__m256 core, __m256i k) noexcept
{
	__m256i expbits = _mm256_slli_epi32(
		_mm256_add_epi32(k, _mm256_set1_epi32(127)),
		23
	);

	__m256 twopk = _mm256_castsi256_ps(expbits);
	return _mm256_mul_ps(core, twopk);
}

static inline __m256 exp10_scale_result_slow(__m256 core, __m256i k) noexcept
{
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 two = _mm256_set1_ps(2.0f);

	const __m256 subnorm_scale = _mm256_set1_ps(0x1p-24f);
	const __m256 scale_2pow127 = _mm256_castsi256_ps(_mm256_set1_epi32(0x7F000000));

	__m256i hi_mask_i = _mm256_cmpgt_epi32(k, _mm256_set1_epi32(127));
	__m256 hi_mask = _mm256_castsi256_ps(hi_mask_i);

	__m256i hi_one = _mm256_srli_epi32(hi_mask_i, 31);
	__m256i k_adj_hi = _mm256_sub_epi32(k, hi_one);
	__m256 extra_hi = _mm256_blendv_ps(one, two, hi_mask);

	__m256i sub_mask_i = _mm256_cmpgt_epi32(_mm256_set1_epi32(-126), k_adj_hi);
	__m256 sub_mask = _mm256_castsi256_ps(sub_mask_i);

	__m256i k_adj_sub = _mm256_add_epi32(
		k_adj_hi,
		_mm256_blendv_epi8(_mm256_setzero_si256(), _mm256_set1_epi32(24), sub_mask_i)
	);

	__m256i expbits = _mm256_slli_epi32(
		_mm256_add_epi32(k_adj_sub, _mm256_set1_epi32(127)),
		23
	);

	__m256 twopk = _mm256_castsi256_ps(expbits);
	twopk = _mm256_blendv_ps(twopk, _mm256_mul_ps(twopk, subnorm_scale), sub_mask);

	__m256 result = _mm256_mul_ps(core, extra_hi);
	result = _mm256_mul_ps(result, twopk);

	__m256i k_eq_128_i = _mm256_cmpeq_epi32(k, _mm256_set1_epi32(128));
	__m256 result_k128 = _mm256_mul_ps(scale_2pow127, _mm256_mul_ps(two, core));

	return _mm256_blendv_ps(result, result_k128, _mm256_castsi256_ps(k_eq_128_i));
}

static inline __m256 exp10_scale_result(__m256 core, __m256i k) noexcept
{
	__m256i lt_min_i = _mm256_cmpgt_epi32(_mm256_set1_epi32(-126), k);
	__m256i gt_max_i = _mm256_cmpgt_epi32(k, _mm256_set1_epi32(127));

	const int out_of_range_mask = _mm256_movemask_ps(
		_mm256_castsi256_ps(_mm256_or_si256(lt_min_i, gt_max_i))
	);

	if (out_of_range_mask == 0)
	{
		return exp10_scale_result_fast(core, k);
	}

	return exp10_scale_result_slow(core, k);
}

static inline __m256 exp10_finalize_with_masks(
	__m256 result,
	__m256 input,
	__m256 absx,
	__m256 nan_mask,
	__m256 pinf_mask,
	__m256 ninf_mask,
	__m256 over_mask,
	__m256 under_mask,
	__m256 tiny_mask) noexcept
{
	const __m256 zero = _mm256_setzero_ps();
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 two = _mm256_set1_ps(2.0f);
	(void)two;
	(void)absx;

	const __m256 inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());
	const __m256 ln10 = _mm256_set1_ps(2.30258509299404568402f);
	const __m256i qnan_quiet_bit = _mm256_set1_epi32(0x00400000u);

	__m256 nan_quieted = _mm256_castsi256_ps(
		_mm256_or_si256(_mm256_castps_si256(input), qnan_quiet_bit)
	);

	result = _mm256_blendv_ps(result, _mm256_fmadd_ps(input, ln10, one), tiny_mask);
	result = _mm256_blendv_ps(result, zero, under_mask);
	result = _mm256_blendv_ps(result, inf, over_mask);
	result = _mm256_blendv_ps(result, zero, ninf_mask);
	result = _mm256_blendv_ps(result, inf, pinf_mask);
	result = _mm256_blendv_ps(result, nan_quieted, nan_mask);

	return result;
}

__m256 fy::simd::intrinsic::exp10(__m256 input) noexcept
{
	const __m256 zero = _mm256_setzero_ps();
	const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
	const __m256 inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());
	const __m256 ninf = _mm256_set1_ps(-std::numeric_limits<float>::infinity());

	const __m256 tiny_x = _mm256_set1_ps(0x1p-25f);
	const __m256 over_th = _mm256_set1_ps(38.5318394f);
	const __m256 under_th = _mm256_set1_ps(-44.8534698f);
	const __m256 ln10 = _mm256_set1_ps(2.30258509299404568402f);
	const __m256 one = _mm256_set1_ps(1.0f);

	const __m256 fast_norm_lo = _mm256_set1_ps(-37.9391884f);
	const __m256 fast_norm_hi = _mm256_set1_ps(38.5224342f);

	__m256 absx = _mm256_and_ps(input, abs_mask);

	__m256 nan_mask = _mm256_cmp_ps(input, input, _CMP_UNORD_Q);
	__m256 pinf_mask = _mm256_cmp_ps(input, inf, _CMP_EQ_OQ);
	__m256 ninf_mask = _mm256_cmp_ps(input, ninf, _CMP_EQ_OQ);
	__m256 over_mask = _mm256_cmp_ps(input, over_th, _CMP_GT_OQ);
	__m256 under_mask = _mm256_cmp_ps(input, under_th, _CMP_LT_OQ);
	__m256 tiny_mask = _mm256_cmp_ps(absx, tiny_x, _CMP_LT_OQ);

	{
		__m256 ge_lo = _mm256_cmp_ps(input, fast_norm_lo, _CMP_GE_OQ);
		__m256 le_hi = _mm256_cmp_ps(input, fast_norm_hi, _CMP_LE_OQ);
		__m256 in_fast_norm = _mm256_and_ps(ge_lo, le_hi);

		if (_mm256_movemask_ps(in_fast_norm) == 0xFF)
		{
			__m256i k;
			__m256 core = exp10_core_eval(input, &k);
			__m256 result = exp10_scale_result_fast(core, k);

			return _mm256_blendv_ps(result, _mm256_fmadd_ps(input, ln10, one), tiny_mask);
		}
	}

	__m256 special_mask = _mm256_or_ps(
		nan_mask,
		_mm256_or_ps(
			pinf_mask,
			_mm256_or_ps(
				ninf_mask,
				_mm256_or_ps(over_mask, under_mask)
			)
		)
	);

	__m256 safe_input = _mm256_blendv_ps(input, zero, special_mask);

	__m256i k;
	__m256 core = exp10_core_eval(safe_input, &k);
	__m256 result = exp10_scale_result(core, k);

	return exp10_finalize_with_masks(
		result,
		input,
		absx,
		nan_mask,
		pinf_mask,
		ninf_mask,
		over_mask,
		under_mask,
		tiny_mask
	);
}
