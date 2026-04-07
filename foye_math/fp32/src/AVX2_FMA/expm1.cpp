#include <foye_fastmath.hpp>

static __m256 expm1_small_branch(__m256 input) noexcept
{
	const __m256 c2 = _mm256_set1_ps(0.5f);
	const __m256 c3 = _mm256_set1_ps(1.6666667163e-1f);
	const __m256 c4 = _mm256_set1_ps(4.1666667908e-2f);
	const __m256 c5 = _mm256_set1_ps(8.3333337670e-3f);

	__m256 x2 = _mm256_mul_ps(input, input);
	__m256 p = _mm256_fmadd_ps(c5, input, c4);
	p = _mm256_fmadd_ps(p, input, c3);
	p = _mm256_fmadd_ps(p, input, c2);

	return _mm256_fmadd_ps(x2, p, input);
}

static inline __m256 expm1_medium_branch(__m256 input) noexcept
{
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 half = _mm256_set1_ps(0.5f);
	const __m256 three = _mm256_set1_ps(3.0f);
	const __m256 six = _mm256_set1_ps(6.0f);
	const __m256 q1 = _mm256_set1_ps(-3.3333212137e-2f);
	const __m256 q2 = _mm256_set1_ps(1.5807170421e-3f);

	__m256 hfx = _mm256_mul_ps(half, input);
	__m256 hxs = _mm256_mul_ps(input, hfx);
	__m256 r1 = _mm256_fmadd_ps(hxs, _mm256_fmadd_ps(hxs, q2, q1), one);
	__m256 t = _mm256_fnmadd_ps(r1, hfx, three);
	__m256 e = _mm256_mul_ps(hxs, _mm256_div_ps(_mm256_sub_ps(r1, t),
		_mm256_fnmadd_ps(input, t, six)));

	return _mm256_sub_ps(input, _mm256_fmsub_ps(input, e, hxs));
}

static inline __m256i expm1_branchb_special_mask_i(__m256 input, __m256i hx, __m256i mask_pos_i) noexcept
{
	const __m256 overflow_threshold_v = _mm256_set1_ps(88.72283935546875f);
	const __m256i i_inf = _mm256_set1_epi32(0x7f800000);
	const __m256i i_27ln2 = _mm256_set1_epi32(0x4195b844);

	__m256i mask_nan_i = _mm256_cmpgt_epi32(hx, i_inf);
	__m256i mask_inf_i = _mm256_cmpeq_epi32(hx, i_inf);

	__m256i mask_huge_i = _mm256_or_si256(
		_mm256_cmpgt_epi32(hx, i_27ln2),
		_mm256_cmpeq_epi32(hx, i_27ln2));

	__m256i mask_negbig_i = _mm256_andnot_si256(mask_pos_i, mask_huge_i);
	__m256i mask_pos_over_i = _mm256_castps_si256(
		_mm256_cmp_ps(input, overflow_threshold_v, _CMP_GT_OQ));

	return _mm256_or_si256(
		_mm256_or_si256(mask_nan_i, mask_inf_i),
		_mm256_or_si256(mask_negbig_i, mask_pos_over_i));
}

static __m256 expm1_finalize_special(__m256 result, __m256 input, __m256i hx, __m256i mask_pos_i) noexcept
{
	const __m256 neg_one = _mm256_set1_ps(-1.0f);
	const __m256 infv = _mm256_set1_ps(std::numeric_limits<float>::infinity());
	const __m256 overflow_threshold_v = _mm256_set1_ps(88.72283935546875f);

	const __m256i i_inf = _mm256_set1_epi32(0x7f800000);
	const __m256i i_27ln2 = _mm256_set1_epi32(0x4195b844);
	const __m256i qnan_quiet_bit = _mm256_set1_epi32(0x00400000);

	__m256i mask_nan_i = _mm256_cmpgt_epi32(hx, i_inf);
	__m256i mask_inf_i = _mm256_cmpeq_epi32(hx, i_inf);
	__m256i mask_posinf_i = _mm256_and_si256(mask_inf_i, mask_pos_i);
	__m256i mask_neginf_i = _mm256_andnot_si256(mask_pos_i, mask_inf_i);

	__m256i mask_huge_i = _mm256_or_si256(
		_mm256_cmpgt_epi32(hx, i_27ln2),
		_mm256_cmpeq_epi32(hx, i_27ln2));

	__m256i mask_negbig_i = _mm256_andnot_si256(mask_pos_i, mask_huge_i);
	__m256 mask_overflow = _mm256_cmp_ps(input, overflow_threshold_v, _CMP_GT_OQ);

	__m256 nan_quieted = _mm256_castsi256_ps(
		_mm256_or_si256(_mm256_castps_si256(input), qnan_quiet_bit)
	);

	result = _mm256_blendv_ps(result, neg_one, _mm256_castsi256_ps(mask_negbig_i));
	result = _mm256_blendv_ps(result, infv, mask_overflow);
	result = _mm256_blendv_ps(result, neg_one, _mm256_castsi256_ps(mask_neginf_i));
	result = _mm256_blendv_ps(result, input, _mm256_castsi256_ps(mask_posinf_i));
	result = _mm256_blendv_ps(result, nan_quieted, _mm256_castsi256_ps(mask_nan_i));

	return result;
}

static __m256 expm1_general_branch_a(__m256 input) noexcept
{
	const __m256 zero = _mm256_setzero_ps();
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 half = _mm256_set1_ps(0.5f);
	const __m256 mhalf = _mm256_set1_ps(-0.5f);
	const __m256 two = _mm256_set1_ps(2.0f);
	const __m256 three = _mm256_set1_ps(3.0f);
	const __m256 six = _mm256_set1_ps(6.0f);

	const __m256 mquarter = _mm256_set1_ps(-0.25f);

	const __m256 q1 = _mm256_set1_ps(-3.3333212137e-2f);
	const __m256 q2 = _mm256_set1_ps(1.5807170421e-3f);

	const __m256 ln2_hi_v = _mm256_set1_ps(6.9313812256e-01f);
	const __m256 ln2_lo_v = _mm256_set1_ps(9.0580006145e-06f);
	const __m256 neg_ln2_lo_v = _mm256_set1_ps(-9.0580006145e-06f);
	const __m256 invln2_v = _mm256_set1_ps(1.4426950216e+00f);

	const __m256 twop127 = _mm256_set1_ps(0x1p127f);

	const __m256i sign_mask_i = _mm256_set1_epi32(0x80000000u);
	const __m256i abs_mask_i = _mm256_set1_epi32(0x7fffffffu);
	const __m256i i_zero = _mm256_setzero_si256();
	const __m256i i_one = _mm256_set1_epi32(1);
	const __m256i i_mone = _mm256_set1_epi32(-1);
	const __m256i i_23 = _mm256_set1_epi32(23);
	const __m256i i_56 = _mm256_set1_epi32(56);
	const __m256i i_127 = _mm256_set1_epi32(127);
	const __m256i i_128 = _mm256_set1_epi32(128);
	const __m256i i_bias = _mm256_set1_epi32(0x3f800000);
	const __m256i i_half_ln2 = _mm256_set1_epi32(0x3eb17218);
	const __m256i i_onehalf_ln2 = _mm256_set1_epi32(0x3F851592);
	const __m256i i_tiny = _mm256_set1_epi32(0x33000000);
	const __m256i i_1000000 = _mm256_set1_epi32(0x01000000);

	__m256i ix = _mm256_castps_si256(input);
	__m256i xsb = _mm256_and_si256(ix, sign_mask_i);
	__m256i hx = _mm256_and_si256(ix, abs_mask_i);
	__m256i mask_pos_i = _mm256_cmpeq_epi32(xsb, i_zero);

	__m256i mask_big_i = _mm256_cmpgt_epi32(hx, i_half_ln2);
	__m256i mask_mid_i = _mm256_cmpgt_epi32(i_onehalf_ln2, hx);
	__m256i mask_bigmid_i = _mm256_and_si256(mask_big_i, mask_mid_i);
	__m256i mask_bignmid_i = _mm256_andnot_si256(mask_mid_i, mask_big_i);
	__m256i mask_tiny_i = _mm256_cmpgt_epi32(i_tiny, hx);

	__m256 hi1 = _mm256_blendv_ps(_mm256_add_ps(input, ln2_hi_v),
		_mm256_sub_ps(input, ln2_hi_v),
		_mm256_castsi256_ps(mask_pos_i));
	__m256 lo1 = _mm256_blendv_ps(neg_ln2_lo_v,
		ln2_lo_v,
		_mm256_castsi256_ps(mask_pos_i));
	__m256i k1i = _mm256_blendv_epi8(i_mone, i_one, mask_pos_i);

	__m256 adj = _mm256_blendv_ps(mhalf, half, _mm256_castsi256_ps(mask_pos_i));
	__m256 fk2 = _mm256_fmadd_ps(invln2_v, input, adj);
	__m256i k2i = _mm256_cvttps_epi32(fk2);
	__m256 t2 = _mm256_cvtepi32_ps(k2i);
	__m256 hi2 = _mm256_fnmadd_ps(t2, ln2_hi_v, input);
	__m256 lo2 = _mm256_mul_ps(t2, ln2_lo_v);

	__m256 hi = _mm256_blendv_ps(input, hi2, _mm256_castsi256_ps(mask_bignmid_i));
	hi = _mm256_blendv_ps(hi, hi1, _mm256_castsi256_ps(mask_bigmid_i));

	__m256 lo = _mm256_blendv_ps(zero, lo2, _mm256_castsi256_ps(mask_bignmid_i));
	lo = _mm256_blendv_ps(lo, lo1, _mm256_castsi256_ps(mask_bigmid_i));

	__m256i k = _mm256_blendv_epi8(i_zero, k2i, mask_bignmid_i);
	k = _mm256_blendv_epi8(k, k1i, mask_bigmid_i);

	__m256 x = _mm256_blendv_ps(input, _mm256_sub_ps(hi, lo), _mm256_castsi256_ps(mask_big_i));
	__m256 c = _mm256_blendv_ps(zero,
		_mm256_sub_ps(_mm256_sub_ps(hi, x), lo),
		_mm256_castsi256_ps(mask_big_i));

	__m256 hfx = _mm256_mul_ps(half, x);
	__m256 hxs = _mm256_mul_ps(x, hfx);
	__m256 r1 = _mm256_fmadd_ps(hxs, _mm256_fmadd_ps(hxs, q2, q1), one);
	__m256 t = _mm256_fnmadd_ps(r1, hfx, three);
	__m256 e = _mm256_mul_ps(hxs,
		_mm256_div_ps(_mm256_sub_ps(r1, t),
			_mm256_fnmadd_ps(x, t, six)));

	__m256 res_k0 = _mm256_sub_ps(x, _mm256_fmsub_ps(x, e, hxs));

	__m256 emc = _mm256_sub_ps(e, c);
	__m256 e2 = _mm256_fmsub_ps(x, emc, c);
	e2 = _mm256_sub_ps(e2, hxs);

	__m256i twopk_bits = _mm256_add_epi32(i_bias, _mm256_slli_epi32(k, 23));
	__m256 twopk = _mm256_castsi256_ps(twopk_bits);

	__m256 xe2 = _mm256_sub_ps(x, e2);
	__m256 res_km1 = _mm256_fmsub_ps(half, xe2, half);

	__m256 res_k1a = _mm256_mul_ps(_mm256_set1_ps(-2.0f), _mm256_sub_ps(e2, _mm256_add_ps(x, half)));
	__m256 res_k1b = _mm256_fmadd_ps(two, _mm256_sub_ps(x, e2), one);
	__m256 mask_xlt_mquarter = _mm256_cmp_ps(x, mquarter, _CMP_LT_OQ);
	__m256 res_k1 = _mm256_blendv_ps(res_k1b, res_k1a, mask_xlt_mquarter);

	__m256 y_big = _mm256_sub_ps(_mm256_add_ps(one, x), e2);
	__m256 res_bigk = _mm256_fmsub_ps(y_big, twopk, one);
	__m256i mask_k128_i = _mm256_cmpeq_epi32(k, i_128);
	__m256 y2 = _mm256_mul_ps(y_big, two);
	__m256 res_bigk_128 = _mm256_fmsub_ps(y2, twop127, one);
	res_bigk = _mm256_blendv_ps(res_bigk, res_bigk_128, _mm256_castsi256_ps(mask_k128_i));

	__m256i shr = _mm256_srlv_epi32(i_1000000, k);
	__m256i t_lt23_bits = _mm256_sub_epi32(i_bias, shr);
	__m256 t_lt23 = _mm256_castsi256_ps(t_lt23_bits);
	__m256 res_lt23 = _mm256_mul_ps(_mm256_sub_ps(t_lt23, _mm256_sub_ps(e2, x)), twopk);

	__m256i t_ge23_bits = _mm256_slli_epi32(_mm256_sub_epi32(i_127, k), 23);
	__m256 t_ge23 = _mm256_castsi256_ps(t_ge23_bits);
	__m256 y_ge23 = _mm256_add_ps(_mm256_sub_ps(x, _mm256_add_ps(e2, t_ge23)), one);
	__m256 res_ge23 = _mm256_mul_ps(y_ge23, twopk);

	__m256i mask_k0_i = _mm256_cmpeq_epi32(k, i_zero);
	__m256i mask_km1_i = _mm256_cmpeq_epi32(k, i_mone);
	__m256i mask_k1_i = _mm256_cmpeq_epi32(k, i_one);
	__m256i mask_kle_n2_i = _mm256_cmpgt_epi32(i_mone, k);
	__m256i mask_kgt56_i = _mm256_cmpgt_epi32(k, i_56);
	__m256i mask_kbig_i = _mm256_or_si256(mask_kle_n2_i, mask_kgt56_i);
	__m256i mask_kgt1_i = _mm256_cmpgt_epi32(k, i_one);
	__m256i mask_klt23_i = _mm256_cmpgt_epi32(i_23, k);
	__m256i mask_smallpos_i = _mm256_and_si256(mask_kgt1_i, mask_klt23_i);

	__m256 res_nz = res_ge23;
	res_nz = _mm256_blendv_ps(res_nz, res_lt23, _mm256_castsi256_ps(mask_smallpos_i));
	res_nz = _mm256_blendv_ps(res_nz, res_bigk, _mm256_castsi256_ps(mask_kbig_i));
	res_nz = _mm256_blendv_ps(res_nz, res_k1, _mm256_castsi256_ps(mask_k1_i));
	res_nz = _mm256_blendv_ps(res_nz, res_km1, _mm256_castsi256_ps(mask_km1_i));

	__m256 result = _mm256_blendv_ps(res_k0,
		res_nz,
		_mm256_castsi256_ps(_mm256_xor_si256(mask_k0_i, _mm256_set1_epi32(-1))));
	result = _mm256_blendv_ps(result, input, _mm256_castsi256_ps(mask_tiny_i));

	return expm1_finalize_special(result, input, hx, mask_pos_i);
}

static __m256 expm1_general_branch_b_core(__m256 input) noexcept
{
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 half = _mm256_set1_ps(0.5f);
	const __m256 mhalf = _mm256_set1_ps(-0.5f);
	const __m256 two = _mm256_set1_ps(2.0f);
	const __m256 three = _mm256_set1_ps(3.0f);
	const __m256 six = _mm256_set1_ps(6.0f);

	const __m256 q1 = _mm256_set1_ps(-3.3333212137e-2f);
	const __m256 q2 = _mm256_set1_ps(1.5807170421e-3f);

	const __m256 ln2_hi_v = _mm256_set1_ps(6.9313812256e-01f);
	const __m256 ln2_lo_v = _mm256_set1_ps(9.0580006145e-06f);
	const __m256 invln2_v = _mm256_set1_ps(1.4426950216e+00f);
	const __m256 twop127 = _mm256_set1_ps(0x1p127f);

	const __m256i sign_mask_i = _mm256_set1_epi32(0x80000000u);
	const __m256i abs_mask_i = _mm256_set1_epi32(0x7fffffffu);
	const __m256i i_zero = _mm256_setzero_si256();
	const __m256i i_127 = _mm256_set1_epi32(127);
	const __m256i i_128 = _mm256_set1_epi32(128);

	__m256i ix = _mm256_castps_si256(input);
	__m256i xsb = _mm256_and_si256(ix, sign_mask_i);
	__m256i hx = _mm256_and_si256(ix, abs_mask_i);
	__m256i mask_pos_i = _mm256_cmpeq_epi32(xsb, i_zero);
	__m256 mask_pos = _mm256_castsi256_ps(mask_pos_i);

	__m256 adj = _mm256_blendv_ps(mhalf, half, mask_pos);
	__m256 fk = _mm256_fmadd_ps(invln2_v, input, adj);
	__m256i k = _mm256_cvttps_epi32(fk);
	__m256 kf = _mm256_cvtepi32_ps(k);

	__m256 r = _mm256_fnmadd_ps(kf, ln2_hi_v, input);
	r = _mm256_fnmadd_ps(kf, ln2_lo_v, r);

	__m256 hfx = _mm256_mul_ps(half, r);
	__m256 hxs = _mm256_mul_ps(r, hfx);
	__m256 r1 = _mm256_fmadd_ps(hxs, _mm256_fmadd_ps(hxs, q2, q1), one);
	__m256 t = _mm256_fnmadd_ps(r1, hfx, three);
	__m256 e = _mm256_mul_ps(hxs,
		_mm256_div_ps(_mm256_sub_ps(r1, t),
			_mm256_fnmadd_ps(r, t, six)));

	__m256 em1r = _mm256_sub_ps(r, _mm256_fmsub_ps(r, e, hxs));
	__m256 yr = _mm256_add_ps(one, em1r);

	__m256i twopk_bits = _mm256_slli_epi32(_mm256_add_epi32(k, i_127), 23);
	__m256 twopk = _mm256_castsi256_ps(twopk_bits);

	__m256 result = _mm256_fmsub_ps(yr, twopk, one);

	__m256i mask_k128_i = _mm256_cmpeq_epi32(k, i_128);
	__m256 yr2 = _mm256_mul_ps(yr, two);
	__m256 result_k128 = _mm256_fmsub_ps(yr2, twop127, one);
	result = _mm256_blendv_ps(result, result_k128, _mm256_castsi256_ps(mask_k128_i));

	return expm1_finalize_special(result, input, hx, mask_pos_i);
}

static __m256 expm1_general_branch_b(__m256 input) noexcept
{
	const __m256 zero = _mm256_setzero_ps();
	const __m256 infv = _mm256_set1_ps(std::numeric_limits<float>::infinity());
	const __m256 overflow_threshold_v = _mm256_set1_ps(88.72283935546875f);
	const __m256 neg_27ln2_v = _mm256_set1_ps(-18.7149734497f);

	{
		__m256 mask_gt_negbig = _mm256_cmp_ps(input, neg_27ln2_v, _CMP_GT_OQ);
		__m256 mask_le_ovf = _mm256_cmp_ps(input, overflow_threshold_v, _CMP_LE_OQ);
		__m256 mask_regular = _mm256_and_ps(mask_gt_negbig, mask_le_ovf);

		if (_mm256_movemask_ps(mask_regular) == 0xFF)
		{
			return expm1_general_branch_b_core(input);
		}
	}

	{
		__m256 mask_ord = _mm256_cmp_ps(input, input, _CMP_ORD_Q);
		__m256 mask_pos = _mm256_cmp_ps(input, zero, _CMP_GT_OQ);
		__m256 mask_all_pos_ord = _mm256_and_ps(mask_ord, mask_pos);

		if (_mm256_movemask_ps(mask_all_pos_ord) == 0xFF)
		{
			__m256 mask_le_ovf = _mm256_cmp_ps(input, overflow_threshold_v, _CMP_LE_OQ);
			int mm_le_ovf = _mm256_movemask_ps(mask_le_ovf);

			if (mm_le_ovf != 0xFF)
			{
				__m256 safe_input = _mm256_blendv_ps(zero, input, mask_le_ovf);
				__m256 result = expm1_general_branch_b_core(safe_input);

				return _mm256_blendv_ps(infv, result, mask_le_ovf);
			}
		}
	}

	const __m256i sign_mask_i = _mm256_set1_epi32(0x80000000u);
	const __m256i abs_mask_i = _mm256_set1_epi32(0x7fffffffu);
	const __m256i i_zero = _mm256_setzero_si256();

	__m256i ix = _mm256_castps_si256(input);
	__m256i xsb = _mm256_and_si256(ix, sign_mask_i);
	__m256i hx = _mm256_and_si256(ix, abs_mask_i);
	__m256i mask_pos_i = _mm256_cmpeq_epi32(xsb, i_zero);

	__m256i mask_special_i = expm1_branchb_special_mask_i(input, hx, mask_pos_i);
	int mm_special = _mm256_movemask_ps(_mm256_castsi256_ps(mask_special_i));

	if (mm_special == 0xFF)
	{
		return expm1_finalize_special(zero, input, hx, mask_pos_i);
	}

	__m256 safe_input = _mm256_blendv_ps(input, zero, _mm256_castsi256_ps(mask_special_i));
	__m256 result = expm1_general_branch_b_core(safe_input);

	return expm1_finalize_special(result, input, hx, mask_pos_i);
}

__m256 fy::simd::intrinsic::expm1(__m256 input) noexcept
{
	const __m256 tiny_thr = _mm256_set1_ps(0x1p-25f);
	const __m256 small_thr = _mm256_set1_ps(1.0e-3f);
	const __m256 half_ln2 = _mm256_set1_ps(0.34657359028f);

	__m256 ax = _mm256_and_ps(input, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));

	if ((_mm256_movemask_ps(_mm256_cmp_ps(ax, tiny_thr, _CMP_LT_OQ)) == 0xFF))
	{
		return input;
	}

	if ((_mm256_movemask_ps(_mm256_cmp_ps(ax, small_thr, _CMP_LE_OQ)) == 0xFF))
	{
		return expm1_small_branch(input);
	}

	if ((_mm256_movemask_ps(_mm256_cmp_ps(ax, half_ln2, _CMP_LE_OQ)) == 0xFF))
	{
		return expm1_medium_branch(input);
	}

	{
		const __m256 overflow_threshold_v = _mm256_set1_ps(88.72283935546875f);
		if (_mm256_movemask_ps(_mm256_cmp_ps(input, overflow_threshold_v, _CMP_GT_OQ)) == 0xFF)
		{
			return _mm256_set1_ps(std::numeric_limits<float>::infinity());
		}
	}

	{
		const __m256 neg_27ln2_v = _mm256_set1_ps(-18.7149734497f);
		if (_mm256_movemask_ps(_mm256_cmp_ps(input, neg_27ln2_v, _CMP_LE_OQ)) == 0xFF)
		{
			return _mm256_set1_ps(-1.0f);
		}
	}

	const __m256 zero = _mm256_setzero_ps();
	const __m256 one_point_five = _mm256_set1_ps(1.5f);

	__m256 mask_old_region = _mm256_and_ps(
		_mm256_cmp_ps(input, half_ln2, _CMP_GT_OQ),
		_mm256_cmp_ps(input, one_point_five, _CMP_LE_OQ));

	__m256 mask_new_region = _mm256_or_ps(
		_mm256_cmp_ps(input, _mm256_sub_ps(zero, half_ln2), _CMP_LT_OQ),
		_mm256_cmp_ps(input, one_point_five, _CMP_GT_OQ));

	const int mm_old = _mm256_movemask_ps(mask_old_region);
	const int mm_new = _mm256_movemask_ps(mask_new_region);

	if (mm_old == 0xFF || mm_new != 0xFF)
	{
		return expm1_general_branch_a(input);
	}

	return expm1_general_branch_b(input);
}