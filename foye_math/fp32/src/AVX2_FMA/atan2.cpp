#include "foye_fastmath_fp32.hpp"

__m256 fy::simd::intrinsic::atan2(__m256 y, __m256 x)
{
	const __m256i sign_mask_i = _mm256_set1_epi32(0x80000000u);
	const __m256i abs_mask_i = _mm256_set1_epi32(0x7fffffff);
	const __m256i inf_i = _mm256_set1_epi32(0x7f800000);
	const __m256i one_i = _mm256_set1_epi32(0x3f800000);
	const __m256i zero_i = _mm256_setzero_si256();
	const __m256i all1_i = _mm256_set1_epi32(-1);
	const __m256i c26_i = _mm256_set1_epi32(26);
	const __m256i cneg26_i = _mm256_set1_epi32(-26);

	const __m256 zero_ps = _mm256_setzero_ps();
	const __m256 one_ps = _mm256_castsi256_ps(one_i);

	const __m256 pi_o_4_ps = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f490fdb));
	const __m256 pi_o_2_ps = _mm256_castsi256_ps(_mm256_set1_epi32(0x3fc90fdb));
	const __m256 pi_ps = _mm256_castsi256_ps(_mm256_set1_epi32(0x40490fdb));
	const __m256 pi_lo_ps = _mm256_castsi256_ps(_mm256_set1_epi32(0xb3bbbd2e));
	const __m256 tiny_ps = _mm256_set1_ps(1.0e-30f);

	const __m256 pio2_half_pilo_ps = _mm256_fmadd_ps(_mm256_set1_ps(0.5f), pi_lo_ps, pi_o_2_ps);
	const __m256 pi_tiny_ps = _mm256_add_ps(pi_ps, tiny_ps);
	const __m256 mpi_tiny_ps = _mm256_sub_ps(_mm256_sub_ps(zero_ps, pi_ps), tiny_ps);
	const __m256 pio2_tiny_ps = _mm256_add_ps(pi_o_2_ps, tiny_ps);
	const __m256 mpio2_tiny_ps = _mm256_sub_ps(_mm256_sub_ps(zero_ps, pi_o_2_ps), tiny_ps);
	const __m256 pio4_tiny_ps = _mm256_add_ps(pi_o_4_ps, tiny_ps);
	const __m256 mpio4_tiny_ps = _mm256_sub_ps(_mm256_sub_ps(zero_ps, pi_o_4_ps), tiny_ps);
	const __m256 three_pio4_ps = _mm256_add_ps(pi_o_2_ps, pi_o_4_ps);
	const __m256 three_pio4_tiny_ps = _mm256_add_ps(three_pio4_ps, tiny_ps);
	const __m256 mthree_pio4_tiny_ps = _mm256_sub_ps(_mm256_sub_ps(zero_ps, three_pio4_ps), tiny_ps);

	const __m256i hx = _mm256_castps_si256(x);
	const __m256i hy = _mm256_castps_si256(y);
	const __m256i ix = _mm256_and_si256(hx, abs_mask_i);
	const __m256i iy = _mm256_and_si256(hy, abs_mask_i);

	const __m256i x_nan_i = _mm256_cmpgt_epi32(ix, inf_i);
	const __m256i y_nan_i = _mm256_cmpgt_epi32(iy, inf_i);
	const __m256i y_zero_i = _mm256_cmpeq_epi32(iy, zero_i);
	const __m256i x_zero_i = _mm256_cmpeq_epi32(ix, zero_i);
	const __m256i x_inf_i = _mm256_cmpeq_epi32(ix, inf_i);
	const __m256i y_inf_i = _mm256_cmpeq_epi32(iy, inf_i);
	const __m256i x_one_i = _mm256_cmpeq_epi32(hx, one_i);

	const __m256i special_i = _mm256_or_si256(
		_mm256_or_si256(
			_mm256_or_si256(x_nan_i, y_nan_i),
			_mm256_or_si256(y_zero_i, x_zero_i)
		),
		_mm256_or_si256(
			_mm256_or_si256(x_inf_i, y_inf_i),
			x_one_i
		)
	);

	const __m256i x_neg_i = _mm256_srai_epi32(hx, 31);
	const __m256i y_neg_i = _mm256_srai_epi32(hy, 31);

	if (_mm256_movemask_ps(_mm256_castsi256_ps(special_i)) == 0)
	{
		const __m256 abs_y = _mm256_and_ps(y, _mm256_castsi256_ps(abs_mask_i));
		const __m256 abs_x = _mm256_and_ps(x, _mm256_castsi256_ps(abs_mask_i));

		const __m256 z0 = ::fy::simd::intrinsic::atan(_mm256_div_ps(abs_y, abs_x));

		const __m256i k_i = _mm256_srai_epi32(_mm256_sub_epi32(iy, ix), 23);
		const __m256i k_gt_26_i = _mm256_cmpgt_epi32(k_i, c26_i);
		const __m256i k_lt_neg26_i = _mm256_cmpgt_epi32(cneg26_i, k_i);
		const __m256i tiny_ratio_negx_i = _mm256_and_si256(k_lt_neg26_i, x_neg_i);
		const __m256i x_neg_eff_i = _mm256_andnot_si256(k_gt_26_i, x_neg_i);

		__m256 z = z0;
		z = _mm256_blendv_ps(z, pio2_half_pilo_ps, _mm256_castsi256_ps(k_gt_26_i));
		z = _mm256_blendv_ps(z, zero_ps, _mm256_castsi256_ps(tiny_ratio_negx_i));

		const __m256 y_sign_ps = _mm256_and_ps(y, _mm256_castsi256_ps(sign_mask_i));
		const __m256 res_pos = _mm256_xor_ps(z, y_sign_ps);

		const __m256 t = _mm256_sub_ps(z, pi_lo_ps);
		const __m256 res_neg_posy = _mm256_sub_ps(pi_ps, t);
		const __m256 res_neg_negy = _mm256_sub_ps(t, pi_ps);
		const __m256 res_neg = _mm256_blendv_ps(res_neg_posy, res_neg_negy, _mm256_castsi256_ps(y_neg_i));

		return _mm256_blendv_ps(res_pos, res_neg, _mm256_castsi256_ps(x_neg_eff_i));
	}

	const __m256 abs_y = _mm256_and_ps(y, _mm256_castsi256_ps(abs_mask_i));
	const __m256 abs_x = _mm256_and_ps(x, _mm256_castsi256_ps(abs_mask_i));
	const __m256 normal_mask_ps = _mm256_castsi256_ps(_mm256_andnot_si256(special_i, all1_i));

	const __m256 safe_abs_y = _mm256_blendv_ps(zero_ps, abs_y, normal_mask_ps);
	const __m256 safe_abs_x = _mm256_blendv_ps(one_ps, abs_x, normal_mask_ps);

	__m256 z = ::fy::simd::intrinsic::atan(_mm256_div_ps(safe_abs_y, safe_abs_x));

	const __m256i k_i = _mm256_srai_epi32(_mm256_sub_epi32(iy, ix), 23);
	const __m256i k_gt_26_i = _mm256_cmpgt_epi32(k_i, c26_i);
	const __m256i k_lt_neg26_i = _mm256_cmpgt_epi32(cneg26_i, k_i);
	const __m256i tiny_ratio_negx_i = _mm256_and_si256(k_lt_neg26_i, x_neg_i);
	const __m256i x_neg_eff_i = _mm256_andnot_si256(k_gt_26_i, x_neg_i);

	z = _mm256_blendv_ps(z, pio2_half_pilo_ps, _mm256_castsi256_ps(k_gt_26_i));
	z = _mm256_blendv_ps(z, zero_ps, _mm256_castsi256_ps(tiny_ratio_negx_i));

	const __m256 y_sign_ps = _mm256_and_ps(y, _mm256_castsi256_ps(sign_mask_i));
	const __m256 res_pos = _mm256_xor_ps(z, y_sign_ps);

	const __m256 t = _mm256_sub_ps(z, pi_lo_ps);
	const __m256 res_neg_posy = _mm256_sub_ps(pi_ps, t);
	const __m256 res_neg_negy = _mm256_sub_ps(t, pi_ps);
	const __m256 res_neg = _mm256_blendv_ps(res_neg_posy, res_neg_negy, _mm256_castsi256_ps(y_neg_i));

	__m256 result = _mm256_blendv_ps(res_pos, res_neg, _mm256_castsi256_ps(x_neg_eff_i));

	const __m256 nan_res = _mm256_add_ps(x, y);
	result = _mm256_blendv_ps(result, nan_res, _mm256_castsi256_ps(_mm256_or_si256(x_nan_i, y_nan_i)));

	const __m256 signed_pi = _mm256_blendv_ps(pi_tiny_ps, mpi_tiny_ps, _mm256_castsi256_ps(y_neg_i));
	const __m256 yzero_res = _mm256_blendv_ps(y, signed_pi, _mm256_castsi256_ps(x_neg_i));
	result = _mm256_blendv_ps(result, yzero_res, _mm256_castsi256_ps(y_zero_i));

	const __m256 xzero_res = _mm256_blendv_ps(pio2_tiny_ps, mpio2_tiny_ps, _mm256_castsi256_ps(y_neg_i));
	const __m256i x_zero_only_i = _mm256_andnot_si256(y_zero_i, x_zero_i);
	result = _mm256_blendv_ps(result, xzero_res, _mm256_castsi256_ps(x_zero_only_i));

	const __m256 signed_zero = _mm256_and_ps(y, _mm256_castsi256_ps(sign_mask_i));
	const __m256 finite_xinf_res = _mm256_blendv_ps(signed_zero, signed_pi, _mm256_castsi256_ps(x_neg_i));

	const __m256 infinf_posx_res = _mm256_blendv_ps(pio4_tiny_ps, mpio4_tiny_ps, _mm256_castsi256_ps(y_neg_i));
	const __m256 infinf_negx_res = _mm256_blendv_ps(three_pio4_tiny_ps, mthree_pio4_tiny_ps, _mm256_castsi256_ps(y_neg_i));
	const __m256 infinf_res = _mm256_blendv_ps(infinf_posx_res, infinf_negx_res, _mm256_castsi256_ps(x_neg_i));

	const __m256 xinf_res = _mm256_blendv_ps(finite_xinf_res, infinf_res, _mm256_castsi256_ps(y_inf_i));
	result = _mm256_blendv_ps(result, xinf_res, _mm256_castsi256_ps(x_inf_i));

	const __m256 yinf_res = _mm256_blendv_ps(pio2_tiny_ps, mpio2_tiny_ps, _mm256_castsi256_ps(y_neg_i));
	result = _mm256_blendv_ps(result, yinf_res, _mm256_castsi256_ps(_mm256_andnot_si256(x_inf_i, y_inf_i)));

	if (_mm256_movemask_ps(_mm256_castsi256_ps(x_one_i)))
	{
		const __m256 xone_res = ::fy::simd::intrinsic::atan(y);
		result = _mm256_blendv_ps(result, xone_res, _mm256_castsi256_ps(x_one_i));
	}

	result = _mm256_blendv_ps(result, nan_res, _mm256_castsi256_ps(_mm256_or_si256(x_nan_i, y_nan_i)));

	return result;
}