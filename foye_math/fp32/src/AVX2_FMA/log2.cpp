#include <foye_fastmath.hpp>

static inline __m256 log2_log1p_core(__m256 f) noexcept
{
	const __m256 half_ps = _mm256_set1_ps(0.5f);
	const __m256 two_ps = _mm256_set1_ps(2.0f);

	const __m256 Lp1 = _mm256_set1_ps(6.6666668653e-01f);
	const __m256 Lp2 = _mm256_set1_ps(4.0000000596e-01f);
	const __m256 Lp3 = _mm256_set1_ps(2.8571429849e-01f);
	const __m256 Lp4 = _mm256_set1_ps(2.2222198546e-01f);
	const __m256 Lp5 = _mm256_set1_ps(1.8183572590e-01f);
	const __m256 Lp6 = _mm256_set1_ps(1.5313838422e-01f);
	const __m256 Lp7 = _mm256_set1_ps(1.4798198640e-01f);

	const __m256i abs_mask_i = _mm256_set1_epi32(0x7fffffffu);
	const __m256i tiny_thr_i = _mm256_set1_epi32(0x38000000u);

	__m256i af_i = _mm256_and_si256(_mm256_castps_si256(f), abs_mask_i);
	__m256 tiny_mask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(tiny_thr_i, af_i));

	__m256 ff = _mm256_mul_ps(f, f);
	__m256 hfsq = _mm256_mul_ps(half_ps, ff);
	__m256 s = _mm256_div_ps(f, _mm256_add_ps(two_ps, f));
	__m256 z = _mm256_mul_ps(s, s);

	__m256 R = _mm256_fmadd_ps(Lp7, z, Lp6);
	R = _mm256_fmadd_ps(R, z, Lp5);
	R = _mm256_fmadd_ps(R, z, Lp4);
	R = _mm256_fmadd_ps(R, z, Lp3);
	R = _mm256_fmadd_ps(R, z, Lp2);
	R = _mm256_fmadd_ps(R, z, Lp1);
	R = _mm256_mul_ps(R, z);

	__m256 core = _mm256_sub_ps(
		f,
		_mm256_sub_ps(hfsq, _mm256_mul_ps(s, _mm256_add_ps(hfsq, R)))
	);

	__m256 small = _mm256_fnmadd_ps(ff, half_ps, f);

	return _mm256_blendv_ps(core, small, tiny_mask);
}

static inline __m256 log2_finalize_special(
	__m256 result,
	__m256 input,
	__m256i raw_i,
	__m256i abs_i) noexcept
{
	const __m256 zero_ps = _mm256_setzero_ps();
	const __m256 neg_inf_ps = _mm256_castsi256_ps(_mm256_set1_epi32(0xff800000u));
	const __m256 qnan_ps = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000u));

	const __m256i zero_i = _mm256_setzero_si256();
	const __m256i exp_mask_i = _mm256_set1_epi32(0x7f800000u);

	__m256 zero_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(abs_i, zero_i));
	__m256 neg_mask = _mm256_cmp_ps(input, zero_ps, _CMP_LT_OQ);

	__m256i inf_gt_i = _mm256_cmpgt_epi32(abs_i, exp_mask_i);
	__m256i inf_eq_i = _mm256_cmpeq_epi32(abs_i, exp_mask_i);

	__m256i sign_i = _mm256_srai_epi32(raw_i, 31);
	__m256i posinf_i = _mm256_andnot_si256(sign_i, inf_eq_i);
	__m256i neginf_i = _mm256_and_si256(sign_i, inf_eq_i);

	const __m256i qnan_quiet_bit = _mm256_set1_epi32(0x00400000u);
	__m256 nan_quieted = _mm256_castsi256_ps(
		_mm256_or_si256(raw_i, qnan_quiet_bit)
	);

	result = _mm256_blendv_ps(result, neg_inf_ps, zero_mask);
	result = _mm256_blendv_ps(result, qnan_ps, neg_mask);
	result = _mm256_blendv_ps(result, input, _mm256_castsi256_ps(posinf_i));
	result = _mm256_blendv_ps(result, qnan_ps, _mm256_castsi256_ps(neginf_i));
	result = _mm256_blendv_ps(result, nan_quieted, _mm256_castsi256_ps(inf_gt_i));

	return result;
}

static inline bool all_positive_finite_normal(__m256 x) noexcept
{
	const __m256i sign_mask = _mm256_set1_epi32(0x80000000u);
	const __m256i exp_mask = _mm256_set1_epi32(0x7f800000u);
	const __m256i zero_i = _mm256_setzero_si256();
	const __m256i inf_exp_i = _mm256_set1_epi32(0x7f800000u);

	__m256i ix = _mm256_castps_si256(x);
	__m256i sign = _mm256_and_si256(ix, sign_mask);
	__m256i exp = _mm256_and_si256(ix, exp_mask);

	__m256i sign_ok = _mm256_cmpeq_epi32(sign, zero_i);
	__m256i exp_nz = _mm256_cmpgt_epi32(exp, zero_i);
	__m256i exp_ok = _mm256_cmpgt_epi32(inf_exp_i, exp);

	__m256i ok = _mm256_and_si256(sign_ok, _mm256_and_si256(exp_nz, exp_ok));
	return _mm256_movemask_ps(_mm256_castsi256_ps(ok)) == 0xFF;
}

static inline __m256 log2_core_normal(__m256 input) noexcept
{
	const __m256 one_ps = _mm256_set1_ps(1.0f);
	const __m256 log2e_ps = _mm256_set1_ps(1.4426950408889634074f);

	const __m256i mant_mask_i = _mm256_set1_epi32(0x007fffffu);
	const __m256i bias_i = _mm256_set1_epi32(127);
	const __m256i magic_i = _mm256_set1_epi32(0x004afb0d);
	const __m256i one_bits_i = _mm256_set1_epi32(0x3f800000u);
	const __m256i high_bit_i = _mm256_set1_epi32(0x00800000u);

	__m256i hx_i = _mm256_castps_si256(input);
	__m256i exp_i = _mm256_sub_epi32(_mm256_srli_epi32(hx_i, 23), bias_i);

	__m256i mant_i = _mm256_and_si256(hx_i, mant_mask_i);
	__m256i i_i = _mm256_and_si256(_mm256_add_epi32(mant_i, magic_i), high_bit_i);
	__m256i xb_i = _mm256_or_si256(mant_i, _mm256_xor_si256(i_i, one_bits_i));
	__m256 x = _mm256_castsi256_ps(xb_i);

	__m256i k_i = _mm256_add_epi32(exp_i, _mm256_srli_epi32(i_i, 23));
	__m256 y = _mm256_cvtepi32_ps(k_i);
	__m256 f = _mm256_sub_ps(x, one_ps);

	return _mm256_fmadd_ps(log2_log1p_core(f), log2e_ps, y);
}

__m256 fy::simd::intrinsic::log2(__m256 input) noexcept
{
	const __m256 zero_ps = _mm256_setzero_ps();
	const __m256 one_ps = _mm256_set1_ps(1.0f);
	const __m256 two25_ps = _mm256_set1_ps(3.3554432000e+07f);
	const __m256 log2e_ps = _mm256_set1_ps(1.4426950408889634074f);

	const __m256i zero_i = _mm256_setzero_si256();
	const __m256i abs_mask_i = _mm256_set1_epi32(0x7fffffffu);
	const __m256i mant_mask_i = _mm256_set1_epi32(0x007fffffu);
	const __m256i sub_limit_i = _mm256_set1_epi32(0x00800000u);
	const __m256i bias_i = _mm256_set1_epi32(127);
	const __m256i magic_i = _mm256_set1_epi32(0x004afb0d);
	const __m256i one_bits_i = _mm256_set1_epi32(0x3f800000u);
	const __m256i high_bit_i = _mm256_set1_epi32(0x00800000u);
	const __m256i minus25_i = _mm256_set1_epi32(-25);

	if (all_positive_finite_normal(input))
	{
		return log2_core_normal(input);
	}

	__m256i raw_i = _mm256_castps_si256(input);
	__m256i abs_i = _mm256_and_si256(raw_i, abs_mask_i);

	__m256i sub_lt_i = _mm256_cmpgt_epi32(sub_limit_i, abs_i);
	__m256i sub_nz_i = _mm256_andnot_si256(_mm256_cmpeq_epi32(abs_i, zero_i), sub_lt_i);
	__m256 sub_mask = _mm256_and_ps(
		_mm256_castsi256_ps(sub_nz_i),
		_mm256_cmp_ps(input, zero_ps, _CMP_GT_OQ)
	);

	__m256 scaled_input = _mm256_blendv_ps(input, _mm256_mul_ps(input, two25_ps), sub_mask);

	__m256i k_i = _mm256_and_si256(_mm256_castps_si256(sub_mask), minus25_i);

	__m256i hx_i = _mm256_castps_si256(scaled_input);
	__m256i exp_i = _mm256_sub_epi32(_mm256_srli_epi32(hx_i, 23), bias_i);
	k_i = _mm256_add_epi32(k_i, exp_i);

	__m256i mant_i = _mm256_and_si256(hx_i, mant_mask_i);
	__m256i i_i = _mm256_and_si256(_mm256_add_epi32(mant_i, magic_i), high_bit_i);
	__m256i xb_i = _mm256_or_si256(mant_i, _mm256_xor_si256(i_i, one_bits_i));
	__m256 x = _mm256_castsi256_ps(xb_i);

	k_i = _mm256_add_epi32(k_i, _mm256_srli_epi32(i_i, 23));

	__m256 y = _mm256_cvtepi32_ps(k_i);
	__m256 f = _mm256_sub_ps(x, one_ps);

	__m256 result = _mm256_fmadd_ps(log2_log1p_core(f), log2e_ps, y);

	return log2_finalize_special(result, input, raw_i, abs_i);
}