#include <foye_fastmath.hpp>

__m256 fy::simd::intrinsic::log1p(__m256 input) noexcept
{
	const __m256 zero = _mm256_set1_ps(0.0f);
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 half = _mm256_set1_ps(0.5f);
	const __m256 two = _mm256_set1_ps(2.0f);
	const __m256 neg_one = _mm256_set1_ps(-1.0f);
	const __m256 ln2_hi = _mm256_set1_ps(6.9313812256e-01f);
	const __m256 ln2_lo = _mm256_set1_ps(9.0580006145e-06f);

	const __m256 Lp1 = _mm256_set1_ps(6.6666668653e-01f);
	const __m256 Lp2 = _mm256_set1_ps(4.0000000596e-01f);
	const __m256 Lp3 = _mm256_set1_ps(2.8571429849e-01f);
	const __m256 Lp4 = _mm256_set1_ps(2.2222198546e-01f);
	const __m256 Lp5 = _mm256_set1_ps(1.8183572590e-01f);
	const __m256 Lp6 = _mm256_set1_ps(1.5313838422e-01f);
	const __m256 Lp7 = _mm256_set1_ps(1.4798198640e-01f);

	const __m256 c066666666 = _mm256_set1_ps(0.66666666666666666f);
	const __m256 ninf = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
	const __m256 qnan = _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN());

	const __m256i i_zero = _mm256_setzero_si256();
	const __m256i i_one = _mm256_set1_epi32(1);
	const __m256i i_127 = _mm256_set1_epi32(127);
	const __m256i i_abs = _mm256_set1_epi32(0x7fffffff);
	const __m256i i_mant = _mm256_set1_epi32(0x007fffff);
	const __m256i i_exp1 = _mm256_set1_epi32(0x3f800000);
	const __m256i i_exp05 = _mm256_set1_epi32(0x3f000000);
	const __m256i i_hidden = _mm256_set1_epi32(0x00800000);
	const __m256i i_pos_inf = _mm256_set1_epi32(0x7f800000);
	const __m256i i_neg_inf = _mm256_set1_epi32(0xff800000u);

	const __m256i t_sqrt2 = _mm256_set1_epi32(0x3ed413d0);
	const __m256i t_small = _mm256_set1_epi32(0x38000000);
	const __m256i t_tiny = _mm256_set1_epi32(0x33800000);
	const __m256i t_neg = _mm256_set1_epi32(0xbe95f619u);
	const __m256i t_hu = _mm256_set1_epi32(0x003504f4);

	__m256i hx = _mm256_castps_si256(input);
	__m256i ax = _mm256_and_si256(hx, i_abs);

	__m256i k = i_one;
	__m256i hu = i_one;
	__m256 f = input;
	__m256 c = zero;
	__m256 u = zero;

	__m256i m_lt_sqrt2 = _mm256_cmpgt_epi32(t_sqrt2, hx);
	__m256i m_small = _mm256_cmpgt_epi32(t_small, ax);
	__m256i m_tiny = _mm256_cmpgt_epi32(t_tiny, ax);
	__m256i m_hx_gt_zero = _mm256_cmpgt_epi32(hx, i_zero);
	__m256i m_hx_gt_tneg = _mm256_cmpgt_epi32(hx, t_neg);
	__m256i m_hx_le_tneg = _mm256_andnot_si256(m_hx_gt_tneg, _mm256_cmpeq_epi32(i_zero, i_zero));
	__m256i m_setk0 = _mm256_and_si256(m_lt_sqrt2, _mm256_or_si256(m_hx_gt_zero, m_hx_le_tneg));

	k = _mm256_blendv_epi8(k, i_zero, m_setk0);
	f = _mm256_blendv_ps(f, input, _mm256_castsi256_ps(m_setk0));
	hu = _mm256_blendv_epi8(hu, i_one, m_setk0);

	__m256i m_k_nonzero = _mm256_andnot_si256(_mm256_cmpeq_epi32(k, i_zero), _mm256_cmpeq_epi32(i_zero, i_zero));

	__m256i m_hx_big = _mm256_cmpgt_epi32(hx, _mm256_set1_epi32(0x59ffffff));

	__m256 u_small = _mm256_add_ps(one, input);
	__m256i hu_small_bits = _mm256_castps_si256(u_small);
	__m256i k_small = _mm256_sub_epi32(_mm256_srli_epi32(hu_small_bits, 23), i_127);
	__m256i m_ksmall_pos = _mm256_cmpgt_epi32(k_small, i_zero);
	__m256 c_small_a = _mm256_sub_ps(one, _mm256_sub_ps(u_small, input));
	__m256 c_small_b = _mm256_sub_ps(input, _mm256_sub_ps(u_small, one));
	__m256 c_small = _mm256_div_ps(_mm256_blendv_ps(c_small_b, c_small_a, _mm256_castsi256_ps(m_ksmall_pos)), u_small);

	__m256 u_big = input;
	__m256i hu_big_bits = hx;
	__m256i k_big = _mm256_sub_epi32(_mm256_srli_epi32(hu_big_bits, 23), i_127);
	__m256 c_big = zero;

	__m256 u_pre = _mm256_blendv_ps(u_small, u_big, _mm256_castsi256_ps(m_hx_big));
	__m256i hu_pre = _mm256_blendv_epi8(hu_small_bits, hu_big_bits, m_hx_big);
	__m256i k_pre = _mm256_blendv_epi8(k_small, k_big, m_hx_big);
	__m256 c_pre = _mm256_blendv_ps(c_small, c_big, _mm256_castsi256_ps(m_hx_big));

	__m256i mant = _mm256_and_si256(hu_pre, i_mant);
	__m256i m_mant_small = _mm256_cmpgt_epi32(t_hu, mant);

	__m256i u_bits_a = _mm256_or_si256(mant, i_exp1);
	__m256i u_bits_b = _mm256_or_si256(mant, i_exp05);
	__m256 u_norm = _mm256_castsi256_ps(_mm256_blendv_epi8(u_bits_b, u_bits_a, m_mant_small));
	__m256i hu_norm = _mm256_blendv_epi8(_mm256_srli_epi32(_mm256_sub_epi32(i_hidden, mant), 2), mant, m_mant_small);
	__m256i k_norm = _mm256_add_epi32(k_pre, _mm256_andnot_si256(m_mant_small, i_one));
	__m256 f_norm = _mm256_sub_ps(u_norm, one);

	f = _mm256_blendv_ps(f, f_norm, _mm256_castsi256_ps(m_k_nonzero));
	c = _mm256_blendv_ps(c, c_pre, _mm256_castsi256_ps(m_k_nonzero));
	hu = _mm256_blendv_epi8(hu, hu_norm, m_k_nonzero);
	k = _mm256_blendv_epi8(k, k_norm, m_k_nonzero);
	u = _mm256_blendv_ps(u, u_pre, _mm256_castsi256_ps(m_k_nonzero));

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

	__m256 kf = _mm256_cvtepi32_ps(k);
	__m256 k_ln2_hi = _mm256_mul_ps(kf, ln2_hi);
	__m256 k_ln2_lo = _mm256_mul_ps(kf, ln2_lo);

	__m256 res_gen_k0 = _mm256_sub_ps(f, _mm256_sub_ps(hfsq, _mm256_mul_ps(s, _mm256_add_ps(hfsq, R))));
	__m256 res_gen_k1 = _mm256_sub_ps(k_ln2_hi, _mm256_sub_ps(_mm256_sub_ps(hfsq, _mm256_add_ps(_mm256_mul_ps(s, _mm256_add_ps(hfsq, R)), _mm256_add_ps(k_ln2_lo, c))), f));
	__m256i m_k_zero = _mm256_cmpeq_epi32(k, i_zero);
	__m256 result = _mm256_blendv_ps(res_gen_k1, res_gen_k0, _mm256_castsi256_ps(m_k_zero));

	__m256 R_hu0 = _mm256_mul_ps(hfsq, _mm256_fnmadd_ps(c066666666, f, one));
	__m256 res_hu0_k0 = _mm256_sub_ps(f, R_hu0);
	__m256 res_hu0_k1 = _mm256_sub_ps(k_ln2_hi, _mm256_sub_ps(_mm256_sub_ps(R_hu0, _mm256_add_ps(k_ln2_lo, c)), f));
	__m256 res_hu0 = _mm256_blendv_ps(res_hu0_k1, res_hu0_k0, _mm256_castsi256_ps(m_k_zero));

	__m256i m_hu_zero = _mm256_cmpeq_epi32(hu, i_zero);
	result = _mm256_blendv_ps(result, res_hu0, _mm256_castsi256_ps(m_hu_zero));

	__m256i m_f_zero = _mm256_castps_si256(_mm256_cmp_ps(f, zero, _CMP_EQ_OQ));
	__m256i m_hu0_f0 = _mm256_and_si256(m_hu_zero, m_f_zero);
	__m256 res_hu0_f0_k1 = _mm256_add_ps(k_ln2_hi, _mm256_add_ps(c, k_ln2_lo));
	__m256 res_hu0_f0 = _mm256_blendv_ps(res_hu0_f0_k1, zero, _mm256_castsi256_ps(m_k_zero));
	result = _mm256_blendv_ps(result, res_hu0_f0, _mm256_castsi256_ps(m_hu0_f0));

	__m256 small_poly = _mm256_fnmadd_ps(_mm256_mul_ps(input, input), half, input);
	__m256 small_res = _mm256_blendv_ps(small_poly, input, _mm256_castsi256_ps(m_tiny));
	__m256i m_small_path = _mm256_and_si256(m_lt_sqrt2, m_small);
	result = _mm256_blendv_ps(result, small_res, _mm256_castsi256_ps(m_small_path));

	__m256i m_eq_neg_one = _mm256_castps_si256(_mm256_cmp_ps(input, neg_one, _CMP_EQ_OQ));
	__m256i m_lt_neg_one = _mm256_castps_si256(_mm256_cmp_ps(input, neg_one, _CMP_LT_OQ));
	result = _mm256_blendv_ps(result, qnan, _mm256_castsi256_ps(m_lt_neg_one));
	result = _mm256_blendv_ps(result, ninf, _mm256_castsi256_ps(m_eq_neg_one));

	__m256i m_nan = _mm256_cmpgt_epi32(ax, i_pos_inf);
	__m256i m_pos_inf = _mm256_cmpeq_epi32(hx, i_pos_inf);
	__m256i m_neg_inf = _mm256_cmpeq_epi32(hx, i_neg_inf);

	const __m256i qnan_quiet_i = _mm256_set1_epi32(0x00400000);
	__m256 nan_quieted = _mm256_castsi256_ps(
		_mm256_or_si256(_mm256_castps_si256(input), qnan_quiet_i)
	);

	result = _mm256_blendv_ps(result, nan_quieted, _mm256_castsi256_ps(m_nan));
	result = _mm256_blendv_ps(result, input, _mm256_castsi256_ps(m_pos_inf));
	result = _mm256_blendv_ps(result, qnan, _mm256_castsi256_ps(m_neg_inf));

	return result;
}