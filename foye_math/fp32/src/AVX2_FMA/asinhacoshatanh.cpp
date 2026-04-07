#include <foye_fastmath.hpp>

/*
* acosh
invalid_all_negative_large			[-1.0000000e+03,   -1.00000000e+00]       radio 180.3511 x  max ulp 0
invalid_all_negative_small			[-1.0000000e+00,   -9.99999997e-07]       radio 179.7356 x  max ulp 0
invalid_zero_to_one					[0.00000000e+00,    9.99998987e-01]       radio 49.28237 x  max ulp 0
domain_edge_cross_1_tight			[9.99499977e-01,    1.00049996e+00]       radio 38.84666 x  max ulp 2
domain_edge_cross_1_wide			[8.99999976e-01,    1.10000002e+00]       radio 39.92458 x  max ulp 2
just_above_1_ultra_tight			[1.00000000e+00,    1.00001001e+00]       radio 4.007695 x  max ulp 1
just_above_1_very_tight				[1.00000000e+00,    1.00010002e+00]       radio 4.057442 x  max ulp 1
just_above_1_tight					[1.00000000e+00,    1.00100005e+00]       radio 4.044748 x  max ulp 2
just_above_1_small					[1.00000000e+00,    1.00999999e+00]       radio 4.078637 x  max ulp 2
just_above_1_medium					[1.00000000e+00,    1.04999995e+00]       radio 4.087768 x  max ulp 2
log1p_smallrange_pure				[1.00000000e+00,    1.07000005e+00]       radio 5.238006 x  max ulp 2
log1p_smallrange_boundary_tight		[1.06799996e+00,    1.07400000e+00]       radio 10.91262 x  max ulp 1
log1p_general_subrange				[1.07149994e+00,    1.25000000e+00]       radio 10.87366 x  max ulp 2
one_to_two_full						[1.00000000e+00,    2.00000000e+00]       radio 10.47592 x  max ulp 2
one_to_two_upper_half				[1.50000000e+00,    2.00000000e+00]       radio 10.91883 x  max ulp 1
one_to_two_centered					[1.20000005e+00,    1.79999995e+00]       radio 10.88088 x  max ulp 1
boundary_2_left_tight				[1.95000005e+00,    1.99989998e+00]       radio 10.72293 x  max ulp 1
boundary_2_right_tight				[2.00009990e+00,    2.04999995e+00]       radio 20.76988 x  max ulp 1
boundary_2_cross_tight				[1.99000001e+00,    2.00999999e+00]       radio 8.641905 x  max ulp 1
boundary_2_cross_wide				[1.75000000e+00,    2.25000000e+00]       radio 8.492950 x  max ulp 1
moderate_gt2_small					[2.00000000e+00,    4.00000000e+00]       radio 20.83178 x  max ulp 1
moderate_gt2_common					[2.00000000e+00,    1.00000000e+01]       radio 20.25819 x  max ulp 1
moderate_gt2_wide					[2.00000000e+00,    1.00000000e+02]       radio 20.78186 x  max ulp 1
large_positive_mainpath				[1.00000000e+01,    1.00000000e+06]       radio 20.51859 x  max ulp 1
very_large_mainpath					[1.00000000e+06,    1.00000000e+08]       radio 19.66009 x  max ulp 1
common_ml_like_near_one				[1.00000000e+00,    3.00000000e+00]       radio 8.480430 x  max ulp 2
common_geometry_like				[1.00000000e+00,    1.00000000e+01]       radio 8.731169 x  max ulp 2
common_scientific_moderate			[1.00000000e+00,    1.00000000e+02]       radio 9.082411 x  max ulp 2
common_scientific_wide				[1.00000000e+00,    1.00000000e+06]       radio 11.74500 x  max ulp 2
threshold_2pow28_left_tight			[2.68430000e+08,    2.68435456e+08]       radio 20.13168 x  max ulp 0
threshold_2pow28_right_tight		[2.68435456e+08,    2.68440000e+08]       radio 5.801341 x  max ulp 1
threshold_2pow28_cross_tight		[2.68435296e+08,    2.68435712e+08]       radio 12.18275 x  max ulp 0
threshold_2pow28_cross_wide			[2.68000000e+08,    2.69000000e+08]       radio 12.93822 x  max ulp 1
huge_fastpath_small_span			[2.68435456e+08,    3.00000000e+08]       radio 5.721057 x  max ulp 1
huge_fastpath_mid					[3.00000000e+08,    1.00000000e+10]       radio 5.724977 x  max ulp 1
huge_fastpath_wide					[1.00000000e+09,    1.00000002e+20]       radio 5.886368 x  max ulp 1
mixed_invalid_to_small_valid		[5.00000000e-01,    1.50000000e+00]       radio 41.87912 x  max ulp 2
mixed_invalid_to_gt2				[5.00000000e-01,    4.00000000e+00]       radio 21.30112 x  max ulp 2
mixed_one_to_gt2_cross_2			[1.00000000e+00,    4.00000000e+00]       radio 8.516761 x  max ulp 2
mixed_small_valid_to_mainpath		[1.00000000e+00,    1.00000000e+01]       radio 8.680820 x  max ulp 2
mixed_mainpath_to_hugepath			[2.00000000e+00,    3.00000000e+08]       radio 19.74054 x  max ulp 1
mixed_all_valid_paths				[1.00000000e+00,    3.00000000e+08]       radio 12.93009 x  max ulp 2
mixed_invalid_valid_huge			[-1.0000000e+00,    3.00000000e+08]       radio 27.00808 x  max ulp 2
centered_small_mixed				[-2.0000000e+00,    2.00000000e+00]       radio 47.43734 x  max ulp 2
centered_moderate_mixed				[-1.0000000e+01,    1.00000000e+01]       radio 43.73500 x  max ulp 2
centered_large_mixed				[-1.0000000e+03,    1.00000000e+03]       radio 36.23651 x  max ulp 2
positive_wide_small_to_mid			[1.00000000e+00,    1.00000000e+02]       radio 9.011445 x  max ulp 2
positive_wide_mid_to_large			[1.00000000e+02,    1.00000000e+06]       radio 20.06355 x  max ulp 1
positive_wide_full_valid_no_huge	[1.00000000e+00,    1.00000000e+08]       radio 12.95004 x  max ulp 2
positive_wide_full_valid_with_huge	[1.00000000e+00,    1.00000000e+09]       radio 10.70171 x  max ulp 2

asinh
tiny_pos_full						[0.00000000e+00,	3.70000008e-09]       radio 1.849020 x  max ulp 0
tiny_mixed_full						[-3.70000008e-09,	3.70000008e-09]       radio 1.832462 x  max ulp 0
boundary_2powm28_tight_pos			[3.50000007e-09,	3.89999988e-09]       radio 1.050750 x  max ulp 0
boundary_2powm28_mixed				[-3.89999988e-09,	3.89999988e-09]       radio 1.833827 x  max ulp 0
small_pos_near_zero					[3.79999987e-09,	3.12500000e-02]       radio 1.071689 x  max ulp 1
small_pos_full						[3.79999987e-09,	1.99989998e+00]       radio 3.022043 x  max ulp 2
small_neg_full						[-1.99989998e+00,  -3.79999987e-09]       radio 3.042570 x  max ulp 2
small_mixed_full					[-1.99989998e+00,	1.99989998e+00]       radio 1.001985 x  max ulp 2
small_centered_tight				[-1.25000000e-01,	1.25000000e-01]       radio 0.895304 x  max ulp 1
small_centered_medium				[-5.00000000e-01,	5.00000000e-01]       radio 0.946695 x  max ulp 2
small_centered_wide					[-1.00000000e+00,	1.00000000e+00]       radio 0.968204 x  max ulp 2
small_positive_unit					[0.00000000e+00,	1.00000000e+00]       radio 0.966330 x  max ulp 2
boundary_2_tight_pos				[1.95000005e+00,	2.04999995e+00]       radio 8.392254 x  max ulp 1
boundary_2_tight_neg				[-2.04999995e+00,  -1.95000005e+00]       radio 8.446752 x  max ulp 1
boundary_2_mixed					[-2.04999995e+00,	2.04999995e+00]       radio 0.993633 x  max ulp 2
boundary_2_wide_mixed				[-2.25000000e+00,	2.25000000e+00]       radio 1.003126 x  max ulp 2
medium_pos_full						[2.00009990e+00,	1.00000000e+03]       radio 21.53573 x  max ulp 1
medium_neg_full						[-1.00000000e+03,  -2.00009990e+00]       radio 20.32642 x  max ulp 1
medium_mixed_full					[-1.00000000e+03,	1.00000000e+03]       radio 1.031600 x  max ulp 2
medium_pos_inner					[3.00000000e+00,	1.00000000e+02]       radio 21.37271 x  max ulp 1
medium_neg_inner					[-1.00000000e+02,  -3.00000000e+00]       radio 21.21058 x  max ulp 1
medium_pos_far						[1.00000000e+03,	1.00000000e+07]       radio 21.24368 x  max ulp 1
medium_mixed_far					[-1.00000000e+07,	1.00000000e+07]       radio 1.157632 x  max ulp 2
boundary_2pow28_tight_pos			[2.60000000e+08,	2.76000000e+08]       radio 13.52368 x  max ulp 1
boundary_2pow28_tight_neg			[-2.76000000e+08,  -2.60000000e+08]       radio 13.52758 x  max ulp 1
boundary_2pow28_mixed				[-2.76000000e+08,	2.76000000e+08]       radio 1.226452 x  max ulp 2
large_pos_small_over				[2.68435456e+08,	1.00000000e+09]       radio 5.059366 x  max ulp 1
large_neg_small_over				[-1.00000000e+09,  -2.68435456e+08]       radio 7.201556 x  max ulp 1
large_mixed_full					[-1.00000000e+09,	1.00000000e+09]       radio 1.258276 x  max ulp 2
large_pos_far						[1.00000000e+09,	3.40282347e+38]       radio 5.028131 x  max ulp 1
mixed_tiny_small					[-9.99999997e-07,	9.99999997e-07]       radio 1.011935 x  max ulp 0
mixed_small_medium					[-4.00000000e+00,	4.00000000e+00]       radio 1.011863 x  max ulp 2
mixed_small_to_large				[-1.00000000e+09,	1.00000000e+09]       radio 1.248952 x  max ulp 2
realistic_centered_unit				[-1.00000000e+00,	1.00000000e+00]       radio 0.970418 x  max ulp 2
realistic_centered_wide				[-4.00000000e+00,	4.00000000e+00]       radio 0.994639 x  max ulp 2
realistic_positive_wide				[0.00000000e+00,	8.00000000e+00]       radio 0.996712 x  max ulp 2

atanh
tiny_signed_linear_ultra			[-3.72529030e-09,   3.72529030e-09]		  radio 5.652381 x  max ulp 0
tiny_signed_linear_tight			[-9.99999994e-09,   9.99999994e-09]		  radio 6.285985 x  max ulp 0
tiny_signed_linear_wide				[-9.99999997e-07,   9.99999997e-07]		  radio 2.754112 x  max ulp 0
small_signed_below_half_1e4			[-9.99999975e-05,   9.99999975e-05]		  radio 2.987253 x  max ulp 0
small_signed_below_half_1e2			[-9.99999978e-03,   9.99999978e-03]		  radio 3.144697 x  max ulp 1
small_signed_below_half_common		[-1.00000001e-01,   1.00000001e-01]		  radio 3.201274 x  max ulp 1
mid_signed_below_half_full			[-4.99900013e-01,   4.99900013e-01]		  radio 2.695713 x  max ulp 2
half_boundary_ultra_tight			[4.99900013e-01,    5.00100017e-01]		  radio 0.997386 x  max ulp 1
half_boundary_tight_signed			[-5.00999987e-01,   5.00999987e-01]		  radio 2.817302 x  max ulp 2
half_to_three_quarters_pos			[5.00000000e-01,    7.50000000e-01]		  radio 0.992625 x  max ulp 1
half_to_three_quarters_signed		[-7.50000000e-01,   7.50000000e-01]		  radio 2.666247 x  max ulp 2
moderate_signed_common				[-7.50000000e-01,   7.50000000e-01]		  radio 2.637990 x  max ulp 2
moderate_pos_activation_like		[9.99999978e-03,    8.00000012e-01]		  radio 0.972556 x  max ulp 2
moderate_signed_activation_like		[-8.00000012e-01,   8.00000012e-01]		  radio 2.649229 x  max ulp 2
near_one_loose_pos					[8.00000012e-01,    9.49999988e-01]		  radio 0.975797 x  max ulp 1
near_one_loose_signed				[-9.49999988e-01,   9.49999988e-01]		  radio 2.596695 x  max ulp 2
near_one_tight_pos					[9.49999988e-01,    9.90000010e-01]		  radio 0.971290 x  max ulp 1
near_one_tight_signed				[-9.90000010e-01,   9.90000010e-01]		  radio 2.621335 x  max ulp 2
near_one_very_tight_pos				[9.90000010e-01,    9.99000013e-01]		  radio 0.979469 x  max ulp 1
near_one_very_tight_signed			[-9.99000013e-01,   9.99000013e-01]		  radio 2.583280 x  max ulp 2
near_one_ultra_tight_pos			[9.99000013e-01,    9.99989986e-01]		  radio 0.970763 x  max ulp 1
near_one_ultra_tight_signed			[-9.99989986e-01,   9.99989986e-01]		  radio 2.589743 x  max ulp 2
domain_edge_inside_pos				[9.99899983e-01,    9.99999940e-01]		  radio 0.978115 x  max ulp 1
domain_edge_inside_signed			[-9.99999940e-01,   9.99999940e-01]		  radio 2.588491 x  max ulp 2
*/

static __m256 log_pos_norm(__m256 x) noexcept
{
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 half = _mm256_set1_ps(0.5f);
	const __m256 sqrt_half = _mm256_set1_ps(0.70710678118654752440f);

	const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
	const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);

	const __m256 ce0 = _mm256_set1_ps(7.0376836292e-2f);
	const __m256 ce1 = _mm256_set1_ps(-1.1514610310e-1f);
	const __m256 ce2 = _mm256_set1_ps(1.1676998740e-1f);
	const __m256 ce3 = _mm256_set1_ps(-1.2420140846e-1f);
	const __m256 ce4 = _mm256_set1_ps(1.4249322787e-1f);
	const __m256 ce5 = _mm256_set1_ps(-1.6668057665e-1f);
	const __m256 ce6 = _mm256_set1_ps(2.0000714765e-1f);
	const __m256 ce7 = _mm256_set1_ps(-2.4999993993e-1f);
	const __m256 ce8 = _mm256_set1_ps(3.3333331174e-1f);

	__m256i ix = _mm256_castps_si256(x);
	__m256i expi = _mm256_srli_epi32(ix, 23);

	ix = _mm256_and_si256(ix, _mm256_set1_epi32(0x007fffff));
	ix = _mm256_or_si256(ix, _mm256_set1_epi32(0x3f000000));
	__m256 m = _mm256_castsi256_ps(ix);

	__m256 e = _mm256_cvtepi32_ps(_mm256_sub_epi32(expi, _mm256_set1_epi32(126)));

	__m256 mask = _mm256_cmp_ps(m, sqrt_half, _CMP_LT_OQ);
	__m256 tmp = _mm256_and_ps(m, mask);

	m = _mm256_sub_ps(m, one);
	e = _mm256_sub_ps(e, _mm256_and_ps(mask, one));
	m = _mm256_add_ps(m, tmp);

	__m256 z = _mm256_mul_ps(m, m);

	__m256 p = ce0;
	p = _mm256_fmadd_ps(p, m, ce1);
	p = _mm256_fmadd_ps(p, m, ce2);
	p = _mm256_fmadd_ps(p, m, ce3);
	p = _mm256_fmadd_ps(p, m, ce4);
	p = _mm256_fmadd_ps(p, m, ce5);
	p = _mm256_fmadd_ps(p, m, ce6);
	p = _mm256_fmadd_ps(p, m, ce7);
	p = _mm256_fmadd_ps(p, m, ce8);
	p = _mm256_mul_ps(p, m);
	p = _mm256_mul_ps(p, z);

	__m256 y = _mm256_fmadd_ps(e, ln2_lo, p);
	y = _mm256_sub_ps(y, _mm256_mul_ps(half, z));
	y = _mm256_add_ps(y, m);
	y = _mm256_fmadd_ps(e, ln2_hi, y);

	return y;
}

static __m256 log1p_nonneg(const __m256 x) noexcept
{
	const __m256 zero = _mm256_setzero_ps();
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 half = _mm256_set1_ps(0.5f);
	const __m256 two = _mm256_set1_ps(2.0f);
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

	const __m256i i_zero = _mm256_setzero_si256();
	const __m256i i_one = _mm256_set1_epi32(1);
	const __m256i i_127 = _mm256_set1_epi32(127);
	const __m256i i_mant = _mm256_set1_epi32(0x007fffff);
	const __m256i i_exp1 = _mm256_set1_epi32(0x3f800000);
	const __m256i i_exp05 = _mm256_set1_epi32(0x3f000000);
	const __m256i i_hidden = _mm256_set1_epi32(0x00800000);
	const __m256i t_sqrt2 = _mm256_set1_epi32(0x3ed413d0);
	const __m256i t_small = _mm256_set1_epi32(0x38000000);
	const __m256i t_tiny = _mm256_set1_epi32(0x33800000);
	const __m256i t_hu = _mm256_set1_epi32(0x003504f4);

	__m256i hx = _mm256_castps_si256(x);

	__m256i m_setk0 = _mm256_cmpgt_epi32(t_sqrt2, hx);

	__m256i k = _mm256_blendv_epi8(i_one, i_zero, m_setk0);
	__m256 f = x;
	__m256 c = zero;
	__m256i hu = i_one;

	__m256i m_k_nonzero =
		_mm256_andnot_si256(_mm256_cmpeq_epi32(k, i_zero), _mm256_cmpeq_epi32(i_zero, i_zero));

	__m256 u_small = _mm256_add_ps(one, x);
	__m256i hu_small_bits = _mm256_castps_si256(u_small);
	__m256i k_small = _mm256_sub_epi32(_mm256_srli_epi32(hu_small_bits, 23), i_127);
	__m256i m_ksmall_pos = _mm256_cmpgt_epi32(k_small, i_zero);

	__m256 c_small_a = _mm256_sub_ps(one, _mm256_sub_ps(u_small, x));
	__m256 c_small_b = _mm256_sub_ps(x, _mm256_sub_ps(u_small, one));
	__m256 c_small = _mm256_div_ps(
		_mm256_blendv_ps(c_small_b, c_small_a, _mm256_castsi256_ps(m_ksmall_pos)),
		u_small);

	__m256i mant = _mm256_and_si256(hu_small_bits, i_mant);
	__m256i m_mant_small = _mm256_cmpgt_epi32(t_hu, mant);

	__m256i u_bits_a = _mm256_or_si256(mant, i_exp1);
	__m256i u_bits_b = _mm256_or_si256(mant, i_exp05);
	__m256 u_norm = _mm256_castsi256_ps(_mm256_blendv_epi8(u_bits_b, u_bits_a, m_mant_small));

	__m256i hu_norm = _mm256_blendv_epi8(
		_mm256_srli_epi32(_mm256_sub_epi32(i_hidden, mant), 2),
		mant,
		m_mant_small);

	__m256i k_norm = _mm256_add_epi32(k_small, _mm256_andnot_si256(m_mant_small, i_one));
	__m256 f_norm = _mm256_sub_ps(u_norm, one);

	f = _mm256_blendv_ps(f, f_norm, _mm256_castsi256_ps(m_k_nonzero));
	c = _mm256_blendv_ps(c, c_small, _mm256_castsi256_ps(m_k_nonzero));
	hu = _mm256_blendv_epi8(hu, hu_norm, m_k_nonzero);

	k = _mm256_blendv_epi8(k, k_norm, m_k_nonzero);

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

	__m256 res_gen_k0 =
		_mm256_sub_ps(f, _mm256_sub_ps(hfsq, _mm256_mul_ps(s, _mm256_add_ps(hfsq, R))));
	__m256 res_gen_k1 =
		_mm256_sub_ps(k_ln2_hi,
			_mm256_sub_ps(
				_mm256_sub_ps(hfsq,
					_mm256_add_ps(_mm256_mul_ps(s, _mm256_add_ps(hfsq, R)),
						_mm256_add_ps(k_ln2_lo, c))),
				f));

	__m256i m_k_zero = _mm256_cmpeq_epi32(k, i_zero);
	__m256 result = _mm256_blendv_ps(res_gen_k1, res_gen_k0, _mm256_castsi256_ps(m_k_zero));

	__m256 R_hu0 = _mm256_mul_ps(hfsq, _mm256_fnmadd_ps(c066666666, f, one));
	__m256 res_hu0_k0 = _mm256_sub_ps(f, R_hu0);
	__m256 res_hu0_k1 =
		_mm256_sub_ps(k_ln2_hi, _mm256_sub_ps(_mm256_sub_ps(R_hu0, _mm256_add_ps(k_ln2_lo, c)), f));
	__m256 res_hu0 = _mm256_blendv_ps(res_hu0_k1, res_hu0_k0, _mm256_castsi256_ps(m_k_zero));

	__m256i m_hu_zero = _mm256_cmpeq_epi32(hu, i_zero);
	result = _mm256_blendv_ps(result, res_hu0, _mm256_castsi256_ps(m_hu_zero));

	__m256i m_f_zero = _mm256_castps_si256(_mm256_cmp_ps(f, zero, _CMP_EQ_OQ));
	__m256i m_hu0_f0 = _mm256_and_si256(m_hu_zero, m_f_zero);
	__m256 res_hu0_f0_k1 = _mm256_add_ps(k_ln2_hi, _mm256_add_ps(c, k_ln2_lo));
	__m256 res_hu0_f0 = _mm256_blendv_ps(res_hu0_f0_k1, zero, _mm256_castsi256_ps(m_k_zero));
	result = _mm256_blendv_ps(result, res_hu0_f0, _mm256_castsi256_ps(m_hu0_f0));

	__m256i m_small = _mm256_cmpgt_epi32(t_small, hx);
	__m256i m_tiny = _mm256_cmpgt_epi32(t_tiny, hx);
	__m256 small_poly = _mm256_fnmadd_ps(_mm256_mul_ps(x, x), half, x);
	__m256 small_res = _mm256_blendv_ps(small_poly, x, _mm256_castsi256_ps(m_tiny));
	result = _mm256_blendv_ps(result, small_res, _mm256_castsi256_ps(m_small));

	return result;
}

static inline __m256 log2_pos_norm(__m256 x) noexcept
{
	const __m256 ln2 = _mm256_set1_ps(0.693147182464599609375f);
	return _mm256_add_ps(log_pos_norm(x), ln2);
}

static inline __m256 asinh_pos_small(__m256 ax, __m256 x2) noexcept
{
	const __m256 one = _mm256_set1_ps(1.0f);
	__m256 sqrt1p = _mm256_sqrt_ps(_mm256_add_ps(one, x2));
	__m256 den = _mm256_add_ps(one, sqrt1p);
	__m256 term = _mm256_div_ps(x2, den);
	__m256 arg = _mm256_add_ps(ax, term);
	return log1p_nonneg(arg);
}

static inline __m256 asinh_pos_gt2(__m256 ax, __m256 x2) noexcept
{
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 two = _mm256_set1_ps(2.0f);
	__m256 sqrt1p = _mm256_sqrt_ps(_mm256_add_ps(one, x2));
	__m256 den = _mm256_add_ps(sqrt1p, ax);
	__m256 term = _mm256_div_ps(one, den);
	__m256 arg = _mm256_fmadd_ps(two, ax, term);
	return log_pos_norm(arg);
}

static inline __m256 acosh_1to2(__m256 x) noexcept
{
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 two = _mm256_set1_ps(2.0f);
	__m256 t = _mm256_sub_ps(x, one);
	__m256 sqrt_term = _mm256_sqrt_ps(_mm256_fmadd_ps(t, t, _mm256_mul_ps(two, t)));
	__m256 expr = _mm256_add_ps(t, sqrt_term);
	return log1p_nonneg(expr);
}

static inline __m256 acosh_gt2(__m256 x, __m256 x2) noexcept
{
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 two = _mm256_set1_ps(2.0f);
	__m256 sqrt_term = _mm256_sqrt_ps(_mm256_sub_ps(x2, one));
	__m256 den = _mm256_add_ps(x, sqrt_term);
	__m256 invden = _mm256_div_ps(one, den);
	__m256 expr = _mm256_fnmadd_ps(invden, one, _mm256_mul_ps(two, x));
	return log_pos_norm(expr);
}

void fy::simd::intrinsic::asinhacosh(__m256 input, __m256* asinh_res, __m256* acosh_res) noexcept
{
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 two = _mm256_set1_ps(2.0f);
	const __m256 zero = _mm256_setzero_ps();
	const __m256 two_pow_28 = _mm256_set1_ps(268435456.0f);
	const __m256 two_pow_minus_28 = _mm256_set1_ps(3.7252902984619140625e-9f);

	const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
	const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
	const __m256 all_ones = _mm256_castsi256_ps(_mm256_set1_epi32(-1));

	const __m256 pinf = _mm256_castsi256_ps(_mm256_set1_epi32(0x7f800000));
	const __m256 qnan = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000));

	__m256 ax = _mm256_and_ps(input, abs_mask);
	__m256 x2 = _mm256_mul_ps(input, input);

	__m256 mask_nan = _mm256_cmp_ps(input, input, _CMP_UNORD_Q);
	__m256 mask_abs_inf = _mm256_cmp_ps(ax, pinf, _CMP_EQ_OQ);
	__m256 mask_pos_inf = _mm256_cmp_ps(input, pinf, _CMP_EQ_OQ);

	if (asinh_res)
	{
		__m256 mask_special = _mm256_or_ps(mask_nan, mask_abs_inf);
		__m256 mask_tiny = _mm256_andnot_ps(mask_special, _mm256_cmp_ps(ax, two_pow_minus_28, _CMP_LT_OQ));
		__m256 mask_large = _mm256_andnot_ps(mask_special, _mm256_cmp_ps(ax, two_pow_28, _CMP_GT_OQ));
		__m256 mask_gt2 = _mm256_andnot_ps(_mm256_or_ps(mask_special, mask_large), _mm256_cmp_ps(ax, two, _CMP_GT_OQ));
		__m256 mask_small = _mm256_andnot_ps(_mm256_or_ps(mask_special, _mm256_or_ps(mask_tiny, _mm256_or_ps(mask_large, mask_gt2))), all_ones);

		__m256 result = input;

		if (_mm256_movemask_ps(mask_small))
		{
			__m256 r = asinh_pos_small(ax, x2);
			result = _mm256_blendv_ps(result, r, mask_small);
		}

		if (_mm256_movemask_ps(mask_gt2))
		{
			__m256 r = asinh_pos_gt2(ax, x2);
			result = _mm256_blendv_ps(result, r, mask_gt2);
		}

		if (_mm256_movemask_ps(mask_large))
		{
			__m256 r = log2_pos_norm(ax);
			result = _mm256_blendv_ps(result, r, mask_large);
		}

		if (_mm256_movemask_ps(mask_special))
		{
			result = _mm256_blendv_ps(result, _mm256_add_ps(input, input), mask_special);
		}

		__m256 non_tiny_non_special = _mm256_andnot_ps(mask_special, _mm256_andnot_ps(mask_tiny, all_ones));
		__m256 negmask = _mm256_cmp_ps(input, zero, _CMP_LT_OQ);
		__m256 flipmask = _mm256_and_ps(negmask, non_tiny_non_special);
		result = _mm256_xor_ps(result, _mm256_and_ps(flipmask, sign_mask));

		*asinh_res = result;
	}

	if (acosh_res)
	{
		__m256 mask_lt1 = _mm256_cmp_ps(input, one, _CMP_LT_OQ);
		__m256 mask_eq1 = _mm256_cmp_ps(input, one, _CMP_EQ_OQ);
		__m256 mask_ge1 = _mm256_cmp_ps(input, one, _CMP_GE_OQ);
		__m256 mask_gt2 = _mm256_cmp_ps(input, two, _CMP_GT_OQ);
		__m256 mask_large = _mm256_cmp_ps(input, two_pow_28, _CMP_GE_OQ);

		__m256 mask_gt2_only = _mm256_andnot_ps(mask_large, mask_gt2);
		__m256 mask_1to2 = _mm256_andnot_ps(mask_gt2, mask_ge1);

		__m256 result = zero;

		if (_mm256_movemask_ps(mask_1to2))
		{
			__m256 r = acosh_1to2(input);
			result = _mm256_blendv_ps(result, r, mask_1to2);
		}

		if (_mm256_movemask_ps(mask_gt2_only))
		{
			__m256 r = acosh_gt2(input, x2);
			result = _mm256_blendv_ps(result, r, mask_gt2_only);
		}

		if (_mm256_movemask_ps(mask_large))
		{
			__m256 r = log2_pos_norm(input);
			result = _mm256_blendv_ps(result, r, mask_large);
		}

		result = _mm256_blendv_ps(result, zero, mask_eq1);
		result = _mm256_blendv_ps(result, qnan, mask_lt1);
		result = _mm256_blendv_ps(result, input, mask_pos_inf);

		const __m256i qnan_quiet_i = _mm256_set1_epi32(0x00400000);
		__m256 nan_quieted = _mm256_castsi256_ps(
			_mm256_or_si256(_mm256_castps_si256(input), qnan_quiet_i)
		);
		result = _mm256_blendv_ps(result, nan_quieted, mask_nan);

		*acosh_res = result;
	}
}

__m256 fy::simd::intrinsic::asinh(__m256 input) noexcept
{
	const __m256 two = _mm256_set1_ps(2.0f);
	const __m256 two_pow_28 = _mm256_set1_ps(268435456.0f);
	const __m256 two_pow_minus_28 = _mm256_set1_ps(3.7252902984619140625e-9f);
	const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
	const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
	const __m256 inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());
	const __m256 all_ones = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
	const __m256 zero = _mm256_setzero_ps();

	__m256 ax = _mm256_and_ps(input, abs_mask);
	__m256 x2 = _mm256_mul_ps(input, input);

	__m256 mask_nan = _mm256_cmp_ps(input, input, _CMP_UNORD_Q);
	__m256 mask_inf = _mm256_cmp_ps(ax, inf, _CMP_EQ_OQ);
	__m256 mask_special = _mm256_or_ps(mask_nan, mask_inf);

	__m256 mask_tiny = _mm256_andnot_ps(mask_special, _mm256_cmp_ps(ax, two_pow_minus_28, _CMP_LT_OQ));

	__m256 mask_large = _mm256_andnot_ps(mask_special, _mm256_cmp_ps(ax, two_pow_28, _CMP_GT_OQ));

	__m256 mask_gt2 = _mm256_andnot_ps(_mm256_or_ps(mask_special, mask_large), _mm256_cmp_ps(ax, two, _CMP_GT_OQ));

	__m256 mask_small = _mm256_andnot_ps(_mm256_or_ps(mask_special, _mm256_or_ps(mask_tiny, _mm256_or_ps(mask_large, mask_gt2))), all_ones);

	__m256 result = input;

	if (_mm256_movemask_ps(mask_small))
	{
		__m256 r = asinh_pos_small(ax, x2);
		result = _mm256_blendv_ps(result, r, mask_small);
	}

	if (_mm256_movemask_ps(mask_gt2))
	{
		__m256 r = asinh_pos_gt2(ax, x2);
		result = _mm256_blendv_ps(result, r, mask_gt2);
	}

	if (_mm256_movemask_ps(mask_large))
	{
		__m256 r = log2_pos_norm(ax);
		result = _mm256_blendv_ps(result, r, mask_large);
	}

	if (_mm256_movemask_ps(mask_special))
	{
		result = _mm256_blendv_ps(result, _mm256_add_ps(input, input), mask_special);
	}

	__m256 non_tiny_non_special = _mm256_andnot_ps(mask_special, _mm256_andnot_ps(mask_tiny, all_ones));
	__m256 negmask = _mm256_cmp_ps(input, zero, _CMP_LT_OQ);
	__m256 flipmask = _mm256_and_ps(negmask, non_tiny_non_special);
	result = _mm256_xor_ps(result, _mm256_and_ps(flipmask, sign_mask));

	return result;
}

__m256 fy::simd::intrinsic::acosh(__m256 input) noexcept
{
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 two = _mm256_set1_ps(2.0f);
	const __m256 threshold = _mm256_set1_ps(268435456.0f);
	const __m256 zero = _mm256_setzero_ps();

	const __m256 qnan = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000));
	const __m256 pinf = _mm256_castsi256_ps(_mm256_set1_epi32(0x7f800000));

	__m256 x2 = _mm256_mul_ps(input, input);

	__m256 mask_nan = _mm256_cmp_ps(input, input, _CMP_UNORD_Q);
	__m256 mask_lt1 = _mm256_cmp_ps(input, one, _CMP_LT_OQ);
	__m256 mask_eq1 = _mm256_cmp_ps(input, one, _CMP_EQ_OQ);
	__m256 mask_ge1 = _mm256_cmp_ps(input, one, _CMP_GE_OQ);
	__m256 mask_gt2 = _mm256_cmp_ps(input, two, _CMP_GT_OQ);
	__m256 mask_large = _mm256_cmp_ps(input, threshold, _CMP_GE_OQ);
	__m256 mask_inf = _mm256_cmp_ps(input, pinf, _CMP_EQ_OQ);

	__m256 mask_gt2_only = _mm256_andnot_ps(mask_large, mask_gt2);
	__m256 mask_1to2 = _mm256_andnot_ps(mask_gt2, mask_ge1);

	__m256 result = zero;

	if (_mm256_movemask_ps(mask_1to2))
	{
		__m256 r = acosh_1to2(input);
		result = _mm256_blendv_ps(result, r, mask_1to2);
	}

	if (_mm256_movemask_ps(mask_gt2_only))
	{
		__m256 r = acosh_gt2(input, x2);
		result = _mm256_blendv_ps(result, r, mask_gt2_only);
	}

	if (_mm256_movemask_ps(mask_large))
	{
		__m256 r = log2_pos_norm(input);
		result = _mm256_blendv_ps(result, r, mask_large);
	}

	result = _mm256_blendv_ps(result, zero, mask_eq1);
	result = _mm256_blendv_ps(result, qnan, mask_lt1);
	result = _mm256_blendv_ps(result, input, mask_inf);

	const __m256i qnan_quiet_i = _mm256_set1_epi32(0x00400000);
	__m256 nan_quieted = _mm256_castsi256_ps(
		_mm256_or_si256(_mm256_castps_si256(input), qnan_quiet_i)
	);
	result = _mm256_blendv_ps(result, nan_quieted, mask_nan);

	return result;
}


static inline __m256 log1p_nonneg_ge2(const __m256 x) noexcept
{
	const __m256 zero = _mm256_setzero_ps();
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 half = _mm256_set1_ps(0.5f);
	const __m256 two = _mm256_set1_ps(2.0f);

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

	const __m256i i_zero = _mm256_setzero_si256();
	const __m256i i_one = _mm256_set1_epi32(1);
	const __m256i i_127 = _mm256_set1_epi32(127);
	const __m256i i_mant = _mm256_set1_epi32(0x007fffff);
	const __m256i i_exp1 = _mm256_set1_epi32(0x3f800000);
	const __m256i i_exp05 = _mm256_set1_epi32(0x3f000000);
	const __m256i i_hidden = _mm256_set1_epi32(0x00800000);

	const __m256i t_hu = _mm256_set1_epi32(0x003504f4);

	__m256 u = _mm256_add_ps(one, x);
	__m256i ui = _mm256_castps_si256(u);

	__m256i k = _mm256_sub_epi32(_mm256_srli_epi32(ui, 23), i_127);

	__m256 c = _mm256_div_ps(_mm256_sub_ps(one, _mm256_sub_ps(u, x)), u);

	__m256i mant = _mm256_and_si256(ui, i_mant);
	__m256i m_mant_small = _mm256_cmpgt_epi32(t_hu, mant);

	__m256i u_bits_a = _mm256_or_si256(mant, i_exp1);
	__m256i u_bits_b = _mm256_or_si256(mant, i_exp05);
	__m256 u_norm = _mm256_castsi256_ps(_mm256_blendv_epi8(u_bits_b, u_bits_a, m_mant_small));

	__m256i hu = _mm256_blendv_epi8(
		_mm256_srli_epi32(_mm256_sub_epi32(i_hidden, mant), 2),
		mant,
		m_mant_small);

	k = _mm256_add_epi32(k, _mm256_andnot_si256(m_mant_small, i_one));

	__m256 f = _mm256_sub_ps(u_norm, one);

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

	__m256 result =
		_mm256_sub_ps(
			k_ln2_hi,
			_mm256_sub_ps(
				_mm256_sub_ps(
					hfsq,
					_mm256_add_ps(_mm256_mul_ps(s, _mm256_add_ps(hfsq, R)),
						_mm256_add_ps(k_ln2_lo, c))),
				f));

	__m256 R_hu0 = _mm256_mul_ps(hfsq, _mm256_fnmadd_ps(c066666666, f, one));
	__m256 res_hu0 =
		_mm256_sub_ps(
			k_ln2_hi,
			_mm256_sub_ps(
				_mm256_sub_ps(R_hu0, _mm256_add_ps(k_ln2_lo, c)),
				f));

	__m256i m_hu_zero = _mm256_cmpeq_epi32(hu, i_zero);
	result = _mm256_blendv_ps(result, res_hu0, _mm256_castsi256_ps(m_hu_zero));

	__m256i m_f_zero = _mm256_castps_si256(_mm256_cmp_ps(f, zero, _CMP_EQ_OQ));
	__m256i m_hu0_f0 = _mm256_and_si256(m_hu_zero, m_f_zero);

	__m256 res_hu0_f0 = _mm256_add_ps(k_ln2_hi, _mm256_add_ps(c, k_ln2_lo));
	result = _mm256_blendv_ps(result, res_hu0_f0, _mm256_castsi256_ps(m_hu0_f0));

	return result;
}

static inline __m256 atanh_small_poly(__m256 x) noexcept
{
	const __m256 c1 = _mm256_set1_ps(1.0f / 3.0f);
	const __m256 c2 = _mm256_set1_ps(1.0f / 5.0f);
	const __m256 c3 = _mm256_set1_ps(1.0f / 7.0f);
	const __m256 c4 = _mm256_set1_ps(1.0f / 9.0f);

	__m256 x2 = _mm256_mul_ps(x, x);
	__m256 p = _mm256_fmadd_ps(c4, x2, c3);
	p = _mm256_fmadd_ps(p, x2, c2);
	p = _mm256_fmadd_ps(p, x2, c1);

	__m256 x3 = _mm256_mul_ps(x, x2);
	return _mm256_fmadd_ps(x3, p, x);
}

static inline __m256 log1p_nonneg_atanh_dispatch(__m256 x) noexcept
{
	__m256i xi = _mm256_castps_si256(x);

	__m256i m_ge2 = _mm256_cmpgt_epi32(xi, _mm256_set1_epi32(0x3fffffff));
	if (_mm256_movemask_ps(_mm256_castsi256_ps(m_ge2)) == 0xFF)
	{
		return log1p_nonneg_ge2(x);
	}

	return log1p_nonneg(x);
}

__m256 fy::simd::intrinsic::atanh(__m256 x) noexcept
{
	const __m256 vone = _mm256_set1_ps(1.0f);
	const __m256 vhalf = _mm256_set1_ps(0.5f);
	const __m256 vnzero = _mm256_set1_ps(-0.0f);

	const __m256i abs_mask = _mm256_set1_epi32(0x7fffffff);
	const __m256i thresh_1 = _mm256_set1_epi32(0x3f800000);
	const __m256i thresh_2m28 = _mm256_set1_epi32(0x31800000);
	const __m256i thresh_small = _mm256_set1_epi32(0x3e000000);
	const __m256i pos_inf_i = _mm256_set1_epi32(0x7f800000);
	const __m256i qnan_i = _mm256_set1_epi32(0x7fc00000);

	__m256i xi = _mm256_castps_si256(x);
	__m256i ax_i = _mm256_and_si256(xi, abs_mask);
	__m256  ax = _mm256_castsi256_ps(ax_i);

	__m256i m_tiny_i = _mm256_cmpgt_epi32(thresh_2m28, ax_i);
	__m256i m_eq1_i = _mm256_cmpeq_epi32(ax_i, thresh_1);
	__m256i m_lt1_i = _mm256_cmpgt_epi32(thresh_1, ax_i);
	__m256i m_gt1_i = _mm256_cmpgt_epi32(ax_i, thresh_1);
	__m256i m_small_i = _mm256_cmpgt_epi32(thresh_small, ax_i);

	if (_mm256_movemask_ps(_mm256_castsi256_ps(m_tiny_i)) == 0xFF)
	{
		return x;
	}

	if (_mm256_movemask_ps(_mm256_castsi256_ps(m_small_i)) == 0xFF)
	{
		__m256 y = atanh_small_poly(x);
		__m256 sign = _mm256_and_ps(x, vnzero);
		__m256 inf_res = _mm256_or_ps(_mm256_castsi256_ps(pos_inf_i), sign);
		__m256 nan_res = _mm256_castsi256_ps(qnan_i);

		__m256 result = y;
		result = _mm256_blendv_ps(result, x, _mm256_castsi256_ps(m_tiny_i));
		result = _mm256_blendv_ps(result, inf_res, _mm256_castsi256_ps(m_eq1_i));
		result = _mm256_blendv_ps(result, nan_res, _mm256_castsi256_ps(m_gt1_i));
		return result;
	}

	__m256 sign = _mm256_and_ps(x, vnzero);
	__m256 inf_res = _mm256_or_ps(_mm256_castsi256_ps(pos_inf_i), sign);
	__m256 nan_res = _mm256_castsi256_ps(qnan_i);

	__m256i m_need_i = _mm256_andnot_si256(m_tiny_i, m_lt1_i);
	__m256  y = _mm256_and_ps(ax, _mm256_castsi256_ps(m_need_i));

	__m256 one_minus_y = _mm256_sub_ps(vone, y);
	__m256 two_y = _mm256_add_ps(y, y);
	__m256 arg = _mm256_div_ps(two_y, one_minus_y);

	__m256 t_abs = _mm256_mul_ps(vhalf, log1p_nonneg(arg));
	__m256 t_result = _mm256_or_ps(t_abs, sign);

	__m256 result = t_result;
	result = _mm256_blendv_ps(result, x, _mm256_castsi256_ps(m_tiny_i));
	result = _mm256_blendv_ps(result, inf_res, _mm256_castsi256_ps(m_eq1_i));
	result = _mm256_blendv_ps(result, nan_res, _mm256_castsi256_ps(m_gt1_i));
	return result;
}