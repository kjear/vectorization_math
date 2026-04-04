#include "foye_fastmath_fp32.hpp"

static __m256 rcp_nr_ps(__m256 a)
{
	__m256 x = _mm256_rcp_ps(a);
	return _mm256_mul_ps(x, _mm256_fnmadd_ps(a, x, _mm256_set1_ps(2.0f)));
}

static __m256 exp256_neg_ps(__m256 x)
{
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
	const __m256 ln2_hi = _mm256_set1_ps(-6.93359375E-1f);
	const __m256 ln2_lo = _mm256_set1_ps(2.12194440E-4f);

	const __m256 c0 = _mm256_set1_ps(1.9875691500E-4f);
	const __m256 c1 = _mm256_set1_ps(1.3981999507E-3f);
	const __m256 c2 = _mm256_set1_ps(8.3334519073E-3f);
	const __m256 c3 = _mm256_set1_ps(4.1665795894E-2f);
	const __m256 c4 = _mm256_set1_ps(1.6666665459E-1f);
	const __m256 c5 = _mm256_set1_ps(5.0000001201E-1f);

	const __m256i bias127 = _mm256_set1_epi32(127);

	__m256 t = _mm256_mul_ps(x, log2e);
	__m256i n_i = _mm256_cvtps_epi32(t);
	__m256 n = _mm256_cvtepi32_ps(n_i);

	__m256 r = _mm256_fmadd_ps(n, ln2_hi, x);
	r = _mm256_fmadd_ps(n, ln2_lo, r);

	__m256 r2 = _mm256_mul_ps(r, r);

	__m256 p = _mm256_fmadd_ps(r, c0, c1);
	p = _mm256_fmadd_ps(p, r, c2);
	p = _mm256_fmadd_ps(p, r, c3);
	p = _mm256_fmadd_ps(p, r, c4);
	p = _mm256_fmadd_ps(p, r, c5);

	__m256 er = _mm256_fmadd_ps(p, r2, r);
	er = _mm256_add_ps(er, one);

	__m256i e_i = _mm256_add_epi32(n_i, bias127);
	e_i = _mm256_slli_epi32(e_i, 23);
	__m256 scale = _mm256_castsi256_ps(e_i);

	return _mm256_mul_ps(er, scale);
}

static __m256 erf_small_mag_ps(__m256 ax, __m256 z)
{
	const __m256 one = _mm256_set1_ps(1.0f);

	const __m256 pp0 = _mm256_set1_ps(1.2837916613e-01f);
	const __m256 pp1 = _mm256_set1_ps(-3.2504209876e-01f);
	const __m256 pp2 = _mm256_set1_ps(-2.8481749818e-02f);
	const __m256 pp3 = _mm256_set1_ps(-5.7702702470e-03f);
	const __m256 pp4 = _mm256_set1_ps(-2.3763017452e-05f);

	const __m256 qq1 = _mm256_set1_ps(3.9791721106e-01f);
	const __m256 qq2 = _mm256_set1_ps(6.5022252500e-02f);
	const __m256 qq3 = _mm256_set1_ps(5.0813062117e-03f);
	const __m256 qq4 = _mm256_set1_ps(1.3249473704e-04f);
	const __m256 qq5 = _mm256_set1_ps(-3.9602282413e-06f);

	__m256 r = _mm256_fmadd_ps(pp4, z, pp3);
	r = _mm256_fmadd_ps(r, z, pp2);
	r = _mm256_fmadd_ps(r, z, pp1);
	r = _mm256_fmadd_ps(r, z, pp0);

	__m256 s = _mm256_fmadd_ps(qq5, z, qq4);
	s = _mm256_fmadd_ps(s, z, qq3);
	s = _mm256_fmadd_ps(s, z, qq2);
	s = _mm256_fmadd_ps(s, z, qq1);
	s = _mm256_fmadd_ps(s, z, one);

	__m256 y = _mm256_mul_ps(r, rcp_nr_ps(s));
	return _mm256_fmadd_ps(ax, y, ax);
}

static __m256 erf_mid_mag_ps(__m256 ax)
{
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 erx = _mm256_set1_ps(8.45062911510467529297e-01f);

	const __m256 pa0 = _mm256_set1_ps(-2.3621185683e-03f);
	const __m256 pa1 = _mm256_set1_ps(4.1485610604e-01f);
	const __m256 pa2 = _mm256_set1_ps(-3.7220788002e-01f);
	const __m256 pa3 = _mm256_set1_ps(3.1834661961e-01f);
	const __m256 pa4 = _mm256_set1_ps(-1.1089469492e-01f);
	const __m256 pa5 = _mm256_set1_ps(3.5478305072e-02f);
	const __m256 pa6 = _mm256_set1_ps(-2.1663755178e-03f);

	const __m256 qa1 = _mm256_set1_ps(1.0642088205e-01f);
	const __m256 qa2 = _mm256_set1_ps(5.4039794207e-01f);
	const __m256 qa3 = _mm256_set1_ps(7.1828655899e-02f);
	const __m256 qa4 = _mm256_set1_ps(1.2617121637e-01f);
	const __m256 qa5 = _mm256_set1_ps(1.3637083583e-02f);
	const __m256 qa6 = _mm256_set1_ps(1.1984499916e-02f);

	__m256 s = _mm256_sub_ps(ax, one);

	__m256 p = _mm256_fmadd_ps(pa6, s, pa5);
	p = _mm256_fmadd_ps(p, s, pa4);
	p = _mm256_fmadd_ps(p, s, pa3);
	p = _mm256_fmadd_ps(p, s, pa2);
	p = _mm256_fmadd_ps(p, s, pa1);
	p = _mm256_fmadd_ps(p, s, pa0);

	__m256 q = _mm256_fmadd_ps(qa6, s, qa5);
	q = _mm256_fmadd_ps(q, s, qa4);
	q = _mm256_fmadd_ps(q, s, qa3);
	q = _mm256_fmadd_ps(q, s, qa2);
	q = _mm256_fmadd_ps(q, s, qa1);
	q = _mm256_fmadd_ps(q, s, one);

	return _mm256_add_ps(erx, _mm256_mul_ps(p, rcp_nr_ps(q)));
}

static __m256 erf_tail1_mag_ps(__m256 ax, __m256 z)
{
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 m5625 = _mm256_set1_ps(-5.62500000000000000000e-01f);

	const __m256 ra0 = _mm256_set1_ps(-9.8649440333e-03f);
	const __m256 ra1 = _mm256_set1_ps(-6.9385856390e-01f);
	const __m256 ra2 = _mm256_set1_ps(-1.0558626175e+01f);
	const __m256 ra3 = _mm256_set1_ps(-6.2375331879e+01f);
	const __m256 ra4 = _mm256_set1_ps(-1.6239666748e+02f);
	const __m256 ra5 = _mm256_set1_ps(-1.8460508728e+02f);
	const __m256 ra6 = _mm256_set1_ps(-8.1287437439e+01f);
	const __m256 ra7 = _mm256_set1_ps(-9.8143291473e+00f);

	const __m256 sa1 = _mm256_set1_ps(1.9651271820e+01f);
	const __m256 sa2 = _mm256_set1_ps(1.3765776062e+02f);
	const __m256 sa3 = _mm256_set1_ps(4.3456588745e+02f);
	const __m256 sa4 = _mm256_set1_ps(6.4538726807e+02f);
	const __m256 sa5 = _mm256_set1_ps(4.2900814819e+02f);
	const __m256 sa6 = _mm256_set1_ps(1.0863500214e+02f);
	const __m256 sa7 = _mm256_set1_ps(6.5702495575e+00f);
	const __m256 sa8 = _mm256_set1_ps(-6.0424413532e-02f);

	__m256 inv_z = rcp_nr_ps(z);

	__m256 r = _mm256_fmadd_ps(ra7, inv_z, ra6);
	r = _mm256_fmadd_ps(r, inv_z, ra5);
	r = _mm256_fmadd_ps(r, inv_z, ra4);
	r = _mm256_fmadd_ps(r, inv_z, ra3);
	r = _mm256_fmadd_ps(r, inv_z, ra2);
	r = _mm256_fmadd_ps(r, inv_z, ra1);
	r = _mm256_fmadd_ps(r, inv_z, ra0);

	__m256 s = _mm256_fmadd_ps(sa8, inv_z, sa7);
	s = _mm256_fmadd_ps(s, inv_z, sa6);
	s = _mm256_fmadd_ps(s, inv_z, sa5);
	s = _mm256_fmadd_ps(s, inv_z, sa4);
	s = _mm256_fmadd_ps(s, inv_z, sa3);
	s = _mm256_fmadd_ps(s, inv_z, sa2);
	s = _mm256_fmadd_ps(s, inv_z, sa1);
	s = _mm256_fmadd_ps(s, inv_z, one);

	__m256 rs = _mm256_mul_ps(r, rcp_nr_ps(s));
	__m256 exp_arg = _mm256_add_ps(_mm256_sub_ps(rs, z), m5625);
	__m256 e = exp256_neg_ps(exp_arg);
	__m256 erfc_val = _mm256_mul_ps(e, rcp_nr_ps(ax));
	return _mm256_sub_ps(one, erfc_val);
}

static __m256 erf_tail2_mag_ps(__m256 ax, __m256 z)
{
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 m5625 = _mm256_set1_ps(-5.62500000000000000000e-01f);

	const __m256 rb0 = _mm256_set1_ps(-9.8649431020e-03f);
	const __m256 rb1 = _mm256_set1_ps(-7.9928326607e-01f);
	const __m256 rb2 = _mm256_set1_ps(-1.7757955551e+01f);
	const __m256 rb3 = _mm256_set1_ps(-1.6063638306e+02f);
	const __m256 rb4 = _mm256_set1_ps(-6.3756646729e+02f);
	const __m256 rb5 = _mm256_set1_ps(-1.0250950928e+03f);
	const __m256 rb6 = _mm256_set1_ps(-4.8351919556e+02f);

	const __m256 sb1 = _mm256_set1_ps(3.0338060379e+01f);
	const __m256 sb2 = _mm256_set1_ps(3.2579251099e+02f);
	const __m256 sb3 = _mm256_set1_ps(1.5367296143e+03f);
	const __m256 sb4 = _mm256_set1_ps(3.1998581543e+03f);
	const __m256 sb5 = _mm256_set1_ps(2.5530502930e+03f);
	const __m256 sb6 = _mm256_set1_ps(4.7452853394e+02f);

	__m256 inv_z = rcp_nr_ps(z);

	__m256 r = _mm256_fmadd_ps(rb6, inv_z, rb5);
	r = _mm256_fmadd_ps(r, inv_z, rb4);
	r = _mm256_fmadd_ps(r, inv_z, rb3);
	r = _mm256_fmadd_ps(r, inv_z, rb2);
	r = _mm256_fmadd_ps(r, inv_z, rb1);
	r = _mm256_fmadd_ps(r, inv_z, rb0);

	__m256 s = _mm256_fmadd_ps(sb6, inv_z, sb5);
	s = _mm256_fmadd_ps(s, inv_z, sb4);
	s = _mm256_fmadd_ps(s, inv_z, sb3);
	s = _mm256_fmadd_ps(s, inv_z, sb2);
	s = _mm256_fmadd_ps(s, inv_z, sb1);
	s = _mm256_fmadd_ps(s, inv_z, one);

	__m256 rs = _mm256_mul_ps(r, rcp_nr_ps(s));
	__m256 exp_arg = _mm256_add_ps(_mm256_sub_ps(rs, z), m5625);
	__m256 e = exp256_neg_ps(exp_arg);
	__m256 erfc_val = _mm256_mul_ps(e, rcp_nr_ps(ax));
	return _mm256_sub_ps(one, erfc_val);
}

static __m256 erf_tail12_mag_ps(__m256 ax, __m256 z, __m256 mask_tail1)
{
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 m5625 = _mm256_set1_ps(-5.62500000000000000000e-01f);

	__m256 inv_z = rcp_nr_ps(z);

	const __m256 ra0 = _mm256_set1_ps(-9.8649440333e-03f);
	const __m256 ra1 = _mm256_set1_ps(-6.9385856390e-01f);
	const __m256 ra2 = _mm256_set1_ps(-1.0558626175e+01f);
	const __m256 ra3 = _mm256_set1_ps(-6.2375331879e+01f);
	const __m256 ra4 = _mm256_set1_ps(-1.6239666748e+02f);
	const __m256 ra5 = _mm256_set1_ps(-1.8460508728e+02f);
	const __m256 ra6 = _mm256_set1_ps(-8.1287437439e+01f);
	const __m256 ra7 = _mm256_set1_ps(-9.8143291473e+00f);

	const __m256 sa1 = _mm256_set1_ps(1.9651271820e+01f);
	const __m256 sa2 = _mm256_set1_ps(1.3765776062e+02f);
	const __m256 sa3 = _mm256_set1_ps(4.3456588745e+02f);
	const __m256 sa4 = _mm256_set1_ps(6.4538726807e+02f);
	const __m256 sa5 = _mm256_set1_ps(4.2900814819e+02f);
	const __m256 sa6 = _mm256_set1_ps(1.0863500214e+02f);
	const __m256 sa7 = _mm256_set1_ps(6.5702495575e+00f);
	const __m256 sa8 = _mm256_set1_ps(-6.0424413532e-02f);

	const __m256 rb0 = _mm256_set1_ps(-9.8649431020e-03f);
	const __m256 rb1 = _mm256_set1_ps(-7.9928326607e-01f);
	const __m256 rb2 = _mm256_set1_ps(-1.7757955551e+01f);
	const __m256 rb3 = _mm256_set1_ps(-1.6063638306e+02f);
	const __m256 rb4 = _mm256_set1_ps(-6.3756646729e+02f);
	const __m256 rb5 = _mm256_set1_ps(-1.0250950928e+03f);
	const __m256 rb6 = _mm256_set1_ps(-4.8351919556e+02f);

	const __m256 sb1 = _mm256_set1_ps(3.0338060379e+01f);
	const __m256 sb2 = _mm256_set1_ps(3.2579251099e+02f);
	const __m256 sb3 = _mm256_set1_ps(1.5367296143e+03f);
	const __m256 sb4 = _mm256_set1_ps(3.1998581543e+03f);
	const __m256 sb5 = _mm256_set1_ps(2.5530502930e+03f);
	const __m256 sb6 = _mm256_set1_ps(4.7452853394e+02f);

	__m256 r1 = _mm256_fmadd_ps(ra7, inv_z, ra6);
	r1 = _mm256_fmadd_ps(r1, inv_z, ra5);
	r1 = _mm256_fmadd_ps(r1, inv_z, ra4);
	r1 = _mm256_fmadd_ps(r1, inv_z, ra3);
	r1 = _mm256_fmadd_ps(r1, inv_z, ra2);
	r1 = _mm256_fmadd_ps(r1, inv_z, ra1);
	r1 = _mm256_fmadd_ps(r1, inv_z, ra0);

	__m256 s1 = _mm256_fmadd_ps(sa8, inv_z, sa7);
	s1 = _mm256_fmadd_ps(s1, inv_z, sa6);
	s1 = _mm256_fmadd_ps(s1, inv_z, sa5);
	s1 = _mm256_fmadd_ps(s1, inv_z, sa4);
	s1 = _mm256_fmadd_ps(s1, inv_z, sa3);
	s1 = _mm256_fmadd_ps(s1, inv_z, sa2);
	s1 = _mm256_fmadd_ps(s1, inv_z, sa1);
	s1 = _mm256_fmadd_ps(s1, inv_z, one);

	__m256 rs1 = _mm256_mul_ps(r1, rcp_nr_ps(s1));

	__m256 r2 = _mm256_fmadd_ps(rb6, inv_z, rb5);
	r2 = _mm256_fmadd_ps(r2, inv_z, rb4);
	r2 = _mm256_fmadd_ps(r2, inv_z, rb3);
	r2 = _mm256_fmadd_ps(r2, inv_z, rb2);
	r2 = _mm256_fmadd_ps(r2, inv_z, rb1);
	r2 = _mm256_fmadd_ps(r2, inv_z, rb0);

	__m256 s2 = _mm256_fmadd_ps(sb6, inv_z, sb5);
	s2 = _mm256_fmadd_ps(s2, inv_z, sb4);
	s2 = _mm256_fmadd_ps(s2, inv_z, sb3);
	s2 = _mm256_fmadd_ps(s2, inv_z, sb2);
	s2 = _mm256_fmadd_ps(s2, inv_z, sb1);
	s2 = _mm256_fmadd_ps(s2, inv_z, one);

	__m256 rs2 = _mm256_mul_ps(r2, rcp_nr_ps(s2));

	__m256 rs = _mm256_blendv_ps(rs2, rs1, mask_tail1);

	__m256 exp_arg = _mm256_add_ps(_mm256_sub_ps(rs, z), m5625);
	__m256 e = exp256_neg_ps(exp_arg);
	__m256 erfc_val = _mm256_mul_ps(e, rcp_nr_ps(ax));
	return _mm256_sub_ps(one, erfc_val);
}

__m256 fy::simd::intrinsic::erf(__m256 input) noexcept
{
	const __m256 one = _mm256_set1_ps(1.0f);

	const __m256 small_bound = _mm256_set1_ps(0.84375f);
	const __m256 mid_bound = _mm256_set1_ps(1.25f);
	const __m256 tail_bound = _mm256_set1_ps(2.857143f);
	const __m256 huge_bound = _mm256_set1_ps(4.0f);

	const __m256i abs_mask_i = _mm256_set1_epi32(0x7fffffff);
	const __m256i sign_mask_i = _mm256_set1_epi32(0x80000000);
	const __m256i inf_bits_i = _mm256_set1_epi32(0x7f800000);

	__m256i ux = _mm256_castps_si256(input);
	__m256i ax_i = _mm256_and_si256(ux, abs_mask_i);
	__m256i sx_i = _mm256_and_si256(ux, sign_mask_i);

	__m256 ax = _mm256_castsi256_ps(ax_i);
	__m256 signbits = _mm256_castsi256_ps(sx_i);
	__m256 z = _mm256_mul_ps(ax, ax);

	__m256 mask_small = _mm256_cmp_ps(ax, small_bound, _CMP_LT_OQ);

	__m256 mask_mid = _mm256_and_ps(
		_mm256_cmp_ps(ax, small_bound, _CMP_GE_OQ),
		_mm256_cmp_ps(ax, mid_bound, _CMP_LT_OQ));

	__m256 mask_tail1 = _mm256_and_ps(
		_mm256_cmp_ps(ax, mid_bound, _CMP_GE_OQ),
		_mm256_cmp_ps(ax, tail_bound, _CMP_LT_OQ));

	__m256 mask_tail2 = _mm256_and_ps(
		_mm256_cmp_ps(ax, tail_bound, _CMP_GE_OQ),
		_mm256_cmp_ps(ax, huge_bound, _CMP_LT_OQ));

	__m256 mask_tail = _mm256_or_ps(mask_tail1, mask_tail2);

	const int bits_small = _mm256_movemask_ps(mask_small);
	const int bits_mid = _mm256_movemask_ps(mask_mid);
	const int bits_tail1 = _mm256_movemask_ps(mask_tail1);
	const int bits_tail2 = _mm256_movemask_ps(mask_tail2);
	const int bits_tail = bits_tail1 | bits_tail2;

	if (bits_small == 0xFF)
	{
		__m256 mag = erf_small_mag_ps(ax, z);
		__m256 result = _mm256_xor_ps(mag, signbits);

		__m256 nan_mask = _mm256_cmp_ps(input, input, _CMP_UNORD_Q);
		__m256 nan_result = _mm256_add_ps(input, input);
		return _mm256_blendv_ps(result, nan_result, nan_mask);
	}

	if (bits_mid == 0xFF)
	{
		__m256 mag = erf_mid_mag_ps(ax);
		__m256 result = _mm256_xor_ps(mag, signbits);

		__m256 nan_mask = _mm256_cmp_ps(input, input, _CMP_UNORD_Q);
		__m256 nan_result = _mm256_add_ps(input, input);
		return _mm256_blendv_ps(result, nan_result, nan_mask);
	}

	if (bits_tail1 == 0xFF)
	{
		__m256 mag = erf_tail1_mag_ps(ax, z);
		__m256 result = _mm256_xor_ps(mag, signbits);

		__m256 nan_mask = _mm256_cmp_ps(input, input, _CMP_UNORD_Q);
		__m256 nan_result = _mm256_add_ps(input, input);
		return _mm256_blendv_ps(result, nan_result, nan_mask);
	}

	if (bits_tail2 == 0xFF)
	{
		__m256 mag = erf_tail2_mag_ps(ax, z);
		__m256 result = _mm256_xor_ps(mag, signbits);

		__m256 nan_mask = _mm256_cmp_ps(input, input, _CMP_UNORD_Q);
		__m256 nan_result = _mm256_add_ps(input, input);
		return _mm256_blendv_ps(result, nan_result, nan_mask);
	}

	if (bits_tail == 0xFF)
	{
		__m256 mag = erf_tail12_mag_ps(ax, z, mask_tail1);
		__m256 result = _mm256_xor_ps(mag, signbits);

		__m256 nan_mask = _mm256_cmp_ps(input, input, _CMP_UNORD_Q);
		__m256 nan_result = _mm256_add_ps(input, input);
		return _mm256_blendv_ps(result, nan_result, nan_mask);
	}

	__m256 result_mag = one;

	if (bits_small)
	{
		__m256 mag_small = erf_small_mag_ps(ax, z);
		result_mag = _mm256_blendv_ps(result_mag, mag_small, mask_small);
	}

	if (bits_mid)
	{
		__m256 mag_mid = erf_mid_mag_ps(ax);
		result_mag = _mm256_blendv_ps(result_mag, mag_mid, mask_mid);
	}

	if (bits_tail)
	{
		__m256 ax_safe = _mm256_blendv_ps(one, ax, mask_tail);
		__m256 z_safe = _mm256_blendv_ps(one, z, mask_tail);

		__m256 mag_tail;
		if (bits_tail1 && bits_tail2)
		{
			mag_tail = erf_tail12_mag_ps(ax_safe, z_safe, mask_tail1);
		}
		else if (bits_tail1)
		{
			mag_tail = erf_tail1_mag_ps(ax_safe, z_safe);
		}
		else
		{
			mag_tail = erf_tail2_mag_ps(ax_safe, z_safe);
		}

		result_mag = _mm256_blendv_ps(result_mag, mag_tail, mask_tail);
	}

	__m256 result = _mm256_xor_ps(result_mag, signbits);

	__m256 nan_mask = _mm256_cmp_ps(input, input, _CMP_UNORD_Q);
	__m256 is_inf = _mm256_castsi256_ps(_mm256_cmpeq_epi32(ax_i, inf_bits_i));
	__m256 signed_one = _mm256_xor_ps(one, signbits);

	result = _mm256_blendv_ps(result, signed_one, is_inf);

	__m256 nan_result = _mm256_add_ps(input, input);
	result = _mm256_blendv_ps(result, nan_result, nan_mask);

	return result;
}