#include "foye_fastmath_fp32.hpp"

template<int N>
static inline __m256 __vectorcall horner_desc_ps(__m256 x,
	const float(&c)[N]) noexcept
{
	__m256 y = _mm256_broadcast_ss(&c[0]);
	for (int i = 1; i < N; ++i)
	{
		y = _mm256_fmadd_ps(y, x, _mm256_broadcast_ss(&c[i]));
	}
	return y;
}

template<int NP, int NQ>
static inline __m256 __vectorcall rational_desc_ps(__m256 x,
	const float(&p)[NP], const float(&q)[NQ]) noexcept
{
	__m256 num = horner_desc_ps(x, p);
	__m256 den = horner_desc_ps(x, q);
	return _mm256_div_ps(num, den);
}

static __m256 __vectorcall exp_erfc_tail(const __m256 x) noexcept
{
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 half = _mm256_set1_ps(0.5f);

	const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
	const __m256 ln2_hi = _mm256_set1_ps(-6.93359375E-1f);
	const __m256 ln2_lo = _mm256_set1_ps(2.12194440E-4f);

	alignas(32) static constexpr float exp_poly[] = {
		1.9875691500E-4f,
		1.3981999507E-3f,
		8.3334519073E-3f,
		4.1665795894E-2f,
		1.6666665459E-1f,
		5.0000001201E-1f
	};

	__m256 exponent_float = _mm256_floor_ps(_mm256_fmadd_ps(x, log2e, half));
	__m256i exponent_int = _mm256_cvtps_epi32(exponent_float);

	__m256 reduced = _mm256_fmadd_ps(exponent_float, ln2_hi, x);
	reduced = _mm256_fmadd_ps(exponent_float, ln2_lo, reduced);

	__m256 reduced_sq = _mm256_mul_ps(reduced, reduced);
	__m256 poly = horner_desc_ps(reduced, exp_poly);

	__m256 exp_reduced = _mm256_fmadd_ps(poly, reduced_sq, reduced);
	exp_reduced = _mm256_add_ps(exp_reduced, one);

	const __m256i int_bias_127 = _mm256_set1_epi32(127);
	__m256i normal_exp_bits = _mm256_add_epi32(exponent_int, int_bias_127);
	normal_exp_bits = _mm256_slli_epi32(normal_exp_bits, 23);
	__m256 normal_scale = _mm256_castsi256_ps(normal_exp_bits);
	__m256 normal_result = _mm256_mul_ps(exp_reduced, normal_scale);

	const __m256 two_pow_23 = _mm256_set1_ps(8388608.0f);
	__m256i mantissa_fixed_23 = _mm256_cvtps_epi32(_mm256_mul_ps(exp_reduced, two_pow_23));

	const __m256i int_neg126 = _mm256_set1_epi32(-126);
	const __m256i int_one = _mm256_set1_epi32(1);

	__m256i sub_shift = _mm256_sub_epi32(int_neg126, exponent_int);
	__m256i sub_shift_minus_one = _mm256_sub_epi32(sub_shift, int_one);
	__m256i round_add = _mm256_sllv_epi32(int_one, sub_shift_minus_one);

	__m256i mantissa_rounded = _mm256_add_epi32(mantissa_fixed_23, round_add);
	__m256i subnormal_bits = _mm256_srlv_epi32(mantissa_rounded, sub_shift);

	__m256 subnormal_result = _mm256_castsi256_ps(subnormal_bits);

	__m256i use_subnormal_path_mask_i = _mm256_cmpgt_epi32(_mm256_set1_epi32(-126), exponent_int);

	return _mm256_blendv_ps(
		normal_result,
		subnormal_result,
		_mm256_castsi256_ps(use_subnormal_path_mask_i));
}

static __m256 __vectorcall erfc_tail_eval(__m256 ax, __m256 z, __m256 is_neg, 
	bool tail1) noexcept
{
	const __m256 zero = _mm256_setzero_ps();
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 two = _mm256_set1_ps(2.0f);
	const __m256 m5625 = _mm256_set1_ps(-5.62500000000000000000e-01f);

	__m256 inv_z = _mm256_div_ps(one, z);

	alignas(32) static constexpr float ra[] = {
		-9.8143291473e+00f,
		-8.1287437439e+01f,
		-1.8460508728e+02f,
		-1.6239666748e+02f,
		-6.2375331879e+01f,
		-1.0558626175e+01f,
		-6.9385856390e-01f,
		-9.8649440333e-03f
	};

	alignas(32) static const float sa[] = {
		-6.0424413532e-02f,
		 6.5702495575e+00f,
		 1.0863500214e+02f,
		 4.2900814819e+02f,
		 6.4538726807e+02f,
		 4.3456588745e+02f,
		 1.3765776062e+02f,
		 1.9651271820e+01f,
		 1.0f
	};

	alignas(32) static constexpr float rb[] = {
		-4.8351919556e+02f,
		-1.0250950928e+03f,
		-6.3756646729e+02f,
		-1.6063638306e+02f,
		-1.7757955551e+01f,
		-7.9928326607e-01f,
		-9.8649431020e-03f
	};

	alignas(32) static constexpr float sb[] = {
		 4.7452853394e+02f,
		 2.5530502930e+03f,
		 3.1998581543e+03f,
		 1.5367296143e+03f,
		 3.2579251099e+02f,
		 3.0338060379e+01f,
		 1.0f
	};

	__m256 rs = tail1 
		? rational_desc_ps(inv_z, ra, sa)
		: rational_desc_ps(inv_z, rb, sb);

	const __m256i trunc_mask_i = _mm256_set1_epi32(0xffffe000);

	__m256 s = _mm256_castsi256_ps(_mm256_and_si256(_mm256_castps_si256(ax), trunc_mask_i));
	__m256 ss = _mm256_mul_ps(s, s);

	__m256 corr = _mm256_fmadd_ps(_mm256_sub_ps(s, ax), _mm256_add_ps(s, ax),rs);

	__m256 exp1_arg = _mm256_add_ps(_mm256_sub_ps(zero, ss), m5625);

	__m256 exp1 = exp_erfc_tail(exp1_arg);
	__m256 exp2 = exp_erfc_tail(corr);
	__m256 e = _mm256_mul_ps(exp1, exp2);

	__m256 erfc_abs = _mm256_div_ps(e, ax);

	return _mm256_blendv_ps(erfc_abs, _mm256_sub_ps(two, erfc_abs), is_neg);
}

static __m256 __vectorcall erfc_small_branch(__m256 z, __m256 ax, __m256 neg_mask, 
	__m256 blend_mask)
{
	alignas(32) static constexpr float pp[] = {
		-2.3763017452e-05f,
		-5.7702702470e-03f,
		-2.8481749818e-02f,
		-3.2504209876e-01f,
		 1.2837916613e-01f
	};

	alignas(32) static constexpr float qq[] = {
		-3.9602282413e-06f,
		 1.3249473704e-04f,
		 5.0813062117e-03f,
		 6.5022252500e-02f,
		 3.9791721106e-01f,
		 1.0f
	};

	const __m256 one = _mm256_set1_ps(1.0f);
	__m256 y = rational_desc_ps(z, pp, qq);
	__m256 erf_abs = _mm256_fmadd_ps(ax, y, ax);

	__m256 erfc_pos = _mm256_sub_ps(one, erf_abs);
	__m256 erfc_neg = _mm256_add_ps(one, erf_abs);
	__m256 erfc_small = _mm256_blendv_ps(erfc_pos, erfc_neg, neg_mask);

	__m256 result = _mm256_set1_ps(1.0f);
	result = _mm256_blendv_ps(result, erfc_small, blend_mask);

	return result;
}

static __m256 __vectorcall erfc_mid_branch(__m256 z, __m256 ax, __m256 neg_mask, __m256 blend_mask)
{
	alignas(32) static const float pa[] = {
			-2.1663755178e-03f,
			 3.5478305072e-02f,
			-1.1089469492e-01f,
			 3.1834661961e-01f,
			-3.7220788002e-01f,
			 4.1485610604e-01f,
			-2.3621185683e-03f
	};

	alignas(32) static const float qa[] = {
		 1.1984499916e-02f,
		 1.3637083583e-02f,
		 1.2617121637e-01f,
		 7.1828655899e-02f,
		 5.4039794207e-01f,
		 1.0642088205e-01f,
		 1.0f
	};

	const __m256 one = _mm256_set1_ps(1.0f);
	__m256 s = _mm256_sub_ps(ax, one);
	__m256 frac = rational_desc_ps(s, pa, qa);

	const __m256 one_minus_erx = _mm256_set1_ps(1.54937088489532470703e-01f);
	const __m256 one_plus_erx = _mm256_set1_ps(1.84506297111511230469e+00f);

	__m256 erfc_pos = _mm256_sub_ps(one_minus_erx, frac);
	__m256 erfc_neg = _mm256_add_ps(one_plus_erx, frac);
	__m256 erfc_mid = _mm256_blendv_ps(erfc_pos, erfc_neg, neg_mask);

	__m256 result = _mm256_set1_ps(1.0f);
	result = _mm256_blendv_ps(result, erfc_mid, blend_mask);

	return result;
}

__m256 fy::simd::intrinsic::erfc(__m256 input) noexcept
{
	const __m256 zero = _mm256_setzero_ps();
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 two = _mm256_set1_ps(2.0f);

	const __m256 small_bound = _mm256_set1_ps(0.84375f);
	const __m256 mid_bound = _mm256_set1_ps(1.25f);
	const __m256 tail_bound = _mm256_set1_ps(2.857143f);
	const __m256 huge_bound = _mm256_set1_ps(10.0f);

	const __m256i abs_mask_i = _mm256_set1_epi32(0x7fffffff);
	const __m256i sign_mask_i = _mm256_set1_epi32(0x80000000);
	const __m256i inf_bits_i = _mm256_set1_epi32(0x7f800000);

	__m256i ux = _mm256_castps_si256(input);
	__m256i ax_i = _mm256_and_si256(ux, abs_mask_i);

	__m256 ax = _mm256_castsi256_ps(ax_i);
	__m256 z = _mm256_mul_ps(ax, ax);

	__m256 is_neg = _mm256_castsi256_ps(_mm256_and_si256(ux, sign_mask_i));

	const __m256 mask_small = _mm256_cmp_ps(ax, small_bound, _CMP_LT_OQ);

	const __m256 mask_mid = _mm256_and_ps(
		_mm256_cmp_ps(ax, small_bound, _CMP_GE_OQ),
		_mm256_cmp_ps(ax, mid_bound, _CMP_LT_OQ));

	const __m256 mask_tail1 = _mm256_and_ps(
		_mm256_cmp_ps(ax, mid_bound, _CMP_GE_OQ),
		_mm256_cmp_ps(ax, tail_bound, _CMP_LT_OQ));

	const __m256 mask_tail2 = _mm256_and_ps(
		_mm256_cmp_ps(ax, tail_bound, _CMP_GE_OQ),
		_mm256_cmp_ps(ax, huge_bound, _CMP_LT_OQ));

	const __m256 mask_huge = _mm256_cmp_ps(ax, huge_bound, _CMP_GE_OQ);

	const int bits_tail1 = _mm256_movemask_ps(mask_tail1);
	const int bits_tail2 = _mm256_movemask_ps(mask_tail2);

	__m256 result = one;

	if (_mm256_movemask_ps(mask_small))
	{
		result = erfc_small_branch(z, ax, is_neg, mask_small);
	}

	if (_mm256_movemask_ps(mask_mid))
	{
		result = erfc_mid_branch(z, ax, is_neg, mask_mid);
	}

	if (bits_tail1 && !bits_tail2)
	{
		__m256 erfc_tail1 = erfc_tail_eval(ax, z, is_neg, true);
		result = _mm256_blendv_ps(result, erfc_tail1, mask_tail1);
	}
	else if (!bits_tail1 && bits_tail2)
	{
		__m256 erfc_tail2 = erfc_tail_eval(ax, z, is_neg, false);
		result = _mm256_blendv_ps(result, erfc_tail2, mask_tail2);
	}
	else if (bits_tail1 && bits_tail2)
	{
		__m256 erfc_tail1 = erfc_tail_eval(ax, z, is_neg, true);
		__m256 erfc_tail2 = erfc_tail_eval(ax, z, is_neg, false);

		result = _mm256_blendv_ps(result, erfc_tail1, mask_tail1);
		result = _mm256_blendv_ps(result, erfc_tail2, mask_tail2);
	}
	
	if (_mm256_movemask_ps(mask_huge))
	{
		__m256 huge_res = _mm256_blendv_ps(zero, two, is_neg);
		result = _mm256_blendv_ps(result, huge_res, mask_huge);
	}

	__m256 nan_mask = _mm256_cmp_ps(input, input, _CMP_UNORD_Q);
	__m256 is_inf = _mm256_castsi256_ps(_mm256_cmpeq_epi32(ax_i, inf_bits_i));

	__m256 inf_res = _mm256_blendv_ps(zero, two, is_neg);
	result = _mm256_blendv_ps(result, inf_res, is_inf);

	__m256 nan_res = _mm256_add_ps(input, input);
	result = _mm256_blendv_ps(result, nan_res, nan_mask);

	return result;
}