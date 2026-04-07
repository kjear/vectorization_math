#include <foye_fastmath.hpp>

void fy::simd::intrinsic::sinhcosh(__m256 input, __m256* sinh_result, __m256* cosh_result) noexcept
{
	const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
	__m256 a = _mm256_and_ps(input, abs_mask);
	__m256 sign = _mm256_xor_ps(input, a);

	__m256 mask_nan = _mm256_cmp_ps(input, input, _CMP_UNORD_Q);

	const __m256i qnan_quiet_bit = _mm256_set1_epi32(0x00400000);
	__m256 nan_quieted = _mm256_castsi256_ps(
		_mm256_or_si256(_mm256_castps_si256(input), qnan_quiet_bit)
	);

	__m256 a2 = _mm256_mul_ps(a, a);
	__m256 sinh_small, cosh_small;

	if (sinh_result)
	{
		__m256 p_sinh = _mm256_set1_ps(2.03721912945e-4f);
		p_sinh = _mm256_fmadd_ps(p_sinh, a2, _mm256_set1_ps(8.33028376239e-3f));
		p_sinh = _mm256_fmadd_ps(p_sinh, a2, _mm256_set1_ps(1.66667160211e-1f));
		__m256 a3 = _mm256_mul_ps(a, a2);
		sinh_small = _mm256_fmadd_ps(a3, p_sinh, a);
	}

	if (cosh_result)
	{
		__m256 p_cosh = _mm256_set1_ps(2.4801587e-5f);
		p_cosh = _mm256_fmadd_ps(p_cosh, a2, _mm256_set1_ps(1.3888889e-3f));
		p_cosh = _mm256_fmadd_ps(p_cosh, a2, _mm256_set1_ps(4.1666664e-2f));
		p_cosh = _mm256_fmadd_ps(p_cosh, a2, _mm256_set1_ps(0.5f));
		cosh_small = _mm256_fmadd_ps(p_cosh, a2, _mm256_set1_ps(1.0f));
	}

	const __m256 threshold = _mm256_set1_ps(1.0f);
	__m256 mask_ge = _mm256_cmp_ps(a, threshold, _CMP_GE_OQ);

	const __m256 overflow_threshold = _mm256_set1_ps(89.415985f);
	__m256 mask_overflow = _mm256_cmp_ps(a, overflow_threshold, _CMP_GT_OQ);
	const __m256 inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());

	if (sinh_result || cosh_result)
	{
		const __m256 LOG2E = _mm256_set1_ps(1.4426950408889634f);
		const __m256 LN2_HI = _mm256_set1_ps(0.693145751953125f);
		const __m256 LN2_LO = _mm256_set1_ps(1.428606765330187e-6f);
		const __m256 quarter = _mm256_set1_ps(0.25f);

		__m256 a_clamped = _mm256_min_ps(a, overflow_threshold);

		__m256 k_float = _mm256_round_ps(
			_mm256_mul_ps(a_clamped, LOG2E),
			_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
		);
		__m256i k_int = _mm256_cvtps_epi32(k_float);

		__m256 r = _mm256_fnmadd_ps(k_float, LN2_HI, a_clamped);
		r = _mm256_fnmadd_ps(k_float, LN2_LO, r);

		__m256 p_exp = _mm256_set1_ps(1.3930435e-3f);
		p_exp = _mm256_fmadd_ps(p_exp, r, _mm256_set1_ps(8.3333607e-3f));
		p_exp = _mm256_fmadd_ps(p_exp, r, _mm256_set1_ps(4.16664853e-2f));
		p_exp = _mm256_fmadd_ps(p_exp, r, _mm256_set1_ps(1.666666716e-1f));
		p_exp = _mm256_fmadd_ps(p_exp, r, _mm256_set1_ps(0.5f));
		p_exp = _mm256_fmadd_ps(p_exp, r, _mm256_set1_ps(1.0f));
		__m256 exp_r = _mm256_fmadd_ps(p_exp, r, _mm256_set1_ps(1.0f));

		__m256i k_minus_1 = _mm256_sub_epi32(k_int, _mm256_set1_epi32(1));
		__m256i k1 = _mm256_min_epi32(k_minus_1, _mm256_set1_epi32(127));
		__m256i k2 = _mm256_sub_epi32(k_minus_1, k1);

		__m256 scale1 = _mm256_castsi256_ps(
			_mm256_slli_epi32(
				_mm256_add_epi32(k1, _mm256_set1_epi32(127)), 23
			)
		);
		__m256 scale2 = _mm256_castsi256_ps(
			_mm256_slli_epi32(
				_mm256_add_epi32(k2, _mm256_set1_epi32(127)), 23
			)
		);

		__m256 y_prime = _mm256_mul_ps(_mm256_mul_ps(exp_r, scale1), scale2);
		__m256 inv_y = _mm256_div_ps(quarter, y_prime);

		if (sinh_result)
		{
			__m256 sinh_large = _mm256_sub_ps(y_prime, inv_y);
			__m256 sinh_abs_non_nan = _mm256_blendv_ps(sinh_small, sinh_large, mask_ge);
			sinh_abs_non_nan = _mm256_blendv_ps(sinh_abs_non_nan, inf, mask_overflow);

			__m256 sinh_signed = _mm256_xor_ps(sinh_abs_non_nan, sign);
			__m256 sinh_final = _mm256_blendv_ps(sinh_signed, nan_quieted, mask_nan);
			*sinh_result = sinh_final;
		}

		if (cosh_result)
		{
			__m256 cosh_large = _mm256_add_ps(y_prime, inv_y);
			__m256 cosh_val = _mm256_blendv_ps(cosh_small, cosh_large, mask_ge);
			cosh_val = _mm256_blendv_ps(cosh_val, inf, mask_overflow);
			cosh_val = _mm256_blendv_ps(cosh_val, nan_quieted, mask_nan);
			*cosh_result = cosh_val;
		}
	}
}

__m256 fy::simd::intrinsic::sinh(__m256 input) noexcept
{
	__m256 sinh_val;
	::fy::simd::intrinsic::sinhcosh(input, &sinh_val, nullptr);
	return sinh_val;
}

__m256 fy::simd::intrinsic::cosh(__m256 input) noexcept
{
	__m256 cosh_val;
	::fy::simd::intrinsic::sinhcosh(input, nullptr, &cosh_val);
	return cosh_val;
}

__m256 fy::simd::intrinsic::tanh(__m256 input) noexcept
{
	const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
	const __m256 a = _mm256_and_ps(input, abs_mask);
	const __m256 sign = _mm256_xor_ps(input, a);

	const __m256i ix = _mm256_castps_si256(input);
	const __m256i hx = _mm256_and_si256(ix, _mm256_set1_epi32(0x7fffffff));
	const __m256i i_inf = _mm256_set1_epi32(0x7f800000);

	const __m256 mask_nan = _mm256_cmp_ps(input, input, _CMP_UNORD_Q);
	const __m256i mask_inf_i = _mm256_cmpeq_epi32(hx, i_inf);

	const __m256 a2 = _mm256_mul_ps(a, a);

	__m256 p_small = _mm256_set1_ps(0.0005900274f);
	p_small = _mm256_fmadd_ps(p_small, a2, _mm256_set1_ps(-0.0014558344f));
	p_small = _mm256_fmadd_ps(p_small, a2, _mm256_set1_ps(0.0035921280f));
	p_small = _mm256_fmadd_ps(p_small, a2, _mm256_set1_ps(-0.0088632355f));
	p_small = _mm256_fmadd_ps(p_small, a2, _mm256_set1_ps(0.0218694885f));
	p_small = _mm256_fmadd_ps(p_small, a2, _mm256_set1_ps(-0.0539682540f));
	p_small = _mm256_fmadd_ps(p_small, a2, _mm256_set1_ps(0.1333333333f));
	p_small = _mm256_fmadd_ps(p_small, a2, _mm256_set1_ps(-0.3333333333f));

	const __m256 a3 = _mm256_mul_ps(a, a2);
	const __m256 tanh_small = _mm256_fmadd_ps(a3, p_small, a);

	const __m256 LOG2E = _mm256_set1_ps(1.4426950408889634f);
	const __m256 LN2_HI = _mm256_set1_ps(0.693145751953125f);
	const __m256 LN2_LO = _mm256_set1_ps(1.428606765330187e-6f);

	const __m256 max_z = _mm256_set1_ps(20.0f);
	__m256 z = _mm256_min_ps(_mm256_add_ps(a, a), max_z);

	__m256 k_float = _mm256_round_ps(
		_mm256_mul_ps(z, LOG2E),
		_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
	);
	__m256i k_int = _mm256_cvtps_epi32(k_float);

	__m256 r = _mm256_fnmadd_ps(k_float, LN2_HI, z);
	r = _mm256_fnmadd_ps(k_float, LN2_LO, r);

	__m256 p_exp = _mm256_set1_ps(1.3930435e-3f);
	p_exp = _mm256_fmadd_ps(p_exp, r, _mm256_set1_ps(8.3333607e-3f));
	p_exp = _mm256_fmadd_ps(p_exp, r, _mm256_set1_ps(4.16664853e-2f));
	p_exp = _mm256_fmadd_ps(p_exp, r, _mm256_set1_ps(1.666666716e-1f));
	p_exp = _mm256_fmadd_ps(p_exp, r, _mm256_set1_ps(0.5f));
	p_exp = _mm256_fmadd_ps(p_exp, r, _mm256_set1_ps(1.0f));
	__m256 exp_r = _mm256_fmadd_ps(p_exp, r, _mm256_set1_ps(1.0f));

	__m256 scale = _mm256_castsi256_ps(
		_mm256_slli_epi32(
			_mm256_add_epi32(k_int, _mm256_set1_epi32(127)), 23
		)
	);

	__m256 exp_z = _mm256_mul_ps(exp_r, scale);

	__m256 denom = _mm256_add_ps(exp_z, _mm256_set1_ps(1.0f));
	__m256 frac = _mm256_div_ps(_mm256_set1_ps(2.0f), denom);
	__m256 tanh_large = _mm256_sub_ps(_mm256_set1_ps(1.0f), frac);

	const __m256 threshold = _mm256_set1_ps(0.625f);
	const __m256 mask_large = _mm256_cmp_ps(a, threshold, _CMP_GE_OQ);
	__m256 abs_result = _mm256_blendv_ps(tanh_small, tanh_large, mask_large);

	__m256 signed_result = _mm256_xor_ps(abs_result, sign);

	const __m256 one = _mm256_set1_ps(1.0f);
	__m256 signed_one = _mm256_xor_ps(one, sign);

	signed_result = _mm256_blendv_ps(
		signed_result,
		signed_one,
		_mm256_castsi256_ps(mask_inf_i)
	);

	const __m256i qnan_quiet_bit = _mm256_set1_epi32(0x00400000);
	const __m256 nan_quieted = _mm256_castsi256_ps(_mm256_or_si256(ix, qnan_quiet_bit));
	signed_result = _mm256_blendv_ps(signed_result, nan_quieted, mask_nan);

	return signed_result;
}