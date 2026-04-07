#include <foye_fastmath.hpp>

static __m256 exp_build_poly(__m256 reduced, __m256 reduced_sq) noexcept
{
	const __m256 poly_c0 = _mm256_set1_ps(1.9875691500E-4f);
	const __m256 poly_c1 = _mm256_set1_ps(1.3981999507E-3f);
	const __m256 poly_c2 = _mm256_set1_ps(8.3334519073E-3f);
	const __m256 poly_c3 = _mm256_set1_ps(4.1665795894E-2f);
	const __m256 poly_c4 = _mm256_set1_ps(1.6666665459E-1f);
	const __m256 poly_c5 = _mm256_set1_ps(5.0000001201E-1f);

	__m256 reduced_4 = _mm256_mul_ps(reduced_sq, reduced_sq);

	__m256 p0 = _mm256_fmadd_ps(poly_c4, reduced, poly_c5);
	__m256 p1 = _mm256_fmadd_ps(poly_c2, reduced, poly_c3);
	__m256 p2 = _mm256_fmadd_ps(poly_c0, reduced, poly_c1);

	return _mm256_fmadd_ps(
		p2, reduced_4,
		_mm256_fmadd_ps(p1, reduced_sq, p0));
}

static __m256 exp_special_only(__m256 input) noexcept
{
	const __m256 zero = _mm256_setzero_ps();
	const __m256 overflow_input = _mm256_set1_ps(88.72283935546875f);
	const __m256 plus_inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());

	__m256 overflow_mask = _mm256_cmp_ps(input, overflow_input, _CMP_GT_OQ);
	__m256 nan_mask = _mm256_cmp_ps(input, input, _CMP_UNORD_Q);

	const __m256i qnan_quiet_bit = _mm256_set1_epi32(0x00400000);
	__m256 nan_quieted = _mm256_castsi256_ps(
		_mm256_or_si256(_mm256_castps_si256(input), qnan_quiet_bit));

	__m256 result = zero;
	result = _mm256_blendv_ps(result, plus_inf, overflow_mask);
	result = _mm256_blendv_ps(result, nan_quieted, nan_mask);
	return result;
}

static __m256 exp_finalize_special(__m256 result, __m256 input) noexcept
{
	const __m256 zero = _mm256_setzero_ps();
	const __m256 overflow_input = _mm256_set1_ps(88.72283935546875f);
	const __m256 underflow_input = _mm256_set1_ps(-103.97208404541015625f);
	const __m256 plus_inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());

	__m256 underflow_mask = _mm256_cmp_ps(input, underflow_input, _CMP_LT_OQ);
	__m256 overflow_mask = _mm256_cmp_ps(input, overflow_input, _CMP_GT_OQ);
	__m256 nan_mask = _mm256_cmp_ps(input, input, _CMP_UNORD_Q);

	const __m256i qnan_quiet_bit = _mm256_set1_epi32(0x00400000);
	__m256 nan_quieted = _mm256_castsi256_ps(
		_mm256_or_si256(_mm256_castps_si256(input), qnan_quiet_bit));

	result = _mm256_blendv_ps(result, zero, underflow_mask);
	result = _mm256_blendv_ps(result, plus_inf, overflow_mask);
	result = _mm256_blendv_ps(result, nan_quieted, nan_mask);

	return result;
}

static __m256 exp_finalize_subnormal(
	__m256 result,
	__m256 exp_reduced,
	__m256i exponent_int,
	__m256i active_mask_i) noexcept
{
	const __m256 two_pow_23 = _mm256_set1_ps(8388608.0f);

	const __m256i int_neg126 = _mm256_set1_epi32(-126);
	const __m256i int_one = _mm256_set1_epi32(1);

	__m256i subnormal_mask_i = _mm256_and_si256(
		active_mask_i,
		_mm256_cmpgt_epi32(int_neg126, exponent_int));

	if (_mm256_testz_si256(subnormal_mask_i, subnormal_mask_i))
	{
		return result;
	}

	__m256i mantissa_fixed_23 =
		_mm256_cvtps_epi32(_mm256_mul_ps(exp_reduced, two_pow_23));

	__m256i sub_shift = _mm256_sub_epi32(int_neg126, exponent_int);
	__m256i sub_shift_minus_one = _mm256_sub_epi32(sub_shift, int_one);
	__m256i round_add = _mm256_sllv_epi32(int_one, sub_shift_minus_one);
	__m256i mantissa_rounded = _mm256_add_epi32(mantissa_fixed_23, round_add);
	__m256i subnormal_bits = _mm256_srlv_epi32(mantissa_rounded, sub_shift);
	__m256 subnormal_result = _mm256_castsi256_ps(subnormal_bits);

	return _mm256_blendv_ps(result, subnormal_result, _mm256_castsi256_ps(subnormal_mask_i));
}

static __m256 exp_slow_path(
	__m256 input,
	__m256 exp_reduced,
	__m256 normal_result,
	__m256i exponent_int) noexcept
{
	const __m256 two = _mm256_set1_ps(2.0f);
	const __m256 scale_2pow127 = _mm256_castsi256_ps(_mm256_set1_epi32(0x7F000000));

	const __m256 overflow_input = _mm256_set1_ps(88.72283935546875f);
	const __m256 underflow_input = _mm256_set1_ps(-103.97208404541015625f);

	const __m256i int_128 = _mm256_set1_epi32(128);

	__m256 underflow_mask = _mm256_cmp_ps(input, underflow_input, _CMP_LT_OQ);
	__m256 overflow_mask = _mm256_cmp_ps(input, overflow_input, _CMP_GT_OQ);
	__m256 nan_mask = _mm256_cmp_ps(input, input, _CMP_UNORD_Q);

	__m256i special_mask_i = _mm256_or_si256(
		_mm256_or_si256(_mm256_castps_si256(underflow_mask), _mm256_castps_si256(overflow_mask)),
		_mm256_castps_si256(nan_mask));

	if (_mm256_movemask_ps(_mm256_castsi256_ps(special_mask_i)) == 0xFF)
	{
		return exp_special_only(input);
	}

	__m256 result = exp_finalize_special(normal_result, input);

	__m256i active_mask_i = _mm256_andnot_si256(special_mask_i, _mm256_set1_epi32(-1));

	__m256i n128_mask_i = _mm256_and_si256(
		active_mask_i,
		_mm256_cmpeq_epi32(exponent_int, int_128));

	if (!_mm256_testz_si256(n128_mask_i, n128_mask_i))
	{
		__m256 n128_result = _mm256_mul_ps(_mm256_mul_ps(exp_reduced, two), scale_2pow127);
		result = _mm256_blendv_ps(result, n128_result, _mm256_castsi256_ps(n128_mask_i));
	}

	result = exp_finalize_subnormal(result, exp_reduced, exponent_int, active_mask_i);
	return result;
}

__m256 fy::simd::intrinsic::exp(__m256 input) noexcept
{
	const __m256 one = _mm256_set1_ps(1.0f);

	const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
	const __m256 ln2_hi = _mm256_set1_ps(-6.93359375E-1f);
	const __m256 ln2_lo = _mm256_set1_ps(2.12194440E-4f);

	const __m256 overflow_input = _mm256_set1_ps(88.72283935546875f);
	const __m256 underflow_input = _mm256_set1_ps(-103.97208404541015625f);

	const __m256i int_bias_127 = _mm256_set1_epi32(127);
	const __m256i int_neg126 = _mm256_set1_epi32(-126);
	const __m256i int_127 = _mm256_set1_epi32(127);

	__m256 clamped_input = _mm256_max_ps(input, underflow_input);
	clamped_input = _mm256_min_ps(clamped_input, overflow_input);

	__m256 special_input_mask = _mm256_cmp_ps(input, clamped_input, _CMP_NEQ_UQ);

	if (_mm256_movemask_ps(special_input_mask) == 0xFF)
	{
		return exp_special_only(input);
	}

	__m256 exponent_float = _mm256_round_ps(
		_mm256_mul_ps(clamped_input, log2e),
		_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

	__m256i exponent_int = _mm256_cvttps_epi32(exponent_float);

	__m256 reduced = _mm256_fmadd_ps(exponent_float, ln2_hi, clamped_input);
	reduced = _mm256_fmadd_ps(exponent_float, ln2_lo, reduced);

	__m256 reduced_sq = _mm256_mul_ps(reduced, reduced);
	__m256 poly = exp_build_poly(reduced, reduced_sq);

	__m256 exp_reduced = _mm256_fmadd_ps(poly, reduced_sq, reduced);
	exp_reduced = _mm256_add_ps(exp_reduced, one);

	__m256i normal_exp_bits = _mm256_add_epi32(exponent_int, int_bias_127);
	normal_exp_bits = _mm256_slli_epi32(normal_exp_bits, 23);
	__m256 normal_scale = _mm256_castsi256_ps(normal_exp_bits);
	__m256 normal_result = _mm256_mul_ps(exp_reduced, normal_scale);

	__m256i subnormal_mask_i = _mm256_cmpgt_epi32(int_neg126, exponent_int);
	__m256i large_exp_mask_i = _mm256_cmpgt_epi32(exponent_int, int_127);

	__m256i slow_mask_i = _mm256_or_si256(
		_mm256_castps_si256(special_input_mask),
		_mm256_or_si256(subnormal_mask_i, large_exp_mask_i));

	if (_mm256_testz_si256(slow_mask_i, slow_mask_i))
	{
		return normal_result;
	}

	return exp_slow_path(input, exp_reduced, normal_result, exponent_int);
}