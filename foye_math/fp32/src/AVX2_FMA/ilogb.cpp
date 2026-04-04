#include "foye_fastmath_fp32.hpp"

__m256i fy::simd::intrinsic::ilogb(__m256 x) noexcept
{
	const __m256i signless_mask = _mm256_set1_epi32(0x7fffffff);
	const __m256i exp_mask = _mm256_set1_epi32(0x7f800000);
	const __m256i mant_mask = _mm256_set1_epi32(0x007fffff);

	const __m256i int_min_v = _mm256_set1_epi32(std::numeric_limits<std::int32_t>::min());
	const __m256i int_max_v = _mm256_set1_epi32(std::numeric_limits<std::int32_t>::max());

	const __m256i bias_127 = _mm256_set1_epi32(127);
	const __m256i bias_150 = _mm256_set1_epi32(150);
	const __m256i zero_i = _mm256_setzero_si256();

	const __m256  scale_2p23 = _mm256_set1_ps(8388608.0f);

	__m256i ux = _mm256_castps_si256(x);
	__m256i ax = _mm256_and_si256(ux, signless_mask);
	__m256i exp = _mm256_and_si256(ax, exp_mask);
	__m256i man = _mm256_and_si256(ax, mant_mask);

	__m256i is_zero = _mm256_cmpeq_epi32(ax, zero_i);
	__m256i is_exp_zero = _mm256_cmpeq_epi32(exp, zero_i);
	__m256i is_man_zero = _mm256_cmpeq_epi32(man, zero_i);
	__m256i is_exp_all_ones = _mm256_cmpeq_epi32(exp, exp_mask);

	__m256i not_man_zero = _mm256_xor_si256(is_man_zero, _mm256_set1_epi32(-1));
	__m256i is_subnormal = _mm256_and_si256(is_exp_zero, not_man_zero);

	__m256i is_nan = _mm256_castps_si256(_mm256_cmp_ps(x, x, _CMP_UNORD_Q));
	__m256i is_inf = _mm256_andnot_si256(is_nan, is_exp_all_ones);

	__m256i normal_exp = _mm256_srli_epi32(ax, 23);
	normal_exp = _mm256_sub_epi32(normal_exp, bias_127);

	__m256  scaled = _mm256_mul_ps(x, scale_2p23);
	__m256i uscaled = _mm256_castps_si256(scaled);
	__m256i ascaled = _mm256_and_si256(uscaled, signless_mask);
	__m256i sub_exp = _mm256_srli_epi32(ascaled, 23);
	sub_exp = _mm256_sub_epi32(sub_exp, bias_150);

	__m256i result = normal_exp;

	result = _mm256_blendv_epi8(result, sub_exp, is_subnormal);
	result = _mm256_blendv_epi8(result, int_min_v, is_zero);
	result = _mm256_blendv_epi8(result, int_min_v, is_nan);
	result = _mm256_blendv_epi8(result, int_max_v, is_inf);

	return result;
}