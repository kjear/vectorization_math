#include "foye_fastmath_fp32.hpp"

__m256 fy::simd::intrinsic::logb(__m256 input) noexcept
{
	__m256i bits = _mm256_castps_si256(input);
	__m256i abits = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff));

	__m256i zero_mask = _mm256_cmpeq_epi32(abits, _mm256_setzero_si256());

	__m256i inf_bits = _mm256_set1_epi32(0x7f800000);
	__m256i gt_inf = _mm256_cmpgt_epi32(abits, inf_bits);
	__m256i eq_inf = _mm256_cmpeq_epi32(abits, inf_bits);

	__m256i sub_thresh = _mm256_set1_epi32(0x00800000);
	__m256i sub_mask = _mm256_andnot_si256(zero_mask, _mm256_cmpgt_epi32(sub_thresh, abits));

	__m256i e_norm = _mm256_sub_epi32(_mm256_srli_epi32(abits, 23), _mm256_set1_epi32(127));
	__m256 r_norm = _mm256_cvtepi32_ps(e_norm);

	__m256 scaled = _mm256_mul_ps(input, _mm256_set1_ps(3.355443200e+07f));
	__m256i sbits = _mm256_and_si256(_mm256_castps_si256(scaled), _mm256_set1_epi32(0x7fffffff));
	__m256i e_sub = _mm256_sub_epi32(_mm256_srli_epi32(sbits, 23), _mm256_set1_epi32(152));
	__m256 r_sub = _mm256_cvtepi32_ps(e_sub);

	__m256 result = _mm256_blendv_ps(r_norm, r_sub, _mm256_castsi256_ps(sub_mask));

	const __m256 neg_inf = _mm256_castsi256_ps(_mm256_set1_epi32(0xff800000));
	const __m256 pos_inf = _mm256_castsi256_ps(_mm256_set1_epi32(0x7f800000));
	const __m256i qnan_quiet_bit = _mm256_set1_epi32(0x00400000);

	__m256 nan_quieted = _mm256_castsi256_ps(
		_mm256_or_si256(abits, qnan_quiet_bit)
	);

	result = _mm256_blendv_ps(result, neg_inf, _mm256_castsi256_ps(zero_mask));
	result = _mm256_blendv_ps(result, pos_inf, _mm256_castsi256_ps(eq_inf));
	result = _mm256_blendv_ps(result, nan_quieted, _mm256_castsi256_ps(gt_inf));

	return result;
}