#include "foye_fastmath_fp32.hpp"

__m256 fy::simd::intrinsic::hypot(__m256 x, __m256 y) noexcept
{
	const __m256 sign = _mm256_set1_ps(-0.0f);
	const __m256 inf = _mm256_castsi256_ps(_mm256_set1_epi32(0x7f800000));

	__m256 ax = _mm256_andnot_ps(sign, x);
	__m256 ay = _mm256_andnot_ps(sign, y);

	__m128 axl = _mm256_castps256_ps128(ax);
	__m128 axh = _mm256_extractf128_ps(ax, 1);
	__m128 ayl = _mm256_castps256_ps128(ay);
	__m128 ayh = _mm256_extractf128_ps(ay, 1);

	__m256d dxl = _mm256_cvtps_pd(axl);
	__m256d dxh = _mm256_cvtps_pd(axh);
	__m256d dyl = _mm256_cvtps_pd(ayl);
	__m256d dyh = _mm256_cvtps_pd(ayh);

	__m256d sl = _mm256_fmadd_pd(dxl, dxl, _mm256_mul_pd(dyl, dyl));
	__m256d sh = _mm256_fmadd_pd(dxh, dxh, _mm256_mul_pd(dyh, dyh));

	__m256d rl = _mm256_sqrt_pd(sl);
	__m256d rh = _mm256_sqrt_pd(sh);

	__m128 resl = _mm256_cvtpd_ps(rl);
	__m128 resh = _mm256_cvtpd_ps(rh);

	__m256 res = _mm256_castps128_ps256(resl);
	res = _mm256_insertf128_ps(res, resh, 1);

	__m256 ax_is_inf = _mm256_cmp_ps(ax, inf, _CMP_EQ_OQ);
	__m256 ay_is_inf = _mm256_cmp_ps(ay, inf, _CMP_EQ_OQ);
	__m256 any_inf = _mm256_or_ps(ax_is_inf, ay_is_inf);

	__m256 x_is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
	__m256 y_is_nan = _mm256_cmp_ps(y, y, _CMP_UNORD_Q);
	__m256 any_nan = _mm256_or_ps(x_is_nan, y_is_nan);

	__m256 nanv = _mm256_add_ps(x, y);
	res = _mm256_blendv_ps(res, nanv, any_nan);
	res = _mm256_blendv_ps(res, inf, any_inf);

	return res;
}