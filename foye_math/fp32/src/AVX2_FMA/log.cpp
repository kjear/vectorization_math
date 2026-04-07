#include <foye_fastmath.hpp>

static inline __m256 log_near1_branch(__m256 t) noexcept
{
	const __m256 c2 = _mm256_set1_ps(-1.0f / 2.0f);
	const __m256 c3 = _mm256_set1_ps(1.0f / 3.0f);
	const __m256 c4 = _mm256_set1_ps(-1.0f / 4.0f);
	const __m256 c5 = _mm256_set1_ps(1.0f / 5.0f);
	const __m256 c6 = _mm256_set1_ps(-1.0f / 6.0f);
	const __m256 c7 = _mm256_set1_ps(1.0f / 7.0f);
	const __m256 c8 = _mm256_set1_ps(-1.0f / 8.0f);
	const __m256 c9 = _mm256_set1_ps(1.0f / 9.0f);
	const __m256 c10 = _mm256_set1_ps(-1.0f / 10.0f);
	const __m256 c11 = _mm256_set1_ps(1.0f / 11.0f);
	const __m256 c12 = _mm256_set1_ps(-1.0f / 12.0f);

	__m256 x2 = _mm256_mul_ps(t, t);

	__m256 p = c12;
	p = _mm256_fmadd_ps(p, t, c11);
	p = _mm256_fmadd_ps(p, t, c10);
	p = _mm256_fmadd_ps(p, t, c9);
	p = _mm256_fmadd_ps(p, t, c8);
	p = _mm256_fmadd_ps(p, t, c7);
	p = _mm256_fmadd_ps(p, t, c6);
	p = _mm256_fmadd_ps(p, t, c5);
	p = _mm256_fmadd_ps(p, t, c4);
	p = _mm256_fmadd_ps(p, t, c3);
	p = _mm256_fmadd_ps(p, t, c2);

	return _mm256_fmadd_ps(x2, p, t);
}

static inline __m256 log_core_normal(__m256 x) noexcept
{
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 half = _mm256_set1_ps(0.5f);
	const __m256 sqrt_half = _mm256_set1_ps(0.707106781186547524f);

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

	ix = _mm256_and_si256(ix, _mm256_set1_epi32(0x007FFFFF));
	ix = _mm256_or_si256(ix, _mm256_set1_epi32(0x3F000000));
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

static inline __m256 log_core_full(__m256 x, __m256 mask_sub) noexcept
{
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 half = _mm256_set1_ps(0.5f);
	const __m256 min_norm = _mm256_castsi256_ps(_mm256_set1_epi32(0x00800000));
	const __m256 sqrt_half = _mm256_set1_ps(0.707106781186547524f);
	const __m256 two23 = _mm256_set1_ps(8388608.0f);

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

	__m256 x_scaled = _mm256_mul_ps(x, two23);
	x = _mm256_blendv_ps(x, x_scaled, mask_sub);
	x = _mm256_max_ps(x, min_norm);

	__m256i ix = _mm256_castps_si256(x);
	__m256i expi = _mm256_srli_epi32(ix, 23);

	ix = _mm256_and_si256(ix, _mm256_set1_epi32(0x007FFFFF));
	ix = _mm256_or_si256(ix, _mm256_set1_epi32(0x3F000000));
	__m256 m = _mm256_castsi256_ps(ix);

	__m256 e = _mm256_cvtepi32_ps(_mm256_sub_epi32(expi, _mm256_set1_epi32(127)));
	e = _mm256_add_ps(e, one);
	e = _mm256_add_ps(e, _mm256_and_ps(mask_sub, _mm256_set1_ps(-23.0f)));

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

static inline __m256 log_finalize_special(__m256 result, __m256 x0) noexcept
{
	const __m256 zero = _mm256_setzero_ps();
	const __m256 qnan = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FC00000));
	const __m256 ninf = _mm256_castsi256_ps(_mm256_set1_epi32(0xFF800000));
	const __m256 pinf = _mm256_castsi256_ps(_mm256_set1_epi32(0x7F800000));

	__m256 mask_nan = _mm256_cmp_ps(x0, x0, _CMP_UNORD_Q);
	__m256 mask_zero = _mm256_cmp_ps(x0, zero, _CMP_EQ_OQ);
	__m256 mask_neg = _mm256_cmp_ps(x0, zero, _CMP_LT_OQ);
	__m256 mask_inf = _mm256_cmp_ps(x0, pinf, _CMP_EQ_OQ);

	const __m256i qnan_quiet_i = _mm256_set1_epi32(0x00400000);
	__m256 nan_quieted = _mm256_castsi256_ps(
		_mm256_or_si256(_mm256_castps_si256(x0), qnan_quiet_i)
	);

	result = _mm256_blendv_ps(result, ninf, mask_zero);
	result = _mm256_blendv_ps(result, qnan, mask_neg);
	result = _mm256_blendv_ps(result, pinf, mask_inf);
	result = _mm256_blendv_ps(result, nan_quieted, mask_nan);

	return result;
}

static inline bool all_positive_finite_normal(__m256 x) noexcept
{
	const __m256i sign_mask = _mm256_set1_epi32(0x80000000);
	const __m256i exp_mask = _mm256_set1_epi32(0x7F800000);
	const __m256i zero = _mm256_setzero_si256();
	const __m256i inf_exp = _mm256_set1_epi32(0x7F800000);

	__m256i ix = _mm256_castps_si256(x);
	__m256i sign = _mm256_and_si256(ix, sign_mask);
	__m256i exp = _mm256_and_si256(ix, exp_mask);

	__m256i sign_ok = _mm256_cmpeq_epi32(sign, zero);
	__m256i exp_nz = _mm256_cmpgt_epi32(exp, zero);
	__m256i exp_ok = _mm256_cmpgt_epi32(inf_exp, exp);

	__m256i ok = _mm256_and_si256(sign_ok, _mm256_and_si256(exp_nz, exp_ok));
	return _mm256_movemask_ps(_mm256_castsi256_ps(ok)) == 0xFF;
}

__m256 fy::simd::intrinsic::log(__m256 x) noexcept
{
	const __m256 zero = _mm256_setzero_ps();
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 min_norm = _mm256_castsi256_ps(_mm256_set1_epi32(0x00800000));
	const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

	__m256 x0 = x;

	if (all_positive_finite_normal(x))
	{
		__m256 t = _mm256_sub_ps(x, one);
		__m256 at = _mm256_and_ps(t, abs_mask);
		__m256 mask_near1 = _mm256_cmp_ps(at, _mm256_set1_ps(0.25f), _CMP_LT_OQ);

		if (_mm256_movemask_ps(mask_near1) == 0xFF)
		{
			return log_near1_branch(t);
		}

		return log_core_normal(x);
	}

	__m256 t = _mm256_sub_ps(x0, one);
	__m256 at = _mm256_and_ps(t, abs_mask);
	__m256 mask_near1 = _mm256_and_ps(
		_mm256_cmp_ps(at, _mm256_set1_ps(0.25f), _CMP_LT_OQ),
		_mm256_cmp_ps(x0, zero, _CMP_GT_OQ)
	);

	__m256 y_near1 = log_near1_branch(t);

	__m256 mask_sub = _mm256_and_ps(
		_mm256_cmp_ps(x, min_norm, _CMP_LT_OQ),
		_mm256_cmp_ps(x, zero, _CMP_GT_OQ)
	);

	__m256 y = log_core_full(x, mask_sub);
	y = _mm256_blendv_ps(y, y_near1, mask_near1);

	return log_finalize_special(y, x0);
}