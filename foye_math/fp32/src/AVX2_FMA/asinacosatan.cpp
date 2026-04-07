#include <foye_fastmath.hpp>

static inline __m256 asin_remez_r(__m256 z)
{
    const __m256 pS0 = _mm256_set1_ps(1.6666586697e-01f);
    const __m256 pS1 = _mm256_set1_ps(-4.2743422091e-02f);
    const __m256 pS2 = _mm256_set1_ps(-8.6563630030e-03f);
    const __m256 qS1 = _mm256_set1_ps(-7.0662963390e-01f);

    __m256 p = _mm256_fmadd_ps(pS2, z, pS1);
    p = _mm256_fmadd_ps(p, z, pS0);
    p = _mm256_mul_ps(p, z);

    __m256 q = _mm256_fmadd_ps(qS1, z, _mm256_set1_ps(1.0f));
    return _mm256_div_ps(p, q);
}

static inline __m256 quiet_nan_ps(__m256 x)
{
    const __m256i qnan_quiet_bit = _mm256_set1_epi32(0x00400000);
    return _mm256_castsi256_ps(
        _mm256_or_si256(_mm256_castps_si256(x), qnan_quiet_bit));
}

static inline __m256 asin_approx_no_special(__m256 x) noexcept
{
    const __m256 ONE = _mm256_set1_ps(1.0f);
    const __m256 HALF = _mm256_set1_ps(0.5f);
    const __m256 TWO = _mm256_set1_ps(2.0f);

    const __m256 PIO2_HI = _mm256_set1_ps(1.5707962513e+00f);
    const __m256 PIO2_LO = _mm256_set1_ps(7.5497894159e-08f);

    const __m256 ABS_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    __m256 ax = _mm256_and_ps(x, ABS_MASK);
    __m256 sgn = _mm256_andnot_ps(ABS_MASK, x);

    __m256 mask_small = _mm256_cmp_ps(ax, HALF, _CMP_LE_OQ);

    __m256 z0 = _mm256_mul_ps(ax, ax);
    __m256 r0 = asin_remez_r(z0);
    __m256 asin_small = _mm256_fmadd_ps(ax, r0, ax);

    __m256 t = _mm256_mul_ps(HALF, _mm256_sub_ps(ONE, ax));
    __m256 s = _mm256_sqrt_ps(t);
    __m256 rs = asin_remez_r(t);
    __m256 asin_large = _mm256_sub_ps(
        PIO2_HI,
        _mm256_sub_ps(_mm256_mul_ps(TWO, _mm256_fmadd_ps(s, rs, s)), PIO2_LO));

    __m256 asin_abs = _mm256_blendv_ps(asin_large, asin_small, mask_small);
    return _mm256_xor_ps(asin_abs, sgn);
}

static inline __m256 asin_core(__m256 x) noexcept
{
    const __m256 ONE = _mm256_set1_ps(1.0f);
    const __m256 HALF_PI = _mm256_set1_ps(1.5707963705e+00f);
    const __m256 NEG_ONE = _mm256_set1_ps(-1.0f);

    const __m256 ABS_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 QNAN = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FC00000));

    __m256 ax = _mm256_and_ps(x, ABS_MASK);

    __m256 mask_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    __m256 mask_oob = _mm256_cmp_ps(ax, ONE, _CMP_GT_OQ);

    __m256 axc = _mm256_min_ps(ax, ONE);
    __m256 xc = _mm256_or_ps(axc, _mm256_andnot_ps(ABS_MASK, x));

    __m256 asin_val = asin_approx_no_special(xc);

    __m256 mask_eq_p1 = _mm256_cmp_ps(x, ONE, _CMP_EQ_OQ);
    __m256 mask_eq_n1 = _mm256_cmp_ps(x, NEG_ONE, _CMP_EQ_OQ);

    asin_val = _mm256_blendv_ps(asin_val, HALF_PI, mask_eq_p1);
    asin_val = _mm256_blendv_ps(
        asin_val,
        _mm256_sub_ps(_mm256_setzero_ps(), HALF_PI),
        mask_eq_n1);

    asin_val = _mm256_blendv_ps(asin_val, QNAN, mask_oob);
    asin_val = _mm256_blendv_ps(asin_val, quiet_nan_ps(x), mask_nan);

    return asin_val;
}

static inline __m256 acos_core(__m256 x) noexcept
{
    const __m256 ONE = _mm256_set1_ps(1.0f);
    const __m256 HALF = _mm256_set1_ps(0.5f);
    const __m256 TWO = _mm256_set1_ps(2.0f);
    const __m256 NEG_ONE = _mm256_set1_ps(-1.0f);

    const __m256 PIO2_HI = _mm256_set1_ps(1.5707962513e+00f);
    const __m256 PIO2_LO = _mm256_set1_ps(7.5497894159e-08f);
    const __m256 PI_NEAR = _mm256_set1_ps(3.1415927410e+00f);

    const __m256 ABS_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 QNAN = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FC00000));

    __m256 ax = _mm256_and_ps(x, ABS_MASK);

    __m256 mask_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    __m256 mask_oob = _mm256_cmp_ps(ax, ONE, _CMP_GT_OQ);

    __m256 xc = _mm256_max_ps(_mm256_min_ps(x, ONE), NEG_ONE);

    __m256 mask_pos = _mm256_cmp_ps(xc, HALF, _CMP_GT_OQ);

    __m256 asin_mid = asin_approx_no_special(xc);
    __m256 acos_mid = _mm256_add_ps(_mm256_sub_ps(PIO2_HI, asin_mid), PIO2_LO);

    __m256 z = _mm256_mul_ps(HALF, _mm256_sub_ps(ONE, xc));
    __m256 s = _mm256_sqrt_ps(z);
    __m256 rz = asin_remez_r(z);
    __m256 apos = _mm256_mul_ps(TWO, _mm256_fmadd_ps(s, rz, s));

    __m256 acos_val = _mm256_blendv_ps(acos_mid, apos, mask_pos);

    __m256 mask_eq_p1 = _mm256_cmp_ps(x, ONE, _CMP_EQ_OQ);
    __m256 mask_eq_n1 = _mm256_cmp_ps(x, NEG_ONE, _CMP_EQ_OQ);

    acos_val = _mm256_blendv_ps(acos_val, _mm256_setzero_ps(), mask_eq_p1);
    acos_val = _mm256_blendv_ps(acos_val, PI_NEAR, mask_eq_n1);

    acos_val = _mm256_blendv_ps(acos_val, QNAN, mask_oob);
    acos_val = _mm256_blendv_ps(acos_val, quiet_nan_ps(x), mask_nan);

    return acos_val;
}

void fy::simd::intrinsic::asinacos(__m256 input, __m256* asin_result, __m256* acos_result) noexcept
{
    if (!asin_result && !acos_result)
    {
        return;
    }

    if (asin_result && !acos_result)
    {
        *asin_result = asin_core(input);
        return;
    }

    if (!asin_result && acos_result)
    {
        *acos_result = acos_core(input);
        return;
    }

    const __m256 ONE = _mm256_set1_ps(1.0f);
    const __m256 HALF = _mm256_set1_ps(0.5f);
    const __m256 TWO = _mm256_set1_ps(2.0f);
    const __m256 NEG_ONE = _mm256_set1_ps(-1.0f);

    const __m256 PIO2_HI = _mm256_set1_ps(1.5707962513e+00f);
    const __m256 PIO2_LO = _mm256_set1_ps(7.5497894159e-08f);
    const __m256 PI_NEAR = _mm256_set1_ps(3.1415927410e+00f);
    const __m256 HALF_PI = _mm256_set1_ps(1.5707963705e+00f);

    const __m256 ABS_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 QNAN = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FC00000));

    __m256 x = input;
    __m256 ax = _mm256_and_ps(x, ABS_MASK);
    __m256 sgn = _mm256_andnot_ps(ABS_MASK, x);

    __m256 mask_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    __m256 mask_oob = _mm256_cmp_ps(ax, ONE, _CMP_GT_OQ);
    __m256 axc = _mm256_min_ps(ax, ONE);
    __m256 mask_small = _mm256_cmp_ps(axc, HALF, _CMP_LE_OQ);

    __m256 z0 = _mm256_mul_ps(axc, axc);
    __m256 r0 = asin_remez_r(z0);
    __m256 asin_small = _mm256_fmadd_ps(axc, r0, axc);

    __m256 t = _mm256_mul_ps(HALF, _mm256_sub_ps(ONE, axc));
    __m256 s = _mm256_sqrt_ps(t);
    __m256 rs = asin_remez_r(t);
    __m256 asin_large = _mm256_sub_ps(
        PIO2_HI,
        _mm256_sub_ps(_mm256_mul_ps(TWO, _mm256_fmadd_ps(s, rs, s)), PIO2_LO));

    __m256 asin_abs = _mm256_blendv_ps(asin_large, asin_small, mask_small);
    __m256 asin_val = _mm256_xor_ps(asin_abs, sgn);

    __m256 mask_eq_p1 = _mm256_cmp_ps(x, ONE, _CMP_EQ_OQ);
    __m256 mask_eq_n1 = _mm256_cmp_ps(x, NEG_ONE, _CMP_EQ_OQ);

    __m256 asin_out = asin_val;
    asin_out = _mm256_blendv_ps(asin_out, HALF_PI, mask_eq_p1);
    asin_out = _mm256_blendv_ps(asin_out, _mm256_sub_ps(_mm256_setzero_ps(), HALF_PI), mask_eq_n1);
    asin_out = _mm256_blendv_ps(asin_out, QNAN, mask_oob);
    asin_out = _mm256_blendv_ps(asin_out, quiet_nan_ps(x), mask_nan);

    *asin_result = asin_out;

    __m256 xc = _mm256_max_ps(_mm256_min_ps(x, ONE), NEG_ONE);
    __m256 mask_pos = _mm256_cmp_ps(xc, HALF, _CMP_GT_OQ);

    __m256 acos_mid = _mm256_add_ps(_mm256_sub_ps(PIO2_HI, asin_val), PIO2_LO);

    __m256 zpos = _mm256_mul_ps(HALF, _mm256_sub_ps(ONE, xc));
    __m256 spos = _mm256_sqrt_ps(zpos);
    __m256 rpos = asin_remez_r(zpos);
    __m256 apos = _mm256_mul_ps(TWO, _mm256_fmadd_ps(spos, rpos, spos));

    __m256 acos_out = _mm256_blendv_ps(acos_mid, apos, mask_pos);
    acos_out = _mm256_blendv_ps(acos_out, _mm256_setzero_ps(), mask_eq_p1);
    acos_out = _mm256_blendv_ps(acos_out, PI_NEAR, mask_eq_n1);
    acos_out = _mm256_blendv_ps(acos_out, QNAN, mask_oob);
    acos_out = _mm256_blendv_ps(acos_out, quiet_nan_ps(x), mask_nan);

    *acos_result = acos_out;
}

__m256 fy::simd::intrinsic::asin(__m256 input) noexcept
{
    return asin_core(input);
}

__m256 fy::simd::intrinsic::acos(__m256 input) noexcept
{
    return acos_core(input);
}

static inline __m256 atan_poly_eval(__m256 z) noexcept
{
    const __m256 aT0 = _mm256_set1_ps(3.3333334327e-01f);
    const __m256 aT1 = _mm256_set1_ps(-2.0000000298e-01f);
    const __m256 aT2 = _mm256_set1_ps(1.4285714924e-01f);
    const __m256 aT3 = _mm256_set1_ps(-1.1111110449e-01f);
    const __m256 aT4 = _mm256_set1_ps(9.0908870101e-02f);
    const __m256 aT5 = _mm256_set1_ps(-7.6918758452e-02f);
    const __m256 aT6 = _mm256_set1_ps(6.6610731184e-02f);
    const __m256 aT7 = _mm256_set1_ps(-5.8335702866e-02f);
    const __m256 aT8 = _mm256_set1_ps(4.9768779427e-02f);
    const __m256 aT9 = _mm256_set1_ps(-3.6531571299e-02f);
    const __m256 aT10 = _mm256_set1_ps(1.6285819933e-02f);

    __m256 zz = _mm256_mul_ps(z, z);
    __m256 w = _mm256_mul_ps(zz, zz);

    __m256 s1 = aT10;
    s1 = _mm256_fmadd_ps(s1, w, aT8);
    s1 = _mm256_fmadd_ps(s1, w, aT6);
    s1 = _mm256_fmadd_ps(s1, w, aT4);
    s1 = _mm256_fmadd_ps(s1, w, aT2);
    s1 = _mm256_fmadd_ps(s1, w, aT0);
    s1 = _mm256_mul_ps(s1, zz);

    __m256 s2 = aT9;
    s2 = _mm256_fmadd_ps(s2, w, aT7);
    s2 = _mm256_fmadd_ps(s2, w, aT5);
    s2 = _mm256_fmadd_ps(s2, w, aT3);
    s2 = _mm256_fmadd_ps(s2, w, aT1);
    s2 = _mm256_mul_ps(s2, w);

    return _mm256_mul_ps(z, _mm256_add_ps(s1, s2));
}

__m256 fy::simd::intrinsic::atan(__m256 input) noexcept
{
    const __m256 ABS_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 ZERO = _mm256_setzero_ps();
    const __m256 ONE = _mm256_set1_ps(1.0f);
    const __m256 TWO = _mm256_set1_ps(2.0f);
    const __m256 THREE_HALVES = _mm256_set1_ps(1.5f);
    const __m256 TINY_BOUND = _mm256_set1_ps(0x1p-12f);

    const __m256 C0 = _mm256_set1_ps(0.4375f);
    const __m256 C1 = _mm256_set1_ps(0.6875f);
    const __m256 C2 = _mm256_set1_ps(1.1875f);
    const __m256 C3 = _mm256_set1_ps(2.4375f);

    const __m256 ATANHI0 = _mm256_set1_ps(4.6364760399e-01f);
    const __m256 ATANHI1 = _mm256_set1_ps(7.8539812565e-01f);
    const __m256 ATANHI2 = _mm256_set1_ps(9.8279368877e-01f);
    const __m256 ATANHI3 = _mm256_set1_ps(1.5707962513e+00f);

    const __m256 ATANLO0 = _mm256_set1_ps(5.0121582440e-09f);
    const __m256 ATANLO1 = _mm256_set1_ps(3.7748947079e-08f);
    const __m256 ATANLO2 = _mm256_set1_ps(3.4473217170e-08f);
    const __m256 ATANLO3 = _mm256_set1_ps(7.5497894159e-08f);

    __m256 x = input;
    __m256 ax = _mm256_and_ps(x, ABS_MASK);
    __m256 sign = _mm256_andnot_ps(ABS_MASK, x);

    __m256 mask_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    __m256 m_tiny = _mm256_cmp_ps(ax, TINY_BOUND, _CMP_LT_OQ);

    __m256 m_small = _mm256_cmp_ps(ax, C0, _CMP_LT_OQ);
    __m256 m_lt1 = _mm256_cmp_ps(ax, C1, _CMP_LT_OQ);
    __m256 m_lt2 = _mm256_cmp_ps(ax, C2, _CMP_LT_OQ);
    __m256 m_lt3 = _mm256_cmp_ps(ax, C3, _CMP_LT_OQ);

    __m256 m0 = _mm256_andnot_ps(m_small, m_lt1);
    __m256 m1 = _mm256_andnot_ps(m_lt1, m_lt2);
    __m256 m2 = _mm256_andnot_ps(m_lt2, m_lt3);
    __m256 m3 = _mm256_andnot_ps(m_lt3, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));

    auto finish_small = [&](const __m256 z_abs) -> __m256
    {
        __m256 poly = atan_poly_eval(z_abs);
        __m256 res_abs = _mm256_sub_ps(z_abs, poly);
        res_abs = _mm256_blendv_ps(res_abs, ax, m_tiny);

        __m256 res = _mm256_xor_ps(res_abs, sign);
        __m256 nanv = _mm256_add_ps(x, x);
        res = _mm256_blendv_ps(res, nanv, mask_nan);
        return res;
    };

    auto finish_large = [&](const __m256 z, const __m256 hi, const __m256 lo) -> __m256
    {
        __m256 poly = atan_poly_eval(z);
        __m256 res_abs = _mm256_sub_ps(hi, _mm256_sub_ps(_mm256_sub_ps(poly, lo), z));
        res_abs = _mm256_blendv_ps(res_abs, ax, m_tiny);

        __m256 res = _mm256_xor_ps(res_abs, sign);
        __m256 nanv = _mm256_add_ps(x, x);
        res = _mm256_blendv_ps(res, nanv, mask_nan);
        return res;
    };

    {
        if (_mm256_movemask_ps(m_small) == 0xFF)
        {
            return finish_small(ax);
        }

        if (_mm256_movemask_ps(m0) == 0xFF)
        {
            __m256 z = _mm256_div_ps(_mm256_sub_ps(_mm256_add_ps(ax, ax), ONE), _mm256_add_ps(TWO, ax));
            return finish_large(z, ATANHI0, ATANLO0);
        }

        if (_mm256_movemask_ps(m1) == 0xFF)
        {
            __m256 z = _mm256_div_ps(_mm256_sub_ps(ax, ONE), _mm256_add_ps(ax, ONE));
            return finish_large(z, ATANHI1, ATANLO1);
        }

        if (_mm256_movemask_ps(m2) == 0xFF)
        {
            __m256 z = _mm256_div_ps(_mm256_sub_ps(ax, THREE_HALVES), _mm256_fmadd_ps(THREE_HALVES, ax, ONE));
            return finish_large(z, ATANHI2, ATANLO2);
        }

        if (_mm256_movemask_ps(m3) == 0xFF)
        {
            __m256 z = _mm256_sub_ps(ZERO, _mm256_div_ps(ONE, ax));
            return finish_large(z, ATANHI3, ATANLO3);
        }
    }

    __m256 m01 = _mm256_andnot_ps(m_small, m_lt2);

    __m256 z_small = ax;
    __m256 z01 = _mm256_div_ps(_mm256_sub_ps(ax, ONE), _mm256_add_ps(ax, ONE));
    __m256 z2v = _mm256_div_ps(
        _mm256_sub_ps(ax, THREE_HALVES),
        _mm256_fmadd_ps(THREE_HALVES, ax, ONE));
    __m256 z3v = _mm256_sub_ps(ZERO, _mm256_div_ps(ONE, ax));

    __m256 z = z3v;
    z = _mm256_blendv_ps(z, z2v, m2);
    z = _mm256_blendv_ps(z, z01, m01);
    z = _mm256_blendv_ps(z, z_small, m_small);

    __m256 hi = ATANHI3;
    __m256 lo = ATANLO3;
    hi = _mm256_blendv_ps(hi, ATANHI2, m2);
    lo = _mm256_blendv_ps(lo, ATANLO2, m2);
    hi = _mm256_blendv_ps(hi, ATANHI1, m01);
    lo = _mm256_blendv_ps(lo, ATANLO1, m01);

    __m256 poly = atan_poly_eval(z);

    __m256 res_abs_small = _mm256_sub_ps(z, poly);
    __m256 res_abs_large = _mm256_sub_ps(hi, _mm256_sub_ps(_mm256_sub_ps(poly, lo), z));

    __m256 res_abs = _mm256_blendv_ps(res_abs_large, res_abs_small, m_small);
    res_abs = _mm256_blendv_ps(res_abs, ax, m_tiny);

    __m256 res = _mm256_xor_ps(res_abs, sign);
    __m256 nanv = _mm256_add_ps(x, x);
    res = _mm256_blendv_ps(res, nanv, mask_nan);

    return res;
}