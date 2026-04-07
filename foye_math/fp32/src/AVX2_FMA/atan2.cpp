#include <foye_fastmath.hpp>

static inline __m256 atan2_poly_eval(__m256 z) noexcept
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

static inline __m256 atan2_pos_no_nan(__m256 u) noexcept
{
    const __m256 ZERO = _mm256_setzero_ps();
    const __m256 ONE = _mm256_set1_ps(1.0f);
    const __m256 TWO = _mm256_set1_ps(2.0f);
    const __m256 THREE_HALVES = _mm256_set1_ps(1.5f);

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

    __m256 m_small = _mm256_cmp_ps(u, C0, _CMP_LT_OQ);
    __m256 m_lt1 = _mm256_cmp_ps(u, C1, _CMP_LT_OQ);
    __m256 m_lt2 = _mm256_cmp_ps(u, C2, _CMP_LT_OQ);
    __m256 m_lt3 = _mm256_cmp_ps(u, C3, _CMP_LT_OQ);

    __m256 m0 = _mm256_andnot_ps(m_small, m_lt1);
    __m256 m1 = _mm256_andnot_ps(m_lt1, m_lt2);
    __m256 m2 = _mm256_andnot_ps(m_lt2, m_lt3);
    __m256 m3 = _mm256_andnot_ps(m_lt3, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));

    if (_mm256_movemask_ps(m_small) == 0xFF)
    {
        __m256 poly = atan2_poly_eval(u);
        return _mm256_sub_ps(u, poly);
    }
    if (_mm256_movemask_ps(m0) == 0xFF)
    {
        __m256 z = _mm256_div_ps(_mm256_sub_ps(_mm256_add_ps(u, u), ONE), _mm256_add_ps(TWO, u));
        __m256 poly = atan2_poly_eval(z);
        return _mm256_sub_ps(ATANHI0, _mm256_sub_ps(_mm256_sub_ps(poly, ATANLO0), z));
    }
    if (_mm256_movemask_ps(m1) == 0xFF)
    {
        __m256 z = _mm256_div_ps(_mm256_sub_ps(u, ONE), _mm256_add_ps(u, ONE));
        __m256 poly = atan2_poly_eval(z);
        return _mm256_sub_ps(ATANHI1, _mm256_sub_ps(_mm256_sub_ps(poly, ATANLO1), z));
    }
    if (_mm256_movemask_ps(m2) == 0xFF)
    {
        __m256 z = _mm256_div_ps(
            _mm256_sub_ps(u, THREE_HALVES),
            _mm256_fmadd_ps(THREE_HALVES, u, ONE));
        __m256 poly = atan2_poly_eval(z);
        return _mm256_sub_ps(ATANHI2, _mm256_sub_ps(_mm256_sub_ps(poly, ATANLO2), z));
    }
    if (_mm256_movemask_ps(m3) == 0xFF)
    {
        __m256 z = _mm256_sub_ps(ZERO, _mm256_div_ps(ONE, u));
        __m256 poly = atan2_poly_eval(z);
        return _mm256_sub_ps(ATANHI3, _mm256_sub_ps(_mm256_sub_ps(poly, ATANLO3), z));
    }

    __m256 z_small = u;
    __m256 z0 = _mm256_div_ps(_mm256_sub_ps(_mm256_add_ps(u, u), ONE), _mm256_add_ps(TWO, u));
    __m256 z1 = _mm256_div_ps(_mm256_sub_ps(u, ONE), _mm256_add_ps(u, ONE));
    __m256 z2 = _mm256_div_ps(
        _mm256_sub_ps(u, THREE_HALVES),
        _mm256_fmadd_ps(THREE_HALVES, u, ONE));
    __m256 z3 = _mm256_sub_ps(ZERO, _mm256_div_ps(ONE, u));

    __m256 z = z3;
    z = _mm256_blendv_ps(z, z2, m2);
    z = _mm256_blendv_ps(z, z1, m1);
    z = _mm256_blendv_ps(z, z0, m0);
    z = _mm256_blendv_ps(z, z_small, m_small);

    __m256 hi = ATANHI3;
    __m256 lo = ATANLO3;
    hi = _mm256_blendv_ps(hi, ATANHI2, m2);
    lo = _mm256_blendv_ps(lo, ATANLO2, m2);
    hi = _mm256_blendv_ps(hi, ATANHI1, m1);
    lo = _mm256_blendv_ps(lo, ATANLO1, m1);
    hi = _mm256_blendv_ps(hi, ATANHI0, m0);
    lo = _mm256_blendv_ps(lo, ATANLO0, m0);

    __m256 poly = atan2_poly_eval(z);
    __m256 res_small = _mm256_sub_ps(z, poly);
    __m256 res_large = _mm256_sub_ps(hi, _mm256_sub_ps(_mm256_sub_ps(poly, lo), z));
    return _mm256_blendv_ps(res_large, res_small, m_small);
}

static inline __m256 atan2_normal_from_abs_ratio(
    __m256 abs_y,
    __m256 abs_x,
    __m256 y,
    __m256 pi_ps,
    __m256 pi_lo_ps,
    __m256 pio2_half_pilo_ps,
    __m256 zero_ps,
    __m256 sign_mask_ps,
    __m256i x_neg_i,
    __m256i y_neg_i,
    __m256i ix,
    __m256i iy,
    __m256i c26_i,
    __m256i cneg26_i) noexcept
{
    __m256 z = atan2_pos_no_nan(_mm256_div_ps(abs_y, abs_x));

    const __m256i k_i = _mm256_srai_epi32(_mm256_sub_epi32(iy, ix), 23);
    const __m256i k_gt_26_i = _mm256_cmpgt_epi32(k_i, c26_i);
    const __m256i k_lt_neg26_i = _mm256_cmpgt_epi32(cneg26_i, k_i);

    const __m256i tiny_ratio_negx_i = _mm256_and_si256(k_lt_neg26_i, x_neg_i);
    const __m256i x_neg_eff_i = _mm256_andnot_si256(k_gt_26_i, x_neg_i);

    z = _mm256_blendv_ps(z, pio2_half_pilo_ps, _mm256_castsi256_ps(k_gt_26_i));
    z = _mm256_blendv_ps(z, zero_ps, _mm256_castsi256_ps(tiny_ratio_negx_i));

    const __m256 y_sign_ps = _mm256_and_ps(y, sign_mask_ps);
    const __m256 res_pos = _mm256_xor_ps(z, y_sign_ps);

    const __m256 t = _mm256_sub_ps(z, pi_lo_ps);
    const __m256 res_neg_posy = _mm256_sub_ps(pi_ps, t);
    const __m256 res_neg_negy = _mm256_sub_ps(t, pi_ps);
    const __m256 res_neg = _mm256_blendv_ps(res_neg_posy, res_neg_negy, _mm256_castsi256_ps(y_neg_i));

    return _mm256_blendv_ps(res_pos, res_neg, _mm256_castsi256_ps(x_neg_eff_i));
}

__m256 fy::simd::intrinsic::atan2(__m256 y, __m256 x) noexcept
{
    const __m256i sign_mask_i = _mm256_set1_epi32(0x80000000u);
    const __m256i abs_mask_i = _mm256_set1_epi32(0x7fffffff);
    const __m256i inf_i = _mm256_set1_epi32(0x7f800000);
    const __m256i one_i = _mm256_set1_epi32(0x3f800000);
    const __m256i zero_i = _mm256_setzero_si256();
    const __m256i all1_i = _mm256_set1_epi32(-1);
    const __m256i c26_i = _mm256_set1_epi32(26);
    const __m256i cneg26_i = _mm256_set1_epi32(-26);

    const __m256 zero_ps = _mm256_setzero_ps();
    const __m256 one_ps = _mm256_castsi256_ps(one_i);
    const __m256 sign_mask_ps = _mm256_castsi256_ps(sign_mask_i);
    const __m256 abs_mask_ps = _mm256_castsi256_ps(abs_mask_i);

    const __m256 pi_o_4_ps = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f490fdb));
    const __m256 pi_o_2_ps = _mm256_castsi256_ps(_mm256_set1_epi32(0x3fc90fdb));
    const __m256 pi_ps = _mm256_castsi256_ps(_mm256_set1_epi32(0x40490fdb));
    const __m256 pi_lo_ps = _mm256_castsi256_ps(_mm256_set1_epi32(0xb3bbbd2e));
    const __m256 tiny_ps = _mm256_set1_ps(1.0e-30f);

    const __m256 pio2_half_pilo_ps = _mm256_fmadd_ps(_mm256_set1_ps(0.5f), pi_lo_ps, pi_o_2_ps);
    const __m256 pi_tiny_ps = _mm256_add_ps(pi_ps, tiny_ps);
    const __m256 mpi_tiny_ps = _mm256_sub_ps(_mm256_sub_ps(zero_ps, pi_ps), tiny_ps);
    const __m256 pio2_tiny_ps = _mm256_add_ps(pi_o_2_ps, tiny_ps);
    const __m256 mpio2_tiny_ps = _mm256_sub_ps(_mm256_sub_ps(zero_ps, pi_o_2_ps), tiny_ps);
    const __m256 pio4_tiny_ps = _mm256_add_ps(pi_o_4_ps, tiny_ps);
    const __m256 mpio4_tiny_ps = _mm256_sub_ps(_mm256_sub_ps(zero_ps, pi_o_4_ps), tiny_ps);
    const __m256 three_pio4_ps = _mm256_add_ps(pi_o_2_ps, pi_o_4_ps);
    const __m256 three_pio4_tiny_ps = _mm256_add_ps(three_pio4_ps, tiny_ps);
    const __m256 mthree_pio4_tiny_ps = _mm256_sub_ps(_mm256_sub_ps(zero_ps, three_pio4_ps), tiny_ps);

    const __m256i hx = _mm256_castps_si256(x);
    const __m256i hy = _mm256_castps_si256(y);
    const __m256i ix = _mm256_and_si256(hx, abs_mask_i);
    const __m256i iy = _mm256_and_si256(hy, abs_mask_i);

    const __m256i x_nan_i = _mm256_cmpgt_epi32(ix, inf_i);
    const __m256i y_nan_i = _mm256_cmpgt_epi32(iy, inf_i);
    const __m256i y_zero_i = _mm256_cmpeq_epi32(iy, zero_i);
    const __m256i x_zero_i = _mm256_cmpeq_epi32(ix, zero_i);
    const __m256i x_inf_i = _mm256_cmpeq_epi32(ix, inf_i);
    const __m256i y_inf_i = _mm256_cmpeq_epi32(iy, inf_i);

    const __m256i special_i = _mm256_or_si256(
        _mm256_or_si256(
            _mm256_or_si256(x_nan_i, y_nan_i),
            _mm256_or_si256(y_zero_i, x_zero_i)
        ),
        _mm256_or_si256(x_inf_i, y_inf_i)
    );

    const __m256i x_neg_i = _mm256_srai_epi32(hx, 31);
    const __m256i y_neg_i = _mm256_srai_epi32(hy, 31);

    const __m256 abs_y = _mm256_and_ps(y, abs_mask_ps);
    const __m256 abs_x = _mm256_and_ps(x, abs_mask_ps);

    if (_mm256_movemask_ps(_mm256_castsi256_ps(special_i)) == 0)
    {
        return atan2_normal_from_abs_ratio(
            abs_y, abs_x, y,
            pi_ps, pi_lo_ps, pio2_half_pilo_ps, zero_ps,
            sign_mask_ps,
            x_neg_i, y_neg_i,
            ix, iy,
            c26_i, cneg26_i);
    }

    const __m256 normal_mask_ps = _mm256_castsi256_ps(_mm256_andnot_si256(special_i, all1_i));
    const __m256 safe_abs_y = _mm256_blendv_ps(zero_ps, abs_y, normal_mask_ps);
    const __m256 safe_abs_x = _mm256_blendv_ps(one_ps, abs_x, normal_mask_ps);

    __m256 result = atan2_normal_from_abs_ratio(
        safe_abs_y, safe_abs_x, y,
        pi_ps, pi_lo_ps, pio2_half_pilo_ps, zero_ps,
        sign_mask_ps,
        x_neg_i, y_neg_i,
        ix, iy,
        c26_i, cneg26_i);

    const __m256 nan_res = _mm256_add_ps(x, y);
    result = _mm256_blendv_ps(result, nan_res,
        _mm256_castsi256_ps(_mm256_or_si256(x_nan_i, y_nan_i)));

    const __m256 signed_pi = _mm256_blendv_ps(pi_tiny_ps, mpi_tiny_ps, _mm256_castsi256_ps(y_neg_i));
    const __m256 yzero_res = _mm256_blendv_ps(y, signed_pi, _mm256_castsi256_ps(x_neg_i));
    result = _mm256_blendv_ps(result, yzero_res, _mm256_castsi256_ps(y_zero_i));

    const __m256 xzero_res = _mm256_blendv_ps(pio2_tiny_ps, mpio2_tiny_ps, _mm256_castsi256_ps(y_neg_i));
    const __m256i x_zero_only_i = _mm256_andnot_si256(y_zero_i, x_zero_i);
    result = _mm256_blendv_ps(result, xzero_res, _mm256_castsi256_ps(x_zero_only_i));

    const __m256 signed_zero = _mm256_and_ps(y, sign_mask_ps);
    const __m256 finite_xinf_res = _mm256_blendv_ps(signed_zero, signed_pi, _mm256_castsi256_ps(x_neg_i));

    const __m256 infinf_posx_res = _mm256_blendv_ps(pio4_tiny_ps, mpio4_tiny_ps, _mm256_castsi256_ps(y_neg_i));
    const __m256 infinf_negx_res = _mm256_blendv_ps(three_pio4_tiny_ps, mthree_pio4_tiny_ps, _mm256_castsi256_ps(y_neg_i));
    const __m256 infinf_res = _mm256_blendv_ps(infinf_posx_res, infinf_negx_res, _mm256_castsi256_ps(x_neg_i));

    const __m256 xinf_res = _mm256_blendv_ps(finite_xinf_res, infinf_res, _mm256_castsi256_ps(y_inf_i));
    result = _mm256_blendv_ps(result, xinf_res, _mm256_castsi256_ps(x_inf_i));

    const __m256 yinf_res = _mm256_blendv_ps(pio2_tiny_ps, mpio2_tiny_ps, _mm256_castsi256_ps(y_neg_i));
    result = _mm256_blendv_ps(result, yinf_res, _mm256_castsi256_ps(_mm256_andnot_si256(x_inf_i, y_inf_i)));

    result = _mm256_blendv_ps(result, nan_res,
        _mm256_castsi256_ps(_mm256_or_si256(x_nan_i, y_nan_i)));

    return result;
}


