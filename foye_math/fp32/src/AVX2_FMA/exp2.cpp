#include <foye_fastmath.hpp>

static __m256 exp2_eval_reduced(
    __m256 input,
    __m256 tbl_lo,
    __m256 tbl_hi,
    __m256i* out_k) noexcept
{
    const __m256 scale16 = _mm256_set1_ps(16.0f);
    const __m256 inv16 = _mm256_set1_ps(0.0625f);

    const __m256 p1 = _mm256_set1_ps(0x1.62e430p-1f);
    const __m256 p2 = _mm256_set1_ps(0x1.ebfbe0p-3f);
    const __m256 p3 = _mm256_set1_ps(0x1.c6b356p-5f);
    const __m256 p4 = _mm256_set1_ps(0x1.3b2c9cp-7f);
    const __m256 p5 = _mm256_set1_ps(0x1.5d8800p-10f);

    __m256 nfp = _mm256_round_ps(
        _mm256_mul_ps(input, scale16),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
    );

    __m256i n = _mm256_cvtps_epi32(nfp);
    __m256i k = _mm256_srai_epi32(n, 4);

    __m256i idx = _mm256_and_si256(n, _mm256_set1_epi32(15));
    __m256i idxlo = _mm256_and_si256(idx, _mm256_set1_epi32(7));
    __m256 idx_hi_mask =
        _mm256_castsi256_ps(_mm256_cmpgt_epi32(idx, _mm256_set1_epi32(7)));

    __m256 tv_lo = _mm256_permutevar8x32_ps(tbl_lo, idxlo);
    __m256 tv_hi = _mm256_permutevar8x32_ps(tbl_hi, idxlo);
    __m256 tv = _mm256_blendv_ps(tv_lo, tv_hi, idx_hi_mask);

    __m256 z = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(n), inv16, input);

    __m256 vz = _mm256_mul_ps(tv, z);

    __m256 poly = _mm256_fmadd_ps(p5, z, p4);
    poly = _mm256_fmadd_ps(poly, z, p3);
    poly = _mm256_fmadd_ps(poly, z, p2);
    poly = _mm256_fmadd_ps(poly, z, p1);

    *out_k = k;
    return _mm256_fmadd_ps(vz, poly, tv);
}

static inline __m256 exp2_build_normal(__m256 r, __m256i k) noexcept
{
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 two = _mm256_set1_ps(2.0f);

    __m256i hi_mask_i = _mm256_cmpgt_epi32(k, _mm256_set1_epi32(127));
    __m256i hi_one = _mm256_srli_epi32(hi_mask_i, 31);

    __m256 extra_hi = _mm256_blendv_ps(one, two, _mm256_castsi256_ps(hi_mask_i));
    __m256i k_adj = _mm256_sub_epi32(k, hi_one);

    __m256i expbits = _mm256_slli_epi32(
        _mm256_add_epi32(k_adj, _mm256_set1_epi32(127)),
        23
    );
    __m256 twopk = _mm256_castsi256_ps(expbits);

    return _mm256_mul_ps(_mm256_mul_ps(r, extra_hi), twopk);
}

static inline __m256 exp2_build_subnormal_bits(__m256 r, __m256i k) noexcept
{
    const __m256i bias = _mm256_set1_epi32(127);
    const __m256i plus149 = _mm256_set1_epi32(149);

    __m256i shift = _mm256_add_epi32(k, plus149);
    __m256i scale_exp = _mm256_add_epi32(shift, bias);
    __m256i scale_bits = _mm256_slli_epi32(scale_exp, 23);
    __m256 scale = _mm256_castsi256_ps(scale_bits);

    __m256 t = _mm256_mul_ps(r, scale);

    __m256i m = _mm256_cvtps_epi32(t);

    return _mm256_castsi256_ps(m);
}

static __m256 exp2_finalize_special(__m256 result, __m256 input) noexcept
{
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_set1_ps(0.0f);
    const __m256 inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());
    const __m256 neg_inf = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    const __m256 ln2 = _mm256_set1_ps(0x1.62e43p-1f);

    const __m256 tiny_limit = _mm256_set1_ps(0x1p-25f);

    const __m256 overflow_limit = _mm256_set1_ps(128.0f);
    const __m256 underflow_limit = _mm256_set1_ps(-150.0f);

    __m256 ax = _mm256_andnot_ps(sign_mask, input);

    __m256 tiny_mask = _mm256_cmp_ps(ax, tiny_limit, _CMP_LE_OQ);
    __m256 ov_mask = _mm256_cmp_ps(input, overflow_limit, _CMP_GE_OQ);
    __m256 uf_mask = _mm256_cmp_ps(input, underflow_limit, _CMP_LE_OQ);
    __m256 nan_mask = _mm256_cmp_ps(input, input, _CMP_UNORD_Q);
    __m256 posinf_mask = _mm256_cmp_ps(input, inf, _CMP_EQ_OQ);
    __m256 neginf_mask = _mm256_cmp_ps(input, neg_inf, _CMP_EQ_OQ);

    const __m256i qnan_quiet_bit = _mm256_set1_epi32(0x00400000);
    __m256 nan_quieted = _mm256_castsi256_ps(
        _mm256_or_si256(_mm256_castps_si256(input), qnan_quiet_bit)
    );

    result = _mm256_blendv_ps(result, _mm256_fmadd_ps(input, ln2, one), tiny_mask);
    result = _mm256_blendv_ps(result, zero, uf_mask);
    result = _mm256_blendv_ps(result, inf, ov_mask);
    result = _mm256_blendv_ps(result, zero, neginf_mask);
    result = _mm256_blendv_ps(result, inf, posinf_mask);
    result = _mm256_blendv_ps(result, nan_quieted, nan_mask);

    return result;
}

static __m256 exp2_fast_branch(
    __m256 input,
    __m256 tbl_lo,
    __m256 tbl_hi) noexcept
{
    __m256i k;
    __m256 r = exp2_eval_reduced(input, tbl_lo, tbl_hi, &k);

    return exp2_build_normal(r, k);
}

static __m256 exp2_main_branch(
    __m256 input,
    __m256 tbl_lo,
    __m256 tbl_hi) noexcept
{
    const __m256 subnormal_limit = _mm256_set1_ps(-126.0f);
    const __m256 underflow_limit = _mm256_set1_ps(-150.0f);

    __m256i k;
    __m256 r = exp2_eval_reduced(input, tbl_lo, tbl_hi, &k);

    __m256 normal_res = exp2_build_normal(r, k);
    __m256 sub_res = exp2_build_subnormal_bits(r, k);

    __m256 gt_uf = _mm256_cmp_ps(input, underflow_limit, _CMP_GT_OQ);
    __m256 lt_sub = _mm256_cmp_ps(input, subnormal_limit, _CMP_LT_OQ);
    __m256 sub_mask = _mm256_and_ps(gt_uf, lt_sub);

    __m256 result = _mm256_blendv_ps(normal_res, sub_res, sub_mask);
    return exp2_finalize_special(result, input);
}

alignas(32) static constexpr float lut_tbl_lo[8] = {
    0x1.000000p+0f, 0x1.0b5586p+0f,
    0x1.172b84p+0f, 0x1.2387a6p+0f,
    0x1.306fe0p+0f, 0x1.3dea64p+0f,
    0x1.4bfdacp+0f, 0x1.5ab07cp+0f
};

alignas(32) static constexpr float lut_tbl_hi[8] = {
    0x1.6a09e6p+0f, 0x1.7a1148p+0f,
    0x1.8ace54p+0f, 0x1.9c4918p+0f,
    0x1.ae89fap+0f, 0x1.c199bep+0f,
    0x1.d5818ep+0f, 0x1.ea4afap+0f
};

__m256 fy::simd::intrinsic::exp2(__m256 input) noexcept
{
    const __m256 fast_min = _mm256_set1_ps(-126.0f);
    const __m256 fast_max = _mm256_set1_ps(127.9375f);

    const __m256 tbl_lo = _mm256_load_ps(lut_tbl_lo);
    const __m256 tbl_hi = _mm256_load_ps(lut_tbl_hi);

    __m256 ge_min = _mm256_cmp_ps(input, fast_min, _CMP_GE_OQ);
    __m256 le_max = _mm256_cmp_ps(input, fast_max, _CMP_LE_OQ);
    __m256 fast_mask = _mm256_and_ps(ge_min, le_max);

    return (_mm256_movemask_ps(fast_mask) == 0xFF)
        ? exp2_fast_branch(input, tbl_lo, tbl_hi)
        : exp2_main_branch(input, tbl_lo, tbl_hi);
}
