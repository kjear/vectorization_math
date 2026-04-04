#include "foye_fastmath_fp32.hpp"

static __m256d exp2_precise(__m256d input)
{
	alignas(64) static constexpr double exp2_tab16[16] = {
		1.0000000000000000000,
		1.0442737824274138403,
		1.0905077326652576592,
		1.1387886347566916537,
		1.1892071150027210667,
		1.2418578120734840486,
		1.2968395546510096659,
		1.3542555469368926513,
		1.4142135623730950488,
		1.4768261459394993114,
		1.5422108254079408236,
		1.6104903319492543082,
		1.6817928305074290861,
		1.7562521603732994831,
		1.8340080864093424635,
		1.9152065613971472939
	};

	const __m256d zero = _mm256_setzero_pd();
	const __m256d one = _mm256_set1_pd(1.0);
	const __m256d two = _mm256_set1_pd(2.0);
	const __m256d half = _mm256_set1_pd(0.5);
	const __m256d three = _mm256_set1_pd(3.0);
	const __m256d six = _mm256_set1_pd(6.0);

	const __m256d ln2_hi = _mm256_set1_pd(6.93147180369123816490e-01);
	const __m256d ln2_lo = _mm256_set1_pd(1.90821492927058770002e-10);
	const __m256d ln2 = _mm256_set1_pd(6.93147180559945309417e-01);

	const __m256d tiny_x = _mm256_set1_pd(0x1p-54);
	const __m256d over_th = _mm256_set1_pd(1024.0);
	const __m256d under_th = _mm256_set1_pd(-1075.0);

	const __m256d invN = _mm256_set1_pd(1.0 / 16.0);
	const __m256d N = _mm256_set1_pd(16.0);

	const __m256d inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());
	const __m256d ninf = _mm256_set1_pd(-std::numeric_limits<double>::infinity());

	const __m256d q1 = _mm256_set1_pd(-3.33333333333331316428e-02);
	const __m256d q2 = _mm256_set1_pd(1.58730158725481460165e-03);
	const __m256d q3 = _mm256_set1_pd(-7.93650757867487942473e-05);
	const __m256d q4 = _mm256_set1_pd(4.00821782732936239552e-06);
	const __m256d q5 = _mm256_set1_pd(-2.01099218183624371326e-07);

	const __m256i abs_mask_i = _mm256_set1_epi64x(0x7fffffffffffffffULL);
	const __m256i int64_one = _mm256_set1_epi64x(1);
	const __m256i int64_three = _mm256_set1_epi64x(3);
	const __m256i int64_bias_1023 = _mm256_set1_epi64x(1023);
	const __m256i int64_1022 = _mm256_set1_epi64x(1022);
	const __m256i int64_1024 = _mm256_set1_epi64x(1024);
	const __m256i int64_neg1022 = _mm256_set1_epi64x(-1022);
	const __m256i int64_0x7ff = _mm256_set1_epi64x(0x7FF);
	const __m256i int64_frac_mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFFULL);
	const __m256i int64_implicit_one = _mm256_set1_epi64x(0x0010000000000000ULL);

	const __m256d scale_2pow1023 = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FE0000000000000ULL));

	__m256d absx = _mm256_castsi256_pd(_mm256_and_si256(_mm256_castpd_si256(input), abs_mask_i));

	__m256d nan_mask = _mm256_cmp_pd(input, input, _CMP_UNORD_Q);
	__m256d pinf_mask = _mm256_cmp_pd(input, inf, _CMP_EQ_OQ);
	__m256d ninf_mask = _mm256_cmp_pd(input, ninf, _CMP_EQ_OQ);
	__m256d over_mask = _mm256_cmp_pd(input, over_th, _CMP_GE_OQ);
	__m256d under_mask = _mm256_cmp_pd(input, under_th, _CMP_LE_OQ);
	__m256d tiny_mask = _mm256_cmp_pd(absx, tiny_x, _CMP_LT_OQ);

	__m256d special_mask = _mm256_or_pd(
		nan_mask,
		_mm256_or_pd(
			over_mask,
			_mm256_or_pd(under_mask, _mm256_or_pd(pinf_mask, ninf_mask))
		)
	);

	__m256d x = _mm256_blendv_pd(input, zero, special_mask);

	__m256d xm = _mm256_mul_pd(x, N);
	__m256d md_rounded = _mm256_round_pd(
		xm,
		_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
	);

	__m128i m32 = _mm256_cvttpd_epi32(md_rounded);
	__m256i m64 = _mm256_cvtepi32_epi64(m32);
	__m256d md = _mm256_cvtepi32_pd(m32);

	__m256d r = _mm256_fnmadd_pd(md, invN, x);

	__m128i j32 = _mm_and_si128(m32, _mm_set1_epi32(15));
	__m128i k32 = _mm_srai_epi32(m32, 4);
	__m256i k64 = _mm256_cvtepi32_epi64(k32);

	alignas(16) int j_idx[4];
	alignas(32) double t_arr[4];
	_mm_store_si128(reinterpret_cast<__m128i*>(j_idx), j32);
	t_arr[0] = exp2_tab16[j_idx[0]];
	t_arr[1] = exp2_tab16[j_idx[1]];
	t_arr[2] = exp2_tab16[j_idx[2]];
	t_arr[3] = exp2_tab16[j_idx[3]];
	__m256d t = _mm256_load_pd(t_arr);

	__m256d z_hi = _mm256_mul_pd(r, ln2_hi);
	__m256d z_lo = _mm256_mul_pd(r, ln2_lo);
	__m256d z = _mm256_add_pd(z_hi, z_lo);
	__m256d c = _mm256_add_pd(_mm256_sub_pd(z_hi, z), z_lo);

	__m256d hfx = _mm256_mul_pd(half, z);
	__m256d hxs = _mm256_mul_pd(z, hfx);

	__m256d p = _mm256_fmadd_pd(hxs, q5, q4);
	p = _mm256_fmadd_pd(hxs, p, q3);
	p = _mm256_fmadd_pd(hxs, p, q2);
	p = _mm256_fmadd_pd(hxs, p, q1);

	__m256d r1 = _mm256_fmadd_pd(hxs, p, one);
	__m256d tt = _mm256_fnmadd_pd(r1, hfx, three);
	__m256d e = _mm256_mul_pd(
		hxs,
		_mm256_div_pd(
			_mm256_sub_pd(r1, tt),
			_mm256_fnmadd_pd(z, tt, six)
		)
	);

	__m256d emc = _mm256_sub_pd(e, c);
	__m256d e2 = _mm256_fmsub_pd(z, emc, c);
	e2 = _mm256_sub_pd(e2, hxs);

	__m256d u = _mm256_sub_pd(z, e2);

	__m256d absz = _mm256_castsi256_pd(_mm256_and_si256(_mm256_castpd_si256(z), abs_mask_i));
	__m256d tiny_z_mask = _mm256_cmp_pd(absz, tiny_x, _CMP_LT_OQ);
	u = _mm256_blendv_pd(u, z, tiny_z_mask);

	__m256d core = _mm256_fmadd_pd(t, u, t);

	__m256i normal_exponent_bits = _mm256_add_epi64(k64, int64_bias_1023);
	normal_exponent_bits = _mm256_slli_epi64(normal_exponent_bits, 52);
	__m256d normal_scale = _mm256_castsi256_pd(normal_exponent_bits);
	__m256d normal_result = _mm256_mul_pd(core, normal_scale);

	__m256i mask_k_eq_1024_i = _mm256_cmpeq_epi64(k64, int64_1024);
	__m256d k1024_result = _mm256_mul_pd(_mm256_mul_pd(core, two), scale_2pow1023);

	__m256d result = _mm256_blendv_pd(
		normal_result,
		k1024_result,
		_mm256_castsi256_pd(mask_k_eq_1024_i)
	);

	__m256i core_bits = _mm256_castpd_si256(core);
	__m256i core_exp = _mm256_and_si256(_mm256_srli_epi64(core_bits, 52), int64_0x7ff);
	__m256i core_frac = _mm256_and_si256(core_bits, int64_frac_mask);
	__m256i significand53 = _mm256_or_si256(core_frac, int64_implicit_one);

	__m256i half_floor = _mm256_srli_epi64(significand53, 1);
	__m256i low2 = _mm256_and_si256(significand53, int64_three);
	__m256i tie_odd_mask = _mm256_cmpeq_epi64(low2, int64_three);
	__m256i half_rounded_even = _mm256_add_epi64(
		half_floor,
		_mm256_and_si256(tie_odd_mask, int64_one)
	);

	__m256i mask_exp_is_1022_i = _mm256_cmpeq_epi64(core_exp, int64_1022);
	__m256i fixed52 = _mm256_blendv_epi8(significand53, half_rounded_even, mask_exp_is_1022_i);

	__m256i sub_shift = _mm256_sub_epi64(int64_neg1022, k64);
	__m256i sub_shift_minus_one = _mm256_sub_epi64(sub_shift, int64_one);

	__m256i round_add = _mm256_sllv_epi64(int64_one, sub_shift_minus_one);
	__m256i sub_mantissa_bits = _mm256_srlv_epi64(
		_mm256_add_epi64(fixed52, round_add),
		sub_shift
	);

	__m256d subnormal_result = _mm256_castsi256_pd(sub_mantissa_bits);

	__m256i mask_subnormal_range_i = _mm256_cmpgt_epi64(int64_neg1022, k64);
	result = _mm256_blendv_pd(
		result,
		subnormal_result,
		_mm256_castsi256_pd(mask_subnormal_range_i)
	);

	__m256d tiny_result = _mm256_fmadd_pd(input, ln2, one);
	result = _mm256_blendv_pd(result, tiny_result, tiny_mask);

	result = _mm256_blendv_pd(result, zero, under_mask);
	result = _mm256_blendv_pd(result, inf, over_mask);
	result = _mm256_blendv_pd(result, zero, ninf_mask);
	result = _mm256_blendv_pd(result, inf, pinf_mask);
	result = _mm256_blendv_pd(result, input, nan_mask);

	return result;
};

static __m256d exp2_fast_ordinary(__m256d input)
{
	alignas(64) static constexpr double exp2_tab32[32] = {
		1.0000000000000000000,
		1.0218971486541166782,
		1.0442737824274138403,
		1.0671404006768236972,
		1.0905077326652576592,
		1.1143867425958924329,
		1.1387886347566916537,
		1.1637248587775774755,
		1.1892071150027210667,
		1.2152473599804689552,
		1.2418578120734840486,
		1.2690509571917332226,
		1.2968395546510096659,
		1.3252366431597412946,
		1.3542555469368926513,
		1.3839098819638320226,
		1.4142135623730950488,
		1.4451808069770466200,
		1.4768261459394993114,
		1.5091644275934227398,
		1.5422108254079408236,
		1.5759808451078864865,
		1.6104903319492543082,
		1.6457554781539649458,
		1.6817928305074290861,
		1.7186192981224779340,
		1.7562521603732994831,
		1.7947090750031071864,
		1.8340080864093424635,
		1.8741676341102999013,
		1.9152065613971472939,
		1.9571441241754001794
	};

	const __m256d one = _mm256_set1_pd(1.0);

	const __m256d N = _mm256_set1_pd(32.0);
	const __m256d invN = _mm256_set1_pd(1.0 / 32.0);

	const __m256d ln2_hi = _mm256_set1_pd(6.93147180369123816490e-01);
	const __m256d ln2_lo = _mm256_set1_pd(1.90821492927058770002e-10);

	const __m256d c2 = _mm256_set1_pd(1.0 / 2.0);
	const __m256d c3 = _mm256_set1_pd(1.0 / 6.0);
	const __m256d c4 = _mm256_set1_pd(1.0 / 24.0);
	const __m256d c5 = _mm256_set1_pd(1.0 / 120.0);
	const __m256d c6 = _mm256_set1_pd(1.0 / 720.0);
	const __m256d c7 = _mm256_set1_pd(1.0 / 5040.0);

	const __m256i int64_bias_1023 = _mm256_set1_epi64x(1023);

	__m256d xm = _mm256_mul_pd(input, N);
	__m256d md_rounded = _mm256_round_pd(xm, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

	__m128i m32 = _mm256_cvttpd_epi32(md_rounded);
	__m256d md = _mm256_cvtepi32_pd(m32);

	__m256d r = _mm256_fnmadd_pd(md, invN, input);

	__m128i j32 = _mm_and_si128(m32, _mm_set1_epi32(31));
	__m128i k32 = _mm_srai_epi32(m32, 5);
	__m256i k64 = _mm256_cvtepi32_epi64(k32);

	__m256d t = _mm256_i32gather_pd(exp2_tab32, j32, 8);

	__m256d z = _mm256_fmadd_pd(r, ln2_lo, _mm256_mul_pd(r, ln2_hi));

	__m256d p = c7;
	p = _mm256_fmadd_pd(p, z, c6);
	p = _mm256_fmadd_pd(p, z, c5);
	p = _mm256_fmadd_pd(p, z, c4);
	p = _mm256_fmadd_pd(p, z, c3);
	p = _mm256_fmadd_pd(p, z, c2);

	__m256d z2 = _mm256_mul_pd(z, z);
	__m256d u = _mm256_fmadd_pd(z2, p, z);

	__m256d core = _mm256_fmadd_pd(t, u, t);

	__m256i exponent_bits = _mm256_slli_epi64(_mm256_add_epi64(k64, int64_bias_1023), 52);
	__m256d scale = _mm256_castsi256_pd(exponent_bits);

	return _mm256_mul_pd(core, scale);
};

static __m256d exp2_fp64(__m256d input)
{
	const __m256d abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fffffffffffffffULL));
	const __m256d tiny_x = _mm256_set1_pd(0x1p-54);

	const __m256d fast_lo = _mm256_set1_pd(-1022.0);
	const __m256d fast_hi = _mm256_set1_pd(1023.0);

	__m256d absx = _mm256_and_pd(input, abs_mask);

	__m256d mask_not_nan = _mm256_cmp_pd(input, input, _CMP_ORD_Q);
	__m256d mask_ge_lo = _mm256_cmp_pd(input, fast_lo, _CMP_GE_OQ);
	__m256d mask_lt_hi = _mm256_cmp_pd(input, fast_hi, _CMP_LT_OQ);
	__m256d mask_not_tiny = _mm256_cmp_pd(absx, tiny_x, _CMP_GE_OQ);

	__m256d fast_mask = _mm256_and_pd(mask_not_nan,
		_mm256_and_pd(mask_ge_lo, _mm256_and_pd(mask_lt_hi, mask_not_tiny)));

	if (_mm256_movemask_pd(fast_mask) == 0xF)
	{
		return exp2_fast_ordinary(input);
	}

	return exp2_precise(input);
}

static __m256d log2_fp64(__m256d input)
{
	const __m256d two54 = _mm256_set1_pd(1.80143985094819840000e+16);
	const __m256d one = _mm256_set1_pd(1.0);
	const __m256d half = _mm256_set1_pd(0.5);
	const __m256d zero = _mm256_set1_pd(0.0);
	const __m256d two = _mm256_set1_pd(2.0);

	const __m256d Lg1 = _mm256_set1_pd(6.666666666666735130e-01);
	const __m256d Lg2 = _mm256_set1_pd(3.999999999940941908e-01);
	const __m256d Lg3 = _mm256_set1_pd(2.857142874366239149e-01);
	const __m256d Lg4 = _mm256_set1_pd(2.222219843214978396e-01);
	const __m256d Lg5 = _mm256_set1_pd(1.818357216161805012e-01);
	const __m256d Lg6 = _mm256_set1_pd(1.531383769920937332e-01);
	const __m256d Lg7 = _mm256_set1_pd(1.479819860511658591e-01);

	const __m256d invln2_hi = _mm256_set1_pd(1.44269502162933349609e+00);
	const __m256d invln2_lo = _mm256_set1_pd(1.92596299112661746887e-08);
	const __m256d invln2 = _mm256_set1_pd(1.44269504088896340736e+00);

	const __m256d near_thresh = _mm256_set1_pd(3.125e-2);

	const __m256d neg_inf = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
	const __m256d qnan = _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN());

	const __m256i abs_mask = _mm256_set1_epi64x(0x7fffffffffffffffULL);
	const __m256i exp_mask = _mm256_set1_epi64x(0x7ff0000000000000ULL);
	const __m256i frac_mask = _mm256_set1_epi64x(0x000fffffffffffffULL);
	const __m256i min_norm = _mm256_set1_epi64x(0x0010000000000000ULL);
	const __m256i one_bits = _mm256_set1_epi64x(0x3ff0000000000000ULL);

	const __m256i c1023 = _mm256_set1_epi64x(1023);
	const __m256i c54 = _mm256_set1_epi64x(54);
	const __m256i c95f64 = _mm256_set1_epi64x(0x95f64);
	const __m256i c100000 = _mm256_set1_epi64x(0x100000);
	const __m256i c3ff00000 = _mm256_set1_epi64x(0x3ff00000);
	const __m256i cffffffff = _mm256_set1_epi64x(0xffffffffULL);
	const __m256i hi_mask = _mm256_set1_epi64x(0xffffffff00000000ULL);

	const __m256i zero_i = _mm256_setzero_si256();
	const __m256i perm_idx = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
	const __m256i all_ones = _mm256_cmpeq_epi64(zero_i, zero_i);

	__m256d dx = _mm256_sub_pd(input, one);
	__m256d adx = _mm256_castsi256_pd(
		_mm256_and_si256(_mm256_castpd_si256(dx), abs_mask));
	__m256d near_mask = _mm256_cmp_pd(adx, near_thresh, _CMP_LT_OQ);

	__m256i bits = _mm256_castpd_si256(input);
	__m256i absbits = _mm256_and_si256(bits, abs_mask);

	__m256i zero_mask_i = _mm256_cmpeq_epi64(absbits, zero_i);
	__m256i nonzero_mask_i = _mm256_xor_si256(zero_mask_i, all_ones);
	__m256i neg_mask_i = _mm256_cmpgt_epi64(zero_i, bits);
	__m256i neg_nonzero_mask_i = _mm256_and_si256(neg_mask_i, nonzero_mask_i);
	__m256i subnormal_mask_i = _mm256_and_si256(_mm256_cmpgt_epi64(min_norm, absbits), nonzero_mask_i);
	__m256i infnan_mask_i = _mm256_cmpeq_epi64(_mm256_and_si256(absbits, exp_mask), exp_mask);
	__m256i one_mask_i = _mm256_cmpeq_epi64(bits, one_bits);

	__m256d x_scaled = _mm256_blendv_pd(
		input,
		_mm256_mul_pd(input, two54),
		_mm256_castsi256_pd(subnormal_mask_i));

	__m256i bits2 = _mm256_castpd_si256(x_scaled);

	__m256i exp2 = _mm256_srli_epi64(bits2, 52);
	__m256i k_i64 = _mm256_sub_epi64(exp2, c1023);
	k_i64 = _mm256_sub_epi64(k_i64, _mm256_and_si256(subnormal_mask_i, c54));

	__m256i frac = _mm256_and_si256(bits2, frac_mask);
	__m256i hx = _mm256_and_si256(_mm256_srli_epi64(frac, 32), cffffffff);
	__m256i i = _mm256_and_si256(_mm256_add_epi64(hx, c95f64), c100000);

	__m256i frac_lo = _mm256_and_si256(frac, cffffffff);
	__m256i norm_hi = _mm256_or_si256(hx, _mm256_xor_si256(i, c3ff00000));
	__m256i norm_bits = _mm256_or_si256(_mm256_slli_epi64(norm_hi, 32), frac_lo);

	k_i64 = _mm256_add_epi64(k_i64, _mm256_srli_epi64(i, 20));

	__m256d m = _mm256_castsi256_pd(norm_bits);
	__m256d f = _mm256_sub_pd(m, one);

	__m256d hfsq = _mm256_mul_pd(half, _mm256_mul_pd(f, f));
	__m256d s = _mm256_div_pd(f, _mm256_add_pd(two, f));
	__m256d z = _mm256_mul_pd(s, s);
	__m256d w = _mm256_mul_pd(z, z);

	__m256d t1 = _mm256_mul_pd(
		w,
		_mm256_fmadd_pd(w, _mm256_fmadd_pd(w, Lg6, Lg4), Lg2));

	__m256d t2 = _mm256_mul_pd(
		z,
		_mm256_fmadd_pd(
			w,
			_mm256_fmadd_pd(w, _mm256_fmadd_pd(w, Lg7, Lg5), Lg3),
			Lg1));

	__m256d R = _mm256_add_pd(t1, t2);

	__m256d hi = _mm256_sub_pd(f, hfsq);
	hi = _mm256_castsi256_pd(
		_mm256_and_si256(_mm256_castpd_si256(hi), hi_mask));

	__m256d lo = _mm256_add_pd(
		_mm256_sub_pd(_mm256_sub_pd(f, hi), hfsq),
		_mm256_mul_pd(s, _mm256_add_pd(hfsq, R)));

	__m256i k32 = _mm256_permutevar8x32_epi32(k_i64, perm_idx);
	__m256d kd = _mm256_cvtepi32_pd(_mm256_castsi256_si128(k32));

	__m256d val_hi = _mm256_mul_pd(hi, invln2_hi);
	__m256d val_lo = _mm256_fmadd_pd(lo, invln2_hi, _mm256_mul_pd(_mm256_add_pd(hi, lo), invln2_lo));

	__m256d wsum = _mm256_add_pd(kd, val_hi);
	val_lo = _mm256_add_pd(val_lo, _mm256_add_pd(_mm256_sub_pd(kd, wsum), val_hi));
	__m256d core_res = _mm256_add_pd(wsum, val_lo);

	__m256d infnan_result = _mm256_add_pd(input, input);

	core_res = _mm256_blendv_pd(core_res, zero, _mm256_castsi256_pd(one_mask_i));
	core_res = _mm256_blendv_pd(core_res, neg_inf, _mm256_castsi256_pd(zero_mask_i));
	core_res = _mm256_blendv_pd(core_res, infnan_result, _mm256_castsi256_pd(infnan_mask_i));
	core_res = _mm256_blendv_pd(core_res, qnan, _mm256_castsi256_pd(neg_nonzero_mask_i));

	__m256d nf = dx;
	__m256d nhfsq = _mm256_mul_pd(half, _mm256_mul_pd(nf, nf));
	__m256d ns = _mm256_div_pd(nf, _mm256_add_pd(two, nf));
	__m256d nz = _mm256_mul_pd(ns, ns);
	__m256d nw = _mm256_mul_pd(nz, nz);

	__m256d nt1 = _mm256_mul_pd(
		nw,
		_mm256_fmadd_pd(nw, _mm256_fmadd_pd(nw, Lg6, Lg4), Lg2));

	__m256d nt2 = _mm256_mul_pd(
		nz,
		_mm256_fmadd_pd(
			nw,
			_mm256_fmadd_pd(nw, _mm256_fmadd_pd(nw, Lg7, Lg5), Lg3),
			Lg1));

	__m256d nR = _mm256_add_pd(nt1, nt2);

	__m256d nhi = _mm256_sub_pd(nf, nhfsq);
	nhi = _mm256_castsi256_pd(
		_mm256_and_si256(_mm256_castpd_si256(nhi), hi_mask));

	__m256d nlo = _mm256_add_pd(
		_mm256_sub_pd(_mm256_sub_pd(nf, nhi), nhfsq),
		_mm256_mul_pd(ns, _mm256_add_pd(nhfsq, nR)));

	__m256d near_hi = _mm256_mul_pd(nhi, invln2_hi);
	__m256d near_lo = _mm256_fmadd_pd(nlo, invln2_hi,
		_mm256_mul_pd(_mm256_add_pd(nhi, nlo), invln2_lo));

	__m256d near_res = _mm256_add_pd(near_hi, near_lo);

	return _mm256_blendv_pd(core_res, near_res, near_mask);
}

__m256 fy::simd::intrinsic::pow(__m256 x, __m256 y) noexcept
{
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 zero = _mm256_set1_ps(0.0f);
	const __m256 inf = _mm256_castsi256_ps(_mm256_set1_epi32(0x7f800000));
	const __m256 qnan = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000));
	const __m256 all1 = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
	const __m256 sign_mask_ps = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
	const __m256 abs_mask_ps = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

	const __m256i zero_i = _mm256_setzero_si256();
	const __m256i sign_mask_i = _mm256_set1_epi32(0x80000000);
	const __m256i abs_mask_i = _mm256_set1_epi32(0x7fffffff);
	const __m256i one_bits_i = _mm256_set1_epi32(0x3f800000);
	const __m256i inf_bits_i = _mm256_set1_epi32(0x7f800000);
	const __m256 two24 = _mm256_set1_ps(16777216.0f);

	__m256i xi = _mm256_castps_si256(x);
	__m256i yi = _mm256_castps_si256(y);
	__m256i ix = _mm256_and_si256(xi, abs_mask_i);
	__m256i iy = _mm256_and_si256(yi, abs_mask_i);
	__m256 ax = _mm256_and_ps(x, abs_mask_ps);

	__m256 x_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
	__m256 y_nan = _mm256_cmp_ps(y, y, _CMP_UNORD_Q);
	__m256 any_nan = _mm256_or_ps(x_nan, y_nan);

	__m256 y_is_zero = _mm256_castsi256_ps(_mm256_cmpeq_epi32(iy, zero_i));
	__m256 x_is_one = _mm256_castsi256_ps(_mm256_cmpeq_epi32(xi, one_bits_i));

	__m256 x_is_zero = _mm256_castsi256_ps(_mm256_cmpeq_epi32(ix, zero_i));
	__m256 x_is_inf = _mm256_castsi256_ps(_mm256_cmpeq_epi32(ix, inf_bits_i));
	__m256 ax_eq_1 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(ix, one_bits_i));
	__m256 x_ax_special = _mm256_or_ps(_mm256_or_ps(x_is_zero, x_is_inf), ax_eq_1);

	__m256 y_is_inf = _mm256_castsi256_ps(_mm256_cmpeq_epi32(iy, inf_bits_i));
	__m256 y_pos = _mm256_cmp_ps(y, zero, _CMP_GT_OQ);
	__m256 y_neg = _mm256_cmp_ps(y, zero, _CMP_LT_OQ);

	__m256 x_neg = _mm256_castsi256_ps(_mm256_cmpgt_epi32(zero_i, xi));
	__m256 x_nonneg_sign = _mm256_castsi256_ps(
		_mm256_cmpeq_epi32(_mm256_and_si256(xi, sign_mask_i), zero_i));

	__m256 y_abs = _mm256_and_ps(y, abs_mask_ps);
	__m256 y_abs_ge_1 = _mm256_cmp_ps(y_abs, one, _CMP_GE_OQ);
	__m256 y_abs_ge_2p24 = _mm256_cmp_ps(y_abs, two24, _CMP_GE_OQ);

	__m256 y_trunc = _mm256_round_ps(y, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
	__m256 y_is_int_small = _mm256_cmp_ps(y, y_trunc, _CMP_EQ_OQ);
	__m256 y_is_int = _mm256_or_ps(y_abs_ge_2p24, _mm256_and_ps(y_abs_ge_1, y_is_int_small));

	__m256i y_trunc_i = _mm256_cvttps_epi32(y_trunc);
	__m256i y_odd_i = _mm256_and_si256(y_trunc_i, _mm256_set1_epi32(1));
	__m256 y_is_odd_small = _mm256_castsi256_ps(
		_mm256_cmpeq_epi32(y_odd_i, _mm256_set1_epi32(1)));
	__m256 y_is_odd = _mm256_and_ps(y_is_int_small, y_is_odd_small);

	__m256 y_is_pos1 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(yi, _mm256_set1_epi32(0x3f800000)));
	__m256 y_is_neg1 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(yi, _mm256_set1_epi32((int)0xbf800000)));
	__m256 y_is_two = _mm256_castsi256_ps(_mm256_cmpeq_epi32(yi, _mm256_set1_epi32(0x40000000)));
	__m256 y_is_half = _mm256_castsi256_ps(_mm256_cmpeq_epi32(yi, _mm256_set1_epi32(0x3f000000)));

	__m256 ax_gt_1 = _mm256_cmp_ps(ax, one, _CMP_GT_OQ);
	__m256 ax_lt_1 = _mm256_cmp_ps(ax, one, _CMP_LT_OQ);

	__m256 res_yinf_hi = _mm256_blendv_ps(zero, inf, y_pos);
	__m256 res_yinf_lo = _mm256_blendv_ps(inf, zero, y_pos);
	__m256 res_yinf = one;
	res_yinf = _mm256_blendv_ps(res_yinf, res_yinf_hi, ax_gt_1);
	res_yinf = _mm256_blendv_ps(res_yinf, res_yinf_lo, ax_lt_1);

	__m256 z_special = ax;
	__m256 z_special_inv = _mm256_div_ps(one, z_special);
	z_special = _mm256_blendv_ps(z_special, z_special_inv, y_neg);

	__m256 z_special_signed = _mm256_xor_ps(
		z_special,
		_mm256_and_ps(_mm256_and_ps(sign_mask_ps, x), y_is_odd)
	);

	__m256 neg_one_nonint = _mm256_and_ps(
		_mm256_and_ps(x_neg, ax_eq_1),
		_mm256_andnot_ps(y_is_int, all1)
	);
	z_special = _mm256_blendv_ps(z_special_signed, qnan, neg_one_nonint);

	__m256 neg_nonint_general = _mm256_and_ps(
		_mm256_and_ps(x_neg, _mm256_andnot_ps(y_is_int, all1)),
		_mm256_andnot_ps(x_ax_special, all1)
	);

	__m256 safe_ax = ax;
	__m256 safe_y = y;
	__m256 special_for_general = _mm256_or_ps(
		_mm256_or_ps(x_ax_special, y_is_inf),
		any_nan
	);

	safe_ax = _mm256_blendv_ps(safe_ax, one, special_for_general);
	safe_y = _mm256_blendv_ps(safe_y, zero, special_for_general);

	__m128 ax_lo_ps = _mm256_castps256_ps128(safe_ax);
	__m128 ax_hi_ps = _mm256_extractf128_ps(safe_ax, 1);
	__m128 y_lo_ps = _mm256_castps256_ps128(safe_y);
	__m128 y_hi_ps = _mm256_extractf128_ps(safe_y, 1);

	__m256d ax_lo_pd = _mm256_cvtps_pd(ax_lo_ps);
	__m256d ax_hi_pd = _mm256_cvtps_pd(ax_hi_ps);
	__m256d y_lo_pd = _mm256_cvtps_pd(y_lo_ps);
	__m256d y_hi_pd = _mm256_cvtps_pd(y_hi_ps);

	__m256d lg_lo_pd = log2_fp64(ax_lo_pd);
	__m256d lg_hi_pd = log2_fp64(ax_hi_pd);

	__m256d z_lo_pd = _mm256_mul_pd(y_lo_pd, lg_lo_pd);
	__m256d z_hi_pd = _mm256_mul_pd(y_hi_pd, lg_hi_pd);

	__m256d r_lo_pd = exp2_fp64(z_lo_pd);
	__m256d r_hi_pd = exp2_fp64(z_hi_pd);

	__m128 r_lo_ps = _mm256_cvtpd_ps(r_lo_pd);
	__m128 r_hi_ps = _mm256_cvtpd_ps(r_hi_pd);

	__m256 res_general = _mm256_castps128_ps256(r_lo_ps);
	res_general = _mm256_insertf128_ps(res_general, r_hi_ps, 1);

	__m256 sign_flip = _mm256_and_ps(_mm256_and_ps(sign_mask_ps, x), y_is_odd);
	res_general = _mm256_xor_ps(res_general, sign_flip);

	res_general = _mm256_blendv_ps(res_general, qnan, neg_nonint_general);

	__m256 res_ytwo = _mm256_mul_ps(x, x);
	__m256 res_yhalf = _mm256_sqrt_ps(x);

	__m256 res = res_general;
	res = _mm256_blendv_ps(res, z_special, x_ax_special);
	res = _mm256_blendv_ps(res, res_yinf, y_is_inf);

	res = _mm256_blendv_ps(res, x, y_is_pos1);
	res = _mm256_blendv_ps(res, _mm256_div_ps(one, x), y_is_neg1);
	res = _mm256_blendv_ps(res, res_ytwo, y_is_two);
	res = _mm256_blendv_ps(res, res_yhalf, _mm256_and_ps(y_is_half, x_nonneg_sign));

	res = _mm256_blendv_ps(res, qnan, neg_nonint_general);
	res = _mm256_blendv_ps(res, qnan, any_nan);

	res = _mm256_blendv_ps(res, one, x_is_one);
	res = _mm256_blendv_ps(res, one, y_is_zero);

	return res;
}