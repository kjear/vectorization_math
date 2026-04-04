#include "foye_fastmath_fp32.hpp"

static __m128i pack_low32_from_u64x4(__m256i x) noexcept
{
	const __m128i lo = _mm256_castsi256_si128(x);
	const __m128i hi = _mm256_extracti128_si256(x, 1);

	const __m128i lo_sh = _mm_shuffle_epi32(lo, _MM_SHUFFLE(2, 0, 2, 0));
	const __m128i hi_sh = _mm_shuffle_epi32(hi, _MM_SHUFFLE(2, 0, 2, 0));

	return _mm_unpacklo_epi64(lo_sh, hi_sh);
}

static __m128i pack_high32_from_u64x4(__m256i x) noexcept
{
	return pack_low32_from_u64x4(_mm256_srli_epi64(x, 32));
}

static __m256d cvtepu32_pd_4x(__m128i x) noexcept
{
	const __m128i lo31 = _mm_and_si128(x, _mm_set1_epi32(0x7fffffff));
	const __m128i hi1 = _mm_srli_epi32(x, 31);

	__m256d d0 = _mm256_cvtepi32_pd(lo31);
	__m256d d1 = _mm256_cvtepi32_pd(hi1);

	return _mm256_fmadd_pd(d1, _mm256_set1_pd(2147483648.0), d0);
}

struct u256_4x64x4
{
	__m256i v[4];
};

struct u256_nz_prefix_4x64
{
	__m256i limb_nz[4];
	__m256i below[4];
};

struct bit3_sticky_4x64
{
	__m128i bits3;
	__m128i sticky;
};

static __m256i nz_to_one_64(__m256i x) noexcept
{
	const __m256i z = _mm256_setzero_si256();
	const __m256i is_zero = _mm256_cmpeq_epi64(x, z);
	return _mm256_andnot_si256(is_zero, _mm256_set1_epi64x(1));
}

static __m256i cmplt_epu64(__m256i a, __m256i b) noexcept
{
	const __m256i sign = _mm256_set1_epi64x(std::numeric_limits<std::int64_t>::min());
	const __m256i ax = _mm256_xor_si256(a, sign);
	const __m256i bx = _mm256_xor_si256(b, sign);
	return _mm256_cmpgt_epi64(bx, ax);
}

static __m256i select_limb64_4x_from4(
	__m128i idx,
	__m256i v0, __m256i v1, __m256i v2, __m256i v3) noexcept
{
	const __m128i one32 = _mm_set1_epi32(1);
	const __m128i two32 = _mm_set1_epi32(2);

	const __m128i bit0_32 = _mm_cmpeq_epi32(_mm_and_si128(idx, one32), one32);
	const __m128i bit1_32 = _mm_cmpeq_epi32(_mm_and_si128(idx, two32), two32);

	const __m256i bit0_64 = _mm256_cvtepi32_epi64(bit0_32);
	const __m256i bit1_64 = _mm256_cvtepi32_epi64(bit1_32);

	const __m256i a01 = _mm256_blendv_epi8(v0, v1, bit0_64);
	const __m256i a23 = _mm256_blendv_epi8(v2, v3, bit0_64);

	return _mm256_blendv_epi8(a01, a23, bit1_64);
}

static __m256i lowbits_mask_64_from_b_4x(__m128i b32) noexcept
{
	const __m128i zero32 = _mm_setzero_si128();
	const __m128i gt0_32 = _mm_cmpgt_epi32(b32, zero32);
	const __m256i gt0_64 = _mm256_cvtepi32_epi64(gt0_32);

	const __m256i b64 = _mm256_cvtepu32_epi64(b32);
	__m256i t = _mm256_sllv_epi64(_mm256_set1_epi64x(1), b64);
	t = _mm256_sub_epi64(t, _mm256_set1_epi64x(1));
	return _mm256_and_si256(t, gt0_64);
}

static void muladd_u24_u64_limb_4x(
	__m256i mant_u32_64,
	__m256i k_lo_u32_64,
	__m256i k_hi_u32_64,
	__m256i carry_in,
	__m256i* low_out,
	__m256i* carry_out) noexcept
{
	const __m256i p_lo = _mm256_mul_epu32(mant_u32_64, k_lo_u32_64);
	const __m256i p_hi = _mm256_mul_epu32(mant_u32_64, k_hi_u32_64);

	const __m256i p_hi_lo = _mm256_slli_epi64(p_hi, 32);
	const __m256i p_hi_hi = _mm256_srli_epi64(p_hi, 32);

	const __m256i sum1 = _mm256_add_epi64(p_lo, p_hi_lo);
	const __m256i carry1 = _mm256_and_si256(
		cmplt_epu64(sum1, p_lo),
		_mm256_set1_epi64x(1));

	const __m256i sum2 = _mm256_add_epi64(sum1, carry_in);
	const __m256i carry2 = _mm256_and_si256(
		cmplt_epu64(sum2, sum1),
		_mm256_set1_epi64x(1));

	*low_out = sum2;
	*carry_out = _mm256_add_epi64(
		_mm256_add_epi64(p_hi_hi, carry1),
		carry2);
}

static u256_4x64x4 mul_u192_u24_4x_64(__m128i mant) noexcept
{
	u256_4x64x4 out{};
	const __m256i mant_u32_64 = _mm256_cvtepu32_epi64(mant);
	const __m256i K0_LO = _mm256_set1_epi64x(0x000000003C439041ull);
	const __m256i K0_HI = _mm256_set1_epi64x(0x00000000DB629599ull);

	const __m256i K1_LO = _mm256_set1_epi64x(0x00000000F534DDC0ull);
	const __m256i K1_HI = _mm256_set1_epi64x(0x00000000FC2757D1ull);

	const __m256i K2_LO = _mm256_set1_epi64x(0x000000004E441529ull);
	const __m256i K2_HI = _mm256_set1_epi64x(0x00000000A2F9836Eull);

	__m256i carry = _mm256_setzero_si256();

	muladd_u24_u64_limb_4x(mant_u32_64, K0_LO, K0_HI, carry, &out.v[0], &carry);
	muladd_u24_u64_limb_4x(mant_u32_64, K1_LO, K1_HI, carry, &out.v[1], &carry);
	muladd_u24_u64_limb_4x(mant_u32_64, K2_LO, K2_HI, carry, &out.v[2], &carry);

	out.v[3] = carry;
	return out;
}

static u256_nz_prefix_4x64 build_nz_prefix_4x64(const u256_4x64x4& x) noexcept
{
	u256_nz_prefix_4x64 p{};

	for (int i = 0; i < 4; ++i)
	{
		p.limb_nz[i] = nz_to_one_64(x.v[i]);
	}

	p.below[0] = _mm256_setzero_si256();
	for (int i = 1; i < 4; ++i)
	{
		p.below[i] = _mm256_or_si256(p.below[i - 1], p.limb_nz[i - 1]);
	}

	return p;
}

static __m256i get_bits64_4x_64rep(const u256_4x64x4& x, __m128i bitpos) noexcept
{
	const __m128i zero32 = _mm_setzero_si128();
	const __m128i neg_mask32 = _mm_cmpgt_epi32(zero32, bitpos);

	const __m128i w = _mm_srai_epi32(bitpos, 6);
	const __m128i b32 = _mm_and_si128(bitpos, _mm_set1_epi32(63));

	const __m256i b64 = _mm256_cvtepu32_epi64(b32);
	const __m256i s64 = _mm256_sub_epi64(_mm256_set1_epi64x(64), b64);

	const __m256i z = _mm256_setzero_si256();

	const __m256i a0 = x.v[0];
	const __m256i a1 = x.v[1];
	const __m256i a2 = x.v[2];
	const __m256i a3 = x.v[3];

	auto mk = [&](const __m256i& a, const __m256i& c) -> __m256i
	{
		return _mm256_or_si256(
			_mm256_srlv_epi64(a, b64),
			_mm256_sllv_epi64(c, s64));
	};

	const __m256i r0 = mk(a0, a1);
	const __m256i r1 = mk(a1, a2);
	const __m256i r2 = mk(a2, a3);
	const __m256i r3 = mk(a3, z);

	__m256i r = select_limb64_4x_from4(w, r0, r1, r2, r3);

	const __m256i neg_mask64 = _mm256_cvtepi32_epi64(neg_mask32);
	return _mm256_andnot_si256(neg_mask64, r);
}

static bit3_sticky_4x64 analyze_bitpos_3bits_sticky_4x64(
	const u256_4x64x4& x,
	const u256_nz_prefix_4x64& p,
	__m128i bitpos) noexcept
{
	const __m128i zero32 = _mm_setzero_si128();
	const __m128i one32 = _mm_set1_epi32(1);
	const __m128i neg_mask32 = _mm_cmpgt_epi32(zero32, bitpos);

	const __m128i w = _mm_srai_epi32(bitpos, 6);
	const __m128i b32 = _mm_and_si128(bitpos, _mm_set1_epi32(63));

	const __m256i cur = select_limb64_4x_from4(w, x.v[0], x.v[1], x.v[2], x.v[3]);
	const __m256i nxt = select_limb64_4x_from4(
		_mm_add_epi32(w, one32),
		x.v[0], x.v[1],
		x.v[2], x.v[3]);

	const __m256i b64 = _mm256_cvtepu32_epi64(b32);
	const __m256i s64 = _mm256_sub_epi64(_mm256_set1_epi64x(64), b64);

	__m256i bits64 = _mm256_or_si256(
		_mm256_srlv_epi64(cur, b64),
		_mm256_sllv_epi64(nxt, s64));

	const __m128i bits3 = _mm_and_si128(
		pack_low32_from_u64x4(bits64),
		_mm_set1_epi32(7));

	const __m256i cur_mask = lowbits_mask_64_from_b_4x(b32);
	const __m256i cur_bits = _mm256_and_si256(cur, cur_mask);
	const __m256i cur_nz = nz_to_one_64(cur_bits);

	const __m256i lower_nz = select_limb64_4x_from4(
		w,
		p.below[0], p.below[1],
		p.below[2], p.below[3]);

	__m256i sticky64 = _mm256_or_si256(cur_nz, lower_nz);
	sticky64 = _mm256_and_si256(sticky64, _mm256_set1_epi64x(1));

	__m128i sticky = _mm_and_si128(
		pack_low32_from_u64x4(sticky64),
		_mm_set1_epi32(1));

	const __m128i keep = _mm_andnot_si128(neg_mask32, _mm_set1_epi32(-1));

	return {
		_mm_and_si128(bits3, keep),
		_mm_and_si128(sticky, keep)
	};
}

static void rem_pio2f_float_payne_hanek_4x_64rep(
	__m128 ax,
	__m128i* q_out,
	__m256d* y_out) noexcept
{
	const __m128i ux = _mm_castps_si128(ax);

	const __m128i exp = _mm_sub_epi32(
		_mm_and_si128(_mm_srli_epi32(ux, 23), _mm_set1_epi32(0xff)),
		_mm_set1_epi32(127));

	const __m128i mant = _mm_or_si128(
		_mm_and_si128(ux, _mm_set1_epi32(0x007fffff)),
		_mm_set1_epi32(0x00800000));

	const u256_4x64x4 P = mul_u192_u24_4x_64(mant);
	const u256_nz_prefix_4x64 Pnz = build_nz_prefix_4x64(P);

	const __m128i shift = _mm_sub_epi32(_mm_set1_epi32(215), exp);
	const __m128i shift_m1 = _mm_sub_epi32(shift, _mm_set1_epi32(1));

	const bit3_sticky_4x64 a = analyze_bitpos_3bits_sticky_4x64(P, Pnz, shift_m1);

	const __m128i round_bit = _mm_and_si128(a.bits3, _mm_set1_epi32(1));
	const __m128i floor_odd = _mm_and_si128(_mm_srli_epi32(a.bits3, 1), _mm_set1_epi32(1));
	const __m128i q0 = _mm_and_si128(_mm_srli_epi32(a.bits3, 1), _mm_set1_epi32(3));
	const __m256i frac64 = get_bits64_4x_64rep(P, _mm_sub_epi32(shift, _mm_set1_epi32(64)));

	const __m128i sticky = a.sticky;

	const __m128i sticky_or_odd = _mm_or_si128(sticky, floor_odd);
	const __m128i carry = _mm_and_si128(round_bit, sticky_or_odd);

	const __m128i q = _mm_and_si128(_mm_add_epi32(q0, carry), _mm_set1_epi32(3));

	const __m128i frac_hi32 = pack_high32_from_u64x4(frac64);
	const __m128i frac_lo32 = pack_low32_from_u64x4(frac64);

	__m256d f = _mm256_mul_pd(cvtepu32_pd_4x(frac_hi32), _mm256_set1_pd(0x1.0p-32));
	f = _mm256_fmadd_pd(cvtepu32_pd_4x(frac_lo32), _mm256_set1_pd(0x1.0p-64), f);
	f = _mm256_sub_pd(f, _mm256_cvtepi32_pd(carry));

	const __m256d PIO2_1 = _mm256_set1_pd(1.570796326734125614166);
	const __m256d PIO2_2 = _mm256_set1_pd(6.07710050650619224932e-11);
	const __m256d PIO2_3 = _mm256_set1_pd(2.02226624879595063154e-21);

	__m256d y = _mm256_fmadd_pd(f, PIO2_1,
		_mm256_fmadd_pd(f, PIO2_2, _mm256_mul_pd(f, PIO2_3)));

	*q_out = q;
	*y_out = y;
}

static void sincos_poly_kernel_4x_pd(__m256d x, __m128* s, __m128* c) noexcept
{
	const __m256d S1 = _mm256_set1_pd(-0x15555554cbac77.0p-55);
	const __m256d S2 = _mm256_set1_pd(0x111110896efbb2.0p-59);
	const __m256d S3 = _mm256_set1_pd(-0x1a00f9e2cae774.0p-65);
	const __m256d S4 = _mm256_set1_pd(0x16cd878c3b46a7.0p-71);

	const __m256d C0 = _mm256_set1_pd(-0x1ffffffd0c5e81.0p-54);
	const __m256d C1 = _mm256_set1_pd(0x155553e1053a42.0p-57);
	const __m256d C2 = _mm256_set1_pd(-0x16c087e80f1e27.0p-62);
	const __m256d C3 = _mm256_set1_pd(0x199342e0ee5069.0p-68);

	const __m256d one = _mm256_set1_pd(1.0);

	const __m256d z = _mm256_mul_pd(x, x);
	const __m256d w = _mm256_mul_pd(z, z);
	const __m256d vz = _mm256_mul_pd(z, x);
	const __m256d wz = _mm256_mul_pd(w, z);

	const __m256d cp_hi = _mm256_fmadd_pd(C3, z, C2);
	const __m256d sp_hi = _mm256_fmadd_pd(S4, z, S3);

	const __m256d cp_lo = _mm256_fmadd_pd(C1, w,
		_mm256_fmadd_pd(C0, z, one));

	const __m256d sp_lo = _mm256_fmadd_pd(S2, z, S1);

	const __m256d cos_p = _mm256_fmadd_pd(wz, cp_hi, cp_lo);
	const __m256d sin_p = _mm256_fmadd_pd(_mm256_mul_pd(vz, w), sp_hi,
		_mm256_fmadd_pd(vz, sp_lo, x));

	*c = _mm256_cvtpd_ps(cos_p);
	*s = _mm256_cvtpd_ps(sin_p);
}

static void apply_quadrant_8x(
	__m256 x,
	__m256 sin_k,
	__m256 cos_k,
	__m256i q,
	__m256* s,
	__m256* c) noexcept
{
	const __m256i one = _mm256_set1_epi32(1);
	const __m256i two = _mm256_set1_epi32(2);

	const __m256 odd_mask = _mm256_castsi256_ps(
		_mm256_cmpeq_epi32(_mm256_and_si256(q, one), one));

	__m256 sv = _mm256_blendv_ps(sin_k, cos_k, odd_mask);
	__m256 cv = _mm256_blendv_ps(cos_k, sin_k, odd_mask);

	const __m256i sin_sign_i = _mm256_slli_epi32(_mm256_and_si256(q, two), 30);
	const __m256i cos_sign_i = _mm256_slli_epi32(
		_mm256_and_si256(_mm256_add_epi32(q, one), two), 30);

	sv = _mm256_xor_ps(sv, _mm256_castsi256_ps(sin_sign_i));
	cv = _mm256_xor_ps(cv, _mm256_castsi256_ps(cos_sign_i));

	const __m256 signbit = _mm256_set1_ps(-0.0f);
	const __m256 x_sign = _mm256_and_ps(x, signbit);
	sv = _mm256_xor_ps(sv, x_sign);

	*s = sv;
	*c = cv;
}

static void apply_quadrant_4x(
	__m128 x,
	__m128 sin_k,
	__m128 cos_k,
	__m128i q,
	__m128* s,
	__m128* c) noexcept
{
	const __m128i one = _mm_set1_epi32(1);
	const __m128i two = _mm_set1_epi32(2);

	const __m128 swap_mask = _mm_castsi128_ps(
		_mm_cmpeq_epi32(_mm_and_si128(q, one), one));

	__m128 sv = _mm_blendv_ps(sin_k, cos_k, swap_mask);
	__m128 cv = _mm_blendv_ps(cos_k, sin_k, swap_mask);

	const __m128i sin_sign_i = _mm_slli_epi32(_mm_and_si128(q, two), 30);
	const __m128i cos_sign_i = _mm_slli_epi32(
		_mm_and_si128(_mm_add_epi32(q, one), two), 30);

	sv = _mm_xor_ps(sv, _mm_castsi128_ps(sin_sign_i));
	cv = _mm_xor_ps(cv, _mm_castsi128_ps(cos_sign_i));

	const __m128 signbit = _mm_set1_ps(-0.0f);
	const __m128 x_sign = _mm_and_ps(x, signbit);
	sv = _mm_xor_ps(sv, x_sign);

	*s = sv;
	*c = cv;
}

static void sincos_large_finite_4x(__m128 x, __m128* s, __m128* c) noexcept
{
	const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
	const __m128 ax = _mm_and_ps(x, abs_mask);

	__m128i q;
	__m256d y;

	rem_pio2f_float_payne_hanek_4x_64rep(ax, &q, &y);

	__m128 ks, kc;
	sincos_poly_kernel_4x_pd(y, &ks, &kc);

	apply_quadrant_4x(x, ks, kc, q, s, c);
}

static void sincos_medium_finite_4x(__m128 x, __m128* s, __m128* c) noexcept
{
	const __m128 abs_mask_f = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
	const __m128 ax_f = _mm_and_ps(x, abs_mask_f);

	const __m256d ax = _mm256_cvtps_pd(ax_f);

	const __m256d INVPIO2_HI = _mm256_set1_pd(0.63661977236758138);
	const __m256d INVPIO2_LO = _mm256_set1_pd(-3.9357353350364972e-17);

	__m256d t = _mm256_mul_pd(ax, INVPIO2_HI);
	t = _mm256_fmadd_pd(ax, INVPIO2_LO, t);

	const __m128i n_i = _mm256_cvtpd_epi32(t);
	const __m256d n = _mm256_cvtepi32_pd(n_i);

	const __m256d MPIO2_1 = _mm256_set1_pd(-1.570796326734125614166);
	const __m256d MPIO2_2 = _mm256_set1_pd(-6.07710050650619224932e-11);
	const __m256d MPIO2_3 = _mm256_set1_pd(-2.02226624879595063154e-21);
	const __m256d MPIO2_4 = _mm256_set1_pd(-8.47842766036889956997e-32);

	__m256d r = ax;
	r = _mm256_fmadd_pd(n, MPIO2_1, r);
	r = _mm256_fmadd_pd(n, MPIO2_2, r);
	r = _mm256_fmadd_pd(n, MPIO2_3, r);
	r = _mm256_fmadd_pd(n, MPIO2_4, r);

	__m128 ks, kc;
	sincos_poly_kernel_4x_pd(r, &ks, &kc);

	apply_quadrant_4x(x, ks, kc, n_i, s, c);
}

static void sincos_medium_finite(__m256 x, __m256* s, __m256* c) noexcept
{
	const __m128 x0 = _mm256_castps256_ps128(x);
	const __m128 x1 = _mm256_extractf128_ps(x, 1);

	__m128 s0, c0, s1, c1;
	sincos_medium_finite_4x(x0, &s0, &c0);
	sincos_medium_finite_4x(x1, &s1, &c1);

	*s = _mm256_insertf128_ps(_mm256_castps128_ps256(s0), s1, 1);
	*c = _mm256_insertf128_ps(_mm256_castps128_ps256(c0), c1, 1);
}

static void sincos_large_finite(__m256 x, __m256* s, __m256* c) noexcept
{
	const __m128 x0 = _mm256_castps256_ps128(x);
	const __m128 x1 = _mm256_extractf128_ps(x, 1);

	__m128 s0, c0, s1, c1;
	sincos_large_finite_4x(x0, &s0, &c0);
	sincos_large_finite_4x(x1, &s1, &c1);

	*s = _mm256_insertf128_ps(_mm256_castps128_ps256(s0), s1, 1);
	*c = _mm256_insertf128_ps(_mm256_castps128_ps256(c0), c1, 1);
}

static void sincos_poly_fast_8x(__m256 r, __m256* s, __m256* c) noexcept
{
	const __m256 c0 = _mm256_set1_ps(2.443315711809948e-005f);
	const __m256 c1 = _mm256_set1_ps(-1.388731625493765e-003f);
	const __m256 c2 = _mm256_set1_ps(4.166664568298827e-002f);

	const __m256 s0 = _mm256_set1_ps(-1.9515295891e-4f);
	const __m256 s1 = _mm256_set1_ps(8.3321608736e-3f);
	const __m256 s2 = _mm256_set1_ps(-1.6666654611e-1f);

	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 half = _mm256_set1_ps(0.5f);

	const __m256 z = _mm256_mul_ps(r, r);
	const __m256 z2 = _mm256_mul_ps(z, z);

	__m256 sp = _mm256_fmadd_ps(s0, z, s1);
	__m256 cp = _mm256_fmadd_ps(c0, z, c1);

	sp = _mm256_fmadd_ps(sp, z, s2);
	cp = _mm256_fmadd_ps(cp, z, c2);

	const __m256 sin_tail = _mm256_mul_ps(sp, z);
	const __m256 cos_tail = _mm256_mul_ps(cp, z2);

	*s = _mm256_fmadd_ps(sin_tail, r, r);

	__m256 cos_base = _mm256_fnmadd_ps(z, half, one);
	*c = _mm256_add_ps(cos_base, cos_tail);
}

void fy::simd::intrinsic::sincos(__m256 input, __m256* sin_result, __m256* cos_result) noexcept
{
	if (!sin_result && !cos_result)
	{
		return;
	}

	const __m256 sign_mask_ps = _mm256_set1_ps(-0.0f);
	const __m256 abs_mask_ps = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
	const __m256 pinf = _mm256_castsi256_ps(_mm256_set1_epi32(0x7f800000));

	const __m256 v_fast_max = _mm256_set1_ps(8388608.0f);
	const __m256 v_medium_max = _mm256_set1_ps(16777216.0f);

	const __m256 ax = _mm256_and_ps(input, abs_mask_ps);

	const __m256 mask_nan = _mm256_cmp_ps(input, input, _CMP_UNORD_Q);
	const __m256 mask_inf = _mm256_cmp_ps(ax, pinf, _CMP_EQ_OQ);
	const __m256 mask_special = _mm256_or_ps(mask_nan, mask_inf);

	const __m256 mask_medium_range = _mm256_and_ps(
		_mm256_cmp_ps(ax, v_fast_max, _CMP_GT_OQ),
		_mm256_cmp_ps(ax, v_medium_max, _CMP_LE_OQ));

	const __m256 mask_slow_range = _mm256_cmp_ps(ax, v_medium_max, _CMP_GT_OQ);

	const __m256 mask_medium = _mm256_andnot_ps(mask_special, mask_medium_range);
	const __m256 mask_slow = _mm256_andnot_ps(mask_special, mask_slow_range);
	const __m256 mask_not_fast = _mm256_or_ps(mask_medium, mask_slow);

	const __m256 ax_fast = _mm256_andnot_ps(mask_not_fast, ax);

	const __m256d invpio2 = _mm256_set1_pd(0.636619772367581343075535053490057448);
	const __m256d mpio2_1 = _mm256_set1_pd(-1.570796326734125614166);
	const __m256d mpio2_2 = _mm256_set1_pd(-6.07710050650619224932e-11);
	const __m256d mpio2_3 = _mm256_set1_pd(-2.02226624879595063154e-21);

	const __m128 ax_lo_f = _mm256_castps256_ps128(ax_fast);
	const __m128 ax_hi_f = _mm256_extractf128_ps(ax_fast, 1);

	const __m256d ax_lo_d = _mm256_cvtps_pd(ax_lo_f);
	const __m256d ax_hi_d = _mm256_cvtps_pd(ax_hi_f);

	const __m128i n_lo_i = _mm256_cvtpd_epi32(_mm256_mul_pd(ax_lo_d, invpio2));
	const __m128i n_hi_i = _mm256_cvtpd_epi32(_mm256_mul_pd(ax_hi_d, invpio2));

	const __m256d n_lo_d = _mm256_cvtepi32_pd(n_lo_i);
	const __m256d n_hi_d = _mm256_cvtepi32_pd(n_hi_i);

	__m256d r_lo_d = ax_lo_d;
	__m256d r_hi_d = ax_hi_d;

	r_lo_d = _mm256_fmadd_pd(n_lo_d, mpio2_1, r_lo_d);
	r_hi_d = _mm256_fmadd_pd(n_hi_d, mpio2_1, r_hi_d);

	r_lo_d = _mm256_fmadd_pd(n_lo_d, mpio2_2, r_lo_d);
	r_hi_d = _mm256_fmadd_pd(n_hi_d, mpio2_2, r_hi_d);

	r_lo_d = _mm256_fmadd_pd(n_lo_d, mpio2_3, r_lo_d);
	r_hi_d = _mm256_fmadd_pd(n_hi_d, mpio2_3, r_hi_d);

	const __m128 r_lo_f = _mm256_cvtpd_ps(r_lo_d);
	const __m128 r_hi_f = _mm256_cvtpd_ps(r_hi_d);

	__m256 r = _mm256_castps128_ps256(r_lo_f);
	r = _mm256_insertf128_ps(r, r_hi_f, 1);

	__m256i n_i = _mm256_castsi128_si256(n_lo_i);
	n_i = _mm256_insertf128_si256(n_i, n_hi_i, 1);

	__m256 sin_k_fast, cos_k_fast;
	sincos_poly_fast_8x(r, &sin_k_fast, &cos_k_fast);

	__m256 sin_out, cos_out;
	apply_quadrant_8x(input, sin_k_fast, cos_k_fast, n_i, &sin_out, &cos_out);

	const int medium_bits = _mm256_movemask_ps(mask_medium);
	if (medium_bits != 0)
	{
		const int med_lo = medium_bits & 0x0F;
		const int med_hi = (medium_bits >> 4) & 0x0F;

		__m256 sin_med = _mm256_setzero_ps();
		__m256 cos_med = _mm256_setzero_ps();

		if (med_lo != 0)
		{
			const __m128 in_lo = _mm256_castps256_ps128(input);
			__m128 s_lo, c_lo;
			sincos_medium_finite_4x(in_lo, &s_lo, &c_lo);

			sin_med = _mm256_castps128_ps256(s_lo);
			cos_med = _mm256_castps128_ps256(c_lo);
		}

		if (med_hi != 0)
		{
			const __m128 in_hi = _mm256_extractf128_ps(input, 1);
			__m128 s_hi, c_hi;
			sincos_medium_finite_4x(in_hi, &s_hi, &c_hi);

			sin_med = _mm256_insertf128_ps(sin_med, s_hi, 1);
			cos_med = _mm256_insertf128_ps(cos_med, c_hi, 1);
		}

		sin_out = _mm256_blendv_ps(sin_out, sin_med, mask_medium);
		cos_out = _mm256_blendv_ps(cos_out, cos_med, mask_medium);
	}

	const int slow_bits = _mm256_movemask_ps(mask_slow);
	if (slow_bits != 0)
	{
		const int slow_lo = slow_bits & 0x0F;
		const int slow_hi = (slow_bits >> 4) & 0x0F;

		__m256 sin_slow = _mm256_setzero_ps();
		__m256 cos_slow = _mm256_setzero_ps();

		if (slow_lo != 0)
		{
			const __m128 in_lo = _mm256_castps256_ps128(input);
			__m128 s_lo, c_lo;
			sincos_large_finite_4x(in_lo, &s_lo, &c_lo);

			sin_slow = _mm256_castps128_ps256(s_lo);
			cos_slow = _mm256_castps128_ps256(c_lo);
		}

		if (slow_hi != 0)
		{
			const __m128 in_hi = _mm256_extractf128_ps(input, 1);
			__m128 s_hi, c_hi;
			sincos_large_finite_4x(in_hi, &s_hi, &c_hi);

			sin_slow = _mm256_insertf128_ps(sin_slow, s_hi, 1);
			cos_slow = _mm256_insertf128_ps(cos_slow, c_hi, 1);
		}

		sin_out = _mm256_blendv_ps(sin_out, sin_slow, mask_slow);
		cos_out = _mm256_blendv_ps(cos_out, cos_slow, mask_slow);

		_mm256_zeroupper();
	}

	const __m256 inf_nan = _mm256_sub_ps(input, input);
	sin_out = _mm256_blendv_ps(sin_out, inf_nan, mask_inf);
	cos_out = _mm256_blendv_ps(cos_out, inf_nan, mask_inf);

	const __m256i qnan_quiet_i = _mm256_set1_epi32(0x00400000);
	const __m256 nan_quieted = _mm256_castsi256_ps(
		_mm256_or_si256(_mm256_castps_si256(input), qnan_quiet_i));

	sin_out = _mm256_blendv_ps(sin_out, nan_quieted, mask_nan);
	cos_out = _mm256_blendv_ps(cos_out, nan_quieted, mask_nan);

	if (sin_result) *sin_result = sin_out;
	if (cos_result) *cos_result = cos_out;
}

__m256 fy::simd::intrinsic::sin(__m256 input) noexcept
{
	__m256 sin_val;
	::fy::simd::intrinsic::sincos(input, &sin_val, nullptr);
	return sin_val;
}

__m256 fy::simd::intrinsic::cos(__m256 input) noexcept
{
	__m256 cos_val;
	::fy::simd::intrinsic::sincos(input, nullptr, &cos_val);
	return cos_val;
}

struct reduce_pio2_result
{
	__m256 r;
	__m256i n;
};

struct reduce_pio2_result_4x
{
	__m128 r;
	__m128i n;
};

static reduce_pio2_result reduce_pio2(__m256 x_fast) noexcept
{
	const __m256d invpio2 = _mm256_set1_pd(0.636619772367581343075535053490057448);
	const __m256d mpio2_1 = _mm256_set1_pd(-1.570796326734125614166);
	const __m256d mpio2_2 = _mm256_set1_pd(-6.07710050650619224932e-11);
	const __m256d mpio2_3 = _mm256_set1_pd(-2.02226624879595063154e-21);

	const __m128 x_lo_f = _mm256_castps256_ps128(x_fast);
	const __m128 x_hi_f = _mm256_extractf128_ps(x_fast, 1);

	const __m256d x_lo_d = _mm256_cvtps_pd(x_lo_f);
	const __m256d x_hi_d = _mm256_cvtps_pd(x_hi_f);

	const __m128i n_lo_i = _mm256_cvtpd_epi32(_mm256_mul_pd(x_lo_d, invpio2));
	const __m128i n_hi_i = _mm256_cvtpd_epi32(_mm256_mul_pd(x_hi_d, invpio2));

	const __m256d n_lo_d = _mm256_cvtepi32_pd(n_lo_i);
	const __m256d n_hi_d = _mm256_cvtepi32_pd(n_hi_i);

	__m256d r_lo_d = x_lo_d;
	r_lo_d = _mm256_fmadd_pd(n_lo_d, mpio2_1, r_lo_d);
	r_lo_d = _mm256_fmadd_pd(n_lo_d, mpio2_2, r_lo_d);
	r_lo_d = _mm256_fmadd_pd(n_lo_d, mpio2_3, r_lo_d);

	__m256d r_hi_d = x_hi_d;
	r_hi_d = _mm256_fmadd_pd(n_hi_d, mpio2_1, r_hi_d);
	r_hi_d = _mm256_fmadd_pd(n_hi_d, mpio2_2, r_hi_d);
	r_hi_d = _mm256_fmadd_pd(n_hi_d, mpio2_3, r_hi_d);

	const __m128 r_lo_f = _mm256_cvtpd_ps(r_lo_d);
	const __m128 r_hi_f = _mm256_cvtpd_ps(r_hi_d);

	__m256 r = _mm256_castps128_ps256(r_lo_f);
	r = _mm256_insertf128_ps(r, r_hi_f, 1);

	__m256i n = _mm256_castsi128_si256(n_lo_i);
	n = _mm256_insertf128_si256(n, n_hi_i, 1);

	return { r, n };
}

static reduce_pio2_result_4x reduce_pio2_4x(__m128 x_fast) noexcept
{
	const __m256d invpio2 = _mm256_set1_pd(0.636619772367581343075535053490057448);
	const __m256d mpio2_1 = _mm256_set1_pd(-1.570796326734125614166);
	const __m256d mpio2_2 = _mm256_set1_pd(-6.07710050650619224932e-11);
	const __m256d mpio2_3 = _mm256_set1_pd(-2.02226624879595063154e-21);

	const __m256d x_d = _mm256_cvtps_pd(x_fast);
	const __m128i n_i = _mm256_cvtpd_epi32(_mm256_mul_pd(x_d, invpio2));
	const __m256d n_d = _mm256_cvtepi32_pd(n_i);

	__m256d r_d = x_d;
	r_d = _mm256_fmadd_pd(n_d, mpio2_1, r_d);
	r_d = _mm256_fmadd_pd(n_d, mpio2_2, r_d);
	r_d = _mm256_fmadd_pd(n_d, mpio2_3, r_d);

	return { _mm256_cvtpd_ps(r_d), n_i };
}

static __m256 tan_poly_pio8(__m256 u) noexcept
{
	const __m256 z = _mm256_mul_ps(u, u);

	const __m256 c0 = _mm256_set1_ps(0.3333333333333f);
	const __m256 c1 = _mm256_set1_ps(0.1333333333333f);
	const __m256 c2 = _mm256_set1_ps(0.0539682539683f);
	const __m256 c3 = _mm256_set1_ps(0.0218694885362f);
	const __m256 c4 = _mm256_set1_ps(0.0088632355299f);
	const __m256 c5 = _mm256_set1_ps(0.0035921280366f);

	__m256 p = c5;
	p = _mm256_fmadd_ps(p, z, c4);
	p = _mm256_fmadd_ps(p, z, c3);
	p = _mm256_fmadd_ps(p, z, c2);
	p = _mm256_fmadd_ps(p, z, c1);
	p = _mm256_fmadd_ps(p, z, c0);

	return _mm256_fmadd_ps(_mm256_mul_ps(u, z), p, u);
}

static __m128 tan_poly_pio8_4x(__m128 u) noexcept
{
	const __m128 z = _mm_mul_ps(u, u);

	const __m128 c0 = _mm_set1_ps(0.3333333333333f);
	const __m128 c1 = _mm_set1_ps(0.1333333333333f);
	const __m128 c2 = _mm_set1_ps(0.0539682539683f);
	const __m128 c3 = _mm_set1_ps(0.0218694885362f);
	const __m128 c4 = _mm_set1_ps(0.0088632355299f);
	const __m128 c5 = _mm_set1_ps(0.0035921280366f);

	__m128 p = c5;
	p = _mm_fmadd_ps(p, z, c4);
	p = _mm_fmadd_ps(p, z, c3);
	p = _mm_fmadd_ps(p, z, c2);
	p = _mm_fmadd_ps(p, z, c1);
	p = _mm_fmadd_ps(p, z, c0);

	return _mm_fmadd_ps(_mm_mul_ps(u, z), p, u);
}

static __m128 tan_from_reduced_4x(__m128 r, __m128i n) noexcept
{
	const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
	const __m128 sign_mask = _mm_set1_ps(-0.0f);

	const __m128 pi_over_8 = _mm_set1_ps(0.39269908169872415481f);
	const __m128 pio4_hi = _mm_set1_ps(0.78539812564849853515625f);
	const __m128 pio4_lo = _mm_set1_ps(3.77489497744594108e-08f);

	const __m128 one = _mm_set1_ps(1.0f);
	const __m128 minus_one = _mm_set1_ps(-1.0f);

	const __m128 abs_r = _mm_and_ps(r, abs_mask);
	const __m128 sign_r = _mm_and_ps(r, sign_mask);

	const __m128 mask_big = _mm_cmp_ps(abs_r, pi_over_8, _CMP_GT_OQ);
	const int big_bits = _mm_movemask_ps(mask_big);

	__m128 u;
	if (big_bits == 0)
	{
		u = r;
	}
	else if (big_bits == 0xF)
	{
		const __m128 u_big_mag = _mm_add_ps(_mm_sub_ps(pio4_hi, abs_r), pio4_lo);
		u = _mm_xor_ps(u_big_mag, sign_r);
	}
	else
	{
		const __m128 u_big_mag = _mm_add_ps(_mm_sub_ps(pio4_hi, abs_r), pio4_lo);
		const __m128 u_big = _mm_xor_ps(u_big_mag, sign_r);
		u = _mm_blendv_ps(r, u_big, mask_big);
	}

	const __m128 t = tan_poly_pio8_4x(u);

	__m128 tan_r;
	if (big_bits == 0)
	{
		tan_r = t;
	}
	else if (big_bits == 0xF)
	{
		const __m128 abs_t = _mm_and_ps(t, abs_mask);
		const __m128 num = _mm_sub_ps(one, abs_t);
		const __m128 den = _mm_add_ps(one, abs_t);
		const __m128 big_mag = _mm_div_ps(num, den);
		tan_r = _mm_xor_ps(big_mag, sign_r);
	}
	else
	{
		const __m128 abs_t = _mm_and_ps(t, abs_mask);
		const __m128 num = _mm_sub_ps(one, abs_t);
		const __m128 den = _mm_add_ps(one, abs_t);
		const __m128 big_mag = _mm_div_ps(num, den);
		const __m128 tan_r_big = _mm_xor_ps(big_mag, sign_r);
		tan_r = _mm_blendv_ps(t, tan_r_big, mask_big);
	}

	const __m128 odd_mask = _mm_castsi128_ps(
		_mm_cmpeq_epi32(
			_mm_and_si128(n, _mm_set1_epi32(1)),
			_mm_set1_epi32(1)));

	const int odd_bits = _mm_movemask_ps(odd_mask);

	if (odd_bits == 0)
	{
		return tan_r;
	}
	if (odd_bits == 0xF)
	{
		return _mm_div_ps(minus_one, tan_r);
	}

	const __m128 tan_odd = _mm_div_ps(minus_one, tan_r);
	return _mm_blendv_ps(tan_r, tan_odd, odd_mask);
}

static __m256 tan_from_reduced_8x(__m256 r, __m256i n) noexcept
{
	const __m256 sign_mask = _mm256_set1_ps(-0.0f);
	const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

	const __m256 pi_over_8 = _mm256_set1_ps(0.39269908169872415481f);
	const __m256 pio4_hi = _mm256_set1_ps(0.78539812564849853515625f);
	const __m256 pio4_lo = _mm256_set1_ps(3.77489497744594108e-08f);

	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 minus_one = _mm256_set1_ps(-1.0f);

	const __m256 abs_r = _mm256_and_ps(r, abs_mask);
	const __m256 sign_r = _mm256_and_ps(r, sign_mask);

	const __m256 mask_big = _mm256_cmp_ps(abs_r, pi_over_8, _CMP_GT_OQ);
	const int big_bits = _mm256_movemask_ps(mask_big);

	__m256 u;
	if (big_bits == 0)
	{
		u = r;
	}
	else if (big_bits == 0xFF)
	{
		const __m256 u_big_mag = _mm256_add_ps(_mm256_sub_ps(pio4_hi, abs_r), pio4_lo);
		u = _mm256_xor_ps(u_big_mag, sign_r);
	}
	else
	{
		const __m256 u_big_mag = _mm256_add_ps(_mm256_sub_ps(pio4_hi, abs_r), pio4_lo);
		const __m256 u_big = _mm256_xor_ps(u_big_mag, sign_r);
		u = _mm256_blendv_ps(r, u_big, mask_big);
	}

	const __m256 t = tan_poly_pio8(u);

	__m256 tan_r;
	if (big_bits == 0)
	{
		tan_r = t;
	}
	else if (big_bits == 0xFF)
	{
		const __m256 abs_t = _mm256_and_ps(t, abs_mask);
		const __m256 num = _mm256_sub_ps(one, abs_t);
		const __m256 den = _mm256_add_ps(one, abs_t);
		const __m256 big_mag = _mm256_div_ps(num, den);
		tan_r = _mm256_xor_ps(big_mag, sign_r);
	}
	else
	{
		const __m256 abs_t = _mm256_and_ps(t, abs_mask);
		const __m256 num = _mm256_sub_ps(one, abs_t);
		const __m256 den = _mm256_add_ps(one, abs_t);
		const __m256 big_mag = _mm256_div_ps(num, den);
		const __m256 tan_r_big = _mm256_xor_ps(big_mag, sign_r);
		tan_r = _mm256_blendv_ps(t, tan_r_big, mask_big);
	}

	const __m256i one_i = _mm256_set1_epi32(1);
	const __m256i odd_i = _mm256_and_si256(n, one_i);
	const __m256 odd_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(odd_i, one_i));
	const int odd_bits = _mm256_movemask_ps(odd_mask);

	if (odd_bits == 0)
	{
		return tan_r;
	}
	if (odd_bits == 0xFF)
	{
		return _mm256_div_ps(minus_one, tan_r);
	}

	const __m256 tan_odd = _mm256_div_ps(minus_one, tan_r);
	return _mm256_blendv_ps(tan_r, tan_odd, odd_mask);
}

static __m128 tan_fast_reduced_4x(__m128 x) noexcept
{
	const reduce_pio2_result_4x red = reduce_pio2_4x(x);
	return tan_from_reduced_4x(red.r, red.n);
}

static __m256 tan_fast_reduced_8x(__m256 x) noexcept
{
	const reduce_pio2_result red = reduce_pio2(x);
	return tan_from_reduced_8x(red.r, red.n);
}

static __m128 tan_poly_kernel_4x_pd(__m256d x) noexcept
{
	const __m256d T0 = _mm256_set1_pd(0x15554d3418c99f.0p-54);
	const __m256d T1 = _mm256_set1_pd(0x1112fd38999f72.0p-55);
	const __m256d T2 = _mm256_set1_pd(0x1b54c91d865afe.0p-57);
	const __m256d T3 = _mm256_set1_pd(0x191df3908c33ce.0p-58);
	const __m256d T4 = _mm256_set1_pd(0x185dadfcecf44e.0p-61);
	const __m256d T5 = _mm256_set1_pd(0x1362b9bf971bcd.0p-59);

	const __m256d z = _mm256_mul_pd(x, x);
	const __m256d w = _mm256_mul_pd(z, z);
	const __m256d s = _mm256_mul_pd(z, x);

	__m256d r = _mm256_fmadd_pd(T5, z, T4);
	const __m256d t = _mm256_fmadd_pd(T3, z, T2);
	const __m256d u = _mm256_fmadd_pd(T1, z, T0);

	r = _mm256_add_pd(
		_mm256_fmadd_pd(s, u, x),
		_mm256_mul_pd(
			_mm256_mul_pd(s, w),
			_mm256_fmadd_pd(w, r, t)));

	return _mm256_cvtpd_ps(r);
}

static void tan_large_finite_4x(__m128 x, __m128* out) noexcept
{
	const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
	const __m128 signbit = _mm_set1_ps(-0.0f);

	const __m128 ax = _mm_and_ps(x, abs_mask);

	__m128i q;
	__m256d y;
	rem_pio2f_float_payne_hanek_4x_64rep(ax, &q, &y);

	const __m128 t = tan_poly_kernel_4x_pd(y);

	const __m128i odd_i = _mm_and_si128(q, _mm_set1_epi32(1));
	const __m128 odd_mask = _mm_castsi128_ps(
		_mm_cmpeq_epi32(odd_i, _mm_set1_epi32(1)));

	const __m128 tan_odd = _mm_div_ps(_mm_set1_ps(-1.0f), t);
	__m128 r = _mm_blendv_ps(t, tan_odd, odd_mask);

	const __m128 x_sign = _mm_and_ps(x, signbit);
	r = _mm_xor_ps(r, x_sign);

	*out = r;
}

static void tan_medium_finite_4x(__m128 x, __m128* out) noexcept
{
	const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
	const __m128 signbit = _mm_set1_ps(-0.0f);

	const __m128 ax_f = _mm_and_ps(x, abs_mask);
	const __m256d ax = _mm256_cvtps_pd(ax_f);

	const __m256d INVPIO2_HI = _mm256_set1_pd(0.63661977236758138);
	const __m256d INVPIO2_LO = _mm256_set1_pd(-3.9357353350364972e-17);

	__m256d t = _mm256_mul_pd(ax, INVPIO2_HI);
	t = _mm256_fmadd_pd(ax, INVPIO2_LO, t);

	const __m128i n_i = _mm256_cvtpd_epi32(t);
	const __m256d n = _mm256_cvtepi32_pd(n_i);

	const __m256d MPIO2_1 = _mm256_set1_pd(-1.570796326734125614166);
	const __m256d MPIO2_2 = _mm256_set1_pd(-6.07710050650619224932e-11);
	const __m256d MPIO2_3 = _mm256_set1_pd(-2.02226624879595063154e-21);
	const __m256d MPIO2_4 = _mm256_set1_pd(-8.47842766036889956997e-32);

	__m256d r = ax;
	r = _mm256_fmadd_pd(n, MPIO2_1, r);
	r = _mm256_fmadd_pd(n, MPIO2_2, r);
	r = _mm256_fmadd_pd(n, MPIO2_3, r);
	r = _mm256_fmadd_pd(n, MPIO2_4, r);

	const __m128 t_r = tan_poly_kernel_4x_pd(r);

	const __m128 odd_mask = _mm_castsi128_ps(
		_mm_cmpeq_epi32(
			_mm_and_si128(n_i, _mm_set1_epi32(1)),
			_mm_set1_epi32(1)));

	const __m128 tan_odd = _mm_div_ps(_mm_set1_ps(-1.0f), t_r);
	__m128 y = _mm_blendv_ps(t_r, tan_odd, odd_mask);

	const __m128 x_sign = _mm_and_ps(x, signbit);
	y = _mm_xor_ps(y, x_sign);

	*out = y;
}

__m256 fy::simd::intrinsic::tan(__m256 input) noexcept
{
	const __m256 sign_mask = _mm256_set1_ps(-0.0f);
	const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
	const __m256 pinf = _mm256_castsi256_ps(_mm256_set1_epi32(0x7f800000));

	const __m256 one = _mm256_set1_ps(1.0f);

	const __m256 v_fast_max = _mm256_set1_ps(8388608.0f);
	const __m256 v_medium_max = _mm256_set1_ps(16777216.0f);

	const __m256 pi_over_8 = _mm256_set1_ps(0.39269908169872415481f);
	const __m256 pi_over_4 = _mm256_set1_ps(0.78539816339744830962f);
	const __m256 pio4_hi = _mm256_set1_ps(0.78539812564849853515625f);
	const __m256 pio4_lo = _mm256_set1_ps(3.77489497744594108e-08f);

	const __m256 ax = _mm256_and_ps(input, abs_mask);

	const __m256 mask_nan = _mm256_cmp_ps(input, input, _CMP_UNORD_Q);
	const __m256 mask_inf = _mm256_cmp_ps(ax, pinf, _CMP_EQ_OQ);
	const __m256 mask_special = _mm256_or_ps(mask_nan, mask_inf);

	const __m256 mask_medium_range = _mm256_and_ps(
		_mm256_cmp_ps(ax, v_fast_max, _CMP_GT_OQ),
		_mm256_cmp_ps(ax, v_medium_max, _CMP_LE_OQ));

	const __m256 mask_slow_range = _mm256_cmp_ps(ax, v_medium_max, _CMP_GT_OQ);

	const __m256 mask_medium = _mm256_andnot_ps(mask_special, mask_medium_range);
	const __m256 mask_slow = _mm256_andnot_ps(mask_special, mask_slow_range);

	const __m256 mask_fast_finite = _mm256_andnot_ps(
		_mm256_or_ps(mask_special, _mm256_or_ps(mask_medium, mask_slow)),
		_mm256_castsi256_ps(_mm256_set1_epi32(-1)));

	const __m256 mask_fast_le_pio4 = _mm256_and_ps(
		mask_fast_finite,
		_mm256_cmp_ps(ax, pi_over_4, _CMP_LE_OQ));

	const __m256 mask_fast_gt_pio4 = _mm256_and_ps(
		mask_fast_finite,
		_mm256_cmp_ps(ax, pi_over_4, _CMP_GT_OQ));

	__m256 result = _mm256_setzero_ps();

	const int fast_pio4_bits = _mm256_movemask_ps(mask_fast_le_pio4);
	if (fast_pio4_bits != 0)
	{
		const __m256 sign_x = _mm256_and_ps(input, sign_mask);

		const __m256 mask_big = _mm256_and_ps(
			mask_fast_le_pio4,
			_mm256_cmp_ps(ax, pi_over_8, _CMP_GT_OQ));

		const int big_bits = _mm256_movemask_ps(mask_big);

		__m256 u;
		if (big_bits == 0)
		{
			u = input;
		}
		else if (big_bits == fast_pio4_bits)
		{
			const __m256 u_big_mag = _mm256_add_ps(_mm256_sub_ps(pio4_hi, ax), pio4_lo);
			u = _mm256_xor_ps(u_big_mag, sign_x);
		}
		else
		{
			const __m256 u_big_mag = _mm256_add_ps(_mm256_sub_ps(pio4_hi, ax), pio4_lo);
			const __m256 u_big = _mm256_xor_ps(u_big_mag, sign_x);
			u = _mm256_blendv_ps(input, u_big, mask_big);
		}

		const __m256 t = tan_poly_pio8(u);

		__m256 fast_val;
		if (big_bits == 0)
		{
			fast_val = t;
		}
		else if (big_bits == fast_pio4_bits)
		{
			const __m256 abs_t = _mm256_and_ps(t, abs_mask);
			const __m256 num = _mm256_sub_ps(one, abs_t);
			const __m256 den = _mm256_add_ps(one, abs_t);
			const __m256 big_mag = _mm256_div_ps(num, den);
			fast_val = _mm256_xor_ps(big_mag, sign_x);
		}
		else
		{
			const __m256 abs_t = _mm256_and_ps(t, abs_mask);
			const __m256 num = _mm256_sub_ps(one, abs_t);
			const __m256 den = _mm256_add_ps(one, abs_t);
			const __m256 big_mag = _mm256_div_ps(num, den);
			const __m256 big_val = _mm256_xor_ps(big_mag, sign_x);
			fast_val = _mm256_blendv_ps(t, big_val, mask_big);
		}

		result = _mm256_blendv_ps(result, fast_val, mask_fast_le_pio4);
	}

	const int fast_reduce_bits = _mm256_movemask_ps(mask_fast_gt_pio4);
	if (fast_reduce_bits != 0)
	{
		__m256 fast_red_val = _mm256_setzero_ps();

		const int fr_lo = fast_reduce_bits & 0x0F;
		const int fr_hi = (fast_reduce_bits >> 4) & 0x0F;

		if (fast_reduce_bits == 0xFF)
		{
			fast_red_val = tan_fast_reduced_8x(input);
		}
		else if (fr_lo != 0 && fr_hi == 0)
		{
			const __m128 in_lo = _mm256_castps256_ps128(input);
			const __m128 mask_lo = _mm256_castps256_ps128(mask_fast_gt_pio4);
			const __m128 x_lo = _mm_and_ps(in_lo, mask_lo);

			const __m128 out_lo = tan_fast_reduced_4x(x_lo);
			fast_red_val = _mm256_castps128_ps256(out_lo);
		}
		else if (fr_lo == 0 && fr_hi != 0)
		{
			const __m128 in_hi = _mm256_extractf128_ps(input, 1);
			const __m128 mask_hi = _mm256_extractf128_ps(mask_fast_gt_pio4, 1);
			const __m128 x_hi = _mm_and_ps(in_hi, mask_hi);

			const __m128 out_hi = tan_fast_reduced_4x(x_hi);
			fast_red_val = _mm256_insertf128_ps(fast_red_val, out_hi, 1);
		}
		else
		{
			if (fr_lo != 0)
			{
				const __m128 in_lo = _mm256_castps256_ps128(input);
				const __m128 mask_lo = _mm256_castps256_ps128(mask_fast_gt_pio4);
				const __m128 x_lo = _mm_and_ps(in_lo, mask_lo);

				const __m128 out_lo = tan_fast_reduced_4x(x_lo);
				fast_red_val = _mm256_castps128_ps256(out_lo);
			}

			if (fr_hi != 0)
			{
				const __m128 in_hi = _mm256_extractf128_ps(input, 1);
				const __m128 mask_hi = _mm256_extractf128_ps(mask_fast_gt_pio4, 1);
				const __m128 x_hi = _mm_and_ps(in_hi, mask_hi);

				const __m128 out_hi = tan_fast_reduced_4x(x_hi);
				fast_red_val = _mm256_insertf128_ps(fast_red_val, out_hi, 1);
			}
		}

		result = _mm256_blendv_ps(result, fast_red_val, mask_fast_gt_pio4);
	}

	const int medium_bits = _mm256_movemask_ps(mask_medium);
	if (medium_bits != 0)
	{
		const int med_lo = medium_bits & 0x0F;
		const int med_hi = (medium_bits >> 4) & 0x0F;

		__m256 med_vec = _mm256_setzero_ps();

		if (med_lo != 0)
		{
			const __m128 in_lo = _mm256_castps256_ps128(input);
			__m128 out_lo;
			tan_medium_finite_4x(in_lo, &out_lo);

			med_vec = _mm256_castps128_ps256(out_lo);
		}

		if (med_hi != 0)
		{
			const __m128 in_hi = _mm256_extractf128_ps(input, 1);
			__m128 out_hi;
			tan_medium_finite_4x(in_hi, &out_hi);

			med_vec = _mm256_insertf128_ps(med_vec, out_hi, 1);
		}

		result = _mm256_blendv_ps(result, med_vec, mask_medium);
	}

	const int slow_bits = _mm256_movemask_ps(mask_slow);
	if (slow_bits != 0)
	{
		const int slow_lo = slow_bits & 0x0F;
		const int slow_hi = (slow_bits >> 4) & 0x0F;

		__m256 slow_vec = _mm256_setzero_ps();

		if (slow_lo != 0)
		{
			const __m128 in_lo = _mm256_castps256_ps128(input);
			__m128 out_lo;
			tan_large_finite_4x(in_lo, &out_lo);

			slow_vec = _mm256_castps128_ps256(out_lo);
		}

		if (slow_hi != 0)
		{
			const __m128 in_hi = _mm256_extractf128_ps(input, 1);
			__m128 out_hi;
			tan_large_finite_4x(in_hi, &out_hi);

			slow_vec = _mm256_insertf128_ps(slow_vec, out_hi, 1);
		}

		result = _mm256_blendv_ps(result, slow_vec, mask_slow);

		_mm256_zeroupper();
	}

	const __m256 inf_nan = _mm256_sub_ps(input, input);
	result = _mm256_blendv_ps(result, inf_nan, mask_inf);

	const __m256i qnan_quiet_i = _mm256_set1_epi32(0x00400000);
	const __m256 nan_quieted = _mm256_castsi256_ps(
		_mm256_or_si256(_mm256_castps_si256(input), qnan_quiet_i));

	result = _mm256_blendv_ps(result, nan_quieted, mask_nan);

	return result;
}