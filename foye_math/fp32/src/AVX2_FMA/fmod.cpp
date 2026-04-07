#include <foye_fastmath.hpp>

static __m256i ilog2_epu32_nonzero(__m256i v)
{
	const __m256i zero_i = _mm256_setzero_si256();
	const __m256i c16 = _mm256_set1_epi32(16);
	const __m256i c8 = _mm256_set1_epi32(8);
	const __m256i c4 = _mm256_set1_epi32(4);
	const __m256i c2 = _mm256_set1_epi32(2);
	const __m256i c1 = _mm256_set1_epi32(1);

	const __m256i cmp_ffff = _mm256_set1_epi32(0x0000ffff);
	const __m256i cmp_ff = _mm256_set1_epi32(0x000000ff);
	const __m256i cmp_f = _mm256_set1_epi32(0x0000000f);
	const __m256i cmp_3 = _mm256_set1_epi32(0x00000003);
	const __m256i cmp_1 = _mm256_set1_epi32(0x00000001);

	__m256i lg = zero_i;
	__m256i m;

	m = _mm256_cmpgt_epi32(v, cmp_ffff);
	lg = _mm256_blendv_epi8(lg, c16, m);
	v = _mm256_blendv_epi8(v, _mm256_srli_epi32(v, 16), m);

	m = _mm256_cmpgt_epi32(v, cmp_ff);
	lg = _mm256_add_epi32(lg, _mm256_and_si256(m, c8));
	v = _mm256_blendv_epi8(v, _mm256_srli_epi32(v, 8), m);

	m = _mm256_cmpgt_epi32(v, cmp_f);
	lg = _mm256_add_epi32(lg, _mm256_and_si256(m, c4));
	v = _mm256_blendv_epi8(v, _mm256_srli_epi32(v, 4), m);

	m = _mm256_cmpgt_epi32(v, cmp_3);
	lg = _mm256_add_epi32(lg, _mm256_and_si256(m, c2));
	v = _mm256_blendv_epi8(v, _mm256_srli_epi32(v, 2), m);

	m = _mm256_cmpgt_epi32(v, cmp_1);
	lg = _mm256_add_epi32(lg, _mm256_and_si256(m, c1));

	return lg;
}

__m256 fy::simd::intrinsic::fmod(__m256 x, __m256 y) noexcept
{
	const __m256i all_ones = _mm256_set1_epi32(-1);
	const __m256i zero_i = _mm256_setzero_si256();
	const __m256i one_i = _mm256_set1_epi32(1);
	const __m256i sign_mask = _mm256_set1_epi32(0x80000000u);
	const __m256i abs_mask = _mm256_set1_epi32(0x7fffffffu);
	const __m256i frac_mask = _mm256_set1_epi32(0x007fffffu);
	const __m256i min_norm = _mm256_set1_epi32(0x00800000u);
	const __m256i inf_bits = _mm256_set1_epi32(0x7f800000u);
	const __m256i max_finite = _mm256_set1_epi32(0x7f7fffffu);
	const __m256i bias127 = _mm256_set1_epi32(127);
	const __m256i neg126 = _mm256_set1_epi32(-126);
	const __m256i neg127 = _mm256_set1_epi32(-127);
	const __m256i c149 = _mm256_set1_epi32(149);
	const __m256i c23 = _mm256_set1_epi32(23);
	const __m256 one_ps = _mm256_set1_ps(1.0f);
	const __m256 zero_ps = _mm256_setzero_ps();

	__m256i xi = _mm256_castps_si256(x);
	__m256i yi = _mm256_castps_si256(y);

	__m256i sx = _mm256_and_si256(xi, sign_mask);
	__m256i hx = _mm256_and_si256(xi, abs_mask);
	__m256i hy = _mm256_and_si256(yi, abs_mask);

	__m256i y_zero = _mm256_cmpeq_epi32(hy, zero_i);
	__m256i x_inf_nan = _mm256_cmpgt_epi32(hx, max_finite);
	__m256i y_nan = _mm256_cmpgt_epi32(hy, inf_bits);
	__m256i special = _mm256_or_si256(_mm256_or_si256(y_zero, x_inf_nan), y_nan);

	__m256i x_lt_y = _mm256_cmpgt_epi32(hy, hx);
	__m256i x_eq_y = _mm256_cmpeq_epi32(hx, hy);

	__m256i core = _mm256_andnot_si256(_mm256_or_si256(special, _mm256_or_si256(x_lt_y, x_eq_y)), all_ones);

	__m256i hx_norm = hx;
	__m256i hy_norm = hy;
	__m256i ix = zero_i;
	__m256i iy = zero_i;

	{
		__m256i hx_nonzero = _mm256_cmpgt_epi32(hx, zero_i);
		__m256i hy_nonzero = _mm256_cmpgt_epi32(hy, zero_i);

		__m256i x_sub_mask = _mm256_and_si256(hx_nonzero, _mm256_cmpgt_epi32(min_norm, hx));
		__m256i y_sub_mask = _mm256_and_si256(hy_nonzero, _mm256_cmpgt_epi32(min_norm, hy));
		__m256i any_sub = _mm256_or_si256(x_sub_mask, y_sub_mask);

		if (_mm256_testz_si256(any_sub, any_sub))
		{
			ix = _mm256_sub_epi32(_mm256_srli_epi32(hx, 23), bias127);
			iy = _mm256_sub_epi32(_mm256_srli_epi32(hy, 23), bias127);

			hx_norm = _mm256_or_si256(min_norm, _mm256_and_si256(hx, frac_mask));
			hy_norm = _mm256_or_si256(min_norm, _mm256_and_si256(hy, frac_mask));
		}
		else
		{
			{
				__m256i x_norm_mask = _mm256_cmpgt_epi32(hx, _mm256_sub_epi32(min_norm, one_i));
				__m256i x_sub_only = _mm256_andnot_si256(x_norm_mask, _mm256_cmpgt_epi32(hx, zero_i));

				__m256i ex = _mm256_sub_epi32(_mm256_srli_epi32(hx, 23), bias127);

				__m256i v_nz = _mm256_or_si256(hx, one_i);
				__m256i lg = ilog2_epu32_nonzero(v_nz);
				__m256i ex_sub = _mm256_sub_epi32(lg, c149);

				ix = _mm256_blendv_epi8(ix, ex, x_norm_mask);
				ix = _mm256_blendv_epi8(ix, ex_sub, x_sub_only);
			}

			{
				__m256i y_norm_mask = _mm256_cmpgt_epi32(hy, _mm256_sub_epi32(min_norm, one_i));
				__m256i y_sub_only = _mm256_andnot_si256(y_norm_mask, _mm256_cmpgt_epi32(hy, zero_i));

				__m256i ey = _mm256_sub_epi32(_mm256_srli_epi32(hy, 23), bias127);

				__m256i v_nz = _mm256_or_si256(hy, one_i);
				__m256i lg = ilog2_epu32_nonzero(v_nz);
				__m256i ey_sub = _mm256_sub_epi32(lg, c149);

				iy = _mm256_blendv_epi8(iy, ey, y_norm_mask);
				iy = _mm256_blendv_epi8(iy, ey_sub, y_sub_only);
			}

			{
				__m256i x_ge_neg126 = _mm256_cmpgt_epi32(ix, neg127);
				__m256i shx = _mm256_sub_epi32(neg126, ix);
				__m256i hx_sub = _mm256_sllv_epi32(hx, shx);
				__m256i hx_nrm = _mm256_or_si256(min_norm, _mm256_and_si256(hx, frac_mask));
				hx_norm = _mm256_blendv_epi8(hx_sub, hx_nrm, x_ge_neg126);

				__m256i y_ge_neg126 = _mm256_cmpgt_epi32(iy, neg127);
				__m256i shy = _mm256_sub_epi32(neg126, iy);
				__m256i hy_sub = _mm256_sllv_epi32(hy, shy);
				__m256i hy_nrm = _mm256_or_si256(min_norm, _mm256_and_si256(hy, frac_mask));
				hy_norm = _mm256_blendv_epi8(hy_sub, hy_nrm, y_ge_neg126);
			}
		}
	}

	__m256i n = _mm256_sub_epi32(ix, iy);
	__m256i active = core;
	__m256i zero_mask = x_eq_y;

	{
		__m256i cond = _mm256_and_si256(active, _mm256_cmpgt_epi32(n, zero_i));

		while (!_mm256_testz_si256(cond, cond))
		{
			{
				__m256i iter = cond;
				__m256i hz = _mm256_sub_epi32(hx_norm, hy_norm);
				__m256i hz_neg = _mm256_cmpgt_epi32(zero_i, hz);
				__m256i hz_zero = _mm256_cmpeq_epi32(hz, zero_i);
				__m256i hx_dbl = _mm256_add_epi32(hx_norm, hx_norm);
				__m256i hz_dbl = _mm256_add_epi32(hz, hz);
				__m256i next_hx = _mm256_blendv_epi8(hz_dbl, hx_dbl, hz_neg);

				hx_norm = _mm256_blendv_epi8(hx_norm, next_hx, iter);
				zero_mask = _mm256_or_si256(zero_mask, _mm256_and_si256(iter, hz_zero));
				active = _mm256_andnot_si256(_mm256_and_si256(iter, hz_zero), active);
				n = _mm256_sub_epi32(n, _mm256_and_si256(iter, one_i));
			}

			cond = _mm256_and_si256(active, _mm256_cmpgt_epi32(n, zero_i));
			if (_mm256_testz_si256(cond, cond))
			{
				break;
			}

			{
				__m256i iter = cond;
				__m256i hz = _mm256_sub_epi32(hx_norm, hy_norm);
				__m256i hz_neg = _mm256_cmpgt_epi32(zero_i, hz);
				__m256i hz_zero = _mm256_cmpeq_epi32(hz, zero_i);
				__m256i hx_dbl = _mm256_add_epi32(hx_norm, hx_norm);
				__m256i hz_dbl = _mm256_add_epi32(hz, hz);
				__m256i next_hx = _mm256_blendv_epi8(hz_dbl, hx_dbl, hz_neg);

				hx_norm = _mm256_blendv_epi8(hx_norm, next_hx, iter);
				zero_mask = _mm256_or_si256(zero_mask, _mm256_and_si256(iter, hz_zero));
				active = _mm256_andnot_si256(_mm256_and_si256(iter, hz_zero), active);
				n = _mm256_sub_epi32(n, _mm256_and_si256(iter, one_i));
			}

			cond = _mm256_and_si256(active, _mm256_cmpgt_epi32(n, zero_i));
		}
	}

	{
		__m256i hz = _mm256_sub_epi32(hx_norm, hy_norm);
		__m256i hz_neg = _mm256_cmpgt_epi32(zero_i, hz);
		__m256i hz_nonneg = _mm256_andnot_si256(hz_neg, all_ones);
		__m256i m = _mm256_and_si256(active, hz_nonneg);
		hx_norm = _mm256_blendv_epi8(hx_norm, hz, m);
	}

	{
		__m256i hx_zero = _mm256_cmpeq_epi32(hx_norm, zero_i);
		zero_mask = _mm256_or_si256(zero_mask, _mm256_and_si256(active, hx_zero));
		active = _mm256_andnot_si256(_mm256_and_si256(active, hx_zero), active);
	}

	{
		__m256i need_norm = _mm256_and_si256(active, _mm256_cmpgt_epi32(min_norm, hx_norm));
		if (!_mm256_testz_si256(need_norm, need_norm))
		{
			__m256i safe_hx = _mm256_or_si256(hx_norm, one_i);
			__m256i lg = ilog2_epu32_nonzero(safe_hx);
			__m256i shift = _mm256_sub_epi32(c23, lg);
			__m256i hx_shl = _mm256_sllv_epi32(hx_norm, shift);
			__m256i iy_new = _mm256_sub_epi32(iy, shift);

			hx_norm = _mm256_blendv_epi8(hx_norm, hx_shl, need_norm);
			iy = _mm256_blendv_epi8(iy, iy_new, need_norm);
		}
	}

	__m256i normal_res_mask = _mm256_and_si256(active, _mm256_cmpgt_epi32(iy, neg127));
	__m256i normal_bits = _mm256_or_si256(
		_mm256_or_si256(
			_mm256_sub_epi32(hx_norm, min_norm),
			_mm256_slli_epi32(_mm256_add_epi32(iy, bias127), 23)
		),
		sx
	);

	__m256i sub_shift = _mm256_sub_epi32(neg126, iy);
	__m256i sub_bits = _mm256_or_si256(_mm256_srlv_epi32(hx_norm, sub_shift), sx);

	__m256 normal_ps = _mm256_castsi256_ps(normal_bits);
	__m256 sub_ps = _mm256_fmadd_ps(_mm256_castsi256_ps(sub_bits), one_ps, zero_ps);
	__m256 core_res = _mm256_blendv_ps(sub_ps, normal_ps, _mm256_castsi256_ps(normal_res_mask));

	__m256 signed_zero = _mm256_castsi256_ps(sx);
	__m256 nan_res = _mm256_div_ps(_mm256_fmadd_ps(x, y, zero_ps), _mm256_fmadd_ps(x, y, zero_ps));

	__m256 r = x;
	r = _mm256_blendv_ps(r, signed_zero, _mm256_castsi256_ps(zero_mask));
	r = _mm256_blendv_ps(r, core_res, _mm256_castsi256_ps(active));
	r = _mm256_blendv_ps(r, nan_res, _mm256_castsi256_ps(special));

	return r;
}

