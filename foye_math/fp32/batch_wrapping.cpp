#include "foye_fastmath_fp32.hpp"

#include <cstring>

template<typename intrin_type, typename scalar_type,
	typename process_expr,
	typename loadmem_expr,
	typename storemem_expr>
static void intrinsic_1in_1out_dispatch_sequential(
	process_expr&& process, loadmem_expr&& load, storemem_expr&& store,
	std::size_t length,
	const scalar_type* __restrict memaddr_in,
	scalar_type* __restrict memaddr_out)
{
	constexpr std::size_t batch_size = (sizeof(intrin_type) / sizeof(scalar_type));
	std::size_t i = 0;
	for (; i + batch_size <= length; i += batch_size)
	{
		const intrin_type v_src = load(memaddr_in + i);
		const intrin_type v_res = process(v_src);
		store(memaddr_out + i, v_res);
	}

	const std::size_t remain = length - i;
	if (remain > 0)
	{
		alignas(alignof(intrin_type)) scalar_type temp[batch_size];
		std::memcpy(temp, memaddr_in + i, sizeof(scalar_type) * remain);

		const intrin_type v_src = load(temp);
		const intrin_type v_res = process(v_src);
		store(temp, v_res);

		std::memcpy(memaddr_out + i, temp, sizeof(scalar_type) * remain);
	}
}

#define DEFINE_1IN_1OUT_INSTANCE(function_name, vector_type, scalar_type)								\
void fy::function_name(	std::size_t length,																\
	const float* __restrict memaddr_in,																	\
		  float* __restrict memaddr_out) noexcept														\
{																										\
	intrinsic_1in_1out_dispatch_sequential<__m256, float>(												\
		[](__m256 x)					-> __m256 { return ::fy::simd::intrinsic::function_name(x); },	\
		[](const float* memaddr)		-> __m256 { return _mm256_loadu_ps(memaddr); },					\
		[](float* memaddr, __m256 vec)  -> void   { _mm256_storeu_ps(memaddr, vec); },					\
		length,																							\
		memaddr_in,																						\
		memaddr_out																						\
	);																									\
}

DEFINE_1IN_1OUT_INSTANCE(logb, __m256, float)
DEFINE_1IN_1OUT_INSTANCE(log1p, __m256, float)
DEFINE_1IN_1OUT_INSTANCE(log, __m256, float)
DEFINE_1IN_1OUT_INSTANCE(log2, __m256, float)
DEFINE_1IN_1OUT_INSTANCE(log10, __m256, float)

DEFINE_1IN_1OUT_INSTANCE(expm1, __m256, float)
DEFINE_1IN_1OUT_INSTANCE(exp, __m256, float)
DEFINE_1IN_1OUT_INSTANCE(exp2, __m256, float)
DEFINE_1IN_1OUT_INSTANCE(exp10, __m256, float)

DEFINE_1IN_1OUT_INSTANCE(sin, __m256, float)
DEFINE_1IN_1OUT_INSTANCE(cos, __m256, float)
DEFINE_1IN_1OUT_INSTANCE(tan, __m256, float)

DEFINE_1IN_1OUT_INSTANCE(sinh, __m256, float)
DEFINE_1IN_1OUT_INSTANCE(cosh, __m256, float)
DEFINE_1IN_1OUT_INSTANCE(tanh, __m256, float)

DEFINE_1IN_1OUT_INSTANCE(asin, __m256, float)
DEFINE_1IN_1OUT_INSTANCE(acos, __m256, float)
DEFINE_1IN_1OUT_INSTANCE(atan, __m256, float)

DEFINE_1IN_1OUT_INSTANCE(asinh, __m256, float)
DEFINE_1IN_1OUT_INSTANCE(acosh, __m256, float)
DEFINE_1IN_1OUT_INSTANCE(atanh, __m256, float)

DEFINE_1IN_1OUT_INSTANCE(erf, __m256, float)
DEFINE_1IN_1OUT_INSTANCE(erfc, __m256, float)
DEFINE_1IN_1OUT_INSTANCE(cbrt, __m256, float)
DEFINE_1IN_1OUT_INSTANCE(invcbrt, __m256, float)
