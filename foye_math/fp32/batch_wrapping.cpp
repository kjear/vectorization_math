#include <foye_fastmath.hpp>

#include <cstring>
#include <type_traits>

template<typename intrin_type, typename scalar_type,
	typename process_expr,
	typename loadmem_expr,
	typename storemem_expr>
static void intrinsic_1in_1out_dispatch_forward(
	process_expr&& process,
	loadmem_expr&& load,
	storemem_expr&& store,
	std::size_t length,
	const scalar_type* memaddr_in,
	scalar_type* memaddr_out)
{
	constexpr std::size_t batch_size = sizeof(intrin_type) / sizeof(scalar_type);

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
		alignas(alignof(intrin_type)) scalar_type temp[batch_size] = {};
		std::memcpy(temp, memaddr_in + i, remain * sizeof(scalar_type));

		const intrin_type v_src = load(temp);
		const intrin_type v_res = process(v_src);
		store(temp, v_res);

		std::memcpy(memaddr_out + i, temp, remain * sizeof(scalar_type));
	}
}


template<typename intrin_type, typename scalar_type,
	typename process_expr,
	typename loadmem_expr,
	typename storemem_expr>
static void intrinsic_1in_1out_dispatch_backward(
	process_expr&& process,
	loadmem_expr&& load,
	storemem_expr&& store,
	std::size_t length,
	const scalar_type* memaddr_in,
	scalar_type* memaddr_out)
{
	constexpr std::size_t batch_size = sizeof(intrin_type) / sizeof(scalar_type);

	std::size_t i = length;

	while (i >= batch_size)
	{
		i -= batch_size;
		const intrin_type v_src = load(memaddr_in + i);
		const intrin_type v_res = process(v_src);
		store(memaddr_out + i, v_res);
	}

	if (i > 0)
	{
		alignas(alignof(intrin_type)) scalar_type temp[batch_size] = {};
		std::memcpy(temp, memaddr_in, i * sizeof(scalar_type));

		const intrin_type v_src = load(temp);
		const intrin_type v_res = process(v_src);
		store(temp, v_res);

		std::memcpy(memaddr_out, temp, i * sizeof(scalar_type));
	}
}

template<typename intrin_type, typename scalar_type,
	typename process_expr,
	typename loadmem_expr,
	typename storemem_expr>
static void intrinsic_1in_1out_dispatch_sequential(
	process_expr&& process,
	loadmem_expr&& load,
	storemem_expr&& store,
	std::size_t length,
	const scalar_type* memaddr_in,
	scalar_type* memaddr_out)
{
	if (length == 0)
		return;

	const auto in_begin = reinterpret_cast<std::uintptr_t>(memaddr_in);
	const auto in_end = in_begin + length * sizeof(scalar_type);
	const auto out_begin = reinterpret_cast<std::uintptr_t>(memaddr_out);
	const auto out_end = out_begin + length * sizeof(scalar_type);

	const bool overlap = !(out_end <= in_begin || in_end <= out_begin);

	if (!overlap || out_begin <= in_begin)
	{
		intrinsic_1in_1out_dispatch_forward<intrin_type, scalar_type>(
			std::forward<process_expr>(process),
			std::forward<loadmem_expr>(load),
			std::forward<storemem_expr>(store),
			length,
			memaddr_in,
			memaddr_out);
	}
	else
	{
		intrinsic_1in_1out_dispatch_backward<intrin_type, scalar_type>(
			std::forward<process_expr>(process),
			std::forward<loadmem_expr>(load),
			std::forward<storemem_expr>(store),
			length,
			memaddr_in,
			memaddr_out);
	}
}

#define DEFINE_1IN_1OUT_INSTANCE_IMPL(function_name, vector_type, scalar_type, load, store, process)		\
void function_name(std::size_t length,																		\
    const scalar_type* memaddr_in,																			\
          scalar_type* memaddr_out) noexcept																\
{																											\
    intrinsic_1in_1out_dispatch_sequential<vector_type, scalar_type>(										\
        [](vector_type x) -> vector_type { return process(x); },											\
        [](const scalar_type* memaddr) -> vector_type { return load(memaddr); },							\
        [](scalar_type* memaddr, vector_type vec) -> void { store(memaddr, vec); },							\
        length,																								\
        memaddr_in,																							\
        memaddr_out																							\
    );																										\
}

DEFINE_1IN_1OUT_INSTANCE_IMPL(fy::expm1, __m256, float, _mm256_loadu_ps, _mm256_storeu_ps, fy::simd::intrinsic::expm1)
DEFINE_1IN_1OUT_INSTANCE_IMPL(fy::exp,	 __m256, float, _mm256_loadu_ps, _mm256_storeu_ps, fy::simd::intrinsic::exp)
DEFINE_1IN_1OUT_INSTANCE_IMPL(fy::exp2,  __m256, float, _mm256_loadu_ps, _mm256_storeu_ps, fy::simd::intrinsic::exp2)
DEFINE_1IN_1OUT_INSTANCE_IMPL(fy::exp10, __m256, float, _mm256_loadu_ps, _mm256_storeu_ps, fy::simd::intrinsic::exp10)

DEFINE_1IN_1OUT_INSTANCE_IMPL(fy::log1p, __m256, float, _mm256_loadu_ps, _mm256_storeu_ps, fy::simd::intrinsic::log1p)
DEFINE_1IN_1OUT_INSTANCE_IMPL(fy::log, __m256, float, _mm256_loadu_ps, _mm256_storeu_ps, fy::simd::intrinsic::log)
DEFINE_1IN_1OUT_INSTANCE_IMPL(fy::log2, __m256, float, _mm256_loadu_ps, _mm256_storeu_ps, fy::simd::intrinsic::log2)
DEFINE_1IN_1OUT_INSTANCE_IMPL(fy::log10, __m256, float, _mm256_loadu_ps, _mm256_storeu_ps, fy::simd::intrinsic::log10)

DEFINE_1IN_1OUT_INSTANCE_IMPL(fy::sin, __m256, float, _mm256_loadu_ps, _mm256_storeu_ps, fy::simd::intrinsic::sin)
DEFINE_1IN_1OUT_INSTANCE_IMPL(fy::cos, __m256, float, _mm256_loadu_ps, _mm256_storeu_ps, fy::simd::intrinsic::cos)
DEFINE_1IN_1OUT_INSTANCE_IMPL(fy::tan, __m256, float, _mm256_loadu_ps, _mm256_storeu_ps, fy::simd::intrinsic::tan)

DEFINE_1IN_1OUT_INSTANCE_IMPL(fy::sinh, __m256, float, _mm256_loadu_ps, _mm256_storeu_ps, fy::simd::intrinsic::sinh)
DEFINE_1IN_1OUT_INSTANCE_IMPL(fy::cosh, __m256, float, _mm256_loadu_ps, _mm256_storeu_ps, fy::simd::intrinsic::cosh)
DEFINE_1IN_1OUT_INSTANCE_IMPL(fy::tanh, __m256, float, _mm256_loadu_ps, _mm256_storeu_ps, fy::simd::intrinsic::tanh)

DEFINE_1IN_1OUT_INSTANCE_IMPL(fy::asin, __m256, float, _mm256_loadu_ps, _mm256_storeu_ps, fy::simd::intrinsic::asin)
DEFINE_1IN_1OUT_INSTANCE_IMPL(fy::acos, __m256, float, _mm256_loadu_ps, _mm256_storeu_ps, fy::simd::intrinsic::acos)
DEFINE_1IN_1OUT_INSTANCE_IMPL(fy::atan, __m256, float, _mm256_loadu_ps, _mm256_storeu_ps, fy::simd::intrinsic::atan)

DEFINE_1IN_1OUT_INSTANCE_IMPL(fy::asinh, __m256, float, _mm256_loadu_ps, _mm256_storeu_ps, fy::simd::intrinsic::asinh)
DEFINE_1IN_1OUT_INSTANCE_IMPL(fy::acosh, __m256, float, _mm256_loadu_ps, _mm256_storeu_ps, fy::simd::intrinsic::acosh)
DEFINE_1IN_1OUT_INSTANCE_IMPL(fy::atanh, __m256, float, _mm256_loadu_ps, _mm256_storeu_ps, fy::simd::intrinsic::atanh)

DEFINE_1IN_1OUT_INSTANCE_IMPL(fy::cbrt, __m256, float, _mm256_loadu_ps, _mm256_storeu_ps, fy::simd::intrinsic::cbrt)
DEFINE_1IN_1OUT_INSTANCE_IMPL(fy::invcbrt, __m256, float, _mm256_loadu_ps, _mm256_storeu_ps, fy::simd::intrinsic::invcbrt)


template<typename intrin_type, typename scalar_type,
	typename process_expr,
	typename loadmem_expr,
	typename storemem_expr>
static void intrinsic_2in_1out_dispatch_sequential(
	process_expr&& process, loadmem_expr&& load, storemem_expr&& store,
	std::size_t length,
	const scalar_type* __restrict memaddr_in_arg0,
	const scalar_type* __restrict memaddr_in_arg1,
		  scalar_type* __restrict memaddr_out) noexcept
{
	constexpr std::size_t batch_size = (sizeof(intrin_type) / sizeof(scalar_type));
	std::size_t i = 0;
	for (; i + batch_size <= length; i += batch_size)
	{
		const intrin_type varg0 = load(memaddr_in_arg0 + i);
		const intrin_type varg1 = load(memaddr_in_arg1 + i);
		const intrin_type v_res = process(varg0, varg1);
		store(memaddr_out + i, v_res);
	}

	const std::size_t remain = length - i;
	if (remain > 0)
	{
		alignas(alignof(intrin_type)) scalar_type temp[batch_size * 2];
		std::memcpy(temp,			   memaddr_in_arg0 + i, sizeof(scalar_type) * remain);
		std::memcpy(temp + batch_size, memaddr_in_arg1 + i, sizeof(scalar_type) * remain);

		const intrin_type varg0 = load(temp);
		const intrin_type varg1 = load(temp + batch_size);

		const intrin_type v_res = process(varg0, varg1);
		store(temp, v_res);

		std::memcpy(memaddr_out + i, temp, sizeof(scalar_type) * remain);
	}
}

#define DEFINE2IN_1OUT_INSTANCE(function_name, vector_type, scalar_type)									\
void fy::function_name(	std::size_t length,																	\
	const scalar_type* __restrict memaddr_in_arg0,															\
	const scalar_type* __restrict memaddr_in_arg1,															\
		  scalar_type* __restrict memaddr_out) noexcept														\
{																											\
	intrinsic_2in_1out_dispatch_sequential<__m256, float>(													\
		[](__m256 a0, __m256 a1)	   -> __m256 { return ::fy::simd::intrinsic::function_name(a0, a1); },	\
		[](const float* memaddr)	   -> __m256 { return _mm256_loadu_ps(memaddr); },						\
		[](float* memaddr, __m256 vec) -> void   { _mm256_storeu_ps(memaddr, vec); },						\
		length,																								\
		memaddr_in_arg0,																					\
		memaddr_in_arg1,																					\
		memaddr_out																							\
	);																										\
}


DEFINE2IN_1OUT_INSTANCE(atan2, __m256, float)







