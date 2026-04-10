#ifndef FOYE_INTRINSIC_MATH_HPP
#define FOYE_INTRINSIC_MATH_HPP

#include <immintrin.h>
#include <cstdint>
#include <limits>
#include <bit>

#if 0
#define foyemath_experimental [[deprecated(													\
		"The mild random testing used during development has passed, "						\
		"but comprehensive accuracy and performance testing have not yet been conducted,"	\
		"and the code is still in intensive iterations"	)]]

#define foyemath_conditional [[deprecated(													\
		"At present, there is uncertainty in the optimization direction of the code. "		\
		"Since I have not determined the actual application scenario, "						\
		"the optimization direction is uncertain, "											\
		"so there may be serious performance issues,"										\
		" which depend on the distribution pattern of the input data when you use it")]]

#define foyemath_developing [[deprecated(													\
		"This feature is currently in the early stages of development "						\
		"and is not available for implementation at the moment. ")]]
#else
#define foyemath_experimental
#define foyemath_conditional
#define foyemath_developing
#endif

namespace fy::simd::intrinsic
{
	__m256 expm1(__m256) noexcept;
	__m256 exp(__m256) noexcept;
	__m256 exp2(__m256) noexcept;
	__m256 exp10(__m256) noexcept;

	__m256 log1p(__m256) noexcept;
	__m256 log(__m256) noexcept;
	__m256 log2(__m256) noexcept;
	__m256 log10(__m256) noexcept;

	__m256 sin(__m256) noexcept;
	__m256 cos(__m256) noexcept;
	__m256 tan(__m256) noexcept;

	__m256 sinh(__m256) noexcept;
	__m256 cosh(__m256) noexcept;
	__m256 tanh(__m256) noexcept;

	__m256 asin(__m256) noexcept;
	__m256 acos(__m256) noexcept;
	__m256 atan(__m256) noexcept;

	__m256 asinh(__m256) noexcept;
	__m256 acosh(__m256) noexcept;
	__m256 atanh(__m256) noexcept;



	foyemath_conditional __m256 cbrt(__m256) noexcept;
	foyemath_conditional __m256 invcbrt(__m256) noexcept;

	foyemath_conditional __m256 erf(__m256 input) noexcept;
	foyemath_conditional __m256 erfc(__m256 input) noexcept;

	foyemath_developing __m256 erfcx(__m256 input) noexcept;
	foyemath_developing __m256 erfinv(__m256 input) noexcept;
	foyemath_developing __m256 erfcinv(__m256 input) noexcept;
	foyemath_developing __m256 cdfnorm(__m256 input) noexcept;
	foyemath_developing __m256 cdfnorminv(__m256 input) noexcept;

	foyemath_experimental __m256 hypot(__m256 x, __m256 y) noexcept;
	foyemath_experimental __m256 pow(__m256 x, __m256 y) noexcept;
	foyemath_experimental __m256 fmod(__m256 x, __m256 y) noexcept;
	foyemath_experimental __m256 modf(__m256 x, __m256* intpart) noexcept;
	foyemath_experimental __m256 atan2(__m256 y, __m256 x) noexcept;

	void sincos(__m256 input, __m256* sin_result, __m256* cos_result) noexcept;
	void sinhcosh(__m256 input, __m256* sinh_result, __m256* cosh_result) noexcept;
	void asinacos(__m256 input, __m256* asin_result, __m256* acos_result) noexcept;
	void asinhacosh(__m256 input, __m256* asinh_res, __m256* acosh_res) noexcept;
}

namespace fy
{
	void atan2(std::size_t length,
		const float* __restrict memaddr_in_y, 
		const float* __restrict memaddr_in_x,
			  float* __restrict memaddr_out) noexcept;

	void expm1(std::size_t length, const float* memaddr_in, float* memaddr_out) noexcept;
	void exp(std::size_t length, const float* memaddr_in, float* memaddr_out) noexcept;
	void exp2(std::size_t length, const float* memaddr_in, float* memaddr_out) noexcept;
	void exp10(std::size_t length, const float* memaddr_in, float* memaddr_out) noexcept;

	void log1p(std::size_t length, const float* memaddr_in, float* memaddr_out) noexcept;
	void log(std::size_t length, const float* memaddr_in, float* memaddr_out) noexcept;
	void log2(std::size_t length, const float* memaddr_in, float* memaddr_out) noexcept;
	void log10(std::size_t length, const float* memaddr_in, float* memaddr_out) noexcept;

	void sin(std::size_t length, const float* memaddr_in, float* memaddr_out) noexcept;
	void cos(std::size_t length, const float* memaddr_in, float* memaddr_out) noexcept;
	void tan(std::size_t length, const float* memaddr_in, float* memaddr_out) noexcept;

	void sinh(std::size_t length, const float* memaddr_in, float* memaddr_out) noexcept;
	void cosh(std::size_t length, const float* memaddr_in, float* memaddr_out) noexcept;
	void tanh(std::size_t length, const float* memaddr_in, float* memaddr_out) noexcept;

	void asin(std::size_t length, const float* memaddr_in, float* memaddr_out) noexcept;
	void acos(std::size_t length, const float* memaddr_in, float* memaddr_out) noexcept;
	void atan(std::size_t length, const float* memaddr_in, float* memaddr_out) noexcept;

	void asinh(std::size_t length, const float* memaddr_in, float* memaddr_out) noexcept;
	void acosh(std::size_t length, const float* memaddr_in, float* memaddr_out) noexcept;
	void atanh(std::size_t length, const float* memaddr_in, float* memaddr_out) noexcept;

	void cbrt(std::size_t length, const float* memaddr_in, float* memaddr_out) noexcept;
	void invcbrt(std::size_t length, const float* memaddr_in, float* memaddr_out) noexcept;
}


#endif
