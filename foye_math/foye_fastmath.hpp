/*
	Accuracy Validation Test Report

	1. Origin of Golden Reference Values

	The accuracy validation is based on golden reference values generated using the
	GNU MPFR multiple-precision floating-point library. The generation process is as follows:

	Input values are loaded into MPFR with 64 bits of significand precision, 
	which is sufficient to represent every binary32 input exactly.

	Reference values are generated at two levels:

	Rounded Reference

	The target mathematical function is evaluated with 24-bit precision (i.e., the effective
	significand width of IEEE 754 single-precision float) and rounded using Round-to-Nearest-Even.
	This value represents the correctly rounded single-precision result of an ideal infinite-
	precision computation, which is stronger than merely being faithfully rounded.

	High-Precision Reference

	To support accurate ULP error measurement, the system additionally computes the same
	function with adaptive precision. The initial working precision is 256 bits.
	If the current precision is insufficient to reliably round to the audit precision
	(56 bits = 24 effective bits + 32 guard bits), the working precision is doubled
	successively, up to a maximum of 262144 bits. The decision criterion is whether the
	result can be reliably rounded to the audit precision at the current working precision
	(as determined by MPFR's can_round mechanism). If the maximum precision is reached
	and the criterion still cannot be satisfied (typically occurs in extreme cases where
	the result is extremely close to the midpoint between two adjacent floating-point numbers),
	the system uses the current highest-precision result as a best-effort approximation
	and marks the ULP error for that point as inexact.

	For special results such as NaN and infinity, no adaptive refinement is needed; these cases are treated as exact directly.

	precision = 256
	loop:
		result = compute_function(input, precision, round_nearest)
		if can_reliably_round(result, precision, audit_precision=56): // In the implementation, MPFR's can_round test is applied with 
			mark_exact = true										  // a small safety margin relative to the current working precision.
			break
		if precision >= 262144:
			break  // best-effort
		precision = min(precision * 2, 262144)


	2. Description of Metrics

		2.1 ULP Error Calculation

			ULP (Unit in the Last Place) error measures the distance between the actual
			output of the function under test and the ideal correctly rounded result,
			expressed in units of the least precision at the location of the reference value.

			error = |actual_result - high_precision_reference| / ulp(rounded_reference)

			where ulp(rounded_reference) is defined as:
				* If the reference is zero: take the single-precision minimum subnormal number
				  (subnormal_min ≈ 1.4 × 10⁻⁴⁵)
				* If the reference is non-zero, ulp(rounded_reference) is defined as nextafterf(reference, +∞) - reference.
				  i.e., the spacing from the reference to the next representable floating-point number.

			The numerator uses the high-precision reference (rather than the 24-bit rounded reference)
			to avoid contaminating the metric with rounding errors.

			Special value handling:

				Condition									ULP Error
				Both reference and actual result are NaN	0
				Only one is NaN								+∞
				Both reference and actual result are zero	0
				Both are the same signed infinity			0
				Only one is infinity, or infinities differ in sign	+∞

			Accuracy guarantee:

				The ULP error value is accompanied by an exactness flag. When the high-precision
				reference passes the adaptive precision reliable rounding test, the ULP error is
				exact; otherwise it is a best-effort approximation. In practice, the ULP error is exact for most input points, 
				although the exact fraction is function-dependent.

		2.2 ULP Distance Calculation

			ULP distance is the absolute difference between the positions of two floating-point values in a monotonic ordering of IEEE 754 representations.
			Adjacent floating-point numbers have ULP distance 1.

			Computation method:

				Reinterpret the IEEE 754 binary representation of the floating-point number as
				an unsigned integer, then map it to a linear order via a sign-aware ordering:

					to_ordered(bits):
						if sign_bit is set:
							return bitwise_not(bits)      // negative: flip all bits
						else:
							return bits | sign_mask       // positive: flip the sign bit

					distance = |to_ordered(bitcast(a)) - to_ordered(bitcast(b))|

			For non-NaN floating-point values, reinterpret the IEEE 754 bit pattern as an
			unsigned integer and map it to a monotonic integer order via the sign-aware transform above.
			Under this ordering, adjacent representable floating-point values correspond to adjacent integers.

			NaNs are not included in this ordering and are handled separately.
			Signed zeros and infinities are also given explicit special-case rules in the metric.
			
			Special value handling:

				Both are NaN									0
				Only one is NaN									bit_cast<float>(UINT32_MAX)
				Both are zero (including ±0)					0
				Both are the same signed infinity				0
				Only one is infinity, or infinities differ in sign	bit_cast<float>(UINT32_MAX)

			Difference from ULP error:

				ULP distance is a discrete integer count, suitable for quickly deciding whether
				two floating-point results lie within N representable numbers of each other;
				ULP error is a continuous real-valued metric that can distinguish fine-grained
				differences such as "half an ULP" vs. "0.6 ULP". The two metrics complement
				each other.

		2.3 How to understand indicator 'inexact ulp err' ?

			inexact ulp err counts inputs for which ulp_error was not certified by the
			56-bit can_round audit. However, those values were still computed from a
			262144-bit MPFR approximation. Therefore, the additional uncertainty caused
			by this lack of certification is bounded by at most 2^(24-262144) = 2^-262120
			float ulp per affected input.

			Hence:
			- ulp-distance-based metrics are unaffected;
			- max ulp error may differ by at most 2^-262120;
			- mean ulp error may differ by at most (inexact_count / 2^32) * 2^-262120;
			- median ulp error may differ by at most 2^-262120.


	All single-precision functions with one input argument and one return value,
	whose domain is all real numbers, are validated against all IEEE-754 binary32 inputs covering every possible 2^32 bit pattern.

	For signatures that likewise take one input argument and return one result,
	but whose domain is not all real numbers, the validation range is additionally noted near the validation results

	In the per-function summary, “range: full” means that validation covers all 2^32 binary32 bit patterns,
	including NaNs, infinities, signed zeros, subnormals, and mathematically out-of-domain finite inputs.
*/

/*
* Test Platform:
* CPU:		AMD EPYC 9654 96-Core Processor (Zen 4)
* Memory:	256 GB DDR5-4800
* OS:		Ubuntu 22.04.1 LTS
* Kernel:	5.15.0-91-generic
*
* Build toolchain:
*   CMake	3.22.1
*   Clang	14.0.0 (Ubuntu clang version 14.0.0-1ubuntu1.1)
*   Target: x86_64-pc-linux-gnu
*
* Compiler flags:
*			-O3 -mavx2 -mfma -Wall -Wextra -fopenmp
*
* Reference libraries:
*			MPFR 4.1.0 (linked via pkg-config)
*			GMP 6.2.1
*
* Note: Performance tests use Release build type.
*/

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
	/*
		range:					   full
		count inexact ulp err:     2
		max ulp distance:		   1
		mean ulp distance:         0.0032518
		median ulp distance:       0
		max ulp error:		       1.08049
		mean ulp error:            0.0316884
		median ulp error:          1.52753e-30
		ratio ulp distance 1:      0.32518 %
		ratio ulp distance 2:      0 %
		ratio ulp distance 3:      0 %
		ratio ulp distance 4:      0 %
	*/

	__m256 exp(__m256) noexcept;
	/*
		range:					   full
		count inexact ulp err:     827231721
		max ulp distance:		   1
		mean ulp distance:		   0.00401994
		median ulp distance:	   0
		max ulp error:			   1.03921
		mean ulp error:			   0.0331128
		median ulp error:		   2.37515e-30
		ratio ulp distance 1:      0.401994 %
		ratio ulp distance 2:      0 %
		ratio ulp distance 3:      0 %
		ratio ulp distance 4:      0 %
	*/

	__m256 exp2(__m256) noexcept;
	/*
		range:					   full
		count inexact ulp err:     822083584
		max ulp distance:	       1
		mean ulp distance:         0.0126462
		median ulp distance:       0
		max ulp error:		       1.45211
		mean ulp error:            0.037994
		median ulp error:          2.37511e-30
		ratio ulp distance 1:      1.26462 %
		ratio ulp distance 2:      0 %
		ratio ulp distance 3:      0 %
		ratio ulp distance 4:      0 %
	*/

	__m256 exp10(__m256) noexcept;
	/*
		range:					   full
		count inexact ulp err:     0
		max ulp distance:		   2
		mean ulp distance:		   0.0159738
		median ulp distance:	   0
		max ulp error:			   1.6365
		mean ulp error:			   0.0400642
		median ulp error:		   2.37515e-30
		ratio ulp distance 1:      1.59561 %
		ratio ulp distance 2:      0.000882312 %
		ratio ulp distance 3:      0 %
		ratio ulp distance 4:      0 %
	*/

	__m256 log1p(__m256) noexcept;
	/*
		range:					   full
		count inexact ulp err:     2
		max ulp distance:		   1
		mean ulp distance:		   0.00264305
		median ulp distance:	   0
		max ulp error:			   1		   
		mean ulp error:			   0.0886416
		median ulp error:		   3.511e-13
		ratio ulp distance 1:      0.264305 %
		ratio ulp distance 2:      0 %
		ratio ulp distance 3:      0 %
		ratio ulp distance 4:      0 %
	*/

	__m256 log(__m256) noexcept;
	/*
		range:					   full
		count inexact ulp err:     1
		max ulp distance:		   1
		mean ulp distance:		   0.00315526
		median ulp distance:	   0
		max ulp error:			   0.995498
		mean ulp error:			   0.12467
		median ulp error:		   0
		ratio ulp distance 1:      0.315526 %
		ratio ulp distance 2:      0 %
		ratio ulp distance 3:      0 %
		ratio ulp distance 4:      0 %
	*/

	__m256 log2(__m256) noexcept;
	/*
		range:					   full
		count inexact ulp err:     1
		max ulp distance:		   2
		mean ulp distance:		   0.00222797
		median ulp distance:	   0
		max ulp error:			   1.80444
		mean ulp error:			   0.124871
		median ulp error:		   0
		ratio ulp distance 1:      0.222688 %
		ratio ulp distance 2:      5.43892e-05 %
		ratio ulp distance 3:      0 %
		ratio ulp distance 4:      0 %
	*/

	__m256 log10(__m256) noexcept;
	/*
		range:					   full
		count inexact ulp err:     0
		max ulp distance:		   2
		mean ulp distance:		   0.125065
		median ulp distance:	   0
		max ulp error:		       2.42497
		mean ulp error:			   0.166659
		median ulp error:		   0
		ratio ulp distance 1:      12.4975 %
		ratio ulp distance 2:      0.00454613 %
		ratio ulp distance 3:      0 %
		ratio ulp distance 4:      0 %
	*/

	__m256 sin(__m256) noexcept;
	/*
		range:					   full
		count inexact ulp err:     16777218
		max ulp distance:		   3
		mean ulp distance:		   0.0235913
		median ulp distance:	   0
		max ulp error:			   2.74637
		mean ulp error:			   0.145632
		median ulp error:		   0.0462421
		ratio ulp distance 1:      2.35905 %
		ratio ulp distance 2:      3.67407e-05 %
		ratio ulp distance 3:      4.65661e-08 %
		ratio ulp distance 4:      0 %
	*/

	__m256 cos(__m256) noexcept;
	/*
		range:					   full
		count inexact ulp err:     16777216
		max ulp distance:		   3
		mean ulp distance:		   0.0287955
		median ulp distance:	   0
		max ulp error:		       2.74637
		mean ulp error:			   0.147518
		median ulp error:		   0.0478886
		ratio ulp distance 1:      2.87944 %
		ratio ulp distance 2:      5.14556e-05 %
		ratio ulp distance 3:      4.65661e-08 %
		ratio ulp distance 4:      0 %
	*/

	__m256 tan(__m256) noexcept;
	/*
		range:					   full
		count inexact ulp err:     16777218
		max ulp distance:		   4
		mean ulp distance:		   0.129955
		median ulp distance:	   0
		max ulp error:			   4.03596
		mean ulp error:			   0.192904
		median ulp error:          0.0536609
		ratio ulp distance 1:      12.2825 %
		ratio ulp distance 2:      0.337169 %
		ratio ulp distance 3:      0.0127468 %
		ratio ulp distance 4:      9.92324e-05 %
	*/							   
								   
	__m256 sinh(__m256) noexcept;  
	/*							   
		range:					   full
		count inexact ulp err:     2
		max ulp distance:		   3
		mean ulp distance:		   0.0101207
		median ulp distance:	   0
		max ulp error:			   3.39017
		mean ulp error:			   0.0233672
		median ulp error:		   1.34893e-67
		ratio ulp distance 1:      0.878539 %
		ratio ulp distance 2:      0.0635935 %
		ratio ulp distance 3:      0.0021142 %
		ratio ulp distance 4:      0 %
	*/

	__m256 cosh(__m256) noexcept;
	/*
		range:					   full
		count inexact ulp err:     0
		max ulp distance:		   3
		mean ulp distance:		   0.00894142
		median ulp distance:	   0
		max ulp error:			   3.28334
		mean ulp error:			   0.0239164
		median ulp error:		   2.89637e-67
		ratio ulp distance 1:      0.705244 %
		ratio ulp distance 2:      0.0886112 %
		ratio ulp distance 3:      0.00389167 %
		ratio ulp distance 4:      0 %
	*/

	__m256 tanh(__m256) noexcept;
	/*
		range:					   full
		count inexact ulp err:     0
		max ulp distance:		   2
		mean ulp distance:		   0.00141781
		median ulp distance:	   0
		max ulp error:			   1.65245
		mean ulp error:			   0.0160543
		median ulp error:		   2.33419e-67
		ratio ulp distance 1:      0.141753 %
		ratio ulp distance 2:      1.41561e-05 %
		ratio ulp distance 3:      0 %
		ratio ulp distance 4:      0 %
	*/

	__m256 asin(__m256) noexcept;
	__m256 acos(__m256) noexcept;
	__m256 atan(__m256) noexcept;
	/*
		range:					   full
		count inexact ulp err:     0
		max ulp distance:		   2
		mean ulp distance:		   0.00219914
		median ulp distance:	   0
		max ulp error:			   1.86172
		mean ulp error:			   0.186043
		median ulp error:		   0.169574
		ratio ulp distance 1:      0.216039 %
		ratio ulp distance 2:      0.00193752 %
		ratio ulp distance 3:      0 %
		ratio ulp distance 4:      0 %
	*/

	__m256 asinh(__m256) noexcept;
	/*
		range:					   full
		count inexact ulp err:     0
		max ulp distance:		   2
		mean ulp distance:		   0.126256
		median ulp distance:	   0
		max ulp error:			   1.77808
		mean ulp error:			   0.176395
		median ulp error:		   0.0460967
		ratio ulp 1:			   12.6255
		ratio ulp 2:			   5.58794e-05
		ratio ulp 3:			   0
		ratio ulp 4:			   0
	*/

	__m256 acosh(__m256) noexcept;
	__m256 atanh(__m256) noexcept;

	__m256 cbrt(__m256) noexcept;
	/*
		range:					  full
		count inexact ulp err:    0
		max ulp distance:		  1
		mean ulp distance:		  0.0850614
		median ulp distance:	  0
		max ulp error:			  0.733448
		mean ulp error:			  0.259106
		median ulp error:		  0.24903
		ratio ulp distance 1:     8.50614 %
		ratio ulp distance 2:     0 %
		ratio ulp distance 3:     0 %
		ratio ulp distance 4:     0 %
	*/

	__m256 invcbrt(__m256) noexcept;
	/*
		range:					  full
		count inexact ulp err:    0
		max ulp distance:		  2
		mean ulp distance:		  0.356053
		median ulp distance:	  0
		max ulp error:            2.1701
		mean ulp error:           0.424954
		median ulp error:         0.362513
		ratio ulp 1:              34.8579 %
		ratio ulp 2:              0.373656 %
		ratio ulp 3:              0 %
		ratio ulp 4:              0 %
	*/

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