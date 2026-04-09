# foye_fastmath
**A lightweight vectorized libm-style library for x86-64, built on top of AVX2/FMA and AVX-512 intrinsics.** The implementation prioritizes throughput while maintaining a level of accuracy suitable for practical numerical use.

Its core API operates directly on native SIMD vector types such as `__m256` `__m512` vectors and have been **validated for accuracy over the entire IEEE754 binary32 domain (all 2^32 single-precision values)**.

### 🔌 Zero Dependencies & Self-Contained Functions

Every function in this library is implemented as a **standalone, self-contained unit**. 
There are **no internal cross-dependencies** between different mathematical kernels. 

**Trade-off Notice:** This design deliberately leads to duplication of constant tables and boilerplate code across compilation units. 
**Benefit:** You are free to cherry-pick and copy *only the specific `.cpp` files you need* into your project without pulling in the rest of the library. No linker gymnastics, no hidden internal calls—just drop the file and compile.

## ✨ Key Features

- **Exhaustive accuracy validation**: Every completed core function has been rigorously tested over the **entire IEEE-754 binary32 input space**, including NaNs, infinities, signed zeros, subnormals, and all edge cases.
- **High accuracy guarantees**:
  - For all completed functions, the **maximum ULP error** is typically within 1–4 ULPs (Most are below 2), while the **mean ULP error** is far below 0.5 ULP.
  - Accuracy validation is based on GNU MPFR with adaptive precision (up to 262144 bits), ensuring reliable error measurement.
- **High-performance implementation**:
  - Core algorithms use carefully optimized Remez polynomial approximations.
  - Friendly dispatch logic selects the best algorithmic path for different input ranges.
- **Transparent development status**: Macros such as `foyemath_conditional` and `foyemath_experimental` clearly indicate the maturity and stability level of features.

## 📊 Accuracy Overview

For detailed verification logic and interpretation of various indicators in the verification results, please refer to [accuracy-validation.md](accuracy-validation.md)
For detailed error reports, please refer to [accuracy-validation.md](errors.md)

###  Test Environment

- **CPU:** AMD EPYC 9654 96-Core Processor (Zen 4)
- **Memory:** 256 GB DDR5-4800
- **OS:** Ubuntu 22.04.1 LTS
- **Kernel:** 5.15.0-91-generic

###  Build Toolchain

- **CMake:** 3.22.1
- **Compiler:** Clang 14.0.0 (`Ubuntu clang version 14.0.0-1ubuntu1.1`)
- **Target:** `x86_64-pc-linux-gnu`

###  Compiler Flags

- `-O3 -mavx2 -mfma`

###  Reference Libraries

- **MPFR:** 4.1.0 (linked via `pkg-config`)
- **GMP:** 6.2.1

> **Note**
> - Performance tests use the `Release` build type.

> **Notes**
> - **ULP error**: Continuous real-valued error, capable of distinguishing finer differences such as "0.6 ULP".
> - **ULP distance**: Discrete integer distance, i.e. the number of representable floating-point steps between two results.
> - `count inexact ulp err`: Number of inputs for which the exact ULP error cannot be fully certified because the result lies extremely close to a rounding midpoint. The additional uncertainty is negligible (< 2^-262120 ULP) and has no practical impact on the reported accuracy.
>
> For complete accuracy reports, please refer to the comments below each function declaration. Performance benchmarks are still being expanded and refined.

## 🚀 Quick Start

### Requirements

- **Compiler**: A compiler with support for C++20 (`std::bit_cast` is used) and AVX2/FMA.
- **CPU**: An x86-64 processor with AVX2 and FMA support (e.g. Intel Haswell and newer, AMD Excavator and newer).

### Usage

#### Vectorized SIMD interface (`fy::simd::intrinsic`)

Operate directly on `__m256`, suitable for dense SIMD compute loops.

```cpp
#include "foye_fastmath.hpp"
#include <immintrin.h>

int main()
{
    float data[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    __m256 x = _mm256_loadu_ps(data);

    // Compute sin(x) and cos(x)
    __m256 s = fy::simd::intrinsic::sin(x);
    __m256 c = fy::simd::intrinsic::cos(x);

    // Or compute both at once
    __m256 s2, c2;
    fy::simd::intrinsic::sincos(x, &s2, &c2);

    // Store results
    float res_sin[8], res_cos[8];
    _mm256_storeu_ps(res_sin, s);
    _mm256_storeu_ps(res_cos, c);

    return 0;
}
```

## 📖 Implemented Functions

### Elementary functions

- `exp`, `exp2`, `exp10`, `expm1`
- `log`, `log2`, `log10`, `log1p`

### Trigonometric functions

- `sin`, `cos`, `tan`
- `sincos` (compute both simultaneously)

### Hyperbolic functions

- `sinh`, `cosh`, `tanh`
- `sinhcosh` (compute both simultaneously)

### Inverse trigonometric / hyperbolic functions

- `asin`, `acos`, `atan`, `atan2`
- `asinh`, `acosh`, `atanh`
- `asinacos`, `asinhacosh` (paired computation)

### Powers and roots (experimental)

- `cbrt`, `invsqrt` (pattern demonstrated via `invcbrt`), `hypot`, `pow`, `fmod`, `modf`

### Error functions (experimental)

- `erf`, `erfc`

## 🛠️ Development Status and Roadmap

- **Core development**: Most single-precision unary math functions have been implemented, optimized, and exhaustively validated for accuracy.
- **Current status**: Functions marked with `foyemath_experimental` (such as `pow`, `fmod`, `atan2`, `erf`, `erfc`) still require more testing and optimization. Use them with caution in production.

### Future Plans

- [ ] **quick version**: Relax certain requirements for computational accuracy, ignore some boundary conditions, to switch to higher performance algorithms implementation.
- [ ] **SSE fallback path**: Provide a fallback or substitute implementation for older CPUs with only SSE4.2 support.
- [ ] **AVX-512 support**: Optimize and add `__m512`-based implementations for higher throughput.
- [ ] **ARM NEON/SVE support**: Introduce a unified abstraction layer to support ARM vector instruction sets.
- [ ] **Double-precision support**: Develop and validate `__m256d` (and `__m512d`) versions of the core math functions.

## 🤝 Contributing

This project is under active development, and contributions are very welcome.

- **Report issues**: Please open a GitHub Issue with detailed reproduction steps, input data, and expected behavior.
- **Suggest improvements**: New functions, optimization ideas, and platform support proposals are welcome in Issues.
- **Submit code**:
  1. Fork this repository.
  2. Create your feature branch (`git checkout -b feature/amazing-feature`).
  3. Commit your changes.
  4. Make sure your code follows the existing style and passes all tests.
  5. Push the branch and open a Pull Request.

## 📝 Notes on Macros

- `foyemath_conditional`: Marks functions whose optimization strategy depends on the input distribution and may have performance drawbacks in certain scenarios.
- `foyemath_experimental`: Marks functions that are still experimental and have not yet completed full accuracy/performance validation.
- `foyemath_developing`: Marks features that are currently under development and not ready for use.

These macros are currently defined as empty. In future stable releases, they may become `[[deprecated]]` annotations to warn users about maturity or stability concerns.

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
