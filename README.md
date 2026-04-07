```markdown
# foye_fastmath
**A lightweight vectorized math library for x86-64, built on top of the AVX2/FMA instruction set.** It is designed for high-performance, high-accuracy single-precision floating-point math workloads.

Unlike scalar implementations in standard math libraries (such as glibc), all core functions in this library operate on `__m256` vectors and have been **validated for accuracy over the entire float domain (all 2^32 single-precision values)**. It also provides convenient C++ array-based interfaces under `fy::*` for easier integration.

## ✨ Key Features

- **Fully vectorized computation**: Core functions take and return `__m256`, processing 8 single-precision floating-point values at once and making full use of SIMD instructions (AVX2, FMA).
- **Exhaustive accuracy validation**: Every completed core function has been rigorously tested over the **entire IEEE-754 binary32 input space**, including NaNs, infinities, signed zeros, subnormals, and all edge cases.
- **High accuracy guarantees**:
  - For all completed functions, the **maximum ULP error** is typically within 1–4 ULPs, while the **mean ULP error** is far below 0.5 ULP.
  - Accuracy validation is based on GNU MPFR with adaptive precision (up to 262144 bits), ensuring reliable error measurement.
- **High-performance implementation**:
  - Core algorithms use carefully optimized Remez polynomial approximations.
  - Friendly dispatch logic selects the best algorithmic path for different input ranges.
  - Extensive use of FMA instructions such as `_mm256_fmadd_ps` reduces instruction count and improves accuracy.
- **Easy-to-use batch interfaces**: In addition to the low-level `__m256` interface, the library provides C++-style pointer/length interfaces such as `fy::exp(len, in, out)`, which automatically handle tail elements and unaligned memory.
- **Transparent development status**: Macros such as `foyemath_conditional` and `foyemath_experimental` clearly indicate the maturity and stability level of features.

## 📊 Accuracy Overview
[accuracy-validation.md][accuracy-validation.md]
Below is a snapshot of accuracy results for several functions under exhaustive single-precision testing (test platform: AMD EPYC 9654):

| Function | max ULP error | mean ULP error | ratio ULP distance ≤1 | count inexact ulp err |
| :--- | :--- | :--- | :--- | :--- |
| `exp` | 1.04 | 0.033 | 99.60% | 827,231,721 |
| `log` | 1.00 | 0.125 | 99.68% | 1 |
| `sin` | 2.75 | 0.146 | 97.64% | 16,777,218 |
| `cos` | 2.75 | 0.148 | 97.12% | 16,777,216 |
| `tan` | 4.04 | 0.193 | 87.26% | 16,777,218 |
| `asinh` | 1.78 | 0.176 | 99.87% | 0 |

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

### Basic Usage

#### 1. Vectorized SIMD interface (`fy::simd::intrinsic`)
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

#### 2. Batch array interface (`fy::`)
Process arrays of arbitrary length with automatic vectorized loops and tail handling.

```cpp
#include "foye_fastmath.hpp"
#include <vector>

int main() 
{
    const std::size_t N = 1024;
    std::vector<float> in(N), out_exp(N), out_log(N);

    for (size_t i = 0; i < N; ++i) in[i] = i * 0.01f;

    fy::exp(N, in.data(), out_exp.data());
    fy::log(N, in.data(), out_log.data());

    // ... use results

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
