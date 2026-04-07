```markdown
# foye_fastmath

**一个专注于 x86-64 平台、基于 AVX2/FMA 指令集的轻量级向量化数学库。** 为需要高性能、高精度单精度浮点数学计算的场景而设计。

与标准数学库（如 glibc）的标量版本不同，本库所有函数均以 `__m256` 作为核心运算类型，并对每个函数进行了 **全域（2^32 个单精度浮点数）** 精度验证。同时提供了友好的 C++ 数组接口 `fy::*`，方便集成。

## ✨ 核心特性

-   **全向量化运算**：核心函数接收并返回 `__m256` 类型，一次处理 8 个单精度浮点数，充分利用 CPU 的 SIMD 指令集（AVX2, FMA）。
-   **经过全域精度验证**：每个已实现的核心函数都通过了 **全范围** （所有 2^32 个 IEEE-754 binary32 输入）的严格测试，包括 NaN、无穷大、有符号零、非规格化数等所有边界情况。
-   **高精度保证**：
    -   已开发完毕的所有函数的 **最大 ULP 误差** 通常在 1~4 ULPs 之间，**平均 ULP 误差** 远小于 0.5 ULP。
    -   精度验证基于 GNU MPFR 库，采用自适应精度（最高 262144 bits）参考值，确保了误差计算的可靠性。
-   **高性能实现**：
    -   核心算法使用精心优化的 Remez 多项式逼近。
    -   通过友好的分派（dispatch）逻辑，为不同范围的输入选择最优算法路径。
    -   大量使用 `_mm256_fmadd_ps` 等 FMA 指令，减少运算步骤并提升精度。
-   **易用的批量处理接口**：除了底层的 `__m256` 接口，还提供了以指针和长度为参数的 C++ 风格接口 `fy::exp(长度, 输入指针, 输出指针)`，自动处理剩余元素和非对齐内存，方便集成。
-   **开发状态透明**：通过 `foyemath_conditional`、`foyemath_experimental` 等宏，清晰标识各功能的稳定性状态。

## 📊 性能与精度概览

以下为部分函数在全域随机单精度浮点数测试中的精度表现（测试平台：AMD EPYC 9654）：

| 函数 | max ULP error | mean ULP error | ratio ULP distance ≤1 | count inexact ulp err |
| :--- | :--- | :--- | :--- | :--- |
| `exp` | 1.04 | 0.033 | 99.60% | 827,231,721 |
| `log` | 1.00 | 0.125 | 99.68% | 1 |
| `sin` | 2.75 | 0.146 | 97.64% | 16,777,218 |
| `cos` | 2.75 | 0.148 | 97.12% | 16,777,216 |
| `tan` | 4.04 | 0.193 | 87.26% | 16,777,218 |
| `asinh` | 1.78 | 0.176 | 99.87% | 0 |

> **说明**：
> *   **ULP error**：连续实数值误差，可区分“0.6 ULP”等精细差异。
> *   **ULP distance**：离散整数距离，表示两浮点数之间可表示的步数。
> *   `count inexact ulp err`：因结果过于接近舍入中点而无法完全认证的输入数量，但其附加不确定性极小（< 2^-262120 ULP），对整体精度无实质影响。
>
> 完整精度报告请参见各函数声明下方的注释。性能基准测试正在持续完善中。

## 🚀 快速开始

### 环境要求

-   **编译器**：支持 C++20 (使用了std::bit_cast) 及 AVX2/FMA 指令集的编译器。
-   **CPU**：支持 AVX2 和 FMA 的 x86-64 处理器（如 Intel Haswell 及后续，AMD Excavator 及后续）。
```

### 基本用法

#### 1. 向量化 SIMD 接口（`fy::simd::intrinsic`）
直接操作 `__m256` 类型，适用于密集的 SIMD 计算循环。

```cpp
#include "foye_fastmath.hpp"
#include <immintrin.h>

int main() 
{
    float data[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    __m256 x = _mm256_loadu_ps(data);

    // 计算 sin(x) 和 cos(x)
    __m256 s = fy::simd::intrinsic::sin(x);
    __m256 c = fy::simd::intrinsic::cos(x);

    // 或者同时计算
    __m256 s2, c2;
    fy::simd::intrinsic::sincos(x, &s2, &c2);

    // 存储结果
    float res_sin[8], res_cos[8];
    _mm256_storeu_ps(res_sin, s);
    _mm256_storeu_ps(res_cos, c);

    return 0;
}
```

#### 2. 批量数组接口（`fy::`）
处理任意长度的数组，自动进行向量化循环和剩余元素处理。

```cpp
#include "foye_fastmath.hpp"
#include <vector>

int main() {
    const std::size_t N = 1024;
    std::vector<float> in(N), out_exp(N), out_log(N);

    for (size_t i = 0; i < N; ++i) in[i] = i * 0.01f;

    fy::exp(N, in.data(), out_exp.data());

    fy::log(N, in.data(), out_log.data());

    // ... 使用结果

    return 0;
}
```

## 📖 已实现函数列表

### 基础运算
-   `exp`, `exp2`, `exp10`, `expm1`
-   `log`, `log2`, `log10`, `log1p`

### 三角函数
-   `sin`, `cos`, `tan`
-   `sincos` (同时计算)

### 双曲函数
-   `sinh`, `cosh`, `tanh`
-   `sinhcosh` (同时计算)

### 反三角函数
-   `asin`, `acos`, `atan`, `atan2`
-   `asinh`, `acosh`, `atanh`
-   `asinacos`, `asinhacosh` (成对计算)

### 幂函数与根（实验性）
-   `cbrt`, `invsqrt` (通过 `invcbrt` 展示模式), `hypot`, `pow`, `fmod`, `modf`

### 误差函数（实验性）
-   `erf`, `erfc`

## 🛠️ 开发状态与路线图

-   **核心开发**：已完成大部分单精度、单参数数学函数的实现、优化及全域精度验证。
-   **当前状态**：`foyemath_experimental` 标识的函数（如 `pow`, `fmod`, `atan2`, `erf`, `erfc`）需要更多测试和优化，请谨慎用于生产环境。

### 未来计划

-   [ ] **SSE 回退路径**：为仅支持 SSE 4.2 的旧 CPU 提供 AVX2 指令集的仿真或替代实现。
-   [ ] **AVX-512 支持**：优化并实现基于 `__m512` 的版本，进一步提升吞吐量。
-   [ ] **ARM NEON/SVE 支持**：通过统一的接口抽象，支持 ARM 平台的向量化指令集。
-   [ ] **双精度支持**：开发并验证 `__m256d`（和 `__m512d`）版本的核心数学函数。

## 🤝 贡献指南

本项目正处于积极开发中，非常欢迎您的参与！

-   **反馈问题**：请在 GitHub Issues 中详细描述问题，包括复现步骤、输入数据和预期行为。
-   **提出建议**：对于新的函数、优化点或平台支持，欢迎在 Issues 中讨论。
-   **提交代码**：
    1.  Fork 本仓库。
    2.  创建您的特性分支 (`git checkout -b feature/amazing-feature`)。
    3.  提交您的更改。
    4.  确保代码风格与现有代码一致，并通过所有测试。
    5.  推送到分支并创建一个 Pull Request。

## 📝 关于宏定义的说明

-   `foyemath_conditional`：标识优化策略依赖于输入数据分布，在特定场景下可能存在性能问题的函数。
-   `foyemath_experimental`：标识尚处于实验阶段，未经过完整精度和性能测试的函数。
-   `foyemath_developing`：标识正在开发中，尚未就绪的功能。

这些宏当前默认为空，未来在发布稳定版本时会启用为 `[[deprecated]]` 警告。

## 📄 许可证

本项目基于 **MIT 许可证** 开源。详情请见 [LICENSE](LICENSE) 文件。
```
