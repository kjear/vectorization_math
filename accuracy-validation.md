```md
# Accuracy Validation Test Report

This document describes the methodology used to validate the numerical accuracy 
of the library's single-precision floating-point mathematical functions.

---

## 1. Golden Reference Generation

The accuracy validation uses golden reference values generated with the **GNU MPFR** multiple-precision floating-point library.

Input values are loaded into MPFR with **64 bits of significand precision**, which is sufficient to represent every IEEE-754 binary32 input exactly.

Two kinds of reference values are generated:

---

### 1.1 Rounded Reference

The target mathematical function is evaluated with **24-bit precision** 
(i.e. the effective significand width of IEEE-754 single-precision `float`) using **round-to-nearest-even**.

This value is treated as the **correctly rounded binary32 result** of an ideal infinite-precision computation.

> This is a stronger requirement than merely being *faithfully rounded*.

---

### 1.2 High-Precision Reference

To support accurate ULP error measurement, the system also computes the same function using **adaptive precision**.

- Initial working precision: **256 bits**
- Audit precision: **56 bits**  
  (= 24 effective bits + 32 guard bits)
- Maximum working precision: **262144 bits**

At each step, the implementation checks whether the current result can 
be **reliably rounded** to the audit precision using MPFR's `can_round` mechanism.

If not, the working precision is doubled until either:

1. reliable rounding becomes possible, or
2. the maximum precision limit is reached.

If the maximum precision is reached and the result still cannot be certified for reliable rounding 
(which typically happens only in extreme cases where the exact result lies extraordinarily close to a midpoint between adjacent floating-point numbers), 
then the highest-precision result currently available is used as a **best-effort approximation**, 
and the corresponding ULP error is marked as **inexact**.

For special outputs such as **NaN** and **infinity**, no adaptive refinement is required; these cases are treated as exact directly.

---

### 1.3 Adaptive Refinement Procedure

```text
precision = 256
loop:
    result = compute_function(input, precision, round_nearest)
    if can_reliably_round(result, precision, audit_precision=56):
        mark_exact = true
        break
    if precision >= 262144:
        break   // best-effort
    precision = min(precision * 2, 262144)
```

> In the implementation, MPFR's `can_round` test is applied with a small safety margin relative to the current working precision.

---

## 2. Accuracy Metrics

Two complementary metrics are reported:

1. **ULP Error** — a continuous error measure
2. **ULP Distance** — a discrete representable-step distance

---

### 2.1 ULP Error

ULP (Unit in the Last Place) error measures the distance between the actual implementation result 
and the ideal mathematical result, expressed in units of local floating-point spacing.

The metric is defined as:

```text
error = |actual_result - high_precision_reference| / ulp(rounded_reference)
```

where:

- `actual_result` is the output produced by the function under test
- `high_precision_reference` is the adaptive-precision MPFR result
- `rounded_reference` is the correctly rounded binary32 reference

#### Definition of `ulp(rounded_reference)`

- If the reference is **zero**:

```text
ulp(0) = minimum positive binary32 subnormal
       ≈ 1.40129846 × 10^-45
```

- If the reference is **non-zero**:

```text
ulp(reference) = nextafterf(reference, +∞) - reference
```

That is, the ULP is the spacing from the reference value to the next representable single-precision number in the positive direction.

#### Why the numerator uses the high-precision reference

The numerator uses the **high-precision reference** instead of the 24-bit rounded reference so that the metric reflects the true mathematical error, 
rather than mixing in the rounding error of the reference itself.

---

### 2.2 Special Cases for ULP Error

| Condition | ULP Error |
|---|---:|
| Both reference and actual result are NaN | `0` |
| Only one is NaN | `+∞` |
| Both reference and actual result are zero | `0` |
| Both are the same signed infinity | `0` |
| Only one is infinity, or infinities differ in sign | `+∞` |

---

### 2.3 Exactness of ULP Error

Each ULP error value is accompanied by an **exactness flag**.

- If the adaptive-precision reference passes the 56-bit reliable-rounding audit, the reported ULP error is **exact**.
- Otherwise, it is reported as a **best-effort approximation**.

In practice, the ULP error is exact for most test points, although the exact fraction depends on the function.

---

## 3. ULP Distance

ULP distance is the absolute difference between the positions of two floating-point values in a monotonic ordering of IEEE-754 bit patterns.

Under this ordering, adjacent representable floating-point numbers have ULP distance **1**.

Unlike ULP error, ULP distance is a **discrete integer metric**.

---

### 3.1 Computation Method

The IEEE-754 binary representation is first reinterpreted as an unsigned integer, 
then mapped into a linear monotonic order using a sign-aware transformation:

```code
    to_ordered(bits):
        if sign_bit is set:
            return bitwise_not(bits)      // negative numbers
        else:
            return bits | sign_mask       // positive numbers

    distance = |to_ordered(bitcast(a)) - to_ordered(bitcast(b))|
```

For all non-NaN floating-point values, this mapping produces an integer ordering in which adjacent representable floats correspond to adjacent integers.

NaNs are excluded from this ordering and handled separately.

Signed zeros and infinities are also handled by explicit rules.

---

### 3.2 Special Cases for ULP Distance

| Condition | ULP Distance |
|---|---:|
| Both are NaN | `0` |
| Only one is NaN | `bit_cast<float>(UINT32_MAX)` |
| Both are zero (including `+0` and `-0`) | `0` |
| Both are the same signed infinity | `0` |
| Only one is infinity, or infinities differ in sign | `bit_cast<float>(UINT32_MAX)` |

> Note: NaNs are not part of the monotonic ordering, so they are treated explicitly outside the bit-order mapping.

---

## 4. Difference Between ULP Error and ULP Distance

Although their names are similar, the two metrics capture different aspects of numerical behavior:

### ULP Error
- Continuous, real-valued metric
- Measures how far the actual result is from the ideal mathematical value
- Can distinguish fine-grained differences such as:
  - `0.5 ulp`
  - `0.6 ulp`

### ULP Distance
- Discrete, integer metric
- Measures how many representable floating-point numbers lie between two values
- Useful for threshold-style checks such as:
  - “within 1 ULP”
  - “within 2 ULPs”

These two metrics complement each other.

---

## 5. Meaning of `inexact ulp err`

The indicator **`inexact ulp err`** counts input points for which the reported ULP error was **not certified** by the 56-bit `can_round` audit.

Even in such cases, the reference value is still computed using an MPFR approximation with up to **262144 bits** of precision. 
Therefore, the additional uncertainty introduced by the lack of certification is extremely small.

Its worst-case contribution is bounded by:

```text
2^(24 - 262144) = 2^-262120 float ulp
```

per affected input.

Therefore:

- **ULP-distance-based metrics are unaffected**
- **Maximum ULP error** may differ by at most `2^-262120`
- **Mean ULP error** may differ by at most

```text
(inexact_count / 2^32) * 2^-262120
```

- **Median ULP error** may differ by at most `2^-262120`

In other words, even when a ULP error is marked inexact, the remaining uncertainty is negligible for practical purposes.

---

## 6. Validation Coverage

All tested functions in this report are:

- **single-precision**
- take **one input argument**
- return **one result**

For functions whose mathematical domain is **all real numbers**, 
validation is performed over **all IEEE-754 binary32 inputs**, covering every possible `2^32` bit pattern.

This includes:

- NaNs
- infinities
- signed zeros
- subnormals
- all normal finite values
- mathematically out-of-domain finite inputs (when applicable to a “full bit pattern” sweep)

For functions that also have a single input and single output but whose domain is **not all real numbers**, 
the tested input range is stated explicitly near the corresponding validation results.

In per-function summaries:

```text
range: full
```

means the validation covers **all `2^32` binary32 bit patterns**.

---

## 7. Test Platform

### Hardware

- **CPU:** AMD EPYC 9654 96-Core Processor (Zen 4)
- **Memory:** 256 GB DDR5-4800

### Software Environment

- **OS:** Ubuntu 22.04.1 LTS
- **Kernel:** 5.15.0-91-generic

### Build Toolchain

- **CMake:** 3.22.1
- **Compiler:** Clang 14.0.0  
  (`Ubuntu clang version 14.0.0-1ubuntu1.1`)
- **Target:** `x86_64-pc-linux-gnu`

### Compiler Flags

```text
-O3 -mavx2 -mfma -Wall -Wextra -fopenmp
```

### Reference Libraries

- **MPFR:** 4.1.0 (linked via `pkg-config`)
- **GMP:** 6.2.1

> Note: performance tests are run using the `Release` build type.
