# 自动向量化与基于 intrinsic 的手动向量化 实验报告

2024010779 刘家豪

## 运行时间

| baseline | auto simd | instrinsic |
| --- | --- | --- |
| 4441 us | 526 us | 527 us |

## `a_plus_b_intrinsic` 函数的实现代码

```cpp
void a_plus_b_intrinsic(float* a, float* b, float* c, int n) {
    size_t i;
    for (i = 0; i < n; i += 8) {
        __m256 va = _mm256_load_ps(&a[i]);
        __m256 vb = _mm256_load_ps(&b[i]);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_store_ps(&c[i], vc);
    }
}
```