// Copyright (C) 2026.
// Gcd123 CPU 参考实现 + 测试。
//
// 作用：
//  - 用纯 C++ 实现与 op_kernel/gcd123.cpp 相同的语义（broadcast + GCD），
//    作为 NPU 运行结果的 golden 参照。
//  - 内置若干断言，独立验证 broadcast 形状推导与 GCD 数值正确性。
//
// 编译/运行（任意平台，无需 CANN）：
//     g++ -std=c++17 -O2 gcd123_cpu_test.cpp -o gcd123_cpu_test
//     ./gcd123_cpu_test
//
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace gcd123ref {

using Shape = std::vector<int64_t>;

// 与 kernel 的 HalfToI32 对应：float16 先截断取整（向负无穷）。
// 此处用 float 模拟 half 可表示的值（测试输入均为 half 可精确表示的整数/小数）。
inline int32_t HalfToI32(float h) { return static_cast<int32_t>(std::floor(h)); }

inline int32_t AbsI32(int32_t v) { return v < 0 ? -v : v; }

// 与 kernel 的 GcdI32 对应：欧几里得辗转相除（输入取绝对值）。
inline int32_t GcdI32(int32_t a, int32_t b) {
    a = AbsI32(a);
    b = AbsI32(b);
    while (b != 0) {
        int32_t r = a % b;
        a = b;
        b = r;
    }
    return a;
}

// 右对齐 broadcast 形状推导；失败返回 false。
bool BroadcastShape(const Shape &a, const Shape &b, Shape &out) {
    const int dimsA = static_cast<int>(a.size());
    const int dimsB = static_cast<int>(b.size());
    const int dims = std::max(dimsA, dimsB);
    out.assign(dims, 1);
    const int offA = dims - dimsA;
    const int offB = dims - dimsB;
    for (int i = 0; i < dims; ++i) {
        const int64_t da = (i < offA) ? 1 : a[i - offA];
        const int64_t db = (i < offB) ? 1 : b[i - offB];
        if (da != db && da != 1 && db != 1) {
            return false;
        }
        out[i] = std::max(da, db);
    }
    return true;
}

// 给定输出坐标 outCoord（长度 dims），计算输入 src（形状 srcShape，右对齐）的扁平下标。
int64_t FlatIndex(const std::vector<int64_t> &outCoord, const Shape &srcShape) {
    const int dims = static_cast<int>(outCoord.size());
    const int dimsA = static_cast<int>(srcShape.size());
    const int offA = dims - dimsA;
    int64_t idx = 0;
    for (int j = 0; j < dimsA; ++j) {
        const int outDim = offA + j;
        const int64_t c = (srcShape[j] == 1) ? 0 : outCoord[outDim];
        int64_t stride = 1;
        for (int k = j + 1; k < dimsA; ++k) {
            stride *= srcShape[k];
        }
        idx += c * stride;
    }
    return idx;
}

// 与 kernel 语义一致的 golden：out = gcd(broadcast(self, other))，结果为整数值。
std::vector<float> Golden(const std::vector<float> &self, const Shape &selfShape,
                          const std::vector<float> &other, const Shape &otherShape, Shape &outShape) {
    if (!BroadcastShape(selfShape, otherShape, outShape)) {
        return {};
    }
    const int dims = static_cast<int>(outShape.size());
    int64_t total = 1;
    for (int64_t d : outShape) {
        total *= d;
    }
    std::vector<int64_t> stride(dims, 1);
    for (int i = dims - 2; i >= 0; --i) {
        stride[i] = stride[i + 1] * outShape[i + 1];
    }
    std::vector<float> out(static_cast<size_t>(total));
    for (int64_t flat = 0; flat < total; ++flat) {
        std::vector<int64_t> coord(dims);
        int64_t rem = flat;
        for (int i = 0; i < dims; ++i) {
            coord[i] = rem / stride[i];
            rem %= stride[i];
        }
        const int64_t si = FlatIndex(coord, selfShape);
        const int64_t oi = FlatIndex(coord, otherShape);
        out[static_cast<size_t>(flat)] =
            static_cast<float>(GcdI32(HalfToI32(self[static_cast<size_t>(si)]),
                                      HalfToI32(other[static_cast<size_t>(oi)])));
    }
    return out;
}

} // namespace gcd123ref

namespace {
int g_failures = 0;

void Check(bool ok, const char *what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        ++g_failures;
    }
}

// 用例1：同形状，无 broadcast。
void CaseSameShape() {
    std::printf("Case1 same-shape [2,1,2,4]:\n");
    using S = gcd123ref::Shape;
    const S sh = {2, 1, 2, 4};
    // 手工构造：self 与 other 均为已知整数
    std::vector<float> self = {12, 18, 8, 10, 7, 21, 6, 35};
    std::vector<float> other = {4, 12, 6, 15, 14, 7, 9, 14};
    S outShape;
    auto out = gcd123ref::Golden(self, sh, other, sh, outShape);
    Check(outShape == S({2, 1, 2, 4}), "out shape == [2,1,2,4]");
    // 期望 gcd 逐项
    const std::vector<float> expect = {4, 6, 2, 5, 7, 7, 3, 7};
    bool allEq = (out == expect);
    if (!allEq) {
        for (size_t i = 0; i < out.size(); ++i) {
            std::printf("    idx %zu got %.0f expect %.0f\n", i, out[i], expect[i]);
        }
    }
    Check(allEq, "gcd values correct");
}

// 用例2：N1 维 broadcast（other 的 N1=1）。
void CaseBroadcastInner() {
    std::printf("Case2 broadcast on N1, self[2,2,2,4] other[2,2,2,1]:\n");
    using S = gcd123ref::Shape;
    const S sh = {2, 2, 2, 4};
    const S oh = {2, 2, 2, 1};
    std::vector<float> self = {12, 8, 4, 6, 9, 6, 3, 12, 10, 15, 5, 20, 14, 21, 7, 35};
    std::vector<float> other = {4, 12, 6, 3}; // 每个“行”一个值
    S outShape;
    auto out = gcd123ref::Golden(self, sh, other, oh, outShape);
    Check(outShape == sh, "out shape == self shape [2,2,2,4]");
    // 第 0 行 other=4: gcd(12,4)=4, gcd(8,4)=4, gcd(4,4)=4, gcd(6,4)=2
    Check(out[0] == 4.0f && out[1] == 4.0f && out[2] == 4.0f && out[3] == 2.0f, "row0 gcd correct");
    // 第 1 行 other=12: gcd(9,12)=3, gcd(6,12)=6, gcd(3,12)=3, gcd(12,12)=12
    Check(out[4] == 3.0f && out[5] == 6.0f && out[6] == 3.0f && out[7] == 12.0f, "row1 gcd correct");
}

// 用例3：self 为“标量”[1,1,1,1]，other 任意 -> 输出 = gcd(标量, other)。
void CaseScalarSelf() {
    std::printf("Case3 scalar self[1,1,1,1] vs other[1,2,3,2]:\n");
    using S = gcd123ref::Shape;
    const S sh = {1, 1, 1, 1};
    const S oh = {1, 2, 3, 2};
    std::vector<float> self = {6};
    std::vector<float> other = {10, 15, 8, 9, 12, 21, 5, 14};
    S outShape;
    auto out = gcd123ref::Golden(self, sh, other, oh, outShape);
    Check(outShape == oh, "out shape == other shape [1,2,3,2]");
    const std::vector<float> expect = {2, 3, 2, 3, 6, 3, 1, 2}; // gcd(6, x)
    Check(out == expect, "scalar-self gcd correct");
}

// 用例4：负数与 0 的边界。
void CaseNegAndZero() {
    std::printf("Case4 negative & zero, self[1,1,1,4] other[1,1,1,4]:\n");
    using S = gcd123ref::Shape;
    const S sh = {1, 1, 1, 4};
    std::vector<float> self = {-12, 0, 18, 7};
    std::vector<float> other = {8, 5, -24, -7};
    S outShape;
    auto out = gcd123ref::Golden(self, sh, other, sh, outShape);
    // gcd(-12,8)=4, gcd(0,5)=5, gcd(18,-24)=6, gcd(7,-7)=7
    const std::vector<float> expect = {4, 5, 6, 7};
    Check(out == expect, "neg/zero gcd correct");
}

} // namespace

int main() {
    std::printf("==== Gcd123 CPU reference test ====\n");
    CaseSameShape();
    CaseBroadcastInner();
    CaseScalarSelf();
    CaseNegAndZero();
    std::printf("==== %s (%d failure(s)) ====\n", g_failures == 0 ? "ALL PASS" : "FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
}
