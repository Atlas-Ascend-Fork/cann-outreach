/*
 * add_custom CPU 调试入口（-DASCEND_CPU_DEBUG）
 * 在 x86 上直接调用核函数，验证 z = x + y（float16）
 * 运行方式：见 build.sh，生成 build/add_custom_cpu 后直接执行
 */
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>
#include <cmath>

#include "kernel_operator.h"   // CPU 调试模式下提供 half 等类型
#include "add_custom_tiling.h"

// 核函数入口（CPU 调试模式按普通函数调用）
extern "C" void add_custom(void* x, void* y, void* z, void* workspace, void* tiling);

static int Fail = 0;

int main()
{
    const uint32_t N2 = 3, N1 = 1000;   // 2 维 shape [N2, N1]
    const uint32_t coreNum = 1;         // CPU 调试按单核执行

    const uint32_t total = N2 * N1;
    std::vector<AscendC::half> x(total), y(total), z(total, AscendC::half(0));

    std::mt19937 rng(20261018);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    std::vector<float> xr(total), yr(total);
    for (uint32_t i = 0; i < total; ++i) {
        xr[i] = dist(rng);
        yr[i] = dist(rng);
        x[i] = AscendC::half(xr[i]);
        y[i] = AscendC::half(yr[i]);
    }

    AddCustomTilingData tiling;
    if (!AddCustomTiling(N2, N1, coreNum, tiling)) {
        std::printf("tiling failed\n");
        return 1;
    }

    add_custom(reinterpret_cast<void*>(x.data()),
               reinterpret_cast<void*>(y.data()),
               reinterpret_cast<void*>(z.data()),
               nullptr, &tiling);

    for (uint32_t i = 0; i < total; ++i) {
        float got = float(z[i]);
        float exp = float(AscendC::half(xr[i]) + AscendC::half(yr[i]));
        if (std::fabs(got - exp) > 1e-2f) {
            if (Fail < 8) {
                std::printf("mismatch @%u: got %.4f, exp %.4f\n", i, got, exp);
            }
            ++Fail;
        }
    }

    if (Fail == 0) {
        std::printf("add_custom CPU test PASS, shape=[%u, %u]\n", N2, N1);
        return 0;
    }
    std::printf("add_custom CPU test FAIL, %u mismatches\n", Fail);
    return 1;
}
