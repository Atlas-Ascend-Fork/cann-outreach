/*
 * \file main.cpp
 * \brief add_custom 算子主机端（Host）测试程序
 *
 * 流程：
 *   1. 生成 2 维 [N2, N1]、ND 格式、float16 的输入 x、y，并计算 CPU 参考结果；
 *   2. 通过 ACL 将 x、y 拷贝到 NPU 设备；
 *   3. 启动 add_custom 算子（z = x + y）；
 *   4. 将结果 z 拷回主机，与参考结果逐元素比较（fp16 精度容差）。
 *
 * 说明：
 *   - 内存管理使用稳定的 ACL 接口（aclInit / aclrtMalloc / aclrtMemcpy 等）。
 *   - 算子启动依赖 CANN 的自定义内核启动接口，不同版本 API 有差异，
 *     代码中用【CANN 版本相关】标出，请按实际 CANN 版本对接。
 */
#include <acl/acl.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// 测试形状：2 维 [N2, N1]
// 元素总数 4 * 10363 = 41452，不是 tile 长度(256)的整数倍，可覆盖末尾 tile 的尾数处理
static const int32_t N2 = 4;
static const int32_t N1 = 10363;

// fp16 精度容差（幅值 ~10 时 half 的 1 ULP 约 0.0078，取 0.05 足够宽松）
static const float TOL = 0.05f;

#define CHECK_ACL(ret)                                                       \
    do {                                                                     \
        aclError _r = (ret);                                                 \
        if (_r != ACL_SUCCESS) {                                             \
            std::fprintf(stderr, "[ACL ERROR] ret=%d (%s) at line %d\n",     \
                         (int)_r, aclGetRecentErrMsg(), __LINE__);           \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// float -> half（简化实现：下溢归 0、上溢归 Inf；测试数据不落在这些区间）
static inline uint16_t FloatToHalf(float f)
{
    uint32_t xf = 0;
    std::memcpy(&xf, &f, sizeof(float));
    uint32_t sign = (xf >> 16) & 0x8000u;
    int32_t exp = (int32_t)((xf >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = xf & 0x7FFFFFu;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

// half -> float
static inline float HalfToFloat(uint16_t h)
{
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t xf = 0;
    if (exp == 0) {
        xf = sign;  // 次正规数简化为 0（测试数据不涉及）
    } else if (exp == 31) {
        xf = sign | 0x7F800000u | (mant << 13);
    } else {
        xf = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f = 0;
    std::memcpy(&f, &xf, sizeof(float));
    return f;
}

int main()
{
    const int64_t total = static_cast<int64_t>(N2) * N1;
    const size_t bytes = static_cast<size_t>(total) * sizeof(uint16_t);
    std::printf("[test] shape=[%d, %d], dtype=float16, format=ND, total=%ld elements\n", N2, N1, (long)total);

    // 1. 主机侧生成输入数据与 CPU 参考结果
    std::vector<uint16_t> hx(total), hy(total), hz(total);
    std::vector<float> expected(total);
    std::srand(20261018);
    for (int64_t i = 0; i < total; ++i) {
        float xf = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 10.0f - 5.0f;
        float yf = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 10.0f - 5.0f;
        hx[i] = FloatToHalf(xf);
        hy[i] = FloatToHalf(yf);
        expected[i] = HalfToFloat(hx[i]) + HalfToFloat(hy[i]);  // 与算子输入一致，消除转换误差
    }

    // 2. ACL 初始化
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));

    // 3. 申请设备内存并拷贝输入
    void *dx = nullptr;
    void *dy = nullptr;
    void *dz = nullptr;
    CHECK_ACL(aclrtMalloc(&dx, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dy, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dz, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(dx, bytes, hx.data(), bytes, aclrtMemcpyHostToDevice));
    CHECK_ACL(aclrtMemcpy(dy, bytes, hy.data(), bytes, aclrtMemcpyHostToDevice));

    // 4.【CANN 版本相关】启动 add_custom 算子
    // ---------------------------------------------------------------
    // 方式 A（CANN 7.0+ 自定义内核接口，示意）：
    //   先把 src/add_custom.cpp 用 ccec_compiler 编译为 build/add_custom.o
    //   （见 build.sh），再加载并启动：
    //
    //   aclrtStream stream = nullptr;
    //   CHECK_ACL(aclrtCreateStream(&stream));
    //   const uint32_t blockDim[2] = { 48, 1 };  // 核数，按设备（如 910B 为 48 核）
    //   GM_ADDR args[3] = { (GM_ADDR)dx, (GM_ADDR)dy, (GM_ADDR)dz };
    //   uint32_t n2 = (uint32_t)N2, n1 = (uint32_t)N1;
    //   GM_ADDR shapeArgs[2] = { (GM_ADDR)n2, (GM_ADDR)n1 };
    //   // 参数需与内核签名 (GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t n2, uint32_t n1) 一一对应
    //   CHECK_ACL(aclrtLoadKernel("build/add_custom.o", "add_custom", &kernelFunc));
    //   CHECK_ACL(aclrtLaunchKernel(kernelFunc, blockDim, nullptr, args, 0, stream));
    //
    // 方式 B（注册为自定义算子）：
    //   若通过 CANN 自定义算子注册机制（JSON/tik + 算子注册）接入，
    //   则在此处按注册框架的调用接口启动，参数同样为 x, y, z, n2, n1。
    // ---------------------------------------------------------------
    // TODO: 按你的 CANN 版本取消注释并补全上面某一种启动方式。
    (void)dx; (void)dy; (void)dz;

    CHECK_ACL(aclrtSynchronizeDevice());

    // 5. 拷回结果并逐元素校验
    CHECK_ACL(aclrtMemcpy(hz.data(), bytes, dz, bytes, aclrtMemcpyDeviceToHost));
    double maxErr = 0.0;
    int64_t bad = 0;
    for (int64_t i = 0; i < total; ++i) {
        float zf = HalfToFloat(hz[i]);
        double err = std::fabs(static_cast<double>(zf) - expected[i]);
        if (err > TOL) {
            if (bad < 5) {
                std::fprintf(stderr, "  mismatch at %ld: z=%.6f, expected=%.6f\n",
                             (long)i, zf, expected[i]);
            }
            ++bad;
        }
        maxErr = std::max(maxErr, err);
    }

    if (bad == 0) {
        std::printf("[test] PASS: add_custom 全部 %ld 元素正确（maxErr=%.6f, tol=%.2f）\n", (long)total, maxErr, TOL);
    } else {
        std::printf("[test] FAIL: %ld / %ld 元素错误（maxErr=%.6f）\n", (long)bad, (long)total, maxErr);
    }

    // 6. 资源释放
    CHECK_ACL(aclrtFree(dz));
    CHECK_ACL(aclrtFree(dy));
    CHECK_ACL(aclrtFree(dx));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return bad == 0 ? 0 : 1;
}
