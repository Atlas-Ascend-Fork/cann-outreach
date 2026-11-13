/*!
 * \file leakyrelu_tiling_data.h
 * \brief tiling data struct
 *
 * LeakyReLU 算子的 tiling 数据。
 *
 * ★ 前三个字段名（totalNum / blockFactor / ubFactor）**不能改**：
 *   官方只读单测 tests/ut/op_kernel/test_leakyrelu.cpp 里直接按字段名赋值
 *       tilingData->totalNum    = size;
 *       tilingData->blockFactor = size;
 *       tilingData->ubFactor    = size;
 *   字段名一旦对不上，官方 kernel UT 直接编译不过。
 *
 * ★ negativeSlope 是本题（第 13 期）新增的属性字段：
 *   任务书明确要求「属性需在 op_def 中注册，由 Tiling 函数读取后写入 tilingData，
 *   再下发到 kernel 侧使用」。因此负半轴斜率 α 必须随 tiling 数据下发，
 *   下发链路为：
 *       op_host/leakyrelu_def.cpp     注册属性 negative_slope（Float）
 *       op_host/leakyrelu_tiling.cpp  读取属性 → 写入 tiling->negativeSlope
 *       op_kernel/leakyrelu.h         从 tilingData 读出 → 计算时使用
 *   默认值 0.01f 取自 PyTorch torch.nn.functional.leaky_relu 的缺省值。
 *
 * 三个 int64 字段沿用官方模板的默认值（0 / 1 / 0），运行期由
 * op_host/leakyrelu_tiling.cpp 统一覆盖（与第八/十/十二期已合入实现同一写法）。
 *
 * 类型说明：三个计数字段用 int64_t，kernel 侧用 int64_t 承接 offset / blockLength_，
 * 避免元素个数较大时溢出；negativeSlope 用 float，与 op_def 中 `.Attr("negative_slope").Float()`
 * 注册的类型一致。
 *
 * ★ 为什么末尾还有一个 uint32_t reserved（本可省略）：
 *   官方 host 单测 `tests/ut/common/tiling_case_executor.cpp` 把 tiling buffer
 *   **按 int64 数组序列化**后与期望字符串做 EXPECT_EQ 严格比较。
 *   若结构体是「3 个 int64 + 1 个 float」（sizeof 32），那 4 个字节的**结构体尾部填充**
 *   落在第 4 个 int64 的高 32 位里，而填充字节来自 GE 分配的 tiling buffer
 *   （未初始化）⇒ 序列化结果是**不确定的**，任何硬编码的期望值都可能随机失配。
 *   显式补一个 uint32_t 字段把尾部填充占满（24+4+4 = 32，恰好 8 的整数倍，无尾部填充），
 *   并由 host 侧显式写 0 ⇒ 序列化结果确定，官方 host 单测的断言才有意义。
 */

#ifndef _LEAKYRELU_TILING_DATA_H_
#define _LEAKYRELU_TILING_DATA_H_

#include <cstdint> // int64_t / uint32_t：不依赖调用方的包含顺序

struct LeakyreluTilingData {
    int64_t totalNum = 0;     // 总元素数量
    int64_t blockFactor = 1;  // 每个核处理的元素数量
    int64_t ubFactor = 0;     // 每次 UB 循环处理的元素数量
    float negativeSlope = 0.01f; // 负半轴斜率 α（算子属性 negative_slope）
    uint32_t reserved = 0;    // 占满尾部填充，保证按 int64 序列化 tiling buffer 时结果确定
};

#endif
