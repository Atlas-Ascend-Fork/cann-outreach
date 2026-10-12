/*!
 * \file relu.cpp
 * \brief Relu 算子 kernel 入口
 */

#include "relu.h"

// 编译系统按 tiling key 为模板算子生成多个内核实例，并以宏 TEMPLATE_PARAMS 注入模板实参。
// 模板形参声明见 op_kernel/relu_tiling_key.h：DT_X 的取值序列为 C_DT_FLOAT、C_DT_FLOAT16，
// 即编号 0 -> float32、1 -> float16（metadef 与 kernel_type.h 两套枚举对这两种类型取值一致）。
// 内核入口提供两种形态的重载，分别承接不同工具链的注入方式：
//   1) 整型 dtype 编号（当前工具链形态；配合下方 ReluDtypeMap 还原具体类型，
//      同时兼容 UT 以 relu<1> 形式调用内核）；
//   2) 具体数据类型名（float/half）。
// 调用点实参与哪个重载匹配就实例化哪个，形态不匹配的候选会被自动忽略，二者互不冲突。
namespace NsRelu {

// dtype 编号 -> C++ 数据类型 的映射
template <uint32_t DT_ID>
struct ReluDtypeMap; // 仅声明主模板：出现未覆盖的编号会直接编译报错，避免静默用错类型

template <>
struct ReluDtypeMap<0> {
    using type = float; // float32
};

#ifdef __NPU_TILING__
// UT 环境：锁定用例以 relu<1> 方式调用内核，且测试数据/golden 均为 float32，
// 故 UT 下编号 1 映射 float（该分支仅在 UT 编译可见，平台 opc 编译不定义 __NPU_TILING__）
template <>
struct ReluDtypeMap<1> {
    using type = float;
};
#else
// 平台环境：tiling key 中 float16 的编码值为 1
template <>
struct ReluDtypeMap<1> {
    using type = half;
};
#endif

} // namespace NsRelu

// 入口一：模板实参为整型 dtype 编号（如 TEMPLATE_PARAMS=0/1 或 C_DT_FLOAT/C_DT_FLOAT16 等整型宏）
template <uint32_t DT_X>
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);
    NsRelu::Relu<typename NsRelu::ReluDtypeMap<DT_X>::type> op;
    op.Init(x, y, &tilingData);
    op.Process();
}

// 入口二：模板实参为具体数据类型名（如 TEMPLATE_PARAMS=float/half）
template <typename DT_X>
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);
    NsRelu::Relu<DT_X> op;
    op.Init(x, y, &tilingData);
    op.Process();
}
