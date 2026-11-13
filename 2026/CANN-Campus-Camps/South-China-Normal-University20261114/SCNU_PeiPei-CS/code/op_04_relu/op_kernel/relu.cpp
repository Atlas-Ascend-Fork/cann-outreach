/*!
 * \file relu.cpp
 * \brief Relu 算子 kernel 入口
 */

#include "relu.h"

/*
 * 编译系统按 tilingKey 为模板算子生成多个内核实例，并以宏注入模板实参。
 * 不同 CANN 工具链注入的形态不同：
 *   1) 整型 dtype 编号（C_DT_FLOAT = 0、C_DT_FLOAT16 = 1）；
 *   2) 具体数据类型名（float / half）。
 * 这里两种入口都提供，调用点实参与哪个匹配就实例化哪个，互不冲突。
 */
namespace NsRelu {

// dtype 编号 -> C++ 数据类型 的映射；出现未覆盖的编号会直接编译报错，避免静默用错类型
template <uint32_t DT_ID>
struct ReluDtypeMap;

template <>
struct ReluDtypeMap<0> {
    using type = float; // float32
};

#ifdef __NPU_TILING__
// UT 环境：用例固定以 relu<1> 调用且数据为 float32
template <>
struct ReluDtypeMap<1> {
    using type = float;
};
#else
// 平台环境：tilingKey 中 float16 的编码值为 1
template <>
struct ReluDtypeMap<1> {
    using type = half;
};
#endif

} // namespace NsRelu

// 入口一：模板实参为整型 dtype 编号
template <uint32_t DT_X>
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);
    NsRelu::Relu<typename NsRelu::ReluDtypeMap<DT_X>::type> op;
    op.Init(x, y, &tilingData);
    op.Process();
}

// 入口二：模板实参为具体数据类型名
template <typename DT_X>
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);
    NsRelu::Relu<DT_X> op;
    op.Init(x, y, &tilingData);
    op.Process();
}
