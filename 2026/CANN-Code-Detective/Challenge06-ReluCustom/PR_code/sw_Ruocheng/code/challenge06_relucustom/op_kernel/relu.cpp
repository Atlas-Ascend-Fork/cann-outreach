/*!
 * \file relu.cpp
 * \brief Relu 算子 kernel 入口
 */

#include "relu.h"

enum class ReluTilingKey : uint32_t
{
    TILING_KEY_RELU_MODE_0 = RELU_TPL_SCH_MODE_0,
    TILING_KEY_RELU_MODE_1 = RELU_TPL_SCH_MODE_1,
};

template <typename T>
__aicore__ inline void ReluKernelBody(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);
    NsRelu::Relu<T> op;
    op.Init(x, y, &tilingData);
    op.Process();
}

template <uint32_t schMode>
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(ReluTilingData);
    if constexpr (schMode == static_cast<uint32_t>(ReluTilingKey::TILING_KEY_RELU_MODE_0)) {
        ReluKernelBody<half>(x, y, tiling);
    }
    if constexpr (schMode == static_cast<uint32_t>(ReluTilingKey::TILING_KEY_RELU_MODE_1)) {
        ReluKernelBody<float>(x, y, tiling);
    }
}

// 编译框架依据 ASCENDC_TPL_ARGS_DECL/ASCENDC_TPL_SEL 生成 dispatch 时，
// 会调用 <入口名>_<索引>_tilingkey，索引即 RELU_TPL_SCH_MODE_* 的声明顺序。
__global__ __aicore__ void relu_0_tilingkey(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    ReluKernelBody<half>(x, y, tiling);
}

__global__ __aicore__ void relu_1_tilingkey(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    ReluKernelBody<float>(x, y, tiling);
}
