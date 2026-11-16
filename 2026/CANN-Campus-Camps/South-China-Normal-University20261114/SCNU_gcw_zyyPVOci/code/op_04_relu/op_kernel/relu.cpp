
/*!
 * \file relu.cpp
 * \brief ReLU 算子 Kernel 入口
 */

#include "relu.h"

// =========================================================
// ReLU Kernel
// =========================================================

template <uint32_t schMode>
__global__ __aicore__ void relu(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    // 当前算子不使用 Workspace
    (void)workspace;

    // 注册默认 Tiling
    REGISTER_TILING_DEFAULT(ReluTilingData);

    // 获取 Tiling 参数
    GET_TILING_DATA_WITH_STRUCT(
        ReluTilingData,
        tilingData,
        tiling
    );

    // =====================================================
    // FP16
    // =====================================================

    if constexpr (schMode == RELU_TPL_SCH_MODE_0) {

        NsRelu::Relu<half> op;

        // 初始化
        op.Init(
            x,
            y,
            &tilingData
        );

        // 执行计算
        op.Process();
    }

    // =====================================================
    // FP32
    // =====================================================

    else if constexpr (
        schMode == RELU_TPL_SCH_MODE_1) {

        NsRelu::Relu<float> op;

        // 初始化
        op.Init(
            x,
            y,
            &tilingData
        );

        // 执行计算
        op.Process();
    }
}
