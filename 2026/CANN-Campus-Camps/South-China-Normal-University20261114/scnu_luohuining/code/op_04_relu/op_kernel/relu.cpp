/*!
 * \file relu.cpp
 * \brief Relu 算子 kernel 入口
 */

#include "relu.h"

enum class ReluTilingKey : uint32_t
{
    TILING_KEY_RELU_MODE_0 = 0,
    TILING_KEY_RELU_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    // 只使用 Vector 核，避免为 Cube 核预留/调度
    // 依据：day2上午/07_Troubleshooting/answer/07.05_answer.txt:85 的同款写法
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    REGISTER_TILING_DEFAULT(ReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(ReluTilingKey::TILING_KEY_RELU_MODE_0)) {
        NsRelu::Relu<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(ReluTilingKey::TILING_KEY_RELU_MODE_1)) {
        NsRelu::Relu<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}