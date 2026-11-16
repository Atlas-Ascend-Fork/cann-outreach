/*!
 * \file relu.cpp
 * \brief Relu 算子 kernel 入口
 */

#include "relu.h"

namespace {
// 0 -> half，1 -> float；与 op_host 侧 GET_TPL_TILING_KEY 的取值一一对应
enum class ReluSchedule : uint32_t {
    kHalf = 0,
    kFloat = 1,
};
}  // namespace

template <uint32_t schMode>
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);

    if constexpr (schMode == static_cast<uint32_t>(ReluSchedule::kHalf)) {
        ReluAct::ReluKernel<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(ReluSchedule::kFloat)) {
        ReluAct::ReluKernel<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}