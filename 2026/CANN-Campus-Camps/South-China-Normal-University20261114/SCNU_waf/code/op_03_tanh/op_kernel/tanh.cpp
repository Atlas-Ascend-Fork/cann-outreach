/*!
 * \file tanh.cpp
 * \brief Tanh 算子 kernel 入口（按 schMode 在编译期选 dtype 分支）
 */

#include "tanh.h"

namespace {
// 0 -> half，1 -> float；数值必须与 op_host 侧 GET_TPL_TILING_KEY 一致
enum class TanhSchedule : uint32_t {
    kHalf = 0,
    kFloat = 1,
};
}  // namespace

template <uint32_t schMode>
__global__ __aicore__ void tanh(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TanhTilingData);
    GET_TILING_DATA_WITH_STRUCT(TanhTilingData, tilingData, tiling);

    if constexpr (schMode == static_cast<uint32_t>(TanhSchedule::kHalf)) {
        TanhVec::TanhKernel<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(TanhSchedule::kFloat)) {
        TanhVec::TanhKernel<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}