/*!
 * \file exp.cpp
 * \brief Exp 算子 kernel 入口
 */

#include "exp.h"

enum class ExpTilingKey : uint32_t
{
    TILING_KEY_EXP_MODE_0 = 0,
    TILING_KEY_EXP_MODE_1 = 1,
    TILING_KEY_EXP_FIXED_HALF = 2,
    TILING_KEY_EXP_FIXED_FLOAT = 3,
};

template <uint32_t schMode>
__global__ __aicore__ void exp(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(ExpTilingData);
    if constexpr (schMode == static_cast<uint32_t>(ExpTilingKey::TILING_KEY_EXP_FIXED_HALF)) {
        NsExp::RunFixed2048<half>(x, y);
    } else if constexpr (schMode == static_cast<uint32_t>(ExpTilingKey::TILING_KEY_EXP_FIXED_FLOAT)) {
        NsExp::RunFixed2048<float>(x, y);
    } else {
        GET_TILING_DATA_WITH_STRUCT(ExpTilingData, tilingData, tiling);
        if constexpr (schMode == static_cast<uint32_t>(ExpTilingKey::TILING_KEY_EXP_MODE_0)) {
            NsExp::RunExp<half>(x, y, &tilingData);
        }
        if constexpr (schMode == static_cast<uint32_t>(ExpTilingKey::TILING_KEY_EXP_MODE_1)) {
            NsExp::RunExp<float>(x, y, &tilingData);
        }
    }
    if constexpr (schMode == static_cast<uint32_t>(ExpTilingKey::TILING_KEY_EXP_FIXED_HALF) ||
        schMode == static_cast<uint32_t>(ExpTilingKey::TILING_KEY_EXP_FIXED_FLOAT)) {
        // The fixed path already establishes MTE2 -> V -> MTE3 ordering.
        // Wait for the final GM write before scalar execution leaves the kernel.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
    } else {
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}
