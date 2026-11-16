/*!
 * \file relu.cpp
 * \brief Relu 算子 kernel 入口
 */

#include "relu.h"

namespace {
constexpr uint32_t SMALL_ELEMENTS_PER_CORE = 2048;

template <typename T>
__aicore__ inline void RunSmallRelu(GM_ADDR x, GM_ADDR y)
{
    AscendC::InitSocState();

    const uint32_t offset = AscendC::GetBlockIdx() * SMALL_ELEMENTS_PER_CORE;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, SMALL_ELEMENTS_PER_CORE);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, SMALL_ELEMENTS_PER_CORE);

    constexpr uint32_t bufferBytes = SMALL_ELEMENTS_PER_CORE * sizeof(T);
    static_assert(bufferBytes % 32 == 0, "UB buffers must be 32-byte aligned");
    AscendC::LocalTensor<T> xLocal(AscendC::TPosition::VECCALC, 0, SMALL_ELEMENTS_PER_CORE);
    AscendC::LocalTensor<T> yLocal(AscendC::TPosition::VECCALC, bufferBytes, SMALL_ELEMENTS_PER_CORE);

    AscendC::DataCopy(xLocal, xGm, SMALL_ELEMENTS_PER_CORE);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

    AscendC::Relu(yLocal, xLocal, SMALL_ELEMENTS_PER_CORE);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::DataCopy(yGm, yLocal, SMALL_ELEMENTS_PER_CORE);
}
template <typename T>
__aicore__ inline void RunTiledRelu(GM_ADDR x, GM_ADDR y, const ReluTilingData& data)
{
    AscendC::InitSocState();
    const int64_t start = static_cast<int64_t>(AscendC::GetBlockIdx()) * data.blockFactor;
    const int64_t remaining = data.totalNum - start;
    const int64_t length = remaining < data.blockFactor ? remaining : data.blockFactor;
    const uint32_t tile = static_cast<uint32_t>(data.ubFactor);
    const uint32_t tileBytes = tile * sizeof(T);
    AscendC::GlobalTensor<T> input, output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + start, length);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + start, length);

    // Input and output slots have separate reuse dependencies.
    uint32_t slot = 0;
    const int64_t reuseDistance = 2 * static_cast<int64_t>(tile);
    for (int64_t offset = 0; offset < length; offset += tile) {
        const uint32_t count = length - offset < tile ? static_cast<uint32_t>(length - offset) : tile;
        const uint32_t ubOffset = slot * 2 * tileBytes;
        AscendC::LocalTensor<T> xLocal(AscendC::TPosition::VECCALC, ubOffset, tile);
        AscendC::LocalTensor<T> yLocal(AscendC::TPosition::VECCALC, ubOffset + tileBytes, tile);
        const bool reused = offset >= reuseDistance;
        const bool willReuse = offset + reuseDistance < length;

        if (reused) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
        }
        AscendC::DataCopy(xLocal, input[offset], count);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);

        if (reused) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
        }
        AscendC::Relu(yLocal, xLocal, static_cast<int32_t>(count));
        if (willReuse) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
        AscendC::DataCopy(output[offset], yLocal, count);
        if (willReuse) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot);
        }
        slot ^= 1U;
    }
}
} // namespace

template <uint32_t schMode>
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluTilingData);
    if constexpr (schMode == RELU_TPL_SCH_MODE_2) {
        RunSmallRelu<half>(x, y);
    } else if constexpr (schMode == RELU_TPL_SCH_MODE_3) {
        RunSmallRelu<float>(x, y);
    } else {
        GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);
        if constexpr (schMode == RELU_TPL_SCH_MODE_0) {
            RunTiledRelu<half>(x, y, tilingData);
        } else if constexpr (schMode == RELU_TPL_SCH_MODE_1) {
            RunTiledRelu<float>(x, y, tilingData);
        }
    }
}
