/*!
 * \file relu.cpp
 * \brief Relu 算子 kernel 入口
 */

#include "relu.h"

// One-shot tensors avoid queue allocation, enqueue/dequeue, and a tile loop.
template <typename DT_X>
__aicore__ inline void ReluOneShot(GM_ADDR x, GM_ADDR y, uint32_t blockLength)
{
    using namespace AscendC;
    TPipe pipe;
    TBuf<QuePosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, blockLength * sizeof(DT_X));
    auto local = buffer.Get<DT_X>();
    GlobalTensor<DT_X> input;
    GlobalTensor<DT_X> output;
    const uint32_t offset = GetBlockIdx() * blockLength;
    input.SetGlobalBuffer((__gm__ DT_X*)x + offset, blockLength);
    output.SetGlobalBuffer((__gm__ DT_X*)y + offset, blockLength);
    DataCopy(local, input, blockLength);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::Relu(local, local, blockLength);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(output, local, blockLength);
}

template <typename DT_X>
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);
    if (tilingData.tileNum == 1) {
        ReluOneShot<DT_X>(x, y, tilingData.blockLength);
        return;
    }
    NsRelu::Relu<DT_X> op;
    op.Init(x, y, &tilingData);
    op.Process();
}
