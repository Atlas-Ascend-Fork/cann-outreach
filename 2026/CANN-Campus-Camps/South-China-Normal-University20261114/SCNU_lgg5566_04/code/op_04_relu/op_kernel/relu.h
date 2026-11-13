/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // InitBuffer 的物理双缓冲数量

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
namespace NsReluDirect {

// 每核单 tile 时走原地静态 UB 直通，省掉 TPipe/TQue 和输出缓冲；
// 上限覆盖 40 核 [45, 20480] FP16 场景的 45KB 单块。
constexpr uint32_t DIRECT_MAX_TILE_BYTES = 49152U;

template <typename DT_X>
__aicore__ inline bool CanUseDirectPath(const ReluTilingData& tilingData)
{
    const uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
    const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
    const uint64_t tileBytes =
        static_cast<uint64_t>(tilingData.blockLength) * sizeof(DT_X);
    return blockNum != 0U &&
           blockIdx < blockNum &&
           tilingData.totalLength != 0U &&
           tilingData.blockLength == tilingData.tileLength &&
           static_cast<uint64_t>(tilingData.blockLength) * blockNum ==
               tilingData.totalLength &&
           tileBytes <= DIRECT_MAX_TILE_BYTES &&
           (tileBytes % 32U) == 0U;
}

template <typename DT_X>
__aicore__ inline bool TryRunDirect(GM_ADDR x, GM_ADDR y,
                                    const ReluTilingData& tilingData)
{
    const bool canRun = CanUseDirectPath<DT_X>(tilingData);
    if (canRun) {
        const uint32_t length = tilingData.blockLength;
        const uint32_t start = static_cast<uint32_t>(GetBlockIdx()) * length;

        GlobalTensor<DT_X> inputGm;
        GlobalTensor<DT_X> outputGm;
        inputGm.SetGlobalBuffer((__gm__ DT_X*)x, tilingData.totalLength);
        outputGm.SetGlobalBuffer((__gm__ DT_X*)y, tilingData.totalLength);

        // ReLU 允许原地计算。输入/输出共用同一 UB 地址，省一块 4KB buffer
        // 和一次 buffer 地址切换；bank 冲突收益需与错位双 buffer 上板对拍。
        LocalTensor<DT_X> xLocal(TPosition::VECCALC, 0U, length);

        DataCopy(xLocal, inputGm[start], length);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        AscendC::Relu(xLocal, xLocal, length);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);

        DataCopy(outputGm[start], xLocal, length);
        // 输出 DMA 异步，任务退出前必须排空，避免连续调用时残留访问。
        PipeBarrier<PIPE_MTE3>();
    }
    return canRun;
}

} // namespace NsReluDirect
#endif

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        const uint32_t totalLength = tilingData->totalLength;
        this->tileLength = tilingData->tileLength;
        if (totalLength == 0) {
            return;
        }

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint64_t blockOffset =
            static_cast<uint64_t>(blockIdx) * static_cast<uint64_t>(tilingData->blockLength);
        if (blockOffset >= totalLength) {
            return;
        }
        this->startOffset = static_cast<uint32_t>(blockOffset);
        this->blockLength = totalLength - this->startOffset;
        if (this->blockLength > tilingData->blockLength) {
            this->blockLength = tilingData->blockLength;
        }
        if (this->blockLength == 0) {
            return;
        }

        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + this->startOffset, this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + this->startOffset, this->blockLength);

        const uint32_t tileBytes = this->tileLength * sizeof(DT_X);
        // 单 tile 没有可重叠的下一块，只申请一份 UB；多 tile 才启用双缓冲。
        const int32_t queueDepth = (this->blockLength > this->tileLength) ? BUFFER_NUM : 1;
        pipe.InitBuffer(inputQueueX, queueDepth, tileBytes);
        pipe.InitBuffer(outputQueueY, queueDepth, tileBytes);
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) {
            return;
        }

        // 16K 小 shape 的常见路径：每核恰好一个 tile，省掉循环和尾块分支。
        if (this->blockLength <= this->tileLength) {
            ProcessTile(0, this->blockLength);
            return;
        }

        uint32_t offset = 0;
        while (this->blockLength - offset > this->tileLength) {
            ProcessTile(offset, this->tileLength);
            offset += this->tileLength;
        }
        ProcessTile(offset, this->blockLength - offset);
    }

private:
    __aicore__ inline void ProcessTile(uint32_t offset, uint32_t length)
    {
        CopyIn(offset, length);
        Compute(length);
        CopyOut(offset, length);
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        const uint64_t globalByteOffset =
            (static_cast<uint64_t>(this->startOffset) + offset) * sizeof(DT_X);
        const uint64_t copyBytes = static_cast<uint64_t>(length) * sizeof(DT_X);
        if ((globalByteOffset % 32U) == 0 && (copyBytes % 32U) == 0) {
            DataCopy(xLocal, inputGMX[offset], length);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyBytes), 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};
            DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
        }
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t length)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        AscendC::Relu(yLocal, xLocal, length);
        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        const uint64_t globalByteOffset =
            (static_cast<uint64_t>(this->startOffset) + offset) * sizeof(DT_X);
        const uint64_t copyBytes = static_cast<uint64_t>(length) * sizeof(DT_X);
        if ((globalByteOffset % 32U) == 0 && (copyBytes % 32U) == 0) {
            DataCopy(outputGMY[offset], yLocal, length);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyBytes), 0, 0, 0};
            DataCopyPad(outputGMY[offset], yLocal, copyParams);
        }
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    // 大 shape 使用双缓冲；小 shape 由 InitBuffer 的 queueDepth 自动收缩为单缓冲。
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t startOffset = 0; // 本核在全局的起始元素偏移
    uint32_t blockLength = 0; // 本核实际处理的元素数
    uint32_t tileLength = 0;  // 单次处理的元素数
};

} // namespace NsRelu
#endif // RELU_H
