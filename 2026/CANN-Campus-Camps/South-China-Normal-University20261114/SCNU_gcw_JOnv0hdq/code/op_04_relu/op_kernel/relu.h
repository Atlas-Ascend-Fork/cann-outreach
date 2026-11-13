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

constexpr uint32_t BLOCK_ALIGN_ELEMENTS = 512;
constexpr uint32_t TILE_ALIGN_BYTES = 512;
constexpr uint32_t SMALL_INPUT_MAX_BYTES = 64 * 1024;

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        const uint32_t totalLength = tilingData->totalLength;
        const uint32_t coreCount = AscendC::GetBlockNum();
        const uint32_t coreIndex = AscendC::GetBlockIdx();
        const uint32_t blockGroups = (totalLength - 1) / BLOCK_ALIGN_ELEMENTS + 1;
        const uint32_t groupsPerCore = blockGroups / coreCount;
        const uint32_t extraGroups = blockGroups % coreCount;
        const uint32_t extraBefore = coreIndex < extraGroups ? coreIndex : extraGroups;
        const uint32_t blockOffset = (groupsPerCore * coreIndex + extraBefore) * BLOCK_ALIGN_ELEMENTS;
        const uint64_t plannedLength = uint64_t(groupsPerCore + (coreIndex < extraGroups)) * BLOCK_ALIGN_ELEMENTS;
        const uint32_t remaining = totalLength - blockOffset;
        this->blockLength = remaining < plannedLength ? remaining : static_cast<uint32_t>(plannedLength);

        // tileNum 是最长数据段需要的实际批次数，按 512B 对齐每批的上限。
        const uint64_t maxPlannedLength = uint64_t(groupsPerCore + (extraGroups != 0)) * BLOCK_ALIGN_ELEMENTS;
        const uint32_t maxBlockLength = totalLength < maxPlannedLength
            ? totalLength : static_cast<uint32_t>(maxPlannedLength);
        const uint32_t tileElements = (maxBlockLength - 1) / tilingData->tileNum + 1;
        const uint32_t tileAlign = TILE_ALIGN_BYTES / sizeof(DT_X);
        this->tileLength = (tileElements + tileAlign - 1) / tileAlign * tileAlign;

        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + blockOffset, this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + blockOffset, this->blockLength);
        const bool hasMultipleTiles = tilingData->tileNum > 1;
        const bool smallInput = static_cast<uint64_t>(totalLength) * sizeof(DT_X) <= SMALL_INPUT_MAX_BYTES;
        const uint8_t inputBuffers = hasMultipleTiles && smallInput ? 2 : 1;
        const uint8_t outputBuffers = hasMultipleTiles ? 2 : 1;
        pipe.InitBuffer(inputQueueX, inputBuffers, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, outputBuffers, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < this->blockLength;) {
            const uint32_t remaining = this->blockLength - offset;
            const uint32_t count = remaining < this->tileLength ? remaining : this->tileLength;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
            offset += count;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, inputGMX[offset], count);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        AscendC::Relu(yLocal, xLocal, static_cast<int32_t>(count));
        outputQueueY.EnQue(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        AscendC::DataCopy(outputGMY[offset], yLocal, count);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inputQueueX;
    TQue<QuePosition::VECOUT, 1> outputQueueY;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength; // 每核处理的元素数
    uint32_t tileLength;  // 单核内每块元素数
};

} // namespace NsRelu
#endif // RELU_H
