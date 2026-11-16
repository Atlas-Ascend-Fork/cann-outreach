/*!
 * \file relu.h
 * \brief Relu 算子 kernel 实现
 */
#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // 硬件原生 Ping-Pong 双缓冲

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        uint32_t totalLength = tilingData->totalLength;
        if (totalLength == 0U) {
            this->loopCount = 0U;
            return;
        }

        // 位运算解包 Host 预计算参数，AI Core 标量单元 0 除法
        uint32_t packed = tilingData->tileNum;
        uint32_t shift = (packed >> 31U) << 2U; // 512B对齐时左移4位(x16)，否则左移0位
        uint32_t baseBlocks = (packed & 0x3FFFU) << shift;
        uint32_t extraCount = (packed >> 14U) & 0x7FU;
        uint32_t tileBlocks = ((packed >> 21U) & 0x3FFU) << shift;
        uint32_t stepBlocks = 1U << shift;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t coreBlocks = baseBlocks + (blockIdx < extraCount ? stepBlocks : 0U);
        uint32_t startBlock = blockIdx * baseBlocks + (blockIdx < extraCount ? blockIdx : extraCount) * stepBlocks;

        constexpr uint32_t ELEMS_PER_BLOCK = 32U / sizeof(DT_X);
        uint32_t gmOffset = startBlock * ELEMS_PER_BLOCK;
        uint32_t rawCoreLength = coreBlocks * ELEMS_PER_BLOCK;

        if (gmOffset >= totalLength || rawCoreLength == 0U) {
            this->loopCount = 0U;
            return;
        }
        uint32_t maxRemain = totalLength - gmOffset;
        uint32_t coreLength = (rawCoreLength < maxRemain) ? rawCoreLength : maxRemain;
        this->tileLength = tileBlocks * ELEMS_PER_BLOCK;

        // 纯加减法计算单核循环次数与末块长度，确保 Process() 只有单一规范流水循环
        uint32_t count = 0U;
        uint32_t rem = coreLength;
        while (rem > this->tileLength) {
            rem -= this->tileLength;
            count++;
        }
        this->loopCount = count + 1U;
        this->lastTileLength = rem;

        // 设置当前核的 Global Memory 起始地址（512B L2 Cache Line 对齐）
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + gmOffset, coreLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + gmOffset, coreLength);

        // 申请 DoubleBuffer UB 空间
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        uint32_t count = this->loopCount;
        if (count == 0U) {
            return;
        }
        uint32_t offset = 0U;
        for (uint32_t i = 0U; i < count; i++) {
            uint32_t curLen = (i + 1U == count) ? this->lastTileLength : this->tileLength;
            CopyIn(offset, curLen);
            Compute();
            CopyOut(offset, curLen);
            offset += this->tileLength;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t curLen)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGMX[offset], curLen);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        AscendC::Relu(yLocal, xLocal, this->tileLength);
        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curLen)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopy(outputGMY[offset], yLocal, curLen);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t loopCount = 0U;      // 当前核迭代次数
    uint32_t tileLength = 0U;     // 满块元素个数（512B 对齐）
    uint32_t lastTileLength = 0U; // 末块元素个数
};
} // namespace NsRelu

#endif // RELU_H