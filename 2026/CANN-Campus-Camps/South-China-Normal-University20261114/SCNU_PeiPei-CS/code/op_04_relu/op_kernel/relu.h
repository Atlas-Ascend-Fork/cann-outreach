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

constexpr int32_t BUFFER_NUM = 8;    // 每队列 buffer 块数（双缓冲）
constexpr int32_t OUT_BUFFER_NUM = 2;
constexpr uint32_t BLOCK_BYTES = 32; // GM/UB 单次搬运的字节对齐粒度

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        this->totalLength = tilingData->totalNum;
        // 每核处理的数据长度（host 侧已按核数切分并按 32B 对齐）
        this->blockLength = tilingData->blockFactor;
        // 单次搬入/计算的元素数：ubFactor 是 BUFFER_NUM 块的总长，故需折半
        this->tileLength = tilingData->ubFactor / BUFFER_NUM;
        if (this->tileLength == 0U) {
            this->tileLength = tilingData->ubFactor;
        }
        if (this->tileLength == 0U) {
            this->tileLength = 1U;
        }

        // 设置每个核的 Global Memory 起始地址（关键的多核切分逻辑）：
        // 总数据按 blockLength 等长切分，第 blockIdx 个核从 blockLength * blockIdx 处开始
        this->gmOffset = static_cast<uint64_t>(this->blockLength) * AscendC::GetBlockIdx();
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + this->gmOffset, this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + this->gmOffset, this->blockLength);

        // 为队列分配 UB 内存（输入、输出各 BUFFER_NUM 块）
        pipe.InitBuffer(inputQueueX, OUT_BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (this->gmOffset >= this->totalLength) {
            return;
        }
        // 最后一个核可能不足 blockLength（host 侧用 ceil 切分时）
        uint32_t curBlockLength = this->totalLength - static_cast<uint32_t>(this->gmOffset);
        if (curBlockLength > this->blockLength) {
            curBlockLength = this->blockLength;
        }

        // 按 tileLength 分块；最后一块可能不足 tileLength（非对齐时走 DataCopyPad 兜底）
        int32_t loopCount = static_cast<int32_t>((curBlockLength + this->tileLength - 1) / this->tileLength);
        for (int32_t i = 0; i < loopCount; i++) {
            uint32_t offset = static_cast<uint32_t>(i) * this->tileLength;
            uint32_t curLength = this->tileLength;
            if (curBlockLength - offset < curLength) {
                curLength = curBlockLength - offset;
            }
            CopyIn(offset, curLength);
            Compute(curLength);
            CopyOut(offset, curLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t curLength)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        if ((curLength * sizeof(DT_X)) % BLOCK_BYTES == 0U) {
            DataCopy(xLocal, inputGMX[offset], curLength);
        } else {
            // 非 32B 对齐的尾块用 DataCopyPad，右填充 0
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{true, 0, 0, static_cast<DT_X>(0)};
            DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
        }
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t curLength)
    {
        /* 计算逻辑：y = max(x, 0)
         * Maxs 是向量级“逐元素与标量取大”指令，一次处理 curLength 个元素；
         * 对 float32 / float16 都成立，且不引入额外精度误差。
         */
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        AscendC::Maxs(yLocal, xLocal, (DT_X)0, curLength);
        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curLength)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        if ((curLength * sizeof(DT_X)) % BLOCK_BYTES == 0U) {
            DataCopy(outputGMY[offset], yLocal, curLength);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(outputGMY[offset], yLocal, copyParams);
        }
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t totalLength; // 总元素数
    uint32_t blockLength; // 每核处理的元素数
    uint32_t tileLength;  // 单核内每块元素数
    uint64_t gmOffset;    // 当前核在 GM 上的起始元素偏移
};

} // namespace NsRelu
#endif // RELU_H
