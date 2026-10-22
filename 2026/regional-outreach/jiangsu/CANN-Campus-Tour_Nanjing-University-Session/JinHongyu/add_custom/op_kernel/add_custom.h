/*!
 * \file add_custom.h
 * \brief add_custom 算子 kernel 类定义
 *
 * 逐元素加法：z = x + y
 * - 输入输出均为 float16 / ND / 二维 [N2, N1]
 * - 多核切分（余数分摊到前 rem 个核）+ 单核内分块（tile）
 * - 尾块（tail）边界处理：最后一个 tile 的长度用 curLen 截断，避免越界与丢元素
 * - GMEM<->UB 搬运（DataCopy）+ 向量计算（AscendC::Add）+ 双缓冲（BUFFER_NUM=2）
 */

#ifndef ADD_CUSTOM_H
#define ADD_CUSTOM_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "add_custom_tiling_data.h"
#include "add_custom_tiling_key.h"

namespace NsAddCustom {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // 每队列 buffer 块数（双缓冲）

template <typename DT_X>
class AddCustom {
public:
    __aicore__ inline AddCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const AddCustomTilingData* tilingData)
    {
        this->totalLength = tilingData->totalLength;
        this->tileLength = tilingData->tileLength;

        // 多核切分：每个核负责一段连续区间；totalLength 不能被核数整除时，
        // 余数 rem 分摊到前 rem 个核（这些核多处理 1 个元素），保证无遗漏、无重叠。
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t base = this->totalLength / blockNum;
        const uint32_t rem = this->totalLength % blockNum;
        if (blockIdx < rem) {
            this->blockStart = blockIdx * (base + 1);
            this->blockLen = base + 1;
        } else {
            this->blockStart = rem * (base + 1) + (blockIdx - rem) * base;
            this->blockLen = base;
        }

        // 设置本核的 Global Memory 视图（全局偏移 + 本核长度）
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + this->blockStart, this->blockLen);
        inputGMY.SetGlobalBuffer((__gm__ DT_X*)y + this->blockStart, this->blockLen);
        outputGMZ.SetGlobalBuffer((__gm__ DT_X*)z + this->blockStart, this->blockLen);

        // 为队列分配 UB 内存（按最大 tile 长度分配，尾块只用到其中一部分）
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));

        // 单核内分块数（向上取整）；最后一个块即尾块，长度可能小于 tileLength
        this->tileNum = (this->blockLen + this->tileLength - 1) / this->tileLength;
    }

    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < this->tileNum; i++) {
            // 尾块边界：最后一个 tile 的有效长度 = blockLen - i * tileLength（<= tileLength）
            int32_t curLen = (i == this->tileNum - 1) ? (this->blockLen - i * this->tileLength)
                                                      : this->tileLength;
            CopyIn(i, curLen);
            Compute(curLen);
            CopyOut(i, curLen);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, int32_t curLen)
    {
        // GMEM -> UB：搬入本核第 progress 块的 curLen 个元素
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inputQueueY.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGMX[progress * this->tileLength], curLen);
        DataCopy(yLocal, inputGMY[progress * this->tileLength], curLen);
        inputQueueX.EnQue(xLocal);
        inputQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t curLen)
    {
        // 向量计算：z = x + y
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inputQueueY.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outputQueueZ.AllocTensor<DT_X>();
        AscendC::Add(zLocal, xLocal, yLocal, curLen);
        outputQueueZ.EnQue<DT_X>(zLocal);
        inputQueueX.FreeTensor(xLocal);
        inputQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, int32_t curLen)
    {
        // UB -> GMEM：搬出本核第 progress 块的 curLen 个结果
        LocalTensor<DT_X> zLocal = outputQueueZ.DeQue<DT_X>();
        DataCopy(outputGMZ[progress * this->tileLength], zLocal, curLen);
        outputQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> inputGMY;
    GlobalTensor<DT_X> outputGMZ;

    uint32_t totalLength; // 总元素数
    uint32_t tileLength;  // 每块元素数（host 指定）
    uint32_t blockStart;  // 本核起始元素下标
    uint32_t blockLen;    // 本核元素数
    int32_t tileNum;      // 本核内分块数
};

} // namespace NsAddCustom

#endif // ADD_CUSTOM_H
