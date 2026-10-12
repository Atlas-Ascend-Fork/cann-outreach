/*!
 * \file tanh.h
 * \brief Tanh 算子 kernel 类定义
 */

#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1; // 每队列 buffer 块数
// UB 容量反算的单块元素上限：4 块缓冲（in/out/tmp1/tmp2）× 8192 × 4B = 128KB < 192KB（Ascend910B UB）
constexpr uint32_t UB_TILE_CAP = 8192;
// 数值稳定截断点：tanh(x) = (e^{2x}-1)/(e^{2x}+1)，e^{2x} 在 fp16 下 x>5.5 溢出、fp32 下 x>44 溢出
// fp16 取 5（e^10 = 22026 < 65504），fp32 取 40（e^80 = 5.5e34 < 3.4e38），截断处 tanh 均已饱和到 1
#define TANH_CLAMP_MAX (sizeof(DT_X) == 2 ? 5.0f : 40.0f)

template <typename DT_X>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
    {
        constexpr uint32_t alignElems = 32 / sizeof(DT_X); // 32B 对齐对应的元素数
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t totalLength = tilingData->totalLength;
        this->tileNum = tilingData->tileNum;

        // 多核切分：前 blockNum-1 个核取向下 32B 对齐的等长块，末核承担全部剩余（含 totalLength%blockNum 余量）
        this->blockLength = totalLength / blockNum / alignElems * alignElems;
        this->isLastCore = (blockIdx == blockNum - 1) ? 1 : 0;
        this->totalLen = this->isLastCore ? (totalLength - this->blockLength * (blockNum - 1)) : this->blockLength;

        // 单块长度：按 tileNum 均分并向下 32B 对齐，同时按 UB 容量封顶，至少一个对齐块
        uint32_t t = this->totalLen / this->tileNum / alignElems * alignElems;
        if (t < alignElems) {
            t = alignElems;
        }
        this->tileLength = t > UB_TILE_CAP ? UB_TILE_CAP : t;

        // 设置每个核的 Global Memory 起始地址（blockLength 已对齐，地址必然 32B 对齐）
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + (uint64_t)this->blockLength * blockIdx, this->totalLen);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + (uint64_t)this->blockLength * blockIdx, this->totalLen);
        // 为队列分配 UB 内存
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        // 临时空间：分别存放 e^{2x} 与分母
        pipe.InitBuffer(tmpExp, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(tmpDen, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (this->totalLen == 0) {
            return; // 空载保护：该核无数据（totalLength 小于核数对齐粒度时）
        }
        // 完整块循环 + 尾块单独处理，保证 totalLen 不被整除时不丢数据
        uint32_t loopCount = this->totalLen / this->tileLength;
        uint32_t tailLength = this->totalLen % this->tileLength;
        for (uint32_t i = 0; i < loopCount; i++) {
            CopyIn(i, this->tileLength);
            Compute(this->tileLength);
            CopyOut(i, this->tileLength);
        }
        if (tailLength > 0) {
            CopyInTail(loopCount, tailLength);
            Compute(tailLength);
            CopyOutTail(loopCount, tailLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t len)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGMX[progress * this->tileLength], len);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyInTail(uint32_t progress, uint32_t len)
    {
        // 尾块长度可能非 32B 对齐，使用 DataCopyPad 搬运
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(DT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, inputGMX[progress * this->tileLength], copyParams, padParams);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        LocalTensor<DT_X> expT = tmpExp.Get<DT_X>();
        LocalTensor<DT_X> denT = tmpDen.Get<DT_X>();

        // tanh(x) = (e^{2x}-1)/(e^{2x}+1)，正向截断防止 e^{2x} 溢出产生 NaN（负向 e^{2x}->0 天然饱和到 -1）
        AscendC::Mins(expT, xLocal, static_cast<DT_X>(TANH_CLAMP_MAX), len);
        AscendC::Muls(expT, expT, static_cast<DT_X>(2), len);
        // e^{2*min(x,C)}
        AscendC::Exp(expT, expT, len);
        // 分母：e^{2x} + 1
        AscendC::Adds(denT, expT, static_cast<DT_X>(1), len);
        // 分子：e^{2x} - 1
        AscendC::Adds(expT, expT, static_cast<DT_X>(-1), len);
        // y = 分子 / 分母
        AscendC::Div(yLocal, expT, denT, len);

        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t len)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopy(outputGMY[progress * this->tileLength], yLocal, len);
        outputQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutTail(uint32_t progress, uint32_t len)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(DT_X)), 0, 0, 0};
        DataCopyPad(outputGMY[progress * this->tileLength], yLocal, copyParams);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<QuePosition::VECCALC> tmpExp;
    TBuf<QuePosition::VECCALC> tmpDen;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength; // 非末核处理的对齐等长块
    uint32_t totalLen;    // 本核实际处理的元素数（末核含余量）
    uint32_t tileNum;     // 单核内分块数
    uint32_t tileLength;  // 单核内每块元素数（32B 对齐且按 UB 封顶）
    uint32_t isLastCore;  // 是否末核
};

} // namespace NsTanh
#undef TANH_CLAMP_MAX
#endif // TANH_H
