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

constexpr int32_t BUFFER_NUM = 2;      // 每队列 buffer 块数（双缓冲）
constexpr uint32_t TILE_ELEMS = 1024;  // 单次搬运算例上限（元素）

template <typename DT_X>
class Tanh {
public:
    using CALC_T = float; // 统一用 fp32 计算，避免 fp16 逐步累积误差

    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
    {
        uint32_t totalNum = static_cast<uint32_t>(tilingData->totalNum);
        uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        if (blockNum == 0) {
            blockNum = 1;
        }
        uint32_t align = 32 / sizeof(DT_X); // 32B 对齐对应的元素数（fp32:8, fp16:16）

        // 以实际启动核数为准做多核切分，每核长度向上取整并按 32B 对齐
        uint32_t perCore = (totalNum + blockNum - 1) / blockNum;
        perCore = (perCore + align - 1) / align * align;

        // 当前核负责 [start, start + coreLen)
        uint32_t start = static_cast<uint32_t>(AscendC::GetBlockIdx()) * perCore;
        uint32_t coreLen = 0;
        if (start < totalNum) {
            coreLen = totalNum - start;
            if (coreLen > perCore) {
                coreLen = perCore;
            }
        } else {
            start = totalNum; // 越界核：指针停在尾后，不做任何访问
        }

        this->blockLength = coreLen;
        this->tileLength = TILE_ELEMS;

        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + start, coreLen);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + start, coreLen);

        // 为队列分配 UB 内存
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, TILE_ELEMS * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, TILE_ELEMS * sizeof(DT_X));
        // 为计算补齐临时变量：x(float)、-x、e^x、e^{-x}、分子、分母
        pipe.InitBuffer(tmpBufX, TILE_ELEMS * sizeof(CALC_T));
        pipe.InitBuffer(tmpBufNeg, TILE_ELEMS * sizeof(CALC_T));
        pipe.InitBuffer(tmpBufExpX, TILE_ELEMS * sizeof(CALC_T));
        pipe.InitBuffer(tmpBufExpNeg, TILE_ELEMS * sizeof(CALC_T));
        pipe.InitBuffer(tmpBufNum, TILE_ELEMS * sizeof(CALC_T));
        pipe.InitBuffer(tmpBufDen, TILE_ELEMS * sizeof(CALC_T));
    }

    __aicore__ inline void Process()
    {
        // 按块推进，最后一块按剩余长度裁剪（不依赖 blockFactor / ubFactor）
        for (uint32_t offset = 0; offset < this->blockLength; offset += this->tileLength) {
            uint32_t len = this->blockLength - offset;
            if (len > this->tileLength) {
                len = this->tileLength;
            }
            CopyIn(offset, len);
            Compute(len);
            CopyOut(offset, len);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t len)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGMX[offset], len);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();

        // 计算所需的临时变量（全部为 fp32）
        LocalTensor<CALC_T> xf = tmpBufX.Get<CALC_T>();
        LocalTensor<CALC_T> neg = tmpBufNeg.Get<CALC_T>();
        LocalTensor<CALC_T> expX = tmpBufExpX.Get<CALC_T>();
        LocalTensor<CALC_T> expNeg = tmpBufExpNeg.Get<CALC_T>();
        LocalTensor<CALC_T> num = tmpBufNum.Get<CALC_T>();
        LocalTensor<CALC_T> den = tmpBufDen.Get<CALC_T>();

        // 0) 输入统一转 fp32（float 输入直接拷贝，half 输入做一次 Cast）
        if constexpr (sizeof(DT_X) == sizeof(CALC_T)) {
            DataCopy(xf, xLocal, len);
        } else {
            Cast(xf, xLocal, RoundMode::CAST_NONE, len);
        }

        // 1) 夹取到 [-8, 8]：tanh(8)=0.9999998，不影响精度，但能避免 exp 溢出
        Mins(num, xf, 8.0f, len);
        Maxs(xf, num, -8.0f, len);

        // 2) -x、e^x、e^{-x}
        Muls(neg, xf, -1.0f, len);
        Exp(expX, xf, len);
        Exp(expNeg, neg, len);

        // 3) 分子 = e^x - e^{-x}，分母 = e^x + e^{-x}
        Sub(num, expX, expNeg, len);
        Add(den, expX, expNeg, len);

        // 4) tanh(x) = 分子 / 分母（fp32 内算完）
        Div(neg, num, den, len);

        // 5) 最后一次性转回输出 dtype（half 只在这里舍入一次）
        if constexpr (sizeof(DT_X) == sizeof(CALC_T)) {
            DataCopy(yLocal, neg, len);
        } else {
            Cast(yLocal, neg, RoundMode::CAST_RINT, len);
        }

        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t len)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopy(outputGMY[offset], yLocal, len);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    // 为计算补齐临时变量（fp32 计算缓冲）
    TBuf<TPosition::VECCALC> tmpBufX;
    TBuf<TPosition::VECCALC> tmpBufNeg;
    TBuf<TPosition::VECCALC> tmpBufExpX;
    TBuf<TPosition::VECCALC> tmpBufExpNeg;
    TBuf<TPosition::VECCALC> tmpBufNum;
    TBuf<TPosition::VECCALC> tmpBufDen;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength; // 本核实际负责的元素数（运行期算出来的）
    uint32_t tileLength;  // 每块元素数（= TILE_ELEMS）
};

} // namespace NsTanh
#endif // TANH_H