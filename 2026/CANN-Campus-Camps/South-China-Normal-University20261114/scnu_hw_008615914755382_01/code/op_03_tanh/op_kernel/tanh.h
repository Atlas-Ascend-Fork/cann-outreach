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

constexpr int32_t BUFFER_NUM = 2;

template <typename DT_X>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const TanhTilingData* tilingData)
    {
        // blockFactor：每个核处理的元素数
        this->blockLength =
            static_cast<uint32_t>(tilingData->blockFactor);

        // ubFactor：单次UB循环处理的元素数
        this->tileLength =
            static_cast<uint32_t>(tilingData->ubFactor);

        // 工程中的CPU UT可能传入整个核的数据长度，
        // 限制单次UB分配规模
        if (this->tileLength > 1024U) {
            this->tileLength = 1024U;
        }

        this->tileNum =
            this->blockLength / this->tileLength / BUFFER_NUM;

        const uint32_t offset =
            this->blockLength * AscendC::GetBlockIdx();

        inputGMX.SetGlobalBuffer(
            (__gm__ DT_X*)x + offset,
            this->blockLength);

        outputGMY.SetGlobalBuffer(
            (__gm__ DT_X*)y + offset,
            this->blockLength);

        // 输入、输出队列采用双缓冲
        pipe.InitBuffer(
            inputQueueX,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X));

        pipe.InitBuffer(
            outputQueueY,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X));

        // 中间计算统一使用fp32
        pipe.InitBuffer(
            tmpXBuffer,
            this->tileLength * sizeof(float));

        pipe.InitBuffer(
            expXBuffer,
            this->tileLength * sizeof(float));

        pipe.InitBuffer(
            expNegXBuffer,
            this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int32_t loopCount = this->tileNum * BUFFER_NUM;

        for (int32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DT_X> xLocal =
            inputQueueX.AllocTensor<DT_X>();

        DataCopy(
            xLocal,
            inputGMX[progress * this->tileLength],
            this->tileLength);

        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal =
            inputQueueX.DeQue<DT_X>();

        LocalTensor<DT_X> yLocal =
            outputQueueY.AllocTensor<DT_X>();

        LocalTensor<float> xFloat =
            tmpXBuffer.Get<float>();

        LocalTensor<float> expX =
            expXBuffer.Get<float>();

        LocalTensor<float> expNegX =
            expNegXBuffer.Get<float>();

        // fp32直接复制；fp16先转换成fp32
        LoadFloat(xFloat, xLocal);

        // 防止大数指数溢出
        // |x|超过10时，tanh(x)已经非常接近±1
        Mins(xFloat, xFloat, 10.0f, this->tileLength);
        Maxs(xFloat, xFloat, -10.0f, this->tileLength);

        // expX = exp(x)
        Exp(expX, xFloat, this->tileLength);

        // expNegX = exp(-x)
        Muls(xFloat, xFloat, -1.0f, this->tileLength);
        Exp(expNegX, xFloat, this->tileLength);

        // 分母：exp(x) + exp(-x)
        Add(xFloat, expX, expNegX, this->tileLength);

        // 分子：exp(x) - exp(-x)
        Sub(expX, expX, expNegX, this->tileLength);

        // tanh(x) = 分子 / 分母
        Div(expNegX, expX, xFloat, this->tileLength);

        // 按输出类型复制或转换
        StoreFloat(yLocal, expNegX);

        outputQueueY.EnQue(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DT_X> yLocal =
            outputQueueY.DeQue<DT_X>();

        DataCopy(
            outputGMY[progress * this->tileLength],
            yLocal,
            this->tileLength);

        outputQueueY.FreeTensor(yLocal);
    }

    // fp32输入
    __aicore__ inline void LoadFloat(
        const LocalTensor<float>& dst,
        const LocalTensor<float>& src)
    {
        DataCopy(dst, src, this->tileLength);
    }

    // fp16输入，转换成fp32
    __aicore__ inline void LoadFloat(
        const LocalTensor<float>& dst,
        const LocalTensor<half>& src)
    {
        Cast(
            dst,
            src,
            RoundMode::CAST_NONE,
            this->tileLength);
    }

    // fp32输出
    __aicore__ inline void StoreFloat(
        const LocalTensor<float>& dst,
        const LocalTensor<float>& src)
    {
        DataCopy(dst, src, this->tileLength);
    }

    // fp16输出，从fp32转换回来
    __aicore__ inline void StoreFloat(
        const LocalTensor<half>& dst,
        const LocalTensor<float>& src)
    {
        Cast(
            dst,
            src,
            RoundMode::CAST_NONE,
            this->tileLength);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    TBuf<QuePosition::VECCALC> tmpXBuffer;
    TBuf<QuePosition::VECCALC> expXBuffer;
    TBuf<QuePosition::VECCALC> expNegXBuffer;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

} // namespace NsTanh

#endif // TANH_H