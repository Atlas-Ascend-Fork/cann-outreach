/*!
 * \file tanh.h
 * \brief 使用小算子拼接实现 Tanh，支持 float16 和 float32
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

// float 输入直接作为 FP32 计算张量。
__aicore__ inline LocalTensor<float> TanhToFloat(
    const LocalTensor<float>& src,
    LocalTensor<float>,
    uint32_t)
{
    return src;
}

// half 输入先转换成 FP32。
__aicore__ inline LocalTensor<float> TanhToFloat(
    const LocalTensor<half>& src,
    LocalTensor<float> scratch,
    uint32_t count)
{
    Cast(scratch, src, RoundMode::CAST_NONE, count);
    return scratch;
}

// FP32 结果写入 float 输出张量。
__aicore__ inline void TanhFromFloat(
    const LocalTensor<float>& dst,
    const LocalTensor<float>& src,
    uint32_t count)
{
    Muls(dst, src, 1.0f, count);
}

// FP32 结果转换为 half 输出。
__aicore__ inline void TanhFromFloat(
    const LocalTensor<half>& dst,
    const LocalTensor<float>& src,
    uint32_t count)
{
    Cast(dst, src, RoundMode::CAST_RINT, count);
}

template <typename DT_X>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const TanhTilingData* tilingData)
    {
        // 与平台的三字段 Tiling 结构保持一致。
        blockLength =
            static_cast<uint32_t>(tilingData->blockFactor);

        tileLength =
            static_cast<uint32_t>(tilingData->ubFactor);

        // blockFactor 和 ubFactor 的单位都是元素数。
        const uint32_t offset =
            blockLength * AscendC::GetBlockIdx();

        inputGMX.SetGlobalBuffer(
            (__gm__ DT_X*)x + offset,
            blockLength);

        outputGMY.SetGlobalBuffer(
            (__gm__ DT_X*)y + offset,
            blockLength);

        // 输入和输出各使用两个 buffer。
        pipe.InitBuffer(
            inputQueueX,
            BUFFER_NUM,
            tileLength * sizeof(DT_X));

        pipe.InitBuffer(
            outputQueueY,
            BUFFER_NUM,
            tileLength * sizeof(DT_X));

        // 中间计算统一使用 FP32。
        const uint32_t floatBytes =
            tileLength * sizeof(float);

        pipe.InitBuffer(xBuffer, floatBytes);
        pipe.InitBuffer(absBuffer, floatBytes);
        pipe.InitBuffer(tBuffer, floatBytes);
        pipe.InitBuffer(uBuffer, floatBytes);
        pipe.InitBuffer(resultBuffer, floatBytes);

        // 每个元素对应一个比较结果 bit。
        // 分配大小向上对齐到 32 字节。
        const uint32_t maskBytes =
            ((tileLength + 255U) / 256U) * 32U;

        pipe.InitBuffer(maskBuffer, maskBytes);
    }

    __aicore__ inline void Process()
    {
        // ubFactor 已经是单次处理长度，不再除以 BUFFER_NUM。
        const int32_t loopCount = blockLength / tileLength;

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
            inputGMX[progress * tileLength],
            tileLength);

        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal =
            inputQueueX.DeQue<DT_X>();

        LocalTensor<DT_X> yLocal =
            outputQueueY.AllocTensor<DT_X>();

        LocalTensor<float> x =
            xBuffer.Get<float>();

        LocalTensor<float> a =
            absBuffer.Get<float>();

        LocalTensor<float> t =
            tBuffer.Get<float>();

        LocalTensor<float> u =
            uBuffer.Get<float>();

        LocalTensor<float> r =
            resultBuffer.Get<float>();

        LocalTensor<uint8_t> mask =
            maskBuffer.Get<uint8_t>();

        // FP16 输入转成 FP32；FP32 输入直接参与计算。
        x = TanhToFloat(xLocal, x, tileLength);

        // a = |x|
        // t = exp(-2|x|)
        // 指数参数非正，避免正指数溢出。
        Abs(a, x, tileLength);
        Muls(t, a, -2.0f, tileLength);
        Exp(t, t, tileLength);

        // |tanh(x)| = (1 - t) / (1 + t)
        Adds(u, t, 1.0f, tileLength);
        Muls(r, t, -1.0f, tileLength);
        Adds(r, r, 1.0f, tileLength);
        Div(t, r, u, tileLength);

        // 根据输入的正负恢复符号。
        Muls(u, t, -1.0f, tileLength);

        CompareScalar(
            mask,
            x,
            0.0f,
            CMPMODE::LT,
            tileLength);

        Select(
            r,
            mask,
            u,
            t,
            SELMODE::VSEL_TENSOR_TENSOR_MODE,
            tileLength);

        // |x| < 0.125 时使用多项式，
        // 降低零附近 1-exp(-2|x|) 的相消误差。
        CompareScalar(
            mask,
            a,
            0.125f,
            CMPMODE::LT,
            tileLength);

        // tanh(x) ≈ x*(1 - x²/3 + 2x⁴/15 - 17x⁶/315)
        // 使用 Horner 形式计算。
        // 限制未被选用分支的中间值，避免高次幂溢出。
        Mins(a, a, 0.125f, tileLength);
        Mul(a, a, a, tileLength);

        Muls(t, a, -17.0f / 315.0f, tileLength);
        Adds(t, t, 2.0f / 15.0f, tileLength);
        Mul(t, t, a, tileLength);
        Adds(t, t, -1.0f / 3.0f, tileLength);
        Mul(t, t, a, tileLength);
        Adds(t, t, 1.0f, tileLength);
        Mul(u, t, x, tileLength);

        // 小区间选择多项式结果，其余选择指数公式结果。
        Select(
            t,
            mask,
            u,
            r,
            SELMODE::VSEL_TENSOR_TENSOR_MODE,
            tileLength);

        // 输出类型与输入一致。
        TanhFromFloat(yLocal, t, tileLength);

        outputQueueY.EnQue(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DT_X> yLocal =
            outputQueueY.DeQue<DT_X>();

        DataCopy(
            outputGMY[progress * tileLength],
            yLocal,
            tileLength);

        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    TBuf<QuePosition::VECCALC> xBuffer;
    TBuf<QuePosition::VECCALC> absBuffer;
    TBuf<QuePosition::VECCALC> tBuffer;
    TBuf<QuePosition::VECCALC> uBuffer;
    TBuf<QuePosition::VECCALC> resultBuffer;
    TBuf<QuePosition::VECCALC> maskBuffer;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength;
    uint32_t tileLength;
};

} // namespace NsTanh

#endif // TANH_H