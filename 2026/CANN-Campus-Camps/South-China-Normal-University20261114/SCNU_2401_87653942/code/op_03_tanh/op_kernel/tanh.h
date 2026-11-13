/*!
 * \file tanh.h
 * \brief Tanh 算子核函数
 */

#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

template <typename T>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
    {
        const int64_t start = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
        if (start >= tilingData->totalNum) {
            return;
        }
        const int64_t remaining = tilingData->totalNum - start;
        blockLength = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
        tileLength = static_cast<uint32_t>(tilingData->ubFactor);
        inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + start, blockLength);
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + start, blockLength);
    }

    __aicore__ inline void Process()
    {
        if (blockLength == 0) {
            return;
        }

        TPipe pipe;
        TBuf<TPosition::VECIN> inputBuf;
        TBuf<TPosition::VECOUT> outputBuf;
        TBuf<TPosition::VECCALC> numeratorBuf;
        TBuf<TPosition::VECCALC> denominatorBuf;
        TBuf<TPosition::VECCALC> fp32Buf;
        pipe.InitBuffer(inputBuf, tileLength * sizeof(T));
        pipe.InitBuffer(outputBuf, tileLength * sizeof(T));
        pipe.InitBuffer(numeratorBuf, tileLength * sizeof(float));
        pipe.InitBuffer(denominatorBuf, tileLength * sizeof(float));
        if constexpr (sizeof(T) == sizeof(half)) {
            pipe.InitBuffer(fp32Buf, tileLength * sizeof(float));
        }
        auto xLocal = inputBuf.Get<T>();
        auto yLocal = outputBuf.Get<T>();
        auto numerator = numeratorBuf.Get<float>();
        auto denominator = denominatorBuf.Get<float>();
        LocalTensor<float> fp32;
        if constexpr (sizeof(T) == sizeof(half)) {
            fp32 = fp32Buf.Get<float>();
        }

        const auto inReady = pipe.FetchEventID<HardEvent::MTE2_V>();
        const auto outReady = pipe.FetchEventID<HardEvent::V_MTE3>();
        const bool multiTile = blockLength > tileLength;
        TEventID inputReusable = 0;
        TEventID outputReusable = 0;
        if (multiTile) {
            inputReusable = pipe.FetchEventID<HardEvent::V_MTE2>();
            outputReusable = pipe.FetchEventID<HardEvent::MTE3_V>();
        }

        for (int64_t offset = 0;;) {
            const int64_t remaining = blockLength - offset;
            const bool hasNext = remaining > tileLength;
            const uint32_t count = hasNext ? tileLength : static_cast<uint32_t>(remaining);
            const uint32_t bytes = count * sizeof(T);
            const bool aligned = bytes % 32 == 0;
            if (aligned) {
                DataCopy(xLocal, inputGm[offset], count);
            } else {
                const DataCopyExtParams copy{1, bytes, 0, 0, 0};
                const DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
                DataCopyPad(xLocal, inputGm[offset], copy, pad);
            }
            SetFlag<HardEvent::MTE2_V>(inReady);
            WaitFlag<HardEvent::MTE2_V>(inReady);
            if (offset != 0) {
                // 上一片写回结束后，才能覆盖输出缓冲区。
                WaitFlag<HardEvent::MTE3_V>(outputReusable);
            }

            if constexpr (sizeof(T) == sizeof(half)) {
                AscendC::Cast(fp32, xLocal, RoundMode::CAST_NONE, count);
                ComputeMagnitude(fp32, fp32, numerator, denominator, count);
                AscendC::Cast(yLocal, fp32, RoundMode::CAST_RINT, count);
                CopySign<uint16_t>(yLocal, xLocal, denominator, count);
            } else {
                ComputeMagnitude(yLocal, xLocal, numerator, denominator, count);
                CopySign<uint32_t>(yLocal, xLocal, denominator, count);
            }
            if (hasNext) {
                SetFlag<HardEvent::V_MTE2>(inputReusable);
            }
            SetFlag<HardEvent::V_MTE3>(outReady);
            WaitFlag<HardEvent::V_MTE3>(outReady);
            if (aligned) {
                DataCopy(outputGm[offset], yLocal, count);
            } else {
                const DataCopyExtParams copy{1, bytes, 0, 0, 0};
                DataCopyPad(outputGm[offset], yLocal, copy);
            }

            if (!hasNext) {
                break;
            }
            SetFlag<HardEvent::MTE3_V>(outputReusable);
            // 下一片搬入前，等待向量单元读完本片输入。
            WaitFlag<HardEvent::V_MTE2>(inputReusable);
            offset += count;
        }
    }

private:
    __aicore__ inline void ComputeMagnitude(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                            const LocalTensor<float>& numerator,
                                            const LocalTensor<float>& denominator, uint32_t count)
    {
        // |tanh(x)| = (1 - e) / (1 + e), e = exp(-2 * |x|).
        AscendC::Abs(numerator, src, count);
        AscendC::Muls(numerator, numerator, -2.0f, count);
        AscendC::Exp(numerator, numerator, count);
        AscendC::Adds(denominator, numerator, 1.0f, count);
        AscendC::Muls(numerator, numerator, -1.0f, count);
        AscendC::Adds(numerator, numerator, 1.0f, count);
        AscendC::Div(dst, numerator, denominator, count);
    }

    template <typename Bits>
    __aicore__ inline void CopySign(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                   const LocalTensor<float>& scratch, uint32_t count)
    {
        // 在原始精度上复制符号位，包含负零。分母缓冲在除法后可以复用。
        constexpr Bits signShift = static_cast<Bits>(sizeof(Bits) * 8 - 1);
        auto sign = scratch.template ReinterpretCast<Bits>();
        auto srcBits = src.template ReinterpretCast<Bits>();
        AscendC::ShiftRight(sign, srcBits, signShift, count);
        AscendC::ShiftLeft(sign, sign, signShift, count);

        // A2/A3 的 Or 按 16 位操作；FP32 元素对应两个 16 位字。
        auto dstWords = dst.template ReinterpretCast<uint16_t>();
        auto signWords = sign.template ReinterpretCast<uint16_t>();
        AscendC::Or(dstWords, dstWords, signWords, count * (sizeof(T) / sizeof(uint16_t)));
    }

    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;
    int64_t blockLength = 0;
    uint32_t tileLength = 0;
};

} // namespace NsTanh
#endif // TANH_H
