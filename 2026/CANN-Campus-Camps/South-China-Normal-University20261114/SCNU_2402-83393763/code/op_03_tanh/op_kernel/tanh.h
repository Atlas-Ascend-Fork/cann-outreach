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

#include <type_traits>

namespace NsTanh {

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t COPY_ALIGNMENT = 32;
constexpr uint32_t ELEMENTS_PER_GROUP = 64; // float32 矢量比较的 256 B 对齐
constexpr uint32_t MAX_TILE_LENGTH = 1024;

template <typename DT_X>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
    {
        if (tilingData->totalNum <= 0 || tilingData->blockFactor <= 0 ||
            tilingData->ubFactor <= 0) {
            return;
        }

        const uint64_t totalLength = static_cast<uint64_t>(tilingData->totalNum);
        const uint64_t blockFactor = static_cast<uint64_t>(tilingData->blockFactor);
        const uint64_t coreIndex = AscendC::GetBlockIdx();
        if (coreIndex > (totalLength - 1) / blockFactor) {
            return;
        }

        const uint64_t start = coreIndex * blockFactor;
        const uint64_t remaining = totalLength - start;
        blockLength = remaining < blockFactor ? remaining : blockFactor;

        const uint64_t ubFactor = static_cast<uint64_t>(tilingData->ubFactor);
        const uint32_t boundedTile = ubFactor < MAX_TILE_LENGTH
            ? static_cast<uint32_t>(ubFactor) : MAX_TILE_LENGTH;
        tileLength = CeilDiv(boundedTile, ELEMENTS_PER_GROUP) * ELEMENTS_PER_GROUP;

        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + start, blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + start, blockLength);

        pipe.InitBuffer(inputQueueX, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, tileLength * sizeof(DT_X));

        const uint32_t fp32Bytes = tileLength * sizeof(float);
        pipe.InitBuffer(absBuffer, fp32Bytes);
        pipe.InitBuffer(expBuffer, fp32Bytes);
        pipe.InitBuffer(denomBuffer, fp32Bytes);
        pipe.InitBuffer(magnitudeBuffer, fp32Bytes);
        pipe.InitBuffer(maskBuffer, CeilDiv(tileLength / 8, COPY_ALIGNMENT) * COPY_ALIGNMENT);
        if constexpr (std::is_same_v<DT_X, half>) {
            pipe.InitBuffer(floatInputBuffer, fp32Bytes);
            pipe.InitBuffer(floatOutputBuffer, fp32Bytes);
        }
    }

    __aicore__ inline void Process()
    {
        for (uint64_t offset = 0; offset < blockLength;) {
            const uint64_t remaining = blockLength - offset;
            const uint32_t count = remaining < tileLength ? static_cast<uint32_t>(remaining) : tileLength;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
            offset += count;
        }
    }

private:
    __aicore__ static inline uint32_t CeilDiv(uint32_t value, uint32_t divisor)
    {
        return value / divisor + (value % divisor != 0);
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        const uint32_t bytes = count * sizeof(DT_X);
        if (bytes % COPY_ALIGNMENT == 0) {
            DataCopy(xLocal, inputGMX[offset], count);
        } else {
            const DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
            constexpr uint32_t copyElements = COPY_ALIGNMENT / sizeof(DT_X);
            const auto rightPadding = static_cast<uint8_t>(copyElements - count % copyElements);
            const DataCopyPadExtParams<DT_X> padParams{
                true, 0, rightPadding, static_cast<DT_X>(0)};
            DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
        }
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        constexpr uint32_t copyElements = COPY_ALIGNMENT / sizeof(DT_X);
        const uint32_t dmaCount = CeilDiv(count, copyElements) * copyElements;
        const uint32_t vectorCount = CeilDiv(count, ELEMENTS_PER_GROUP) * ELEMENTS_PER_GROUP;
        // DataCopyPad 补齐到 32 B。矢量比较所需的其余 UB 元素单独补零。
        if (dmaCount < vectorCount) {
            Duplicate(xLocal[dmaCount], static_cast<DT_X>(0), vectorCount - dmaCount);
        }

        if constexpr (std::is_same_v<DT_X, half>) {
            LocalTensor<float> xFloat = floatInputBuffer.Get<float>();
            LocalTensor<float> yFloat = floatOutputBuffer.Get<float>();
            Cast(xFloat, xLocal, RoundMode::CAST_NONE, vectorCount);
            ComputeFloat(yFloat, xFloat, vectorCount);
            Cast(yLocal, yFloat, RoundMode::CAST_RINT, vectorCount);
        } else {
            ComputeFloat(yLocal, xLocal, vectorCount);
        }
        outputQueueY.EnQue(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeFloat(
        const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t count)
    {
        LocalTensor<float> absX = absBuffer.Get<float>();
        LocalTensor<float> expValue = expBuffer.Get<float>();
        LocalTensor<float> denom = denomBuffer.Get<float>();
        LocalTensor<float> magnitude = magnitudeBuffer.Get<float>();
        LocalTensor<uint8_t> mask = maskBuffer.Get<uint8_t>();

        // e = exp(-2 * |x|)，|tanh(x)| = (1 - e) / (1 + e)。
        Abs(absX, src, count);
        Muls(denom, absX, -2.0f, count);
        Exp(expValue, denom, count);
        Adds(denom, expValue, 1.0f, count);
        Muls(expValue, expValue, -1.0f, count);
        Adds(expValue, expValue, 1.0f, count);
        Div(magnitude, expValue, denom, count);

        Muls(expValue, magnitude, -1.0f, count);
        CompareScalar(mask, src, 0.0f, CMPMODE::LT, count);
        Select(denom, mask, expValue, magnitude, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);

        // |x| < 1/16 时用五阶泰勒式，避免 1 - e 在零附近损失有效位。
        Mul(expValue, src, src, count);
        Muls(magnitude, expValue, 2.0f / 15.0f, count);
        Adds(magnitude, magnitude, -1.0f / 3.0f, count);
        CompareScalar(mask, absX, 0.0625f, CMPMODE::LT, count);
        Mul(absX, magnitude, expValue, count);
        Adds(absX, absX, 1.0f, count);
        Mul(expValue, absX, src, count);
        Select(dst, mask, expValue, denom, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        const uint32_t bytes = count * sizeof(DT_X);
        if (bytes % COPY_ALIGNMENT == 0) {
            DataCopy(outputGMY[offset], yLocal, count);
        } else {
            const DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
            DataCopyPad(outputGMY[offset], yLocal, copyParams);
        }
        outputQueueY.FreeTensor(yLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> absBuffer;
    TBuf<TPosition::VECCALC> expBuffer;
    TBuf<TPosition::VECCALC> denomBuffer;
    TBuf<TPosition::VECCALC> magnitudeBuffer;
    TBuf<TPosition::VECCALC> maskBuffer;
    TBuf<TPosition::VECCALC> floatInputBuffer;
    TBuf<TPosition::VECCALC> floatOutputBuffer;
    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint64_t blockLength = 0;
    uint32_t tileLength = 0;
};

} // namespace NsTanh
#endif // TANH_H
