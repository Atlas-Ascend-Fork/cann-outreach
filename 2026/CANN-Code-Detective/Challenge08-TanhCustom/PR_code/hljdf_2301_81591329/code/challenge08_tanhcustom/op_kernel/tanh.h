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

constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const TanhTilingData* tilingData)
    {
        blockLength_ =
            static_cast<uint32_t>(
                tilingData->blockFactor);

        tileLength_ =
            static_cast<uint32_t>(
                tilingData->ubFactor);

        tileNum_ =
            blockLength_ / tileLength_;

        const uint32_t gmOffset =
            GetBlockIdx() * blockLength_;

        xGm_.SetGlobalBuffer(
            (__gm__ T*)x + gmOffset,
            blockLength_);

        yGm_.SetGlobalBuffer(
            (__gm__ T*)y + gmOffset,
            blockLength_);

        pipe_.InitBuffer(
            inQueueX_,
            BUFFER_NUM,
            tileLength_ * sizeof(T));

        pipe_.InitBuffer(
            outQueueY_,
            BUFFER_NUM,
            tileLength_ * sizeof(T));

        /*
         * 中间计算统一准备 float Buffer。
         *
         * 对 FP16 输入：
         * half -> float -> 计算 -> half
         *
         * 避免半精度计算
         * exp(x)-exp(-x) 时精度损失过大。
         */
        pipe_.InitBuffer(
            work0Buf_,
            tileLength_ * sizeof(float));

        pipe_.InitBuffer(
            work1Buf_,
            tileLength_ * sizeof(float));

        pipe_.InitBuffer(
            work2Buf_,
            tileLength_ * sizeof(float));

        pipe_.InitBuffer(
            work3Buf_,
            tileLength_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0;
             i < tileNum_;
             ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(
        uint32_t progress)
    {
        LocalTensor<T> xLocal =
            inQueueX_.AllocTensor<T>();

        DataCopy(
            xLocal,
            xGm_[progress * tileLength_],
            tileLength_);

        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<T> xLocal =
            inQueueX_.DeQue<T>();

        LocalTensor<T> yLocal =
            outQueueY_.AllocTensor<T>();

        LocalTensor<float> work0Local =
            work0Buf_.Get<float>();

        LocalTensor<float> work1Local =
            work1Buf_.Get<float>();

        LocalTensor<float> work2Local =
            work2Buf_.Get<float>();

        LocalTensor<float> work3Local =
            work3Buf_.Get<float>();

        /*
         * tanh(x)
         *
         *       exp(x) - exp(-x)
         * y = -------------------
         *       exp(x) + exp(-x)
         */

        if constexpr (
            IsSameType<T, half>::value) {

            /*
             * ==========================
             * FP16
             * ==========================
             *
             * 先转换为 float，
             * 中间运算全部使用 float。
             */

            // work0 = float(x)
            Cast(
                work0Local,
                xLocal,
                RoundMode::CAST_NONE,
                tileLength_);

            // work1 = exp(x)
            Exp(
                work1Local,
                work0Local,
                tileLength_);

            // work2 = -x
            Muls(
                work2Local,
                work0Local,
                -1.0f,
                tileLength_);

            // work2 = exp(-x)
            Exp(
                work2Local,
                work2Local,
                tileLength_);

            // work3 = exp(x) - exp(-x)
            Sub(
                work3Local,
                work1Local,
                work2Local,
                tileLength_);

            // work1 = exp(x) + exp(-x)
            Add(
                work1Local,
                work1Local,
                work2Local,
                tileLength_);

            // work0 = numerator / denominator
            Div(
                work0Local,
                work3Local,
                work1Local,
                tileLength_);

            // float -> half
            Cast(
                yLocal,
                work0Local,
                RoundMode::CAST_NONE,
                tileLength_);

        } else {

            /*
             * ==========================
             * FP32
             * ==========================
             */

            // work1 = exp(x)
            Exp(
                work1Local,
                xLocal,
                tileLength_);

            // work2 = -x
            Muls(
                work2Local,
                xLocal,
                -1.0f,
                tileLength_);

            // work2 = exp(-x)
            Exp(
                work2Local,
                work2Local,
                tileLength_);

            // work3 = exp(x) - exp(-x)
            Sub(
                work3Local,
                work1Local,
                work2Local,
                tileLength_);

            // work1 = exp(x) + exp(-x)
            Add(
                work1Local,
                work1Local,
                work2Local,
                tileLength_);

            // y = numerator / denominator
            Div(
                yLocal,
                work3Local,
                work1Local,
                tileLength_);
        }

        outQueueY_.EnQue(yLocal);

        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(
        uint32_t progress)
    {
        LocalTensor<T> yLocal =
            outQueueY_.DeQue<T>();

        DataCopy(
            yGm_[progress * tileLength_],
            yLocal,
            tileLength_);

        outQueueY_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;

    TQue<
        QuePosition::VECIN,
        BUFFER_NUM>
        inQueueX_;

    TQue<
        QuePosition::VECOUT,
        BUFFER_NUM>
        outQueueY_;

    TBuf<TPosition::VECCALC>
        work0Buf_;

    TBuf<TPosition::VECCALC>
        work1Buf_;

    TBuf<TPosition::VECCALC>
        work2Buf_;

    TBuf<TPosition::VECCALC>
        work3Buf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    uint32_t blockLength_ = 0;
    uint32_t tileLength_ = 0;
    uint32_t tileNum_ = 0;
};

} // namespace NsTanh

#endif // TANH_H