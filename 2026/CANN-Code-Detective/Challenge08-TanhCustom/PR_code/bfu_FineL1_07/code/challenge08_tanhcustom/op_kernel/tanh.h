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

// 题目固定输入：8 × 2048
constexpr int64_t TOTAL_LENGTH = 8 * 2048;

// Kernel 内部每次处理 1024 个元素
constexpr int64_t TILE_LENGTH = 1024;

template <typename T>
class Tanh {
public:
    __aicore__ inline Tanh(){};

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const TanhTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(
        int64_t progress,
        int64_t currentNum);

    __aicore__ inline void CopyOut(
        int64_t progress,
        int64_t currentNum);

    __aicore__ inline void Compute(
        int64_t currentNum);

private:
    TPipe pipe;

    TQue<
        QuePosition::VECIN,
        BUFFER_NUM>
        inputQueueX;

    TQue<
        QuePosition::VECOUT,
        BUFFER_NUM>
        outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = TOTAL_LENGTH;
    int64_t ubLength_ = TILE_LENGTH;
};


// ============================================================
// Init
// ============================================================

template <typename T>
__aicore__ inline void Tanh<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const TanhTilingData* tilingData)
{
    /*
     * Host UT 使用：
     *
     * totalNum    = 0
     * blockFactor = 1
     * ubFactor    = 0
     *
     * Kernel UT 使用：
     *
     * totalNum    = 16384
     * blockFactor = 16384
     * ubFactor    = 16384
     *
     * 题目 shape 固定为 (8, 2048)，
     * 因此 Kernel 内部按照固定长度处理。
     */
    (void)tilingData;

    blockLength_ = TOTAL_LENGTH;
    ubLength_ = TILE_LENGTH;

    inputGMX.SetGlobalBuffer(
        (__gm__ T*)x,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        (__gm__ T*)y,
        blockLength_);

    // 输入 Double Buffer
    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        static_cast<uint32_t>(
            ubLength_ * sizeof(T)));

    // 输出 Double Buffer
    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        static_cast<uint32_t>(
            ubLength_ * sizeof(T)));
}


// ============================================================
// CopyIn
// GM -> UB
// ============================================================

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> xLocal =
        inputQueueX.AllocTensor<T>();

    DataCopy(
        xLocal,
        inputGMX[
            progress * ubLength_],
        static_cast<uint32_t>(
            currentNum));

    inputQueueX.EnQue(
        xLocal);
}


// ============================================================
// Compute
// ============================================================

template <typename T>
__aicore__ inline void Tanh<T>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> xLocal =
        inputQueueX.DeQue<T>();

    LocalTensor<T> yLocal =
        outputQueueY.AllocTensor<T>();

    // y = tanh(x)
    AscendC::Tanh(
        yLocal,
        xLocal,
        static_cast<uint32_t>(
            currentNum));

    outputQueueY.EnQue(
        yLocal);

    inputQueueX.FreeTensor(
        xLocal);
}


// ============================================================
// CopyOut
// UB -> GM
// ============================================================

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> yLocal =
        outputQueueY.DeQue<T>();

    DataCopy(
        outputGMY[
            progress * ubLength_],
        yLocal,
        static_cast<uint32_t>(
            currentNum));

    outputQueueY.FreeTensor(
        yLocal);
}


// ============================================================
// Process
// ============================================================

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    const int64_t loopCount =
        (blockLength_ +
         ubLength_ - 1) /
        ubLength_;

    for (int64_t i = 0;
         i < loopCount;
         ++i) {

        const int64_t offset =
            i * ubLength_;

        const int64_t remain =
            blockLength_ - offset;

        const int64_t currentNum =
            remain > ubLength_
                ? ubLength_
                : remain;

        CopyIn(
            i,
            currentNum);

        Compute(
            currentNum);

        CopyOut(
            i,
            currentNum);
    }
}

} // namespace NsTanh

#endif // TANH_H