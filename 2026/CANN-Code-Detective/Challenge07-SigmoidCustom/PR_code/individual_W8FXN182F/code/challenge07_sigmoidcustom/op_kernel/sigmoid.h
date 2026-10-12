/*!
 * \file sigmoid.h
 * \brief Sigmoid 算子 kernel 类定义
 */

#ifndef SIGMOID_H
#define SIGMOID_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sigmoid_tiling_data.h"
#include "sigmoid_tiling_key.h"

namespace NsSigmoid {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// 类型特征：编译期判断 half
template <typename T> struct IsHalf { static constexpr bool value = false; };
template <> struct IsHalf<half> { static constexpr bool value = true; };

template <typename T>
class Sigmoid {
public:
    __aicore__ inline Sigmoid(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    // FP32 工作缓冲（half 分支中间计算 + 1.0 常量），共 8B/元素
    TQue<QuePosition::VECCALC, 1> tmpQueue;
    TQue<QuePosition::VECCALC, 1> oneQueue;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t tileNum_ = 0;        // 当前核内分块数
    int64_t lastTileLength_ = 0; // 核内最后一块实际长度
};

// 每个核处理的元素数：尾核用 lastBlockLength，其余用 blockFactor
template <typename T>
__aicore__ inline void Sigmoid<T>::Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData* tilingData)
{
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    blockLength_ = (blockIdx == blockNum - 1) ? tilingData->lastBlockLength : tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    // 核内分块数向上取整，最后一块可能不满 ubLength_
    tileNum_ = (blockLength_ + ubLength_ - 1) / ubLength_;
    lastTileLength_ = blockLength_ - (tileNum_ - 1) * ubLength_;

    // GM 起始地址按标准 blockFactor 步进，保证核间不重叠、不越界
    inputGMX.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * blockIdx, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * blockIdx, blockLength_);
    // 按满块 ubLength_ 分配 UB，尾块 <= ubLength_ 安全
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    // FP32 工作缓冲（tiling 侧已按 ubSize/16 限幅，总占用不超 UB）
    pipe.InitBuffer(tmpQueue, 1, ubLength_ * sizeof(float));
    pipe.InitBuffer(oneQueue, 1, ubLength_ * sizeof(float));
}

template <typename T>
__aicore__ inline void Sigmoid<T>::Process()
{
    int64_t loopCount = tileNum_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t curLen = (i == loopCount - 1) ? lastTileLength_ : ubLength_;
        CopyIn(i, curLen);
        Compute(curLen);
        CopyOut(i, curLen);
    }
}

template <typename T>
__aicore__ inline void Sigmoid<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], currentNum);
    inputQueueX.EnQue(xLocal);
}

// Sigmoid: y = 1 / (1 + exp(-x))
// 统一走 FP32 中间计算（规避 fp16 Exp 底数/溢出问题），并用 Duplicate(1)+Div 精确除法
// （不用 Reciprocal——其为近似指令，float 用例实测存在 ~1.4% 元素超差）
template <typename T>
__aicore__ inline void Sigmoid<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    if constexpr (IsHalf<T>::value) {
        // half: 升精度 FP32 计算，最后舍回 half
        LocalTensor<float> tmp = tmpQueue.AllocTensor<float>();
        LocalTensor<float> one = oneQueue.AllocTensor<float>();

        AscendC::Cast(tmp, xLocal, RoundMode::CAST_NONE, currentNum);  // half -> fp32
        AscendC::Muls(tmp, tmp, -1.0f, currentNum);                    // -x
        AscendC::Exp(tmp, tmp, currentNum);                            // exp(-x)（fp32 为 e 底）
        AscendC::Adds(tmp, tmp, 1.0f, currentNum);                     // 1 + exp(-x)
        AscendC::Duplicate(one, 1.0f, currentNum);
        AscendC::Div(one, one, tmp, currentNum);                       // 1 / (1 + exp(-x))
        AscendC::Cast(yLocal, one, RoundMode::CAST_RINT, currentNum);  // fp32 -> half

        tmpQueue.EnQue<float>(tmp);
        tmpQueue.DeQue<float>();
        tmpQueue.FreeTensor(tmp);
        oneQueue.EnQue<float>(one);
        oneQueue.DeQue<float>();
        oneQueue.FreeTensor(one);
    } else {
        // float: 直接 FP32 计算
        AscendC::Muls(xLocal, xLocal, -1.0f, currentNum);              // -x
        AscendC::Exp(xLocal, xLocal, currentNum);                      // exp(-x)
        AscendC::Adds(xLocal, xLocal, 1.0f, currentNum);               // 1 + exp(-x)
        LocalTensor<float> one = oneQueue.AllocTensor<float>();
        AscendC::Duplicate(one, 1.0f, currentNum);
        AscendC::Div(yLocal, one, xLocal, currentNum);                 // 1 / (1 + exp(-x))
        oneQueue.EnQue<float>(one);
        oneQueue.DeQue<float>();
        oneQueue.FreeTensor(one);
    }

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

} // namespace NsSigmoid
#endif // SIGMOID_H
