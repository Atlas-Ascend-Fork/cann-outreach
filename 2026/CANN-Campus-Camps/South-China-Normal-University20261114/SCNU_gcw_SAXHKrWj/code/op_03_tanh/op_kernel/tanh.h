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
// 单次 UB 循环处理的元素数上限，防止 Tiling 传入过大的 ubFactor 导致 UB 申请越界
constexpr int64_t MAX_UB_LENGTH = 4096;

template <typename T>
class Tanh {
public:
    __aicore__ inline Tanh(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    // Tanh 高阶接口计算过程中的临时缓存
    TBuf<TPosition::VECCALC> tmpBuf;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t offset = blockIdx * tilingData->blockFactor;

    // 本核负责的数据长度；最后一个核可能不足 blockFactor，需要截断防止越界
    blockLength_ = tilingData->blockFactor;
    if (offset >= tilingData->totalNum) {
        blockLength_ = 0;
    } else if (offset + blockLength_ > tilingData->totalNum) {
        blockLength_ = tilingData->totalNum - offset;
    }
    if (blockLength_ <= 0) {
        return;
    }

    // 单次 UB 循环长度不能超过本核数据量，同时受 UB 容量上限约束
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0 || ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }
    if (ubLength_ > MAX_UB_LENGTH) {
        ubLength_ = MAX_UB_LENGTH;
    }

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    // Tanh 内部需要存放中间变量，按每元素 2 份 float 预留临时空间
    pipe.InitBuffer(tmpBuf, static_cast<uint32_t>(ubLength_ * sizeof(float) * 2));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[progress * ubLength_], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    AscendC::LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();

    // 调用 Ascend C 高阶 Tanh 接口：y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    // 第二个模板参数 isReuseSource=false，表示不复用输入空间
    AscendC::Tanh<T, false>(yLocal, xLocal, tmpLocal, static_cast<uint32_t>(currentNum));

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    AscendC::DataCopy(outputGMY[progress * ubLength_], yLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }

    const int64_t loopCount = blockLength_ / ubLength_;
    const int64_t tailNum = blockLength_ - loopCount * ubLength_;

    // 整块循环：搬运、计算、写回流水并行
    for (int64_t i = 0; i < loopCount; i++) {
        CopyIn(i, ubLength_);
        Compute(ubLength_);
        CopyOut(i, ubLength_);
    }

    // 尾块处理
    if (tailNum > 0) {
        CopyIn(loopCount, tailNum);
        Compute(tailNum);
        CopyOut(loopCount, tailNum);
    }
}

} // namespace NsTanh
#endif // TANH_H
