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
    __aicore__ inline void TanhCore(LocalTensor<float> src, LocalTensor<float> dst, uint32_t count);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    // 计算过程所需临时变量（VECCALC 区，统一为 float 计算域）
    //   tmpBuf0：fp16 输入转 float 后的副本（fp32 场景不占用）
    //   tmpBuf1：exp(x)
    //   tmpBuf2：exp(-x)
    //   tmpBuf3：exp(x) + exp(-x)
    //   tmpBuf4：exp(x) - exp(-x)  /  暂存 -x
    TBuf<TPosition::VECCALC> tmpBuf0;
    TBuf<TPosition::VECCALC> tmpBuf1;
    TBuf<TPosition::VECCALC> tmpBuf2;
    TBuf<TPosition::VECCALC> tmpBuf3;
    TBuf<TPosition::VECCALC> tmpBuf4;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// ---------------------------------------------------------------------------
// 工具函数：字节数向上取整到 32B，满足 UB 申请与 DataCopy 的对齐要求
// ---------------------------------------------------------------------------
__aicore__ inline uint32_t AlignBytesUp32(int64_t elems, int64_t elemBytes)
{
    int64_t bytes = elems * elemBytes;
    return static_cast<uint32_t>((bytes + 31) / 32 * 32);
}

// ---------------------------------------------------------------------------
// Init：按 tiling 完成多核切分，并为队列与计算缓存申请 UB
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    int64_t totalNum = tilingData->totalNum;
    int64_t blockFactor = tilingData->blockFactor;
    int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());

    // 多核切分：本核从 blockIdx * blockFactor 开始，最多处理 blockFactor 个元素
    int64_t startOffset = blockIdx * blockFactor;
    if (startOffset >= totalNum) {
        // 该核分不到数据，Process 直接返回
        blockLength_ = 0;
        ubLength_ = 1;
        return;
    }
    blockLength_ = totalNum - startOffset;
    if (blockLength_ > blockFactor) {
        blockLength_ = blockFactor;
    }

    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0) {
        ubLength_ = blockLength_;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + startOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + startOffset, blockLength_);

    // 双缓冲：搬入、搬出各 2 块，使 MTE2 / V / MTE3 三条流水可重叠
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, AlignBytesUp32(ubLength_, sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, AlignBytesUp32(ubLength_, sizeof(T)));

    // 计算域缓存全部按 float 申请，fp32 / fp16 共用同一套 UB 规划
    pipe.InitBuffer(tmpBuf0, AlignBytesUp32(ubLength_, sizeof(float)));
    pipe.InitBuffer(tmpBuf1, AlignBytesUp32(ubLength_, sizeof(float)));
    pipe.InitBuffer(tmpBuf2, AlignBytesUp32(ubLength_, sizeof(float)));
    pipe.InitBuffer(tmpBuf3, AlignBytesUp32(ubLength_, sizeof(float)));
    pipe.InitBuffer(tmpBuf4, AlignBytesUp32(ubLength_, sizeof(float)));
}

// ---------------------------------------------------------------------------
// CopyIn：GM -> UB
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[progress], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

// ---------------------------------------------------------------------------
// TanhCore：在 float 计算域内用小算子拼接实现
//     y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
// 全程只用 Exp / Muls / Add / Sub / Div 五个基础矢量接口，
// 不使用 tanh 高阶 API，也不出现任何原地写回（dst 与源操作数不重叠）。
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void Tanh<T>::TanhCore(LocalTensor<float> src, LocalTensor<float> dst, uint32_t count)
{
    LocalTensor<float> expPos = tmpBuf1.Get<float>();
    LocalTensor<float> expNeg = tmpBuf2.Get<float>();
    LocalTensor<float> sumBuf = tmpBuf3.Get<float>();
    LocalTensor<float> tmpBuf = tmpBuf4.Get<float>();

    // exp(x)
    AscendC::Exp(expPos, src, count);
    // -x，先落到 tmpBuf，避免覆盖 src
    AscendC::Muls(tmpBuf, src, static_cast<float>(-1.0), count);
    // exp(-x)
    AscendC::Exp(expNeg, tmpBuf, count);
    // 分母：exp(x) + exp(-x)
    AscendC::Add(sumBuf, expPos, expNeg, count);
    // 分子：exp(x) - exp(-x)
    AscendC::Sub(tmpBuf, expPos, expNeg, count);
    // 相除得到 tanh
    AscendC::Div(dst, tmpBuf, sumBuf, count);
}

// ---------------------------------------------------------------------------
// Compute：取一块数据，按 dtype 决定是否需要 float 中转，然后回写到输出队列
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    uint32_t count = static_cast<uint32_t>(currentNum);

    if constexpr (sizeof(T) == sizeof(float)) {
        // fp32：输入本身就是 float，可直接原地参与计算
        TanhCore(xLocal, yLocal, count);
    } else {
        // fp16：先提升到 float 计算，避免 exp 在半精度下的精度损失，
        //       算完再一次性降回 fp16，结果等价于「fp32 精度 + 一次舍入」
        LocalTensor<float> xFloat = tmpBuf0.Get<float>();
        AscendC::Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, count);
        TanhCore(xFloat, xFloat, count);
        AscendC::Cast(yLocal, xFloat, AscendC::RoundMode::CAST_RINT, count);
    }

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

// ---------------------------------------------------------------------------
// CopyOut：UB -> GM
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    AscendC::DataCopy(outputGMY[progress], yLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

// ---------------------------------------------------------------------------
// Process：按 ubLength 分块循环，CopyIn -> Compute -> CopyOut
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t offset = i * ubLength_;
        int64_t currentNum = blockLength_ - offset;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H
