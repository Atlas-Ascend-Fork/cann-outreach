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

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    ////////////////////////////////////////////////////////////////////bg
    // 计算过程所需临时变量：统一在 float 域计算（fp16 需先升精度）
    // 迁移依据：day1下午（test2）/03.05_tiling_template_attr_tbuf_workspace.ipynb
    //   "新增三个 TBuf 类型的成员变量 tmpBuf0、tmpBuf1、tmpBuf2"
    TBuf<TPosition::VECCALC> tmpBufSrc;
    TBuf<TPosition::VECCALC> tmpBufExpX;
    TBuf<TPosition::VECCALC> tmpBufExpNegX;
    ////////////////////////////////////////////////////////////////////end
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0; // 本核实际处理的元素数
    int64_t ubLength_ = 0;    // 每次 UB 循环处理的元素数
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    // 迁移依据：test4/op_kernel/relu.h:54-88（读 blockFactor/ubFactor、blockIdx 偏移、UB 划分）
    this->ubLength_ = tilingData->ubFactor;

    // tests/ut 只读，kernel UT 固定传入 ubFactor=16384（=全部元素），远超 UB 预算；
    // 这里按 UB 容量截断单块长度，保证 2 队列 × BUFFER_NUM 块 + 3 个 float 临时块能同时放下。
    // 预算口径与 op_host/tanh_tiling.cpp 的 BLOCK_PER_CORE 一致。
    constexpr int64_t MAX_UB_BYTES = 192 * 1024;   // 910B 单核 UB 192KB
    constexpr int64_t BLOCK_PER_CORE = 2 * BUFFER_NUM + 3;
    int64_t maxElems = MAX_UB_BYTES / BLOCK_PER_CORE / static_cast<int64_t>(sizeof(float));
    if (this->ubLength_ > maxElems) {
        this->ubLength_ = maxElems;
    }
    // 本核区间起点 = blockIdx × 每核长度（每核长度取 host 的 blockFactor）
    const int64_t blockFactor = tilingData->blockFactor;
    int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t startOffset = blockIdx * blockFactor;

    // 尾核实际长度按 totalNum 截断，避免越界（relu.h:69-77 的同款处理）
    int64_t currentLength = tilingData->totalNum - startOffset;
    if (currentLength > blockFactor) {
        currentLength = blockFactor;
    }
    if (currentLength <= 0) {
        this->blockLength_ = 0;
        return;
    }
    this->blockLength_ = currentLength;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + startOffset, this->blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + startOffset, this->blockLength_);

    // 队列与临时空间统一按 ubLength_ 划分（队列 BUFFER_NUM 块，TBuf 一块）
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(this->ubLength_ * sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(this->ubLength_ * sizeof(T)));
    pipe.InitBuffer(tmpBufSrc, static_cast<uint32_t>(this->ubLength_ * sizeof(float)));
    pipe.InitBuffer(tmpBufExpX, static_cast<uint32_t>(this->ubLength_ * sizeof(float)));
    pipe.InitBuffer(tmpBufExpNegX, static_cast<uint32_t>(this->ubLength_ * sizeof(float)));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    // 迁移依据：test4/op_kernel/relu.h:94-111（AllocTensor → DataCopyPad → EnQue）
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();

    // CANN 9.0.0 的 DataCopyExtParams 构造函数需 5 个参数（第 5 位为预留），只能声明一次
    // 依据：cannbot-skills/ops/ascendc-api-best-practices/references/api-datacopy.md:87
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX[progress * this->ubLength_], copyParams, padParams);

    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    // 小算子链依据：day2上午/07_Troubleshooting/answer/07.05_answer.txt:53-64（Muls→Exp→Sub 写法）
    // float 域 + Cast 依据：day1下午（test2）/03.05_tiling_template_attr_tbuf_workspace.ipynb
    // Div 接口依据：cannbot-skills/.../TileLang-AscendC-API-Mapping.md:43
    // 公式：tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<float> srcLocal = tmpBufSrc.Get<float>();
    LocalTensor<float> expXLocal = tmpBufExpX.Get<float>();
    LocalTensor<float> expNegXLocal = tmpBufExpNegX.Get<float>();
    const int32_t count = static_cast<int32_t>(currentNum);

    if constexpr (sizeof(T) == sizeof(float)) {
        // float 输入：直接以输入/输出 tensor 计算
        Exp(expXLocal, xLocal, count);                    // expX = exp(x)
        Muls(expNegXLocal, xLocal, (float)-1.0, count);   // expNegX = -x
        Exp(expNegXLocal, expNegXLocal, count);           // expNegX = exp(-x)
        Sub(yLocal, expXLocal, expNegXLocal, count);      // 分子
        Add(expXLocal, expXLocal, expNegXLocal, count);   // 分母
        Div(yLocal, yLocal, expXLocal, count);            // tanh
    } else {
        // float16 输入：升 float 计算，结果再降回 float16（题目要求支持 Float16）
        Cast(srcLocal, xLocal, RoundMode::CAST_NONE, count);
        Exp(expXLocal, srcLocal, count);
        Muls(expNegXLocal, srcLocal, (float)-1.0, count);
        Exp(expNegXLocal, expNegXLocal, count);
        Sub(srcLocal, expXLocal, expNegXLocal, count);
        Add(expXLocal, expXLocal, expNegXLocal, count);
        Div(srcLocal, srcLocal, expXLocal, count);
        Cast(yLocal, srcLocal, RoundMode::CAST_RINT, count);
    }

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    // 迁移依据：test4/op_kernel/relu.h:138-148（DeQue → DataCopyPad → FreeTensor）
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();

    // 与 CopyIn 一致：5 参构造函数，只声明一次
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[progress * this->ubLength_], yLocal, copyParams);

    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    // TODO: 实现 Process 逻辑
    // 迁移依据：test4/op_kernel/relu.h:154-185（块循环 + DoubleBuffer 预取下一块）
    if (this->blockLength_ <= 0 || this->ubLength_ <= 0) {
        return;
    }

    // 块数向上取整，只有最后一块可能不满
    int64_t tileNum = (this->blockLength_ + this->ubLength_ - 1) / this->ubLength_;
    int64_t lastNum = this->blockLength_ - (tileNum - 1) * this->ubLength_;

    CopyIn(0, (tileNum == 1) ? lastNum : this->ubLength_);
    for (int64_t i = 0; i < tileNum; i++) {
        int64_t curNum = (i == tileNum - 1) ? lastNum : this->ubLength_;

        // 先取下一块，使 MTE2 搬运与 Vector 计算并行（依赖队列深度 BUFFER_NUM = 2）
        if (i + 1 < tileNum) {
            int64_t nextNum = ((i + 1) == tileNum - 1) ? lastNum : this->ubLength_;
            CopyIn(i + 1, nextNum);
        }

        Compute(curNum);
        CopyOut(i, curNum);
    }
}

} // namespace NsTanh
#endif // TANH_H