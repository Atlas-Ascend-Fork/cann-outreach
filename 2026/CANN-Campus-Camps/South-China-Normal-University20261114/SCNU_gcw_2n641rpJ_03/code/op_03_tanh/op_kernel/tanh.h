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

constexpr int32_t BUFFER_NUM = 2; // 每队列 buffer 块数（双缓冲）

// Tanh kernel：
//   - 多核切分：每核处理 tilingData->blockFactor 个元素（host 侧按 8 核均分）
//   - 单核内按 tilingData->ubFactor 分块做「搬入 -> 计算 -> 搬出」，配合 BUFFER_NUM=2
//     的双缓冲，使 MTE2 搬入 / Vector 计算 / MTE3 搬出 三段流水重叠
//   - 计算直接调用 AscendC 高阶接口 Tanh：y = tanh(x) = (e^(2x)-1)/(e^(2x)+1)，
//     接口内部自动申请临时空间，float16 输入由接口内部 Cast 到 float32 计算后再转回
template <typename T>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
    {
        this->blockLength = tilingData->blockFactor; // 每核元素数
        this->ubLength = tilingData->ubFactor;       // 单次处理的元素数
        if ((this->blockLength <= 0) || (this->ubLength <= 0)) {
            this->loopCount = 0; // 空 tensor 保护
        } else {
            this->loopCount = this->blockLength / this->ubLength;
        }
        // 多核并行：按核号设置 Global Memory 偏移
        inputGMX.SetGlobalBuffer((__gm__ T*)x + GetBlockIdx() * this->blockLength, this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ T*)y + GetBlockIdx() * this->blockLength, this->blockLength);
        // 分配 double buffer 队列（片上 UB）
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength * sizeof(T));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        // 单核内按 UB 块循环，覆盖该核负责的全部元素
        for (int64_t i = 0; i < this->loopCount; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress)
    {
        LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
        DataCopy(xLocal, inputGMX[progress * this->ubLength], this->ubLength);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
        // 显式加 AscendC:: 限定，避免与本类名 Tanh 冲突；临时空间由框架内部申请
        AscendC::Tanh(yLocal, xLocal, this->ubLength);
        outputQueueY.EnQue(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress)
    {
        LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
        DataCopy(outputGMY[progress * this->ubLength], yLocal, this->ubLength);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength = 0; // 每核元素数
    int64_t ubLength = 0;    // 单次处理元素数
    int64_t loopCount = 0;   // 单核内循环次数
};

} // namespace NsTanh

#endif // TANH_H
