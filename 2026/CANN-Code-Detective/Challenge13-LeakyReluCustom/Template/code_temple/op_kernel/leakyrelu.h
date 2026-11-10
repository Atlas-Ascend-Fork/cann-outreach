/*!
 * \file leakyrelu.h
 * \brief Leakyrelu 算子 kernel 类定义
 *
 * 计算公式：y = max(x, 0) + negativeSlope * min(x, 0)
 * 与 ReLU 等单指令算子不同，需要临时 Tensor 完成 Max/Min/Mul/Add 复合计算
 */

#ifndef LEAKYRELU_H
#define LEAKYRELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "leakyrelu_tiling_data.h"
#include "leakyrelu_tiling_key.h"

namespace NsLeakyrelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1; // 每队列 buffer 块数（双缓冲）

template <typename DT_X>
class Leakyrelu {
public:
    __aicore__ inline Leakyrelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const LeakyreluTilingData* tilingData)
    {
        // TODO 考生自行补齐：
        // 1. 属性 negativeSlope 经 tilingData 下发到 kernel 侧
        // 2. 计算每核处理的数据长度（多核并行）
        // 3. 设置每个核的 Global Memory 起始地址（关键的多核切分逻辑）
        // 4. 为队列分配 UB 内存，并为复合计算分配临时 Tensor
    }

    __aicore__ inline void Process()
    {
        // 循环次数 = tileNum × BUFFER_NUM（tileLength 已按 BUFFER_NUM 折半）
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // TODO 考生自行补齐
    }

    __aicore__ inline void Compute()
    {
        // TODO 考生自行补齐：
        // y = max(x, 0) + α * min(x, 0)，可用 Duplicate / Max / Min / Mul / Add 复合实现
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        // TODO 考生自行补齐
    }

private:
    // TODO 考生自行补齐（注意：除队列外还需临时 Tensor 用的 TBuf 成员）
};

} // namespace NsLeakyrelu
#endif // LEAKYRELU_H
