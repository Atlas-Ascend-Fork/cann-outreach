/*!
 * \file exp.h
 * \brief Exp 算子 kernel 类定义
 *
 * Exp 为逐元素一元算子： y = e^x
 *   - float32：直接调用 AscendC 向量 API AscendC::Exp
 *   - float16：Exp 本身按 float32 精度计算，先 Cast 到 fp32 求 Exp 再 Cast 回 fp16
 *     （与 CANN 官方 fp16 元素级算子（如官方 sigmoid 示例）的处理方式一致）
 *
 * 模板的 Process() 骨架保持不变：循环次数 = tileNum × BUFFER_NUM，
 * 因此每个 tile 的元素个数 tileLength 必须满足
 *     tileNum × BUFFER_NUM × tileLength == blockLength
 * 本实现中 BUFFER_NUM = 1，tileNum 由 blockLength / tileLength 推出。
 *
 * ★ 对齐说明：
 *   host 侧把 blockFactor / ubFactor 都按 32 字节对齐单位取整；
 *   但末尾核的 blockLength = totalNum - offset 是**截断值**，
 *   当 totalNum 不是对齐单位的整数倍时（例如 1001 个 float32 的尾核只有 9 个元素），
 *   该截断值不再 32 字节对齐 —— 这种尾块走 DataCopyPad（见 CopyIn / CopyOut），
 *   其 DMA 长度按字节给出，不再有 32 字节整数倍约束；
 *   UB 容量则一律按对齐后的 ubFactor 申请（见 Init 的 processNum）。
 */

#ifndef EXP_H
#define EXP_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "exp_tiling_data.h"
#include "exp_tiling_key.h"

namespace NsExp {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1; // 每队列 buffer 块数（双缓冲）

template <typename T>
class Exp {
public:
    __aicore__ inline Exp() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData)
    {
        // 本核负责的数据区间起点
        int64_t offset = AscendC::GetBlockIdx() * tilingData->blockFactor;

        blockLength_ = tilingData->blockFactor;
        // 防止最后一个核越界：blockFactor 超过剩余数据时截断
        if (offset + blockLength_ > tilingData->totalNum) {
            blockLength_ = tilingData->totalNum - offset;
        }
        if (blockLength_ <= 0) {
            blockLength_ = 0;
            tileLength_ = 0;
            tileNum = 0;
            return;
        }

        // 单个 tile 的元素个数（host 侧已按 32 字节对齐）
        tileLength_ = tilingData->ubFactor;
        if (tileLength_ > blockLength_) {
            tileLength_ = blockLength_;
        }
        // 最后一个 tile 可能不满，向上取整得到 tile 数量
        tileNum = (tileLength_ > 0) ? (blockLength_ + tileLength_ - 1) / tileLength_ : 0;

        // ★ UB 缓冲容量用 host 给的 ubFactor（已按 32 字节对齐），
        //   而不是可能被截断的 tileLength_：末尾核 blockLength 可能是非对齐值
        //   （如 1001 个 float32 的尾核为 9），用它申请会让 InitBuffer 收到非 32B 倍数。
        processNum = tilingData->ubFactor;
        if (processNum < tileLength_) {
            processNum = tileLength_;
        }

        inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
        outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);

        pipe.InitBuffer(inputQueueX, BUFFER_NUM, processNum * sizeof(T));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, processNum * sizeof(T));
        // fp16 路径需要一块 fp32 中间缓冲
        if (std::is_same_v<T, float> == false) {
            pipe.InitBuffer(floatBuf, processNum * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        // 不变式：BUFFER_NUM 必须为 1，tileNum 由 blockLength / tileLength 自算，
        // 每轮 progress 对应一个 tile（若把 BUFFER_NUM 改成 2，本循环会多跑一倍并越界）。
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    // 当前 tile 实际需要处理的元素个数（最后一个 tile 可能不满）
    __aicore__ inline int64_t TileCount(int32_t progress)
    {
        int64_t count = blockLength_ - static_cast<int64_t>(progress) * tileLength_;
        if (count > tileLength_) {
            count = tileLength_;
        }
        if (count < 0) {
            count = 0;
        }
        return count;
    }

    __aicore__ inline void CopyIn(int32_t progress)
    {
        int64_t offset = static_cast<int64_t>(progress) * tileLength_;
        curCount_ = TileCount(progress);

        LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
        // DataCopy 要求 32 字节整数倍的 DMA 粒度；末尾 tile 可能不满一个对齐单元
        // （例如 totalNum=1001 的 float32 尾块只有 9 个元素），此时改用 DataCopyPad。
        // 与 Challenge06/08 已合入实现的写法一致。
        constexpr int64_t alignNum = 32 / sizeof(T);
        if (curCount_ % alignNum == 0) {
            DataCopy(xLocal, inputGMX[offset], static_cast<uint32_t>(curCount_));
        } else {
            const DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(curCount_ * sizeof(T)), 0, 0, 0};
            const DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
        }
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        // curCount_ 由紧邻其前的 CopyIn 设置
        uint32_t count = static_cast<uint32_t>(curCount_);

        LocalTensor<T> xLocal = inputQueueX.DeQue<T>();

        if constexpr (std::is_same_v<T, float>) {
            LocalTensor<float> yLocal = outputQueueY.AllocTensor<float>();
            // float32 直接走向量 Exp
            AscendC::Exp(yLocal, xLocal, count);
            outputQueueY.EnQue(yLocal);
        } else {
            LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
            LocalTensor<float> xFloat = floatBuf.Get<float>();
            // fp16 -> fp32 -> Exp -> fp16，保证精度
            Cast(xFloat, xLocal, RoundMode::CAST_NONE, count);
            AscendC::Exp(xFloat, xFloat, count);
            Cast(yLocal, xFloat, RoundMode::CAST_RINT, count);
            outputQueueY.EnQue(yLocal);
        }

        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        int64_t offset = static_cast<int64_t>(progress) * tileLength_;
        int64_t count = TileCount(progress);

        LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
        // 与 CopyIn 对称：非对齐尾块走 DataCopyPad，避免 DMA 粒度违规。
        constexpr int64_t alignNum = 32 / sizeof(T);
        if (count % alignNum == 0) {
            DataCopy(outputGMY[offset], yLocal, static_cast<uint32_t>(count));
        } else {
            const DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            DataCopyPad(outputGMY[offset], yLocal, copyParams);
        }
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<QuePosition::VECCALC> floatBuf;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0; // 本核处理的元素总数
    int64_t tileLength_ = 0;  // 单个 tile 的元素个数
    int64_t tileNum = 0;      // tile 数量（模板 Process() 使用）
    int64_t processNum = 0;   // UB 缓冲容量（元素个数）
    int64_t curCount_ = 0;    // 当前 tile 的实际元素个数
};

} // namespace NsExp
#endif // EXP_H
