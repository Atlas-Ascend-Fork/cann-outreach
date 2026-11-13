
#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const ReluTilingData* tilingData
    )
    {
        const int64_t totalNum = tilingData->totalNum;
        const int64_t blockFactor = tilingData->blockFactor;

        ubLength_ = tilingData->ubFactor;

        // 检查基础参数，避免无效访问
        if (totalNum <= 0 || blockFactor <= 0 || ubLength_ <= 0) {
            blockLength_ = 0;
            return;
        }

        // 计算当前核负责的数据起始位置
        const int64_t offset =
            static_cast<int64_t>(GetBlockIdx()) * blockFactor;

        // 当前核没有需要处理的数据
        if (offset >= totalNum) {
            blockLength_ = 0;
            return;
        }

        // 计算当前核实际处理的数据长度
        const int64_t remaining = totalNum - offset;
        blockLength_ = remaining < blockFactor
            ? remaining
            : blockFactor;

        // 设置输入和输出的 Global Memory 地址
        inputGMX.SetGlobalBuffer(
            (__gm__ T*)x + offset,
            blockLength_
        );

        outputGMY.SetGlobalBuffer(
            (__gm__ T*)y + offset,
            blockLength_
        );

        // 根据数据块数量动态选择 Buffer 数量
        // 只有一块数据时使用单缓冲
        // 存在多块数据时启用双缓冲
        const uint8_t bufferNum =
            (blockLength_ > ubLength_) ? 2 : 1;

        pipe.InitBuffer(
            inputQueueX,
            bufferNum,
            ubLength_ * sizeof(T)
        );

        pipe.InitBuffer(
            outputQueueY,
            bufferNum,
            ubLength_ * sizeof(T)
        );
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ <= 0 || ubLength_ <= 0) {
            return;
        }

        // 预加载第一块数据
        CopyIn(0, GetTileCount(0));

        // 分块处理当前核负责的数据
        for (
            int64_t offset = 0;
            offset < blockLength_;
            offset += ubLength_
        ) {
            const int64_t count = GetTileCount(offset);
            const int64_t nextOffset = offset + ubLength_;

            // 提前加载下一块数据，实现双缓冲
            if (nextOffset < blockLength_) {
                CopyIn(
                    nextOffset,
                    GetTileCount(nextOffset)
                );
            }

            // 执行 ReLU 向量计算
            Compute(count);

            // 将计算结果写回 Global Memory
            CopyOut(offset, count);
        }
    }

private:

    // 获取当前数据块的实际长度
    __aicore__ inline int64_t GetTileCount(int64_t offset)
    {
        const int64_t remaining = blockLength_ - offset;

        return remaining < ubLength_
            ? remaining
            : ubLength_;
    }

    // 数据搬入
    __aicore__ inline void CopyIn(
        int64_t offset,
        int64_t count
    )
    {
        LocalTensor<T> xLocal =
            inputQueueX.AllocTensor<T>();

        DataCopy(
            xLocal,
            inputGMX[offset],
            count
        );

        inputQueueX.EnQue(xLocal);
    }

    // ReLU 向量计算
    __aicore__ inline void Compute(int64_t count)
    {
        LocalTensor<T> xLocal =
            inputQueueX.DeQue<T>();

        LocalTensor<T> yLocal =
            outputQueueY.AllocTensor<T>();

        AscendC::Relu(
            yLocal,
            xLocal,
            static_cast<int32_t>(count)
        );

        outputQueueY.EnQue(yLocal);

        inputQueueX.FreeTensor(xLocal);
    }

    // 数据搬出
    __aicore__ inline void CopyOut(
        int64_t offset,
        int64_t count
    )
    {
        LocalTensor<T> yLocal =
            outputQueueY.DeQue<T>();

        DataCopy(
            outputGMY[offset],
            yLocal,
            count
        );

        outputQueueY.FreeTensor(yLocal);
    }

    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

}  // namespace NsRelu

#endif  // RELU_H
