
/*!
 * \file relu.h
 * \brief ReLU 算子 Kernel
 *
 * y = max(0, x)
 *
 * 支持：
 * 1. FLOAT16
 * 2. FLOAT32
 *
 * 第六轮优化：
 * 1. 多核并行
 * 2. UB 分块
 * 3. 双缓冲
 * 4. 单 Tile 快速路径
 * 5. 多 Tile 预取流水调度
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

// 必须与 Host 端一致
constexpr uint32_t BUFFER_NUM = 2;

constexpr uint32_t BLOCK_BYTES = 32;

template <typename T>
class Relu {

public:

    __aicore__ inline Relu()
    {
    }

    // =====================================================
    // 初始化
    // =====================================================

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const ReluTilingData* tilingData)
    {
        // 读取 Tiling 参数
        totalLength = tilingData->totalLength;

        blockLength = tilingData->blockLength;

        tileLength = tilingData->tileLength;

        // 防止 Tile 长度为零
        if (tileLength == 0U) {
            tileLength =
                BLOCK_BYTES / static_cast<uint32_t>(sizeof(T));
        }

        // 获取当前核编号
        const uint32_t blockIdx =
            static_cast<uint32_t>(GetBlockIdx());

        // 计算当前核 GM 起始位置
        gmOffset =
            static_cast<uint64_t>(blockIdx) *
            static_cast<uint64_t>(blockLength);

        // 空闲核直接退出
        if (gmOffset >= totalLength) {
            curLength = 0U;
            loopCount = 0U;
            return;
        }

        // 当前核剩余数据长度
        const uint64_t remain =
            static_cast<uint64_t>(totalLength) -
            gmOffset;

        // 当前核实际处理长度
        curLength = static_cast<uint32_t>(
            remain < blockLength ? remain : blockLength
        );

        // 计算 Tile 数量
        loopCount =
            (curLength + tileLength - 1U) / tileLength;

        // 设置输入 GlobalTensor
        xGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(x) + gmOffset,
            curLength
        );

        // 设置输出 GlobalTensor
        yGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(y) + gmOffset,
            curLength
        );

        // 初始化输入双缓冲队列
        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            tileLength * sizeof(T)
        );

        // 初始化输出双缓冲队列
        pipe.InitBuffer(
            outQueueY,
            BUFFER_NUM,
            tileLength * sizeof(T)
        );
    }

    // =====================================================
    // Kernel 主函数
    // =====================================================

    __aicore__ inline void Process()
    {
        // 没有数据需要处理
        if (loopCount == 0U) {
            return;
        }

        // -------------------------------------------------
        // 单 Tile 快速路径
        // -------------------------------------------------

        if (loopCount == 1U) {

            CopyIn(0U);

            Compute(0U);

            CopyOut(0U);

            return;
        }

        // -------------------------------------------------
        // 多 Tile 双缓冲预取流水
        // -------------------------------------------------

        // 第一步：预先搬入第一个 Tile
        CopyIn(0U);

        // 第二步：循环处理 Tile
        for (uint32_t i = 0U; i < loopCount; ++i) {

            // 计算当前 Tile
            Compute(i);

            // 预取下一个 Tile
            // 尝试让下一次 GM -> UB 与当前计算交错
            if (i + 1U < loopCount) {
                CopyIn(i + 1U);
            }

            // 延迟写回上一个 Tile
            // 让输出写回与后续 Tile 的计算、搬运交错
            if (i > 0U) {
                CopyOut(i - 1U);
            }
        }

        // 最后一个 Tile 尚未写回
        CopyOut(loopCount - 1U);
    }

private:

    // =====================================================
    // 获取当前 Tile 实际长度
    // =====================================================

    __aicore__ inline uint32_t TileLen(
        uint32_t progress) const
    {
        const uint32_t offset =
            progress * tileLength;

        const uint32_t remain =
            curLength - offset;

        return remain < tileLength
            ? remain
            : tileLength;
    }

    // =====================================================
    // 数据搬入：GM -> UB
    // =====================================================

    __aicore__ inline void CopyIn(
        uint32_t progress)
    {
        // 分配输入 Tensor
        LocalTensor<T> xLocal =
            inQueueX.AllocTensor<T>();

        // 当前 Tile 起始偏移
        const uint32_t offset =
            progress * tileLength;

        // 当前 Tile 实际长度
        const uint32_t len =
            TileLen(progress);

        // 搬运输入数据
        DataCopy(
            xLocal,
            xGm[offset],
            len
        );

        // 输入进入队列
        inQueueX.EnQue(xLocal);
    }

    // =====================================================
    // ReLU 计算
    // =====================================================

    __aicore__ inline void Compute(
        uint32_t progress)
    {
        // 获取输入 Tensor
        LocalTensor<T> xLocal =
            inQueueX.DeQue<T>();

        // 分配输出 Tensor
        LocalTensor<T> yLocal =
            outQueueY.AllocTensor<T>();

        // 当前 Tile 计算长度
        const uint32_t len =
            TileLen(progress);

        // 执行 ReLU
        AscendC::Relu(
            yLocal,
            xLocal,
            len
        );

        // 输出进入队列
        outQueueY.EnQue(yLocal);

        // 释放输入 Tensor
        inQueueX.FreeTensor(xLocal);
    }

    // =====================================================
    // 数据写回：UB -> GM
    // =====================================================

    __aicore__ inline void CopyOut(
        uint32_t progress)
    {
        // 获取输出 Tensor
        LocalTensor<T> yLocal =
            outQueueY.DeQue<T>();

        // 当前 Tile 起始偏移
        const uint32_t offset =
            progress * tileLength;

        // 当前 Tile 实际长度
        const uint32_t len =
            TileLen(progress);

        // 写回 GM
        DataCopy(
            yGm[offset],
            yLocal,
            len
        );

        // 释放输出 Tensor
        outQueueY.FreeTensor(yLocal);
    }

private:

    // =====================================================
    // Pipeline 和双缓冲队列
    // =====================================================

    TPipe pipe;

    // 输入双缓冲队列
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;

    // 输出双缓冲队列
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    // =====================================================
    // GlobalTensor
    // =====================================================

    GlobalTensor<T> xGm;

    GlobalTensor<T> yGm;

    // =====================================================
    // Tiling 参数
    // =====================================================

    uint32_t totalLength{0};

    uint32_t blockLength{0};

    uint32_t curLength{0};

    uint32_t tileLength{0};

    uint32_t loopCount{0};

    // 当前核 GM 偏移
    uint64_t gmOffset{0};
};

} // namespace NsRelu

#endif // RELU_H
