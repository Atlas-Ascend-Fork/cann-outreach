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

template <typename DT_X>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const TanhTilingData* tilingData)
    {
        // 获取当前计算核负责的数据长度
        this->blockLength =
            tilingData->totalLength / AscendC::GetBlockNum();

        this->tileNum = tilingData->tileNum;
        this->tileLength =
            this->blockLength / this->tileNum / BUFFER_NUM;

        this->tmpBufferSize = tilingData->tmpBufferSize;

        // 设置当前计算核的 Global Memory 起始地址
        inputGMX.SetGlobalBuffer(
            (__gm__ DT_X*)x +
            this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength
        );

        outputGMY.SetGlobalBuffer(
            (__gm__ DT_X*)y +
            this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength
        );

        // 初始化输入队列
        pipe.InitBuffer(
            inputQueueX,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X)
        );

        // 初始化输出队列
        pipe.InitBuffer(
            outputQueueY,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X)
        );

        // 初始化 Tanh 专用临时缓冲区
        if (this->tmpBufferSize > 0) {
            pipe.InitBuffer(
                tmpQueue,
                1,
                this->tmpBufferSize
            );
        }
    }

    __aicore__ inline void Process()
    {
        // 每个计算核处理 tileNum * BUFFER_NUM 个数据块
        int32_t loopCount = this->tileNum * BUFFER_NUM;

        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    // 从 Global Memory 搬运输入数据
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DT_X> xLocal =
            inputQueueX.AllocTensor<DT_X>();

        DataCopy(
            xLocal,
            inputGMX[progress * this->tileLength],
            this->tileLength
        );

        inputQueueX.EnQue(xLocal);
    }

    // 执行 Tanh 计算
    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal =
            inputQueueX.DeQue<DT_X>();

        LocalTensor<DT_X> yLocal =
            outputQueueY.AllocTensor<DT_X>();

        if (this->tmpBufferSize > 0) {
            // 申请 Tanh 计算所需的临时空间
            LocalTensor<uint8_t> sharedTmpBuffer =
                tmpQueue.AllocTensor<uint8_t>();

            // 调用 Ascend C 原生 Tanh 接口
            AscendC::Tanh<DT_X, false>(
                yLocal,
                xLocal,
                sharedTmpBuffer,
                this->tileLength
            );

            // 释放临时空间
            tmpQueue.FreeTensor(sharedTmpBuffer);
        } else {
            // 无需临时空间时使用接口框架申请方式
            AscendC::Tanh<DT_X, false>(
                yLocal,
                xLocal,
                this->tileLength
            );
        }

        outputQueueY.EnQue<DT_X>(yLocal);

        inputQueueX.FreeTensor(xLocal);
    }

    // 将计算结果搬运到 Global Memory
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DT_X> yLocal =
            outputQueueY.DeQue<DT_X>();

        DataCopy(
            outputGMY[progress * this->tileLength],
            yLocal,
            this->tileLength
        );

        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TQue<QuePosition::VECCALC, 1> tmpQueue;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength = 0;
    uint32_t tileNum = 0;
    uint32_t tileLength = 0;
    uint32_t tmpBufferSize = 0;
};

} // namespace NsTanh

#endif // TANH_H