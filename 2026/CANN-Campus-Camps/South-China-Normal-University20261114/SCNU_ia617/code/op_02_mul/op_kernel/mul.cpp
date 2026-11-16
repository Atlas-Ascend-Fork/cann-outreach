#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t QUEUE_DEPTH = 1;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t totalLength,
        uint32_t tileNum)
    {
        // 每个核处理的元素个数
        this->blockLength = totalLength / AscendC::GetBlockNum();

        this->tileNum = tileNum;

        // 每次循环处理的元素个数
        this->tileLength =
            this->blockLength / this->tileNum / BUFFER_NUM;

        // 当前核在全局内存中的起始偏移，单位为元素
        uint32_t offset =
            this->blockLength * AscendC::GetBlockIdx();

        xGm.SetGlobalBuffer(
            (__gm__ DT_X*)x + offset, this->blockLength);
        yGm.SetGlobalBuffer(
            (__gm__ DT_X*)y + offset, this->blockLength);
        zGm.SetGlobalBuffer(
            (__gm__ DT_X*)z + offset, this->blockLength);

        // 为输入、输出队列分配 UB 内存，长度单位为字节
        pipe.InitBuffer(
            inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(
            inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(
            outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;

        for (int32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.AllocTensor<DT_X>();

        uint32_t offset = progress * this->tileLength;

        // 将输入数据从 GM 搬入 UB
        AscendC::DataCopy(
            xLocal, xGm[offset], this->tileLength);
        AscendC::DataCopy(
            yLocal, yGm[offset], this->tileLength);

        // 入队，交给计算阶段使用
        inQueueX.EnQue<DT_X>(xLocal);
        inQueueY.EnQue<DT_X>(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.AllocTensor<DT_X>();

        // 逐元素乘法：z = x * y
        AscendC::Mul(
            zLocal, xLocal, yLocal, this->tileLength);

        // 结果入队，交给搬出阶段使用
        outQueueZ.EnQue<DT_X>(zLocal);

        // 释放输入 Tensor
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.DeQue<DT_X>();

        uint32_t offset = progress * this->tileLength;

        // 将结果从 UB 搬回 GM
        AscendC::DataCopy(
            zGm[offset], zLocal, this->tileLength);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<
        AscendC::QuePosition::VECIN, QUEUE_DEPTH> inQueueX;
    AscendC::TQue<
        AscendC::QuePosition::VECIN, QUEUE_DEPTH> inQueueY;
    AscendC::TQue<
        AscendC::QuePosition::VECOUT, QUEUE_DEPTH> outQueueZ;

    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

// 核函数入口，保留题目模板的函数名称和参数
template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);

    KernelMul<DT_X> op;
    op.Init(
        x,
        y,
        z,
        tiling_data.totalLength,
        tiling_data.tileNum);
    op.Process();
}