// Kernel侧核函数实现：逐元素乘法 z = x * y
#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

// 每块处理128个元素
constexpr uint32_t MUL_TILE_LENGTH = 128;

// 每个队列使用两个缓冲区
constexpr int32_t MUL_BUFFER_NUM = 2;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t length)
    {
        this->length = length;

        // 绑定完整输入和输出的全局内存
        xGm.SetGlobalBuffer((__gm__ DT_X*)x, length);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y, length);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z, length);

        // 初始化队列缓冲区，大小单位为字节
        pipe.InitBuffer(
            inQueueX,
            MUL_BUFFER_NUM,
            MUL_TILE_LENGTH * sizeof(DT_X));

        pipe.InitBuffer(
            inQueueY,
            MUL_BUFFER_NUM,
            MUL_TILE_LENGTH * sizeof(DT_X));

        pipe.InitBuffer(
            outQueueZ,
            MUL_BUFFER_NUM,
            MUL_TILE_LENGTH * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        // 本题长度为8*2048=16384，恰好分为128个块
        const uint32_t tileCount = this->length / MUL_TILE_LENGTH;
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum = AscendC::GetBlockNum();

        // 核k依次处理第k、k+blockNum、k+2*blockNum...块
        for (uint32_t tileIdx = blockIdx;
             tileIdx < tileCount;
             tileIdx += blockNum) {
            const uint32_t offset = tileIdx * MUL_TILE_LENGTH;

            CopyIn(offset);
            Compute();
            CopyOut(offset);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset)
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.AllocTensor<DT_X>();

        // 将当前块从全局内存搬入局部内存
        AscendC::DataCopy(xLocal, xGm[offset], MUL_TILE_LENGTH);
        AscendC::DataCopy(yLocal, yGm[offset], MUL_TILE_LENGTH);

        inQueueX.EnQue<DT_X>(xLocal);
        inQueueY.EnQue<DT_X>(yLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.AllocTensor<DT_X>();

        // 对当前块执行逐元素乘法
        AscendC::Mul(zLocal, xLocal, yLocal, MUL_TILE_LENGTH);

        outQueueZ.EnQue<DT_X>(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset)
    {
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.DeQue<DT_X>();

        // 将计算结果写回对应的全局内存位置
        AscendC::DataCopy(zGm[offset], zLocal, MUL_TILE_LENGTH);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, MUL_BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, MUL_BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, MUL_BUFFER_NUM> outQueueZ;

    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;

    uint32_t length;
};

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
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
