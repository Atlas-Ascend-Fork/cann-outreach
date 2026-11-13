#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

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

        xGm.SetGlobalBuffer((__gm__ DT_X*)x, length);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y, length);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z, length);

        pipe.InitBuffer(
            inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(
            inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(
            outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        // 向上取整，避免length + TILE_LENGTH - 1溢出
        const uint32_t tileCount =
            length / TILE_LENGTH +
            (length % TILE_LENGTH != 0 ? 1U : 0U);

        // 核0处理块0、8、16……
        // 核1处理块1、9、17……（以启动8个核为例）
        for (uint32_t tileId = AscendC::GetBlockIdx();
             tileId < tileCount;
             tileId += AscendC::GetBlockNum()) {

            const uint32_t offset = tileId * TILE_LENGTH;
            const uint32_t remaining = length - offset;
            const uint32_t count =
                remaining < TILE_LENGTH ? remaining : TILE_LENGTH;

            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(
        uint32_t offset, uint32_t count)
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.AllocTensor<DT_X>();

        // blockLen单位为字节
        AscendC::DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(count * sizeof(DT_X)),
            0,
            0,
            0
        };

        // 不要求指定填充值，计算时只处理count个有效元素
        AscendC::DataCopyPadExtParams<DT_X> padParams{
            false, 0, 0, static_cast<DT_X>(0)
        };

        AscendC::DataCopyPad(
            xLocal, xGm[offset], copyParams, padParams);
        AscendC::DataCopyPad(
            yLocal, yGm[offset], copyParams, padParams);

        inQueueX.EnQue<DT_X>(xLocal);
        inQueueY.EnQue<DT_X>(yLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.AllocTensor<DT_X>();

        // 逐元素乘法：z = x * y
        AscendC::Mul(zLocal, xLocal, yLocal, count);

        outQueueZ.EnQue<DT_X>(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(
        uint32_t offset, uint32_t count)
    {
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.DeQue<DT_X>();

        AscendC::DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(count * sizeof(DT_X)),
            0,
            0,
            0
        };

        // 只写回有效元素，不越过输出边界
        AscendC::DataCopyPad(
            zGm[offset], zLocal, copyParams);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    static constexpr uint32_t TILE_LENGTH = 1024;
    static constexpr uint32_t BUFFER_NUM = 2;
    static constexpr uint32_t QUEUE_DEPTH = 1;

    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, QUEUE_DEPTH>
        inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, QUEUE_DEPTH>
        outQueueZ;

    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;

    uint32_t length = 0;
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
    GET_TILING_DATA_WITH_STRUCT(
        MulTilingData, tiling_data, tiling);

    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}