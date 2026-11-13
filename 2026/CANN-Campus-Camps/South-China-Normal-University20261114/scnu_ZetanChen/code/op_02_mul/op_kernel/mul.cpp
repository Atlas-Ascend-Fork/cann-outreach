#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <typename T>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        const MulTilingData& tiling)
    {
        tileLength = tiling.tileLength;

        // 以 32 字节为单位分核，保证各核起点对齐。
        const uint64_t alignElements = 32U / sizeof(T);
        const uint64_t totalLength = tiling.length;
        const uint64_t totalBlocks =
            (totalLength + alignElements - 1U) / alignElements;

        const uint64_t coreNum = AscendC::GetBlockNum();
        const uint64_t coreId = AscendC::GetBlockIdx();

        const uint64_t baseBlocks = totalBlocks / coreNum;
        const uint64_t extraBlocks = totalBlocks % coreNum;

        // 前 extraBlocks 个核各多处理一个 32 字节块。
        const uint64_t currentBlocks =
            baseBlocks + (coreId < extraBlocks ? 1U : 0U);

        const uint64_t startBlock =
            coreId * baseBlocks +
            (coreId < extraBlocks ? coreId : extraBlocks);

        const uint64_t offset = startBlock * alignElements;

        if (currentBlocks == 0 || offset >= totalLength) {
            blockLength = 0;
            return;
        }

        const uint64_t capacity = currentBlocks * alignElements;
        const uint64_t remaining = totalLength - offset;

        blockLength = static_cast<uint32_t>(
            capacity < remaining ? capacity : remaining);

        xGm.SetGlobalBuffer((__gm__ T*)x + offset, blockLength);
        yGm.SetGlobalBuffer((__gm__ T*)y + offset, blockLength);
        zGm.SetGlobalBuffer((__gm__ T*)z + offset, blockLength);

        pipe.InitBuffer(inQueueX, 1, tileLength * sizeof(T));
        pipe.InitBuffer(inQueueY, 1, tileLength * sizeof(T));
        pipe.InitBuffer(outQueueZ, 1, tileLength * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        uint32_t offset = 0;

        while (offset < blockLength) {
            const uint32_t remaining = blockLength - offset;
            const uint32_t count =
                remaining < tileLength ? remaining : tileLength;

            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);

            offset += count;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();

        // blockLen 的单位为字节，仅搬运有效数据。
        AscendC::DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(count * sizeof(T)),
            0, 0, 0
        };

        AscendC::DataCopyPadExtParams<T> padParams{
            false, 0, 0, static_cast<T>(0)
        };

        AscendC::DataCopyPad(
            xLocal, xGm[offset], copyParams, padParams);
        AscendC::DataCopyPad(
            yLocal, yGm[offset], copyParams, padParams);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        // 逐元素乘法：z[i] = x[i] * y[i]
        AscendC::Mul(zLocal, xLocal, yLocal, count);

        outQueueZ.EnQue(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        AscendC::LocalTensor<T> zLocal = outQueueZ.DeQue<T>();

        AscendC::DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(count * sizeof(T)),
            0, 0, 0
        };

        AscendC::DataCopyPad(zGm[offset], zLocal, copyParams);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;

    AscendC::GlobalTensor<T> xGm, yGm, zGm;

    uint32_t blockLength = 0;
    uint32_t tileLength = 0;
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
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tilingData, tiling);

    KernelMul<DT_X> op;
    op.Init(x, y, z, tilingData);
    op.Process();
}