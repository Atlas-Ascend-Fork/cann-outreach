#include "kernel_operator.h"
#include "sqrt_tiling_data.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelSqrt {
public:
    __aicore__ inline KernelSqrt() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint64_t totalNum,
                                uint64_t blockFactor,
                                uint64_t ubFactor)
    {
        this->totalNum = totalNum;
        this->blockFactor = blockFactor;
        this->ubFactor = ubFactor;
        this->blockOffset = AscendC::GetBlockIdx() * blockFactor;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, totalNum);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, totalNum);
        pipe.InitBuffer(inQueue, BUFFER_NUM, ubFactor * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueue, BUFFER_NUM, ubFactor * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process()
    {
        if (blockOffset >= totalNum) {
            return;
        }
        uint64_t validNum = totalNum - blockOffset;
        if (validNum > blockFactor) {
            validNum = blockFactor;
        }
        uint64_t processed = 0;
        while (processed < validNum) {
            uint32_t length = static_cast<uint32_t>(
                (validNum - processed) < ubFactor
                    ? (validNum - processed)
                    : ubFactor);
            CopyIn(processed, length);
            Compute(length);
            CopyOut(processed, length);
            processed += length;
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t length)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal =
            inQueue.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[blockOffset + offset], length);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t length)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal =
            inQueue.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal =
            outQueue.AllocTensor<DTYPE_Y>();
        AscendC::Sqrt(yLocal, xLocal, length);
        inQueue.FreeTensor(xLocal);
        outQueue.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t length)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal =
            outQueue.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[blockOffset + offset], yLocal, length);
        outQueue.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint64_t blockOffset = 0;
    uint64_t blockFactor = 0;
    uint64_t ubFactor = 0;
    uint64_t totalNum = 0;
};

__aicore__ inline void RunSqrt(GM_ADDR x, GM_ADDR y,
                               uint64_t totalNum,
                               uint64_t blockFactor,
                               uint64_t ubFactor)
{
    KernelSqrt op;
    op.Init(x, y, totalNum, blockFactor, ubFactor);
    op.Process();
}

__global__ __aicore__ void sqrt(GM_ADDR x, GM_ADDR y,
                                GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SqrtTilingData);
    GET_TILING_DATA(tilingData, tiling);
    RunSqrt(x, y, tilingData.totalNum,
            tilingData.blockFactor, tilingData.ubFactor);
}


template <uint32_t TILING_KEY>
__global__ __aicore__ void sqrt(GM_ADDR x, GM_ADDR y,
                                GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SqrtTilingData);
    GET_TILING_DATA(tilingData, tiling);
    RunSqrt(x, y, tilingData.totalNum,
            tilingData.blockFactor, tilingData.ubFactor);
}
