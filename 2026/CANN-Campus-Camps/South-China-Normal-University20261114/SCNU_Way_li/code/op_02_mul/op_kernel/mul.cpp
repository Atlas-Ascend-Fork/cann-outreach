#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <typename T>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length)
    {
        constexpr uint32_t alignElements =
            MUL_ALIGN_BYTES / sizeof(T);

        const uint32_t coreId = AscendC::GetBlockIdx();
        const uint32_t coreNum = AscendC::GetBlockNum();

        // 以 32 字节块为分配单位，
        // 前 remainder 个核各多分一个块。
        const uint32_t blocks = length / alignElements;
        const uint32_t quotient = blocks / coreNum;
        const uint32_t remainder = blocks % coreNum;

        const uint32_t extraBefore =
            coreId < remainder ? coreId : remainder;

        const uint32_t offset =
            (coreId * quotient + extraBefore) * alignElements;

        blockLength_ =
            (quotient + (coreId < remainder ? 1U : 0U))
            * alignElements;

        xGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(x) + offset,
            blockLength_);

        yGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(y) + offset,
            blockLength_);

        zGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(z) + offset,
            blockLength_);

        pipe_.InitBuffer(
            xQueue_, MUL_BUFFER_NUM,
            MUL_TILE_LENGTH * sizeof(T));

        pipe_.InitBuffer(
            yQueue_, MUL_BUFFER_NUM,
            MUL_TILE_LENGTH * sizeof(T));

        pipe_.InitBuffer(
            zQueue_, MUL_BUFFER_NUM,
            MUL_TILE_LENGTH * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < blockLength_;) {
            const uint32_t remaining = blockLength_ - offset;
            const uint32_t count =
                remaining < MUL_TILE_LENGTH
                    ? remaining
                    : MUL_TILE_LENGTH;

            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);

            offset += count;
        }
    }

private:
    __aicore__ inline void CopyIn(
        uint32_t offset, uint32_t count)
    {
        AscendC::LocalTensor<T> xLocal =
            xQueue_.AllocTensor<T>();

        AscendC::LocalTensor<T> yLocal =
            yQueue_.AllocTensor<T>();

        AscendC::DataCopy(xLocal, xGm_[offset], count);
        AscendC::DataCopy(yLocal, yGm_[offset], count);

        xQueue_.EnQue(xLocal);
        yQueue_.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        AscendC::LocalTensor<T> xLocal =
            xQueue_.DeQue<T>();

        AscendC::LocalTensor<T> yLocal =
            yQueue_.DeQue<T>();

        AscendC::LocalTensor<T> zLocal =
            zQueue_.AllocTensor<T>();

        AscendC::Mul(zLocal, xLocal, yLocal, count);

        zQueue_.EnQue(zLocal);

        xQueue_.FreeTensor(xLocal);
        yQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(
        uint32_t offset, uint32_t count)
    {
        AscendC::LocalTensor<T> zLocal =
            zQueue_.DeQue<T>();

        AscendC::DataCopy(zGm_[offset], zLocal, count);

        zQueue_.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe_;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        MUL_BUFFER_NUM> xQueue_;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        MUL_BUFFER_NUM> yQueue_;

    AscendC::TQue<
        AscendC::QuePosition::VECOUT,
        MUL_BUFFER_NUM> zQueue_;

    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
    AscendC::GlobalTensor<T> zGm_;

    uint32_t blockLength_ = 0;
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
        MulTilingData, tilingData, tiling);

    KernelMul<DT_X> op;
    op.Init(x, y, z, tilingData.length);
    op.Process();
}