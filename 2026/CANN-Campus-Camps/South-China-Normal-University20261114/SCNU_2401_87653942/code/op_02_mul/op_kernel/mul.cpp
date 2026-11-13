#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <typename T>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length, uint32_t tileLength)
    {
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreCount = AscendC::GetBlockNum();
        const uint32_t fullBlocks = length / elementsPerBlock;
        const uint32_t blocksPerCore = fullBlocks / coreCount;
        const uint32_t extraBlocks = fullBlocks % coreCount;
        const uint32_t extraBefore = coreIdx < extraBlocks ? coreIdx : extraBlocks;
        const uint32_t start =
            (coreIdx * blocksPerCore + extraBefore) * elementsPerBlock;

        coreLength =
            (blocksPerCore + (coreIdx < extraBlocks ? 1U : 0U)) * elementsPerBlock;

        // 按完整的 32 字节块分核，末核再处理不足一块的尾部。
        if (coreIdx + 1 == coreCount) {
            coreLength += length % elementsPerBlock;
        }

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x) + start, coreLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y) + start, coreLength);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(z) + start, coreLength);

        this->tileLength = tileLength;
        const uint32_t count = coreLength < tileLength ? coreLength : tileLength;
        const uint32_t bufferLength =
            (count + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;

        pipe.InitBuffer(inQueueX, 1, bufferLength * sizeof(T));
        pipe.InitBuffer(inQueueY, 1, bufferLength * sizeof(T));
        pipe.InitBuffer(outQueueZ, 1, bufferLength * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < coreLength;) {
            const uint32_t remaining = coreLength - offset;
            const uint32_t count = remaining < tileLength ? remaining : tileLength;

            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);

            offset += count;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count)
    {
        auto xLocal = inQueueX.AllocTensor<T>();
        auto yLocal = inQueueY.AllocTensor<T>();

        if (count % elementsPerBlock == 0) {
            AscendC::DataCopy(xLocal, xGm[offset], count);
            AscendC::DataCopy(yLocal, yGm[offset], count);
        } else {
            const AscendC::DataCopyExtParams copy{
                1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            const AscendC::DataCopyPadExtParams<T> pad{
                false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(xLocal, xGm[offset], copy, pad);
            AscendC::DataCopyPad(yLocal, yGm[offset], copy, pad);
        }

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        auto xLocal = inQueueX.DeQue<T>();
        auto yLocal = inQueueY.DeQue<T>();
        auto zLocal = outQueueZ.AllocTensor<T>();

        AscendC::Mul(zLocal, xLocal, yLocal, count);

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        auto zLocal = outQueueZ.DeQue<T>();

        if (count % elementsPerBlock == 0) {
            AscendC::DataCopy(zGm[offset], zLocal, count);
        } else {
            const AscendC::DataCopyExtParams copy{
                1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(zGm[offset], zLocal, copy);
        }

        outQueueZ.FreeTensor(zLocal);
    }

    static constexpr uint32_t elementsPerBlock = MUL_DATA_BLOCK_BYTES / sizeof(T);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<T> xGm, yGm, zGm;
    uint32_t coreLength;
    uint32_t tileLength;
};

template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tilingData, tiling);

    if (tilingData.length == 0) {
        return;
    }

    KernelMul<DT_X> op;
    op.Init(x, y, z, tilingData.length, tilingData.tileLength);
    op.Process();
}