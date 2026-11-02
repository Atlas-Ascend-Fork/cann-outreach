#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <typename DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                               uint32_t length, uint32_t tileLength)
    {
        tileLength_ = tileLength;
        blockLength_ = 0;
        if (length == 0) {
            return;
        }

        // 按 32 字节块均分，各核的 GM 起点均保持 32 字节对齐。
        constexpr uint32_t ALIGN_NUM = 32 / sizeof(DT_X);
        const uint64_t totalBlocks =
            (static_cast<uint64_t>(length) + ALIGN_NUM - 1) / ALIGN_NUM;
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint32_t coreId = AscendC::GetBlockIdx();
        const uint64_t baseBlocks = totalBlocks / coreNum;
        const uint32_t extra = totalBlocks % coreNum;
        const uint64_t startBlock =
            coreId * baseBlocks + ((coreId < extra) ? coreId : extra);
        const uint64_t coreBlocks = baseBlocks + ((coreId < extra) ? 1 : 0);
        const uint64_t start = startBlock * ALIGN_NUM;
        if (start >= length || coreBlocks == 0) {
            return;
        }
        const uint64_t capacity = coreBlocks * ALIGN_NUM;
        const uint64_t remaining = static_cast<uint64_t>(length) - start;
        blockLength_ = static_cast<uint32_t>(
            (capacity < remaining) ? capacity : remaining);

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x + start, blockLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y + start, blockLength_);
        zGm_.SetGlobalBuffer((__gm__ DT_X *)z + start, blockLength_);
        pipe_.InitBuffer(xQueue_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(yQueue_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(zQueue_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        uint32_t offset = 0;
        while (offset < blockLength_) {
            const uint32_t remaining = blockLength_ - offset;
            const uint32_t count =
                (remaining < tileLength_) ? remaining : tileLength_;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
            offset += count;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count)
    {
        auto xLocal = xQueue_.AllocTensor<DT_X>();
        auto yLocal = yQueue_.AllocTensor<DT_X>();
        constexpr uint32_t ALIGN_NUM = 32 / sizeof(DT_X);
        if (count % ALIGN_NUM == 0) {
            AscendC::DataCopy(xLocal, xGm_[offset], count);
            AscendC::DataCopy(yLocal, yGm_[offset], count);
        } else {
            // 只从 GM 读取有效元素，UB 内补零至 32 字节对齐。
            AscendC::DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padParams{
                true, 0, static_cast<uint8_t>(ALIGN_NUM - count % ALIGN_NUM),
                static_cast<DT_X>(0)};
            AscendC::DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
            AscendC::DataCopyPad(yLocal, yGm_[offset], copyParams, padParams);
        }
        xQueue_.EnQue(xLocal);
        yQueue_.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        auto xLocal = xQueue_.DeQue<DT_X>();
        auto yLocal = yQueue_.DeQue<DT_X>();
        auto zLocal = zQueue_.AllocTensor<DT_X>();
        constexpr uint32_t ALIGN_NUM = 32 / sizeof(DT_X);
        const uint32_t alignedCount =
            (count + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
        AscendC::Mul(zLocal, xLocal, yLocal, alignedCount);
        zQueue_.EnQue(zLocal);
        xQueue_.FreeTensor(xLocal);
        yQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        auto zLocal = zQueue_.DeQue<DT_X>();
        constexpr uint32_t ALIGN_NUM = 32 / sizeof(DT_X);
        if (count % ALIGN_NUM == 0) {
            AscendC::DataCopy(zGm_[offset], zLocal, count);
        } else {
            // 只写回有效字节，避免覆盖输出末尾之外的内存。
            AscendC::DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPad(zGm_[offset], zLocal, copyParams);
        }
        zQueue_.FreeTensor(zLocal);
    }

    static constexpr int32_t BUFFER_NUM = 2;
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> xQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> yQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> zQueue_;
    AscendC::GlobalTensor<DT_X> xGm_, yGm_, zGm_;
    uint32_t tileLength_ = 0;
    uint32_t blockLength_ = 0;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                             GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length, tiling_data.tileLength);
    op.Process();
}
