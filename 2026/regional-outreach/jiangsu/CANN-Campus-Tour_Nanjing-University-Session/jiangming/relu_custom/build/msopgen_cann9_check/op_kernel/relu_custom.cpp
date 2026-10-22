#include "kernel_operator.h"

namespace {
constexpr uint32_t kElementsPerBlock = 16;
constexpr int32_t kBufferCount = 2;

class KernelReluCustom {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t totalLength,
                               uint64_t blocksPerCore, uint32_t extraBlocks,
                               uint32_t tileLength)
    {
        const uint64_t core = AscendC::GetBlockIdx();
        const uint64_t precedingExtra = core < extraBlocks ? core : extraBlocks;
        const uint64_t start = (core * blocksPerCore + precedingExtra) * kElementsPerBlock;
        const uint64_t assigned = (blocksPerCore + (core < extraBlocks ? 1 : 0)) *
                                  kElementsPerBlock;
        length_ = start < totalLength ?
            (assigned < totalLength - start ? assigned : totalLength - start) : 0;
        tileLength_ = tileLength;
        if (length_ == 0) {
            return;
        }
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x) + start, length_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y) + start, length_);
        pipe_.InitBuffer(inQueue_, kBufferCount, tileLength_ * sizeof(half));
        pipe_.InitBuffer(outQueue_, kBufferCount, tileLength_ * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        for (uint64_t offset = 0; offset < length_; offset += tileLength_) {
            const uint32_t count = static_cast<uint32_t>(
                length_ - offset < tileLength_ ? length_ - offset : tileLength_);
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count)
    {
        auto local = inQueue_.AllocTensor<half>();
        if (count % kElementsPerBlock == 0) {
            AscendC::DataCopy(local, xGm_[offset], count);
        } else {
            // blockLen is in bytes. Read only valid GM bytes and pad UB with zero.
            AscendC::DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(half)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<half> pad{
                true, 0, static_cast<uint8_t>(kElementsPerBlock - count % kElementsPerBlock),
                static_cast<half>(0)};
            AscendC::DataCopyPad(local, xGm_[offset], copy, pad);
        }
        inQueue_.EnQue(local);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        auto x = inQueue_.DeQue<half>();
        auto y = outQueue_.AllocTensor<half>();
        const uint32_t alignedCount =
            (count + kElementsPerBlock - 1) / kElementsPerBlock * kElementsPerBlock;
        AscendC::Relu(y, x, alignedCount);
        outQueue_.EnQue(y);
        inQueue_.FreeTensor(x);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count)
    {
        auto local = outQueue_.DeQue<half>();
        if (count % kElementsPerBlock == 0) {
            AscendC::DataCopy(yGm_[offset], local, count);
        } else {
            // Never write the padded UB elements back to GM.
            AscendC::DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(half)), 0, 0, 0};
            AscendC::DataCopyPad(yGm_[offset], local, copy);
        }
        outQueue_.FreeTensor(local);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, kBufferCount> inQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kBufferCount> outQueue_;
    AscendC::GlobalTensor<half> xGm_;
    AscendC::GlobalTensor<half> yGm_;
    uint64_t length_ = 0;
    uint32_t tileLength_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void relu_custom(GM_ADDR x, GM_ADDR y,
                                                GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(data, tiling);
    KernelReluCustom op;
    op.Init(x, y, data.totalLength, data.blocksPerCore, data.extraBlocks, data.tileLength);
    op.Process();
}
