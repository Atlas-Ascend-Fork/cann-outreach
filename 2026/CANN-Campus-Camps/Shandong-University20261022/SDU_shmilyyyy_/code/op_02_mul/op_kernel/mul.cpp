#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <typename T>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                               const MulTilingData& tiling, AscendC::TPipe* pipe)
    {
        pipe_ = pipe;
        const uint32_t offset = AscendC::GetBlockIdx() * tiling.blockLength;
        length_ = tiling.length - offset;
        if (length_ > tiling.blockLength) {
            length_ = tiling.blockLength;
        }
        tileLength_ = tiling.tileLength;
        xGm_.SetGlobalBuffer((__gm__ T*)x + offset, length_);
        yGm_.SetGlobalBuffer((__gm__ T*)y + offset, length_);
        zGm_.SetGlobalBuffer((__gm__ T*)z + offset, length_);
        pipe_->InitBuffer(buffer_, 2u * tileLength_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<T> xLocal = buffer_.Get<T>(tileLength_);
        AscendC::LocalTensor<T> yLocal = buffer_.GetWithOffset<T>(
            tileLength_, tileLength_ * sizeof(T));

        // The specified (8, 2048) workload takes this single-tile path.
        if (length_ <= tileLength_) {
            CopyIn(xLocal, yLocal, 0, length_);
            Compute(xLocal, yLocal, length_);
            CopyOut(yLocal, 0, length_);
            return;
        }

        // Capacity-safe fallback for larger shapes or smaller UB budgets.
        const auto reuseEvent = pipe_->FetchEventID(AscendC::HardEvent::MTE3_MTE2);
        uint32_t offset = 0;
        while (offset < length_) {
            const uint32_t remaining = length_ - offset;
            const uint32_t count = remaining < tileLength_ ? remaining : tileLength_;
            CopyIn(xLocal, yLocal, offset, count);
            Compute(xLocal, yLocal, count);
            CopyOut(yLocal, offset, count);
            offset += count;
            if (offset < length_) {
                // The output reuses yLocal; finish reading it before overwriting it.
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(reuseEvent);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(reuseEvent);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(const AscendC::LocalTensor<T>& xLocal,
                                 const AscendC::LocalTensor<T>& yLocal,
                                 uint32_t offset, uint32_t count)
    {
        constexpr uint32_t BLOCK_ELEMS = 32u / sizeof(T);
        if (count % BLOCK_ELEMS == 0) {
            AscendC::DataCopy(xLocal, xGm_[offset], count);
            AscendC::DataCopy(yLocal, yGm_[offset], count);
        } else {
            // Copy only valid bytes; do not read beyond the last GM allocation.
            const AscendC::DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            const AscendC::DataCopyPadExtParams<T> padParams{
                false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
            AscendC::DataCopyPad(yLocal, yGm_[offset], copyParams, padParams);
        }
        const auto inputReady = pipe_->FetchEventID(AscendC::HardEvent::MTE2_V);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady);
    }

    __aicore__ inline void Compute(const AscendC::LocalTensor<T>& xLocal,
                                  const AscendC::LocalTensor<T>& yLocal,
                                  uint32_t count)
    {
        // Exact overlap with src1: reuse y's UB region, preserving native dtype.
        AscendC::Mul(yLocal, xLocal, yLocal, count);
        const auto outputReady = pipe_->FetchEventID(AscendC::HardEvent::V_MTE3);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady);
    }

    __aicore__ inline void CopyOut(const AscendC::LocalTensor<T>& yLocal,
                                  uint32_t offset, uint32_t count)
    {
        constexpr uint32_t BLOCK_ELEMS = 32u / sizeof(T);
        if (count % BLOCK_ELEMS == 0) {
            AscendC::DataCopy(zGm_[offset], yLocal, count);
        } else {
            const AscendC::DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(zGm_[offset], yLocal, copyParams);
        }
    }

private:
    AscendC::TPipe* pipe_;
    uint32_t length_;
    uint32_t tileLength_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
    AscendC::GlobalTensor<T> zGm_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                             GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.length == 0) {
        return;
    }
    AscendC::TPipe pipe;
    KernelMul<DT_X> op;
    op.Init(x, y, z, tilingData, &pipe);
    op.Process();
}
