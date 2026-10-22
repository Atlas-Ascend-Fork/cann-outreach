/**
 * @file inplace_rsqrt_kernel.cpp
 * @brief InplaceRsqrt operator kernel for AscendC (FP16, 4D ND format)
 *
 * InplaceRsqrt(self): self = 1 / sqrt(self), computed in-place
 *
 * Input/Output (in-place):
 *   self: 4D tensor [N4, N3, N2, N1], FP16, ND format
 *   After execution: self[i] = 1/sqrt(self[i]) for all i
 *
 * Implementation:
 *   - Uses VecRsqrt vector instruction (batch vectorized, no scalar loop)
 *   - Single UB buffer: GM -> UB -> VecRsqrt in-place -> UB -> GM
 *   - Directly modifies self memory (in-place semantics)
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace {
// UB size: 192KB for Ascend 910, 64KB for Ascend 310
#ifndef UB_SIZE
#define UB_SIZE (192 * 1024)
#endif

// FP16 = 2 bytes per element
constexpr int32_t FP16_SIZE = 2;

// DMA alignment requirement: 32 bytes
constexpr int32_t COPY_ALIGN_BYTES = 32;
}  // namespace

/**
 * @struct RsqrtTilingData
 * @brief Tiling parameters passed from host to kernel
 */
struct RsqrtTilingData {
    int64_t totalElements;      // N4 * N3 * N2 * N1
    int32_t blockCount;         // Number of processing blocks
    int32_t blockSizeElements;  // Elements per block (32-byte aligned)
};

/**
 * @class InplaceRsqrtOp
 * @brief In-place reciprocal square root: self = 1/sqrt(self)
 *
 * Tiling strategy:
 *   - Total UB per core: UB_SIZE bytes
 *   - In-place: only ONE buffer needed (read + compute + write same location)
 *   - Block size = UB_SIZE / FP16_SIZE elements (max that fits in UB)
 *   - Each core processes a subset of blocks via GetBlockDim()
 */
class InplaceRsqrtOp {
public:
    __aicore__ inline InplaceRsqrtOp() {}

    /**
     * @brief Initialize operator with tiling data and tensor pointer
     * @param tiling Tiling parameters from host
     * @param self Input/output tensor (in-place)
     */
    __aicore__ inline void Init(const RsqrtTilingData *tiling,
                                const __gm__ half *self)
    {
        this->tiling_ = tiling;
        this->self_ = self;
    }

    /**
     * @brief Main processing: iterate over blocks assigned to this core
     */
    __aicore__ inline void Process()
    {
        const int32_t coreId = GetBlockDim();
        const int32_t coreNum = GetNumCore();

        for (int32_t blockIdx = coreId; blockIdx < tiling_->blockCount;
             blockIdx += coreNum) {
            ProcessBlock(blockIdx);
        }
    }

private:
    /**
     * @brief Process one block in-place
     *
     * Steps:
     *   1. BlockDataCopy: GM -> UB (load self chunk)
     *   2. VecRsqrt: compute 1/sqrt(x) in-place on UB buffer
     *   3. BlockDataCopy: UB -> GM (write back to same location)
     *
     * @param blockIdx Global block index
     */
    __aicore__ inline void ProcessBlock(int32_t blockIdx)
    {
        // Global element offset for this block
        int64_t offset = static_cast<int64_t>(blockIdx) * tiling_->blockSizeElements;
        int32_t blockSize = tiling_->blockSizeElements;

        // Handle last block (may be smaller)
        if ((blockIdx + 1) == tiling_->blockCount) {
            int64_t remaining = tiling_->totalElements - offset;
            if (remaining < blockSize) {
                blockSize = static_cast<int32_t>(remaining);
            }
        }

        if (blockSize <= 0) {
            return;
        }

        int32_t bytes = blockSize * FP16_SIZE;
        int32_t vectorBlocks = bytes / COPY_ALIGN_BYTES;  // In 32-byte units

        // Declare single UB tensor (in-place: one buffer suffices)
        LocalTensor<half> localData;
        localData.AllocTensor<UB>(vectorBlocks);

        // Step 1: GM -> UB
        DataCopyPad(localData, self_ + offset, bytes, 0, bytes);

        // Step 2: VecRsqrt in-place (1/sqrt(x))
        // VecRsqrt computes reciprocal square root element-wise
        // Input and output can be the same tensor (in-place)
        VecRsqrt(localData, localData, vectorBlocks);

        // Step 3: UB -> GM (write back to original location)
        DataCopyPad(self_ + offset, localData, bytes, 0, bytes);

        // Release UB
        localData.FreeTensor<UB>();
    }

    const RsqrtTilingData *tiling_;
    const __gm__ half *self_;  // In-place: same pointer for read and write
};

/**
 * @brief Kernel entry point
 *
 * Called by the Ascend runtime dispatches to each AI core.
 * Parameters are accessed via the standard parameter space.
 */
extern "C" __aicore__ void inplace_rsqrt_entry()
{
    // Get tiling data from system workspace
    __gm__ RsqrtTilingData *tiling =
        GetSysWorkSpace<RsqrtTilingData>(GetSysWorkspaceSize());

    // Get in-place tensor from parameter space (single tensor: self)
    const __gm__ half *self =
        reinterpret_cast<const __gm__ half *>(GetParamPtr(0));

    // Execute
    InplaceRsqrtOp op;
    op.Init(tiling, self);
    op.Process();
}
