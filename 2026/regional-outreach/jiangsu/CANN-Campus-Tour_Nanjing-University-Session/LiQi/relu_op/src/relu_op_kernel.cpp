/**
 * @file relu_op_kernel.cpp
 * @brief ReLU operator kernel for AscendC (FP16, 4D ND format)
 *
 * ReLU(x) = max(0, x)
 * Input:  4D tensor [N4, N3, N2, N1], FP16, ND format
 * Output: 4D tensor [N4, N3, N2, N1], FP16, ND format
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace {
// UB size for Ascend 910: 192KB, Ascend 310: 64KB
// Use 192KB as default (910); adjust via CMake define for 310
#ifndef UB_SIZE
#define UB_SIZE (192 * 1024)
#endif

// Each FP16 element is 2 bytes
constexpr int32_t FP16_SIZE = 2;

// BlockDataCopy requires alignment; use 32-byte aligned blocks
constexpr int32_t COPY_ALIGN_BYTES = 32;
}  // namespace

/**
 * @struct ReluTilingData
 * @brief Tiling parameters passed from host to kernel
 */
struct ReluTilingData {
    int64_t totalElements;    // Total number of elements (N4*N3*N2*N1)
    int32_t blockCount;       // Number of blocks to process
    int32_t blockSizeElements; // Elements per block
};

/**
 * @class ReluOp
 * @brief ReLU operator: output = max(0, input)
 *
 * Strategy:
 * 1. Split total elements into blocks that fit in UB
 * 2. For each block: copy GM->UB, compute VecMax with zero, copy UB->GM
 */
class ReluOp {
public:
    __aicore__ inline ReluOp() {}

    /**
     * @brief Initialize with tiling data and tensor pointers
     */
    __aicore__ inline void Init(const ReluTilingData *tiling,
                                const __gm__ half *input,
                                const __gm__ half *output)
    {
        this->tiling_ = tiling;
        this->input_ = input;
        this->output_ = output;
    }

    /**
     * @brief Main processing loop
     */
    __aicore__ inline void Process()
    {
        const int32_t coreId = GetBlockDim();
        const int32_t coreNum = GetNumCore();

        // Each core processes a subset of blocks (data parallelism across cores)
        for (int32_t blockIdx = coreId; blockIdx < tiling_->blockCount; blockIdx += coreNum) {
            ProcessBlock(blockIdx);
        }
    }

private:
    /**
     * @brief Process one block of data
     *
     * @param blockIdx Index of the block to process
     */
    __aicore__ inline void ProcessBlock(int32_t blockIdx)
    {
        // Calculate global offset for this block
        int64_t offset = static_cast<int64_t>(blockIdx) * tiling_->blockSizeElements;
        int32_t blockSize = tiling_->blockSizeElements;

        // Last block may be smaller
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

        // Declare UB tensors
        LocalTensor<half> localIn;
        LocalTensor<half> localOut;
        LocalTensor<half> localZero;

        // Allocate UB space (in units of 32 bytes for vector ops)
        int32_t allocBlocks = (bytes + COPY_ALIGN_BYTES - 1) / COPY_ALIGN_BYTES;

        localIn.AllocTensor<UB>(allocBlocks);
        localOut.AllocTensor<UB>(allocBlocks);
        localZero.AllocTensor<UB>(allocBlocks);

        // Fill zero tensor with 0 (for VecMax)
        VecSet(localZero, 0.0f, allocBlocks);

        // Copy input: GM -> UB
        BlockDataCopy(localIn, input_ + offset, bytes);
        DataCopyPad(localIn, input_ + offset, bytes, 0, bytes);

        // ReLU: output = max(input, 0)
        // VecMax computes element-wise maximum
        int32_t calcBlocks = (bytes + COPY_ALIGN_BYTES - 1) / COPY_ALIGN_BYTES;
        VecMax(localOut, localIn, localZero, calcBlocks);

        // Copy output: UB -> GM
        BlockDataCopy(output_ + offset, localOut, bytes);
        DataCopyPad(output_ + offset, localOut, bytes, 0, bytes);

        // Release UB
        localIn.FreeTensor<UB>();
        localOut.FreeTensor<UB>();
        localZero.FreeTensor<UB>();
    }

    const ReluTilingData *tiling_;
    const __gm__ half *input_;
    const __gm__ half *output_;
};

/**
 * @brief Kernel entry point (called by Ascend runtime)
 */
extern "C" __aicore__ void relu_op_entry()
{
    // Get tiling data from system workspace
    __gm__ ReluTilingData *tiling =
        GetSysWorkSpace<ReluTilingData>(GetSysWorkspaceSize());

    // Get input/output tensors from parameter space
    // For AscendC custom ops, parameters are passed via TQue or directly
    // Here we use the standard parameter access pattern
    const __gm__ half *input =
        reinterpret_cast<const __gm__ half *>(GetParamPtr(0));
    const __gm__ half *output =
        reinterpret_cast<const __gm__ half *>(GetParamPtr(1));

    // Create operator and execute
    ReluOp op;
    op.Init(tiling, input, output);
    op.Process();
}
