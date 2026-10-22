/**
 * add_custom 算子 device 侧实现（op_kernel）
 * CopyIn → Compute → CopyOut 三阶段流水 + 双缓冲
 * Tiling 参数由 host 侧写入 tiling 传入（见 design.md）
 */
#include "kernel_operator.h"

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        totalLength_ = tilingData.totalLength;
        blockLength_ = tilingData.blockLength;
        tileNum_     = tilingData.tileNum;
        tileLength_  = tilingData.tileLength;

        xGm.SetGlobalBuffer((__gm__ half *)x + GetBlockIdx() * blockLength_, blockLength_);
        yGm.SetGlobalBuffer((__gm__ half *)y + GetBlockIdx() * blockLength_, blockLength_);
        zGm.SetGlobalBuffer((__gm__ half *)z + GetBlockIdx() * blockLength_, blockLength_);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(half));
        pipe_.InitBuffer(inQueueY_, BUFFER_NUM, tileLength_ * sizeof(half));
        pipe_.InitBuffer(outQueueZ_, BUFFER_NUM, tileLength_ * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        const int32_t loopCount = tileNum_;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX_.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY_.AllocTensor<half>();
        DataCopy(xLocal, xGm_[progress * tileLength_], tileLength_);
        DataCopy(yLocal, yGm_[progress * tileLength_], tileLength_);
        inQueueX_.EnQue(xLocal);
        inQueueY_.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX_.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY_.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ_.AllocTensor<half>();

        // dst = src0 + src1；fp16 一个 repeat 覆盖 128B，步长 8
        Add(zLocal, xLocal, yLocal, tileLength_, 1, {1, 1, 1, 8, 8, 8});

        outQueueZ_.EnQue<half>(zLocal);
        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<half> zLocal = outQueueZ_.DeQue<half>();
        DataCopy(zGm_[progress * tileLength_], zLocal, tileLength_);
        outQueueZ_.FreeTensor(zLocal);
    }

private:
    static constexpr int32_t BUFFER_NUM = 2;   // 双缓冲

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_, inQueueY_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ_;
    GlobalTensor<half> xGm_, yGm_, zGm_;

    uint32_t totalLength_ = 0;
    uint32_t blockLength_ = 0;
    uint32_t tileNum_ = 0;
    uint32_t tileLength_ = 0;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR tiling)
{
    KernelAdd op;
    op.Init(x, y, z, tiling);
    op.Process();
}
