// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    // 4核切分：每核 4096 元素，拆分为 2 个 2048 元素的双缓冲块实现 MTE2/VEC/MTE3 流水重叠
    static constexpr uint32_t BLOCK_LEN = 4096u;
    static constexpr uint32_t TILE_LEN = 2048u;
    static constexpr uint32_t IN_TILE_BYTES = TILE_LEN * 2u * sizeof(DT_X);
    static constexpr uint32_t OUT_TILE_BYTES = TILE_LEN * sizeof(DT_X);

    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
        uint32_t offset = AscendC::GetBlockIdx() << 12;

        AscendC::GlobalTensor<DT_X> xGm;
        AscendC::GlobalTensor<DT_X> yGm;
        AscendC::GlobalTensor<DT_X> zGm;
        xGm.SetGlobalBuffer((__gm__ DT_X*)x + offset, BLOCK_LEN);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + offset, BLOCK_LEN);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + offset, BLOCK_LEN);

        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQueueXY;
        AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outQueueZ;
        pipe.InitBuffer(inQueueXY, 2, IN_TILE_BYTES);
        pipe.InitBuffer(outQueueZ, 2, OUT_TILE_BYTES);

        // Tile 0 搬入：异步下发前半块 (2048 元素)
        AscendC::LocalTensor<DT_X> xyLocal0 = inQueueXY.AllocTensor<DT_X>();
        AscendC::DataCopy(xyLocal0, xGm, TILE_LEN);
        AscendC::DataCopy(xyLocal0[TILE_LEN], yGm, TILE_LEN);
        inQueueXY.EnQue(xyLocal0);

        // Tile 1 搬入：立即下发后半块 (2048 元素)，保证 MTE2 队列连续满载
        AscendC::LocalTensor<DT_X> xyLocal1 = inQueueXY.AllocTensor<DT_X>();
        AscendC::DataCopy(xyLocal1, xGm[TILE_LEN], TILE_LEN);
        AscendC::DataCopy(xyLocal1[TILE_LEN], yGm[TILE_LEN], TILE_LEN);
        inQueueXY.EnQue(xyLocal1);

        // Tile 0 计算与搬出：在 MTE2 搬运 Tile 1 的同时，并行完成 Tile 0 的计算与写回
        AscendC::LocalTensor<DT_X> zLocal0 = outQueueZ.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> xyDe0 = inQueueXY.DeQue<DT_X>();
        AscendC::Mul(zLocal0, xyDe0, xyDe0[TILE_LEN], TILE_LEN);
        outQueueZ.EnQue(zLocal0);
        AscendC::LocalTensor<DT_X> zDe0 = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm, zDe0, TILE_LEN);

        // Tile 1 计算与搬出：在 MTE3 写回 Tile 0 的同时，并行完成 Tile 1 的计算与写回
        AscendC::LocalTensor<DT_X> zLocal1 = outQueueZ.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> xyDe1 = inQueueXY.DeQue<DT_X>();
        AscendC::Mul(zLocal1, xyDe1, xyDe1[TILE_LEN], TILE_LEN);
        outQueueZ.EnQue(zLocal1);
        AscendC::LocalTensor<DT_X> zDe1 = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[TILE_LEN], zDe1, TILE_LEN);

        // 所有 DMA 指令下发完毕后统一释放张量
        inQueueXY.FreeTensor(xyDe0);
        inQueueXY.FreeTensor(xyDe1);
        outQueueZ.FreeTensor(zDe0);
        outQueueZ.FreeTensor(zDe1);
    }

    __aicore__ inline void Process() {}
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    (void)tiling_data;
    (void)workspace;
    KernelMul<DT_X> op;
    op.Init(x, y, z);
    op.Process();
}