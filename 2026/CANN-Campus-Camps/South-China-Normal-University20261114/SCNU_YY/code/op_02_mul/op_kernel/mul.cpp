// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

template <typename T>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                const MulTilingData* tilingData) {
        tiling = tilingData;

        xGm.SetGlobalBuffer((__gm__ T*)x, tiling->length);
        yGm.SetGlobalBuffer((__gm__ T*)y, tiling->length);
        zGm.SetGlobalBuffer((__gm__ T*)z, tiling->length);

        pipe.InitBuffer(inQueueX, 1, tiling->tileLength * sizeof(T));
        pipe.InitBuffer(inQueueY, 1, tiling->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueZ, 1, tiling->tileLength * sizeof(T));
    }

    __aicore__ inline void Process() {
        uint32_t block_idx = GetBlockIdx();
        uint32_t start = block_idx * tiling->perCoreLength;
        if (start >= tiling->length) {
            return;
        }

        uint32_t length = tiling->length - start;
        if (length > tiling->perCoreLength) {
            length = tiling->perCoreLength;
        }

        for (uint32_t i = 0; i < tiling->tileNum; ++i) {
            uint32_t offset = i * tiling->tileLength;
            if (offset >= length) {
                break;
            }

            uint32_t cur_len = length - offset;
            if (cur_len > tiling->tileLength) {
                cur_len = tiling->tileLength;
            }

            CopyIn(start + offset, cur_len);
            Compute(cur_len);
            CopyOut(start + offset, cur_len);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t gm_offset, uint32_t len) {
        LocalTensor<T> x_local = inQueueX.AllocTensor<T>();
        LocalTensor<T> y_local = inQueueY.AllocTensor<T>();

        DataCopy(x_local, xGm[gm_offset], len);
        DataCopy(y_local, yGm[gm_offset], len);

        inQueueX.EnQue(x_local);
        inQueueY.EnQue(y_local);
    }

    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<T> x_local = inQueueX.DeQue<T>();
        LocalTensor<T> y_local = inQueueY.DeQue<T>();
        LocalTensor<T> z_local = outQueueZ.AllocTensor<T>();

        Mul(z_local, x_local, y_local, len);

        outQueueZ.EnQue(z_local);
        inQueueX.FreeTensor(x_local);
        inQueueY.FreeTensor(y_local);
    }

    __aicore__ inline void CopyOut(uint32_t gm_offset, uint32_t len) {
        LocalTensor<T> z_local = outQueueZ.DeQue<T>();
        DataCopy(zGm[gm_offset], z_local, len);
        outQueueZ.FreeTensor(z_local);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueY;
    TQue<QuePosition::VECOUT, 1> outQueueZ;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> zGm;

    const MulTilingData* tiling = nullptr;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                               GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);

    KernelMul<DT_X> op;
    op.Init(x, y, z, &tiling_data);
    op.Process();
}