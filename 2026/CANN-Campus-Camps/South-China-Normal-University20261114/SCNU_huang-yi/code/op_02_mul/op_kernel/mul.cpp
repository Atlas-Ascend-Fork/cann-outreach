// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;    // 双缓冲
constexpr int32_t TILE_SIZE = 2048;  // 每个 tile 处理的最大元素数

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        this->totalLength = length;
        this->blockNum = static_cast<uint32_t>(GetBlockNum());
        this->coreIndex = static_cast<uint32_t>(GetBlockIdx());
        if (this->blockNum == 0U) {
            this->blockNum = 1U;
        }
        // host 侧保证 blockDim 能整除 length，故每核长度均等、且为 2 的幂（32 字节对齐）
        this->blockLength = this->totalLength / this->blockNum;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, totalLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z, totalLength);

        // 为三个队列各分配 BUFFER_NUM 块 TILE_SIZE 大小的 UB 空间（双缓冲）
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_SIZE * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_SIZE * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_SIZE * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        // 本核负责的全局起始偏移：coreIndex * blockLength
        uint32_t start = coreIndex * blockLength;
        for (uint32_t i = 0; i < blockLength; i += TILE_SIZE) {
            uint32_t calSize =
                (i + static_cast<uint32_t>(TILE_SIZE) <= blockLength) ? static_cast<uint32_t>(TILE_SIZE)
                                                                      : (blockLength - i);
            CopyIn(start + i, calSize);
            Compute(calSize);
            CopyOut(start + i, calSize);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t calSize) {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm[offset], calSize);
        DataCopy(yLocal, yGm[offset], calSize);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t calSize) {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        // 矢量乘法：z = x * y
        Mul(zLocal, xLocal, yLocal, calSize);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t calSize) {
        LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        DataCopy(zGm[offset], zLocal, calSize);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DT_X> xGm, yGm, zGm;

    uint32_t totalLength = 0;
    uint32_t blockNum = 1;
    uint32_t coreIndex = 0;
    uint32_t blockLength = 0;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
