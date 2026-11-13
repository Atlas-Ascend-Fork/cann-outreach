// Kernel侧核函数实现：z = x * y
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;      // 双缓冲队列深度
constexpr uint32_t TILE_LENGTH = 2048; // 单核内单次搬运与计算的元素数，配合双缓冲形成三级流水

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        // DataCopy 要求 GM 起始地址与搬运长度按 32B 对齐，按 dtype 换算为元素粒度
        uint32_t align = 32 / sizeof(DT_X);
        uint32_t blockNum = GetBlockNum();
        if (blockNum == 0) {
            blockNum = 1;
        }
        uint32_t blockLength = ((length + blockNum - 1) / blockNum + align - 1) / align * align;

        uint32_t start = GetBlockIdx() * blockLength;
        if (start >= length) {
            coreLength = 0;
            return;
        }
        uint32_t remain = length - start;
        coreLength = (remain < blockLength) ? remain : blockLength;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + start, coreLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + start, coreLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + start, coreLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (coreLength == 0) {
            return;
        }
        uint32_t loopCount = (coreLength + TILE_LENGTH - 1) / TILE_LENGTH;
        for (uint32_t i = 0; i < loopCount; ++i) {
            uint32_t offset = i * TILE_LENGTH;
            uint32_t curLength = (offset + TILE_LENGTH <= coreLength)
                                     ? TILE_LENGTH : (coreLength - offset);
            CopyIn(offset, curLength);
            Compute(curLength);
            CopyOut(offset, curLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t curLength) {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm[offset], curLength);
        DataCopy(yLocal, yGm[offset], curLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t curLength) {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        Mul(zLocal, xLocal, yLocal, curLength);
        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curLength) {
        LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        DataCopy(zGm[offset], zLocal, curLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DT_X> xGm, yGm, zGm;
    uint32_t coreLength = 0;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
