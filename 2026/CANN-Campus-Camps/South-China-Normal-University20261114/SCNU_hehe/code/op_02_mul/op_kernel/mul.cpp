// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

// 单次搬运的最大元素数：UB 占用 = 3 队列 × BUFFER_NUM × TILE_LENGTH × sizeof(DT_X)
// 取 1024 时 fp32 为 3 × 2 × 4KB = 24KB，远小于 UB 容量
constexpr uint32_t TILE_LENGTH = 1024;
constexpr uint32_t BUFFER_NUM  = 2;
constexpr uint32_t QUEUE_DEPTH = 1;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        // 32 字节对齐所需的元素个数：fp32 为 8，fp16 为 16
        uint32_t alignNum = 32 / sizeof(DT_X);

        uint32_t blockNum = AscendC::GetBlockNum();
        if (blockNum == 0) {
            blockNum = 1;
        }
        uint32_t blockIdx = AscendC::GetBlockIdx();

        // 每核长度向上取整，再按 32B 对齐：
        // 核数不能整除总长度时（例如 16384 / 40），多出来的元素由前面的核均摊，
        // 最后一个核少拿一些，保证既不漏算也不重算
        uint32_t perCore = (length + blockNum - 1) / blockNum;
        perCore = (perCore + alignNum - 1) / alignNum * alignNum;

        this->start = blockIdx * perCore;
        if (this->start >= length) {
            // 该核分不到数据，Process 里直接返回
            this->coreLength = 0;
            return;
        }
        uint32_t remain = length - this->start;
        this->coreLength = (remain < perCore) ? remain : perCore;

        // 每个核只看自己那一段 GM
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + this->start, this->coreLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + this->start, this->coreLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + this->start, this->coreLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (this->coreLength == 0) {
            return;
        }
        uint32_t loopCount = (this->coreLength + TILE_LENGTH - 1) / TILE_LENGTH;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t offset = i * TILE_LENGTH;
            uint32_t remain = this->coreLength - offset;
            uint32_t curLen = (remain < TILE_LENGTH) ? remain : TILE_LENGTH;
            CopyIn(offset, curLen);
            Compute(curLen);
            CopyOut(offset, curLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t curLen) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[offset], curLen);
        AscendC::DataCopy(yLocal, yGm[offset], curLen);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t curLen) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        // 矢量乘法：z = x * y
        AscendC::Mul(zLocal, xLocal, yLocal, curLen);
        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curLen) {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[offset], zLocal, curLen);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, QUEUE_DEPTH> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, QUEUE_DEPTH> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;

    uint32_t start;       // 本核在全局张量中的起始元素下标
    uint32_t coreLength;  // 本核负责的元素个数
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
