// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 2;   // 每个队列的 tensor 数量（双缓冲）

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length,
                                uint32_t blockLength, uint32_t tileLength) {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t offset   = blockIdx * blockLength;
        // 最后一核只处理真实剩余部分，不越界也不漏算
        if (offset >= length) {
            this->blockLength = 0;
        } else {
            uint32_t remain   = length - offset;
            this->blockLength = (remain < blockLength) ? remain : blockLength;
        }
        this->tileLength = tileLength;
        this->tileNum    = (this->blockLength + tileLength - 1) / tileLength;  // 向上取整

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + offset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + offset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + offset, this->blockLength);

        pipe.InitBuffer(inQueueX,  BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY,  BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < this->tileNum; i++) {
            uint32_t done   = i * this->tileLength;
            uint32_t remain = this->blockLength - done;
            uint32_t curLength = (remain < this->tileLength) ? remain : this->tileLength;
            CopyIn(i, curLength);
            Compute(curLength);
            CopyOut(i, curLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t curLength) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], curLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], curLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t curLength) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        // ★ 相对加法唯一的核心改动：Add -> Mul
        AscendC::Mul(zLocal, xLocal, yLocal, curLength);

        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t curLength) {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, curLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;
    uint32_t blockLength;  // 本核实际处理的元素数
    uint32_t tileNum;      // 本核分块数（由 blockLength 向上取整算出）
    uint32_t tileLength;   // 每块元素数
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length, tiling_data.blockLength, tiling_data.tileLength);
    op.Process();
}