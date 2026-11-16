// Kernel侧核函数实现
#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t TILE_LENGTH = 2048;   // 每个 tile 的元素个数

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                const MulTilingData& tiling)
    {
        int64_t totalLength = static_cast<int64_t>(tiling.length);
        int64_t blockNum = static_cast<int64_t>(AscendC::GetBlockNum());
        int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
        if (blockNum <= 0) {
            blockNum = 1;
        }

        // 每核基础长度按 32B 对齐，避免 DataCopy 尾部不满足对齐要求
        int64_t alignElem = 32 / static_cast<int64_t>(sizeof(DT_X));
        int64_t baseLen = (totalLength / blockNum) / alignElem * alignElem;

        int64_t offset = 0;
        if (baseLen == 0) {
            // 核数过多：让最后一个核处理全部数据，其他核为空
            if (blockIdx == blockNum - 1) {
                this->blockLength = totalLength;
                offset = 0;
            } else {
                this->blockLength = 0;
                offset = 0;
            }
        } else {
            if (blockIdx < blockNum - 1) {
                this->blockLength = baseLen;
                offset = blockIdx * baseLen;
            } else {
                offset = (blockNum - 1) * baseLen;
                this->blockLength = totalLength - offset;
            }
        }

        this->tileLength = TILE_LENGTH;

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + offset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + offset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + offset, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) {
            return;
        }

        int64_t tileNum = (this->blockLength + this->tileLength - 1) / this->tileLength;
        for (int64_t i = 0; i < tileNum; ++i) {
            int64_t curLen = this->tileLength;
            if (i == tileNum - 1) {
                curLen = this->blockLength - i * this->tileLength;
            }
            CopyIn(i, curLen);
            Compute(i, curLen);
            CopyOut(i, curLen);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t curLen)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();

        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], curLen);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], curLen);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int64_t progress, int64_t curLen)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        AscendC::Mul(zLocal, xLocal, yLocal, static_cast<uint32_t>(curLen));

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t curLen)
    {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, curLen);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM>  inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM>  inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;

    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;

    int64_t blockLength = 0;
    int64_t tileLength  = TILE_LENGTH;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                               GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data);
    op.Process();
}