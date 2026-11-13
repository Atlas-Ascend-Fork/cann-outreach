#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;   // 使用 double buffer 提高流水效率

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t totalLength,
        uint32_t tileNum
    ) {
        this->tileNum = tileNum;
        this->tileLength = totalLength / tileNum;   // 每个 tile 的元素个数
        this->coreIdx = GetBlockIdx();
        this->coreNum = GetBlockNum();

        // 初始化 GlobalTensor
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), totalLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), totalLength);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(z), totalLength);

        // 初始化队列与 LocalMemory
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        // 将 tileNum 个 tile 按核编号循环分配给各个核
        // 核 0 处理 tile 0, coreNum, 2*coreNum...；核 1 处理 tile 1, ...
        for (int32_t i = this->coreIdx; i < (int32_t)this->tileNum; i += this->coreNum) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        // progress 即 tile 编号
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        int32_t offset = progress * this->tileLength;

        DataCopy(xLocal, xGm[offset], this->tileLength);
        DataCopy(yLocal, yGm[offset], this->tileLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        // 逐元素相乘（注意加 AscendC:: 前缀避免与算子名 Mul 混淆）
        AscendC::Mul(zLocal, xLocal, yLocal, this->tileLength);

        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor<DT_X>(xLocal);
        inQueueY.FreeTensor<DT_X>(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        int32_t offset = progress * this->tileLength;

        DataCopy(zGm[offset], zLocal, this->tileLength);

        outQueueZ.FreeTensor<DT_X>(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DT_X> xGm, yGm, zGm;
    uint32_t tileNum = 0;     // tile 总数
    uint32_t tileLength = 0;  // 每个 tile 的元素数
    uint32_t coreIdx = 0;     // 当前核编号
    uint32_t coreNum = 0;     // 总核数
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}

