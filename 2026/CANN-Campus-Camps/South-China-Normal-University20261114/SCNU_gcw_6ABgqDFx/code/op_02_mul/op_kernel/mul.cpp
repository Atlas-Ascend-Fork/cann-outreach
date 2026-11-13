// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;   // double buffer 流水深度

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
        this->tileLength = totalLength / tileNum;   // 本题 16384/8=2048
        this->coreIdx = GetBlockIdx();
        this->coreNum = GetBlockNum();

        // GlobalTensor 均指向全量起点，靠 tile 编号算全局偏移
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), totalLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), totalLength);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(z), totalLength);

        // 每个 buffer 大小 = tileLength * sizeof(DT_X)
        // float32 下 2048*4=8KB，三队列双 buffer 共 48KB，远小于 UB
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        // 全局 tile 按核编号轮询分配：
        // tileNum=8、coreNum=8 时，每核只处理 1 个 tile，处理 2048 个元素
        for (int32_t i = static_cast<int32_t>(this->coreIdx);
             i < static_cast<int32_t>(this->tileNum);
             i += static_cast<int32_t>(this->coreNum)) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        int32_t offset = progress * static_cast<int32_t>(this->tileLength);

        DataCopy(xLocal, xGm[offset], this->tileLength);
        DataCopy(yLocal, yGm[offset], this->tileLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t /*progress*/) {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        // 逐元素相乘，加 AscendC:: 前缀避免与算子名 Mul 混淆
        AscendC::Mul(zLocal, xLocal, yLocal, this->tileLength);

        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor<DT_X>(xLocal);
        inQueueY.FreeTensor<DT_X>(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        int32_t offset = progress * static_cast<int32_t>(this->tileLength);

        DataCopy(zGm[offset], zLocal, this->tileLength);

        outQueueZ.FreeTensor<DT_X>(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    GlobalTensor<DT_X> zGm;
    uint32_t tileNum = 0;     // 全局 tile 总数
    uint32_t tileLength = 0;  // 每个 tile 的元素数
    uint32_t coreIdx = 0;     // 当前核编号
    uint32_t coreNum = 0;     // 总核数
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                               GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}