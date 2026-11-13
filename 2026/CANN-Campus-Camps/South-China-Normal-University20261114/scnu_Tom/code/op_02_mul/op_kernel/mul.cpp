// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BLOCK_SIZE = 32;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        // 获取当前核的全局偏移，实现多核切分
        uint32_t blockLength = length / GetBlockNum();
        uint32_t remainder = length % GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        
        // 处理尾块，保证最后一个核能处理余数（如果 length 不能被核数整除）
        uint32_t curBlockLength = blockLength + (blockIdx == GetBlockNum() - 1 ? remainder : 0);
        uint32_t globalOffset = blockIdx * blockLength; // 假设余数全给最后一个核

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + globalOffset, curBlockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + globalOffset, curBlockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + globalOffset, curBlockLength);

        // 32B 对齐：计算一个 Block 包含的元素个数
        uint32_t alignElements = BLOCK_SIZE / sizeof(DT_X);
        
        // 根据可用 UB 内存和当前核处理的数据量，计算 Tiling
        // 固定 tileNum 为 8 或者根据长度动态计算
        this->tileNum = 8;
        // 向上对齐到 32B 的整数倍
        uint32_t alignedLength = (curBlockLength + alignElements - 1) / alignElements * alignElements;
        this->tileLength = alignedLength / this->tileNum;
        // 保证每次处理的大小依然是 32B 的倍数
        this->tileLength = (this->tileLength + alignElements - 1) / alignElements * alignElements;

        // 初始化队列
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        // 核心：调用 AscendC::Mul 进行矢量乘法
        Mul(zLocal, xLocal, yLocal, this->tileLength);

        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DT_X> xGm, yGm, zGm;
    uint32_t tileNum;
    uint32_t tileLength;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}