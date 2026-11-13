// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr uint32_t BUFFER_NUM = 1;   // 单缓冲，核内按块串行处理
constexpr uint32_t QUEUE_DEPTH = 1;
constexpr uint32_t TILE_NUM = 8;     // 单核内的数据块个数

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        // 按核数均分总长度，host 侧已保证整除且每核长度满足 32B 对齐
        uint32_t blockLength = length / AscendC::GetBlockNum();
        uint32_t blockOffset = blockLength * AscendC::GetBlockIdx();
        this->tileNum = TILE_NUM;
        this->tileLength = blockLength / TILE_NUM / BUFFER_NUM;
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + blockOffset, blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + blockOffset, blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + blockOffset, blockLength);
        // x 与 y 共用一块输入缓冲区，容量为两个数据块
        pipe.InitBuffer(inQueueXY, BUFFER_NUM, 2 * this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        AscendC::LocalTensor<DT_X> xyLocal = inQueueXY.AllocTensor<DT_X>();
        AscendC::DataCopy(xyLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(xyLocal[this->tileLength], yGm[progress * this->tileLength], this->tileLength);
        inQueueXY.EnQue(xyLocal);
    }

    __aicore__ inline void Compute() {
        AscendC::LocalTensor<DT_X> xyLocal = inQueueXY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        // y 位于同一缓冲区内 x 之后，偏移一个数据块长度
        AscendC::Mul(zLocal, xyLocal, xyLocal[this->tileLength], this->tileLength);
        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueXY.FreeTensor(xyLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, QUEUE_DEPTH> inQueueXY;   // 输入队列，x 与 y 共用
    AscendC::TQue<AscendC::QuePosition::VECOUT, QUEUE_DEPTH> outQueueZ;  // 输出队列
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;

    uint32_t tileNum;     // 单核内的数据块个数
    uint32_t tileLength;  // 每块元素数，其字节数为 32B 的整数倍
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
