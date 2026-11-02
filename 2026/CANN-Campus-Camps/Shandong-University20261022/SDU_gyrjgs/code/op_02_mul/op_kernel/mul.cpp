
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    // 初始化输入、输出和本地缓冲区
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t length,
        uint32_t tileLength) {

        length_ = length;
        tileLength_ = tileLength;

        // 设置全局内存
        xGm.SetGlobalBuffer(
            (__gm__ DT_X *)x, length_);

        yGm.SetGlobalBuffer(
            (__gm__ DT_X *)y, length_);

        zGm.SetGlobalBuffer(
            (__gm__ DT_X *)z, length_);

        // 为两个输入和一个输出申请本地缓冲区
        pipe.InitBuffer(
            inQueueX, 1,
            tileLength_ * sizeof(DT_X));

        pipe.InitBuffer(
            inQueueY, 1,
            tileLength_ * sizeof(DT_X));

        pipe.InitBuffer(
            outQueueZ, 1,
            tileLength_ * sizeof(DT_X));
    }

    // 执行逐元素乘法
    __aicore__ inline void Process() {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();

        // 每个核负责不同的数据块
        uint32_t stride = blockNum * tileLength_;

        for (uint32_t offset = blockIdx * tileLength_;
             offset < length_;
             offset += stride) {

            uint32_t dataLength = tileLength_;

            // 处理最后一个不足一个数据块的情况
            if (offset + dataLength > length_) {
                dataLength = length_ - offset;
            }

            CopyIn(offset, dataLength);
            Compute(dataLength);
            CopyOut(offset, dataLength);
        }
    }

private:
    // 从全局内存读取输入
    __aicore__ inline void CopyIn(
        uint32_t offset,
        uint32_t dataLength) {

        LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();

        LocalTensor<DT_X> yLocal =
            inQueueY.AllocTensor<DT_X>();

        DataCopy(xLocal, xGm[offset], dataLength);
        DataCopy(yLocal, yGm[offset], dataLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    // 计算 z = x * y
    __aicore__ inline void Compute(
        uint32_t dataLength) {

        LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();

        LocalTensor<DT_X> yLocal =
            inQueueY.DeQue<DT_X>();

        LocalTensor<DT_X> zLocal =
            outQueueZ.AllocTensor<DT_X>();

        Mul(zLocal, xLocal, yLocal, dataLength);

        outQueueZ.EnQue<DT_X>(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    // 将计算结果写回全局内存
    __aicore__ inline void CopyOut(
        uint32_t offset,
        uint32_t dataLength) {

        LocalTensor<DT_X> zLocal =
            outQueueZ.DeQue<DT_X>();

        DataCopy(zGm[offset], zLocal, dataLength);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueY;
    TQue<QuePosition::VECOUT, 1> outQueueZ;

    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    GlobalTensor<DT_X> zGm;

    uint32_t length_;
    uint32_t tileLength_;
};


// Kernel 入口
template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling) {

    REGISTER_TILING_DEFAULT(MulTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        MulTilingData, tiling_data, tiling);

    KernelMul<DT_X> op;

    op.Init(
        x,
        y,
        z,
        tiling_data.length,
        tiling_data.tileLength);

    op.Process();
}
