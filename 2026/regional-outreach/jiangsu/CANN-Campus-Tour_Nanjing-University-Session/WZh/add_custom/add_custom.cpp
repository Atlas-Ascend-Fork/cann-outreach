/*
 * @file add_custom.cpp
 * @brief Ascend C 算子：add_custom，对两个 float16 张量执行逐元素加法 z = x + y
 * 数据格式：ND，形状：2 维 [N2, N1]
 */

#include "kernel_operator.h"

using namespace AscendC;

// tiling 数据结构：host 侧填充，通过 GM 传入 kernel
struct AddCustomTilingData {
    uint32_t totalLength; // 张量总元素个数 = N2 * N1
    uint32_t tileLength;  // 每个循环处理的最小单元（对齐到 32 * 16 元素）
    uint32_t blockLength; // 每个 AI Core 处理的元素个数
    uint32_t blockIdx;    // 当前 Core 索引
};

constexpr uint32_t BUFFER_NUM = 2; // double buffer 深度

class KernelAddCustom {
public:
    __aicore__ inline KernelAddCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                const AddCustomTilingData *__restrict tiling) {
        ASSERT(GetBlockNum() == 1 && "blockNum must be 1");
        this->blockLength = tiling->blockLength;
        this->tileLength = tiling->tileLength;
        this->blockIdx = tiling->blockIdx;

        xGm.SetGlobalBuffer((__gm__ half *)x + this->blockIdx * this->blockLength,
                            this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half *)y + this->blockIdx * this->blockLength,
                            this->blockLength);
        zGm.SetGlobalBuffer((__gm__ half *)z + this->blockIdx * this->blockLength,
                            this->blockLength);

        // 为每个 buffer 分配 tileLength * sizeof(half) 空间，BUFFER_NUM 块 double buffer
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
    }

    __aicore__ inline void Process() {
        // 循环次数：blockLength / tileLength
        int32_t loopCount = this->blockLength / this->tileLength;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();

        // z = x + y，逐元素相加
        Add(zLocal, xLocal, yLocal, this->tileLength);

        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    GlobalTensor<half> xGm, yGm, zGm;

    uint32_t blockLength = 0;
    uint32_t tileLength = 0;
    uint32_t blockIdx = 0;
};

// tiling 函数：host 侧调用，计算切分策略
static bool ComputeTiling(uint32_t totalLength, uint32_t coreNum,
                          AddCustomTilingData &tiling) {
    (void)coreNum; // 本算子使用单核实现
    tiling.totalLength = totalLength;
    tiling.blockIdx = 0;
    tiling.blockLength = totalLength;

    // tileLength 向上对齐到 32 元素的整数倍，且不超过 blockLength
    constexpr uint32_t ALIGN_NUM = 32;
    uint32_t tileLength = ALIGN_NUM;
    while (tileLength * 2 <= totalLength) {
        tileLength *= 2;
    }
    tiling.tileLength = tileLength < totalLength ? tileLength : totalLength;
    return true;
}

extern "C" bool AddCustomTilingFunc(uint32_t blockDim, uint32_t totalLength,
                                    uint32_t tileLength, AddCustomTilingData &tiling) {
    (void)blockDim;
    (void)tileLength;
    return ComputeTiling(totalLength, 1, tiling);
}

// kernel 入口
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                 GM_ADDR tiling) {
    KernelAddCustom op;
    op.Init(x, y, z, reinterpret_cast<AddCustomTilingData *>(tiling));
    op.Process();
}

