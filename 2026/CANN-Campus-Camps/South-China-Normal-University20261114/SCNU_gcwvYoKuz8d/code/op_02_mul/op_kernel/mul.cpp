// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;   // 双缓冲
constexpr uint32_t QUEUE_DEPTH = 1;  // 队列深度

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const MulTilingData& tiling) {
        // 1. 获取 Tiling 参数（此时 totalLength 已在 Host 侧做过 32B 对齐）
        this->totalLength = tiling.totalLength;
        this->tileNum = tiling.tileNum;
        
        // 2. 计算单次处理的元素长度（总长 / 切分块数 / 缓冲数）
        this->tileLength = this->totalLength / this->tileNum / BUFFER_NUM;

        // 3. 单核实现，直接绑定 GM 首地址
        xGm.SetGlobalBuffer((__gm__ DT_X *)x, this->totalLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, this->totalLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z, this->totalLength);

        // 4. 为队列分配 UB 内存，大小为 tileLength * sizeof(DT_X)，满足 32B 对齐要求
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        // 循环次数 = tileNum * BUFFER_NUM
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);    
            Compute(i);   
            CopyOut(i);   
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 从队列中申请 LocalTensor 内存
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();

        // 从 Global Memory 搬运数据到 UB
        // progress * tileLength 偏移量保证了每次搬运的起始地址都是 32B 对齐的
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);

        // 数据入队
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        // 从队列中取出数据
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        // 核心计算：矢量乘法 z = x * y
        Mul(zLocal, xLocal, yLocal, this->tileLength);

        // 结果入队
        outQueueZ.EnQue<DT_X>(zLocal);

        // 释放输入内存
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 从队列中取出结果
        LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();

        // 将结果从 UB 搬运回 Global Memory
        DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);

        // 释放输出内存
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> outQueueZ;
    GlobalTensor<DT_X> xGm, yGm, zGm;

    uint32_t totalLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    
    KernelMul<DT_X> op;
    // 传入获取到的 tiling_data
    op.Init(x, y, z, tiling_data); 
    op.Process();
}