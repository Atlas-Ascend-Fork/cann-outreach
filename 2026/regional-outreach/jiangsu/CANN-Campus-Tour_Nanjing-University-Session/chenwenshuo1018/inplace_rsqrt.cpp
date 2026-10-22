// InplaceRsqrt 算子 (AscendC)
// 功能: 原位计算 1/sqrt(x), 覆盖输入张量
// dtype: float16 (half), format: ND, shape: 4维 [N4, N3, N2, N1] (按一维连续内存处理)
// 实现: AscendC 向量指令 Rsqrt (对应 vrsqrt), 分块批量向量化, 无标量循环, 直接修改 self 内存

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;        // double buffer
constexpr uint32_t TILE_LENGTH = 8192;   // 每块元素个数 (8192 * 2B = 16KB)

// tiling 结构: host 侧填充, 设备侧读取
struct TilingInplaceRsqrt {
    uint32_t totalLength;   // 4维张量元素总数 N4*N3*N2*N1
    uint32_t tileNum;       // 分块数
};

class KernelInplaceRsqrt {
public:
    __aicore__ inline KernelInplaceRsqrt() {}
    __aicore__ inline void Init(GM_ADDR self, const TilingInplaceRsqrt &tiling)
    {
        this->totalLength = tiling.totalLength;
        this->tileNum = tiling.tileNum;

        selfGm.SetGlobalBuffer((__gm__ half *)self, totalLength);
        pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(outQueue, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline uint32_t ValidCount(int32_t progress) const
    {
        uint32_t offset = progress * TILE_LENGTH;
        return (offset + TILE_LENGTH <= totalLength) ? TILE_LENGTH : (totalLength - offset);
    }

    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<half> selfLocal = inQueue.AllocTensor<half>();
        // 尾块只读入有效数据, 计算后按同样长度写出
        DataCopy(selfLocal, selfGm[progress * TILE_LENGTH], ValidCount(progress));
        inQueue.EnQue(selfLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<half> selfLocal = inQueue.DeQue<half>();
        LocalTensor<half> dstLocal = outQueue.AllocTensor<half>();

        // vrsqrt: dst = 1/sqrt(src), 按 count 向量化执行, 无标量循环
        Rsqrt(dstLocal, selfLocal, ValidCount(progress));

        outQueue.EnQue<half>(dstLocal);
        inQueue.FreeTensor(selfLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<half> dstLocal = outQueue.DeQue<half>();
        // 原位覆盖 self 的 GM 内存
        DataCopy(selfGm[progress * TILE_LENGTH], dstLocal, ValidCount(progress));
        outQueue.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    GlobalTensor<half> selfGm;

    uint32_t totalLength = 0;
    uint32_t tileNum = 0;
};

extern "C" __global__ __aicore__ void inplace_rsqrt(GM_ADDR self, GM_ADDR tiling)
{
    const TilingInplaceRsqrt *td = reinterpret_cast<const TilingInplaceRsqrt *>(tiling);
    KernelInplaceRsqrt op;
    op.Init(self, *td);
    op.Process();
}
