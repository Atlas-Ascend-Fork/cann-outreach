/**
 * add_custom 算子核函数（Ascend C）
 * 功能：z = x + y，逐元素加法
 * 数据类型：half (float16)，格式 ND，shape [N2, N1]
 * 使用向量 0 级 API：DataCopy / Adds（此处用 Add 双目）/ Tensor 级流水
 */
#include "kernel_operator.h"
#include "add_custom_tiling.h"

// 核函数侧（AI Core 编译）从 tiling buffer 中解析出 tiling 结构
#define GET_TILING_DATA(var, tiling) \
    const AddCustomTilingData var = *reinterpret_cast<const AddCustomTilingData *>(tiling)

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // double buffer

class KernelAddCustom {
public:
    __aicore__ inline KernelAddCustom() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileLength)
    {
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = blockLength / tileLength;
        this->tileLength = tileLength;

        xGm.SetGlobalBuffer((__gm__ half *)x + GetBlockIdx() * this->blockLength,
                            this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half *)y + GetBlockIdx() * this->blockLength,
                            this->blockLength);
        zGm.SetGlobalBuffer((__gm__ half *)z + GetBlockIdx() * this->blockLength,
                            this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();

        Add(zLocal, xLocal, yLocal, this->tileLength); // z = x + y

        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    GlobalTensor<half> xGm, yGm, zGm;

    uint32_t blockLength; // 每个 AI Core 处理的数据长度
    uint32_t tileNum;     // 每个 AI Core 分片的循环次数
    uint32_t tileLength;  // 每次搬运/计算的数据长度
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelAddCustom op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileLength);
    if (TILING_KEY_IS(1)) { // 2维 [N2, N1] 按 ND 展平为一维处理
        op.Process();
    }
}
