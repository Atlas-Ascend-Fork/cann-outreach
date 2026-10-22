/**
 * \file add_custom.cpp
 * \brief add_custom 自定义算子核函数（Ascend C 实现）
 *
 * 功能：对两个输入张量 x、y 执行逐元素加法，输出 z = x + y。
 * 规格：数据类型 float16（half）、数据格式 ND、形状 2 维 [N2, N1]（动态）。
 * 并行：Host 侧 TilingFunc 依据总元素数选择 AICore 核数（SetBlockDim），
 *       各核按元素区间均匀切分（尾核处理余数），核内以 TILE_NUM 为粒度
 *       做 “CopyIn -> Compute -> CopyOut” 流水。
 */
#include "kernel_operator.h"
#include "add_custom_tiling.h"

using namespace AscendC;

namespace NsAddCustom {
constexpr uint32_t TILE_NUM = 2048;  // 单次搬移/计算元素数：2048 * 2B = 4KB

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    __aicore__ inline void Init(const GM_ADDR x, const GM_ADDR y, const GM_ADDR z, const uint32_t totalLength)
    {
        this->totalLength = totalLength;
        xGlobal.SetGlobalBuffer((__gm__ half *)x);
        yGlobal.SetGlobalBuffer((__gm__ half *)y);
        zGlobal.SetGlobalBuffer((__gm__ half *)z);

        // 依据本 AICore 数量对本核负责的元素做均匀切分，余数归靠前的核，保证无遗漏、无越界
        const uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
        const uint32_t coreId = static_cast<uint32_t>(GetBlockIdx());
        const uint32_t coreLen = totalLength / blockNum;
        const uint32_t remain = totalLength % blockNum;
        coreOffset_ = coreLen * coreId + (coreId < remain ? coreId : remain);
        coreCount_ = coreLen + (coreId < remain ? 1 : 0);

        // 为 x / y / z 各申请 TILE_NUM 个 half 的 UB 空间
        pipe.InitBuffer(xQueue, 1, TILE_NUM * sizeof(half));
        pipe.InitBuffer(yQueue, 1, TILE_NUM * sizeof(half));
        pipe.InitBuffer(zQueue, 1, TILE_NUM * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        uint32_t processed = 0;
        while (processed < coreCount_) {
            const uint32_t curLen = (coreCount_ - processed > TILE_NUM) ? TILE_NUM : (coreCount_ - processed);
            CopyIn(coreOffset_ + processed, curLen);
            Compute(curLen);
            CopyOut(coreOffset_ + processed, curLen);
            processed += curLen;
        }
    }

private:
    // 将 GM 上的 x、y 数据搬移到 UB
    __aicore__ inline void CopyIn(const uint32_t offset, const uint32_t len)
    {
        LocalTensor<half> xLocal = xQueue.AllocTensor<half>();
        LocalTensor<half> yLocal = yQueue.AllocTensor<half>();
        DataCopy(xLocal, xGlobal[offset], len);
        DataCopy(yLocal, yGlobal[offset], len);
        xQueue.EnQue(xLocal);
        yQueue.EnQue(yLocal);
    }

    // UB 上执行向量加法 z = x + y
    __aicore__ inline void Compute(const uint32_t len)
    {
        LocalTensor<half> xLocal = xQueue.DeQue<half>();
        LocalTensor<half> yLocal = yQueue.DeQue<half>();
        LocalTensor<half> zLocal = zQueue.AllocTensor<half>();
        Add(zLocal, xLocal, yLocal, len);
        zQueue.EnQue(zLocal);
        xQueue.FreeTensor(xLocal);
        yQueue.FreeTensor(yLocal);
    }

    // 将 UB 上的结果 z 搬回 GM
    __aicore__ inline void CopyOut(const uint32_t offset, const uint32_t len)
    {
        LocalTensor<half> zLocal = zQueue.DeQue<half>();
        DataCopy(zGlobal[offset], zLocal, len);
        zQueue.FreeTensor(zLocal);
    }

    TPipe pipe;
    TQue<TPosition::VECIN, 1> xQueue;
    TQue<TPosition::VECIN, 1> yQueue;
    TQue<TPosition::VECOUT, 1> zQueue;

    GlobalTensor<half> xGlobal;
    GlobalTensor<half> yGlobal;
    GlobalTensor<half> zGlobal;

    uint32_t totalLength = 0;
    uint32_t coreOffset_ = 0;
    uint32_t coreCount_ = 0;
};
}  // namespace NsAddCustom

// 核函数入口：参数顺序与 OpDef 中声明的输入(x,y) + 输出(z) 一致，随后是 workspace 与 tiling
extern "C" __global__ __aicore__ void add_custom(const GM_ADDR x, const GM_ADDR y, const GM_ADDR z,
                                                 const GM_ADDR workspace, const GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    NsAddCustom::KernelAdd op;
    op.Init(x, y, z, tilingData.totalLength);
    op.Process();
}
