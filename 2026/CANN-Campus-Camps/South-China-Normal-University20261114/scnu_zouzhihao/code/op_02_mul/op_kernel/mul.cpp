
#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t TOTAL_CORES = 4;

constexpr uint32_t TOTAL_LENGTH = 8 * 2048;
constexpr uint32_t BLOCK_LENGTH = TOTAL_LENGTH / TOTAL_CORES;
constexpr uint32_t TILE_LENGTH = BLOCK_LENGTH;


template <class DT_X>
class KernelMul {

public:

    __aicore__ inline KernelMul() {}

    // 初始化
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t length)
    {
        // 每个核负责的数据起始位置
        uint32_t offset =
            BLOCK_LENGTH * GetBlockIdx();

        // 设置全局内存
        xGm.SetGlobalBuffer(
            (__gm__ DT_X*)x + offset,
            BLOCK_LENGTH
        );

        yGm.SetGlobalBuffer(
            (__gm__ DT_X*)y + offset,
            BLOCK_LENGTH
        );

        zGm.SetGlobalBuffer(
            (__gm__ DT_X*)z + offset,
            BLOCK_LENGTH
        );

        // 合并x、y输入缓冲区
        pipe.InitBuffer(
            inQueueXY,
            BUFFER_NUM,
            2 * TILE_LENGTH * sizeof(DT_X)
        );

        // 输出缓冲区
        pipe.InitBuffer(
            outQueueZ,
            BUFFER_NUM,
            TILE_LENGTH * sizeof(DT_X)
        );
    }


    // 核心处理函数
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }


private:

    // CopyIn：GM -> UB
    __aicore__ inline void CopyIn()
    {
        // 申请一个连续的输入缓冲区
        LocalTensor<DT_X> xyLocal =
            inQueueXY.AllocTensor<DT_X>();

        // y对应缓冲区的后半部分
        LocalTensor<DT_X> yLocal =
            xyLocal[TILE_LENGTH];

        // 分别搬运x和y
        DataCopy(
            xyLocal,
            xGm,
            TILE_LENGTH
        );

        DataCopy(
            yLocal,
            yGm,
            TILE_LENGTH
        );

        // 只进行一次入队
        inQueueXY.EnQue(xyLocal);
    }


    // Compute：逐元素乘法
    __aicore__ inline void Compute()
    {
        // 获取合并后的输入缓冲区
        LocalTensor<DT_X> xyLocal =
            inQueueXY.DeQue<DT_X>();

        // 获取y所在的后半段
        LocalTensor<DT_X> yLocal =
            xyLocal[TILE_LENGTH];

        // 申请输出空间
        LocalTensor<DT_X> zLocal =
            outQueueZ.AllocTensor<DT_X>();

        // z = x * y
        Mul(
            zLocal,
            xyLocal,
            yLocal,
            TILE_LENGTH
        );

        outQueueZ.EnQue<DT_X>(zLocal);

        // 只释放一次输入缓冲区
        inQueueXY.FreeTensor(xyLocal);
    }


    // CopyOut：UB -> GM
    __aicore__ inline void CopyOut()
    {
        LocalTensor<DT_X> zLocal =
            outQueueZ.DeQue<DT_X>();

        DataCopy(
            zGm,
            zLocal,
            TILE_LENGTH
        );

        outQueueZ.FreeTensor(zLocal);
    }


private:

    TPipe pipe;

    // 两个输入共用一个队列
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueXY;

    // 输出队列
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    GlobalTensor<DT_X> zGm;

};


// Kernel入口
template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        MulTilingData,
        tiling_data,
        tiling
    );

    KernelMul<DT_X> op;

    op.Init(
        x,
        y,
        z,
        tiling_data.length
    );

    op.Process();
}
