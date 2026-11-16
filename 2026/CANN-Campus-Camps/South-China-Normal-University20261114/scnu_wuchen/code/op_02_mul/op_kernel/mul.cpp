#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_NUM = 4;
constexpr int32_t TOTAL_CORES = 8;

constexpr int32_t TOTAL_LENGTH = 8 * 2048;
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / TOTAL_CORES;
constexpr int32_t TILE_LENGTH =
    BLOCK_LENGTH / TILE_NUM / BUFFER_NUM;


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

        // 初始化Pipe缓冲区
        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            TILE_LENGTH * sizeof(DT_X)
        );

        pipe.InitBuffer(
            inQueueY,
            BUFFER_NUM,
            TILE_LENGTH * sizeof(DT_X)
        );

        pipe.InitBuffer(
            outQueueZ,
            BUFFER_NUM,
            TILE_LENGTH * sizeof(DT_X)
        );
    }


    // 核心处理函数
    __aicore__ inline void Process()
    {
        int32_t loopCount = TILE_NUM * BUFFER_NUM;

        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }


private:

    // CopyIn：将输入数据搬运到UB
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();

        LocalTensor<DT_X> yLocal =
            inQueueY.AllocTensor<DT_X>();

        DataCopy(
            xLocal,
            xGm[progress * TILE_LENGTH],
            TILE_LENGTH
        );

        DataCopy(
            yLocal,
            yGm[progress * TILE_LENGTH],
            TILE_LENGTH
        );

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }


    // Compute：执行逐元素乘法
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();

        LocalTensor<DT_X> yLocal =
            inQueueY.DeQue<DT_X>();

        LocalTensor<DT_X> zLocal =
            outQueueZ.AllocTensor<DT_X>();

        // 核心运算：z = x * y
        Mul(
            zLocal,
            xLocal,
            yLocal,
            TILE_LENGTH
        );

        outQueueZ.EnQue<DT_X>(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }


    // CopyOut：将计算结果写回GM
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DT_X> zLocal =
            outQueueZ.DeQue<DT_X>();

        DataCopy(
            zGm[progress * TILE_LENGTH],
            zLocal,
            TILE_LENGTH
        );

        outQueueZ.FreeTensor(zLocal);
    }


private:

    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;

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