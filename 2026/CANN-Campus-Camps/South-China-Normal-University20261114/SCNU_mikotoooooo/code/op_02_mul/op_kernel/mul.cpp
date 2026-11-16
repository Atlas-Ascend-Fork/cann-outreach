
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 2;


// KernelMul：负责 NPU 端的乘法计算
template <class DT_X>
class KernelMul {

public:

    __aicore__ inline KernelMul() {}

    // 初始化输入输出地址和分块参数
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t length,
        uint32_t tileNum
    ) {

        // 每个 Block 负责的数据量
        this->blockLength = length / AscendC::GetBlockNum();

        // 每个 Block 内部的任务数量
        this->tileNum = tileNum;

        // 每次处理的数据量
        this->tileLength =
            this->blockLength / tileNum / BUFFER_NUM;

        // 设置当前 Block 对应的 GM 地址
        xGm.SetGlobalBuffer(
            (__gm__ DT_X*)x +
            this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength
        );

        yGm.SetGlobalBuffer(
            (__gm__ DT_X*)y +
            this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength
        );

        zGm.SetGlobalBuffer(
            (__gm__ DT_X*)z +
            this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength
        );

        // 初始化输入输出队列
        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X)
        );

        pipe.InitBuffer(
            inQueueY,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X)
        );

        pipe.InitBuffer(
            outQueueZ,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X)
        );
    }


    // 执行完整计算流程
    __aicore__ inline void Process() {

        int32_t loopCount = this->tileNum * BUFFER_NUM;

        for (int32_t i = 0; i < loopCount; i++) {

            CopyIn(i);

            Compute(i);

            CopyOut(i);
        }
    }


private:

    // 将输入数据从 GM 搬运到 UB
    __aicore__ inline void CopyIn(int32_t progress) {

        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.AllocTensor<DT_X>();

        AscendC::DataCopy(
            xLocal,
            xGm[progress * this->tileLength],
            this->tileLength
        );

        AscendC::DataCopy(
            yLocal,
            yGm[progress * this->tileLength],
            this->tileLength
        );

        inQueueX.EnQue(xLocal);

        inQueueY.EnQue(yLocal);
    }


    // 在 UB 中执行乘法
    __aicore__ inline void Compute(int32_t progress) {

        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.DeQue<DT_X>();

        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.AllocTensor<DT_X>();

        // 核心计算：z = x * y
        AscendC::Mul(
            zLocal,
            xLocal,
            yLocal,
            this->tileLength
        );

        outQueueZ.EnQue<DT_X>(zLocal);

        inQueueX.FreeTensor(xLocal);

        inQueueY.FreeTensor(yLocal);
    }


    // 将计算结果从 UB 搬运回 GM
    __aicore__ inline void CopyOut(int32_t progress) {

        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.DeQue<DT_X>();

        AscendC::DataCopy(
            zGm[progress * this->tileLength],
            zLocal,
            this->tileLength
        );

        outQueueZ.FreeTensor(zLocal);
    }


private:

    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;

    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    AscendC::GlobalTensor<DT_X> xGm;

    AscendC::GlobalTensor<DT_X> yGm;

    AscendC::GlobalTensor<DT_X> zGm;

    uint32_t blockLength;

    uint32_t tileNum;

    uint32_t tileLength;
};


// Kernel 入口
template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling
) {

    // 注册并读取 Tiling 数据
    REGISTER_TILING_DEFAULT(MulTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        MulTilingData,
        tiling_data,
        tiling
    );

    // 创建 Kernel 对象
    KernelMul<DT_X> op;

    // 初始化并执行计算
    op.Init(
        x,
        y,
        z,
        tiling_data.length,
        tiling_data.tileNum
    );

    op.Process();
}
