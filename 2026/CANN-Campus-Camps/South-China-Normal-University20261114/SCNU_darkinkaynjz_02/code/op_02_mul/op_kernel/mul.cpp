// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

namespace {
constexpr uint32_t Tile = 128;

template <class T>
class KernelMul {
public:
    __aicore__ inline void Execute(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t block = AscendC::GetBlockIdx();
        const uint32_t begin = (length / Tile) * block / cores;
        const uint32_t end = (length / Tile) * (block + 1) / cores;
        AscendC::GlobalTensor<T> gx, gy, gz;
        gx.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + begin * Tile, (end - begin) * Tile);
        gy.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + begin * Tile, (end - begin) * Tile);
        gz.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(z) + begin * Tile, (end - begin) * Tile);
        pipe.InitBuffer(qx, 2, Tile * sizeof(T));
        pipe.InitBuffer(qy, 2, Tile * sizeof(T));
        pipe.InitBuffer(qz, 2, Tile * sizeof(T));
        for (uint32_t i = 0; i < end - begin; ++i) {
            auto a = qx.AllocTensor<T>();
            auto b = qy.AllocTensor<T>();
            AscendC::DataCopy(a, gx[i * Tile], Tile);
            AscendC::DataCopy(b, gy[i * Tile], Tile);
            qx.EnQue(a);
            qy.EnQue(b);
            a = qx.DeQue<T>();
            b = qy.DeQue<T>();
            auto c = qz.AllocTensor<T>();
            AscendC::Mul(c, a, b, Tile);
            qz.EnQue(c);
            qx.FreeTensor(a);
            qy.FreeTensor(b);
            c = qz.DeQue<T>();
            AscendC::DataCopy(gz[i * Tile], c, Tile);
            qz.FreeTensor(c);
        }
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> qx, qy;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> qz;
};
}

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Execute(x, y, z, tiling_data.length);
}
