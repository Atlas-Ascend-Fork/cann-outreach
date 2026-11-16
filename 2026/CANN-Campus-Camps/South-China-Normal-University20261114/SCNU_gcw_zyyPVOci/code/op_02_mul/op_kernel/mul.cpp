#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

// 每个队列开 2 块 buffer，做 double buffer 流水
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelMul {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                            const optiling::MulTilingData& tiling) {
        // 本核编号 / 总核数（host 侧 SetBlockDim 设好）
        block_idx_ = GetBlockIdx();
        block_num_ = GetBlockNum();

        total_length_ = tiling.length;
        tile_length_  = tiling.tile_length;
        core_length_  = total_length_ / block_num_;   // 每核元素数
        tile_num_     = core_length_ / tile_length_;   // 每核内分几块

        // 本核在 GM 上的起始偏移：block_idx * core_length
        uint32_t offset = block_idx_ * core_length_;
        x_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, core_length_);
        y_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, core_length_);
        z_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(z) + offset, core_length_);

        // 在 UB 上为 x/y/z 各分配 BUFFER_NUM 块队列内存
        pipe_.InitBuffer(in_queue_x_, BUFFER_NUM, tile_length_ * sizeof(T));
        pipe_.InitBuffer(in_queue_y_, BUFFER_NUM, tile_length_ * sizeof(T));
        pipe_.InitBuffer(out_queue_z_, BUFFER_NUM, tile_length_ * sizeof(T));
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < tile_num_; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    // 把 x、y 从 GM 搬进 UB
    __aicore__ inline void CopyIn(uint32_t i) {
        LocalTensor<T> x_local = in_queue_x_.AllocTensor<T>();
        LocalTensor<T> y_local = in_queue_y_.AllocTensor<T>();
        DataCopy(x_local, x_gm_[i * tile_length_], tile_length_);
        DataCopy(y_local, y_gm_[i * tile_length_], tile_length_);
        in_queue_x_.EnQue(x_local);
        in_queue_y_.EnQue(y_local);
    }

    // 在 UB 上做 z = x * y
    __aicore__ inline void Compute(uint32_t /*i*/) {
        LocalTensor<T> x_local = in_queue_x_.DeQue<T>();
        LocalTensor<T> y_local = in_queue_y_.DeQue<T>();
        LocalTensor<T> z_local = out_queue_z_.AllocTensor<T>();

        Mul(z_local, x_local, y_local, tile_length_);

        out_queue_z_.EnQue<T>(z_local);
        in_queue_x_.FreeTensor(x_local);
        in_queue_y_.FreeTensor(y_local);
    }

    // 把 z 从 UB 搬回 GM
    __aicore__ inline void CopyOut(uint32_t i) {
        LocalTensor<T> z_local = out_queue_z_.DeQue<T>();
        DataCopy(z_gm_[i * tile_length_], z_local, tile_length_);
        out_queue_z_.FreeTensor(z_local);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN,  BUFFER_NUM> in_queue_x_, in_queue_y_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> out_queue_z_;
    GlobalTensor<T> x_gm_, y_gm_, z_gm_;

    uint32_t block_idx_{0};
    uint32_t block_num_{0};
    uint32_t total_length_{0};
    uint32_t core_length_{0};
    uint32_t tile_length_{0};
    uint32_t tile_num_{0};
};

// ===== Kernel 入口 =====
template <typename D_T_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                               GM_ADDR workspace, GM_ADDR tiling) {
    // 注册本算子使用的 TilingData 结构体，GET_TILING_DATA 才知道怎么解析
    REGISTER_TILING_DEFAULT(optiling::MulTilingData);
    GET_TILING_DATA(tiling_data, tiling);

    KernelMul<D_T_X> op;
    op.Init(x, y, z, tiling_data);
    op.Process();
}