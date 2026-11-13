// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr uint32_t BLOCK_BYTES = 32; // 一次DMA搬运的最小对齐字节数

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length, uint32_t block_factor) {
        total_length_ = length;
        block_factor_ = block_factor;
        x_gm_.SetGlobalBuffer((__gm__ DT_X *)x);
        y_gm_.SetGlobalBuffer((__gm__ DT_X *)y);
        z_gm_.SetGlobalBuffer((__gm__ DT_X *)z);
        pipe_.InitBuffer(in_queue_x_, 1, block_factor_ * sizeof(DT_X));
        pipe_.InitBuffer(in_queue_y_, 1, block_factor_ * sizeof(DT_X));
        pipe_.InitBuffer(out_queue_z_, 1, block_factor_ * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        uint32_t block_id = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t start = block_id * block_factor_;
        if (start >= total_length_) {
            return;
        }
        uint32_t core_len = total_length_ - start;
        core_len = core_len < block_factor_ ? core_len : block_factor_;
        CopyIn(start, core_len);
        Compute(core_len);
        CopyOut(start, core_len);
    }

private:
    __aicore__ inline void CopyIn(uint32_t start, uint32_t core_len) {
        AscendC::LocalTensor<DT_X> x_local = in_queue_x_.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> y_local = in_queue_y_.AllocTensor<DT_X>();
        // 对齐部分走常规DataCopy；尾块按实际字节数搬运
        constexpr uint32_t align_elems = BLOCK_BYTES / sizeof(DT_X);
        uint32_t align_len = (core_len / align_elems) * align_elems;
        if (align_len > 0) {
            AscendC::DataCopy(x_local, x_gm_[start], align_len);
            AscendC::DataCopy(y_local, y_gm_[start], align_len);
        }
        if (core_len > align_len) {
            AscendC::DataCopyExtParams copy_params{1, static_cast<uint32_t>((core_len - align_len) * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> pad_params{false, 0, 0, static_cast<DT_X>(0)};
            AscendC::DataCopyPad(x_local[align_len], x_gm_[start + align_len], copy_params, pad_params);
            AscendC::DataCopyPad(y_local[align_len], y_gm_[start + align_len], copy_params, pad_params);
        }
        in_queue_x_.EnQue(x_local);
        in_queue_y_.EnQue(y_local);
    }
    __aicore__ inline void Compute(uint32_t core_len) {
        AscendC::LocalTensor<DT_X> x_local = in_queue_x_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> y_local = in_queue_y_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> z_local = out_queue_z_.AllocTensor<DT_X>();
        AscendC::Mul(z_local, x_local, y_local, core_len);
        in_queue_x_.FreeTensor(x_local);
        in_queue_y_.FreeTensor(y_local);
        out_queue_z_.EnQue(z_local);
    }
    __aicore__ inline void CopyOut(uint32_t start, uint32_t core_len) {
        AscendC::LocalTensor<DT_X> z_local = out_queue_z_.DeQue<DT_X>();
        constexpr uint32_t align_elems = BLOCK_BYTES / sizeof(DT_X);
        uint32_t align_len = (core_len / align_elems) * align_elems;
        if (align_len > 0) {
            AscendC::DataCopy(z_gm_[start], z_local, align_len);
        }
        if (core_len > align_len) {
            AscendC::DataCopyExtParams copy_params{1, static_cast<uint32_t>((core_len - align_len) * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPad(z_gm_[start + align_len], z_local[align_len], copy_params);
        }
        out_queue_z_.FreeTensor(z_local);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> in_queue_x_, in_queue_y_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> out_queue_z_;
    AscendC::GlobalTensor<DT_X> x_gm_, y_gm_, z_gm_;
    uint32_t total_length_;
    uint32_t block_factor_;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length, tiling_data.block_factor);
    op.Process();
}
