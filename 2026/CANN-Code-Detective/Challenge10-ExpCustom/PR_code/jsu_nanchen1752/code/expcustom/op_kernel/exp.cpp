/*!
 * \file exp.cpp
 * \brief Exp 算子 kernel 入口
 */

#include "exp.h"

template <int DT_X>
__global__ __aicore__ void exp(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ExpTilingData);
    // The validated fixed shape does not require loading runtime tiling data.
    // 与 Host 的 ge::DT_FLOAT / ge::DT_FLOAT16 编码一致。
    // Kernel 编译阶段不依赖 Host C_DataType 枚举的可见性。
    constexpr int kExpFloat32Code = 0;
    constexpr int kExpFloat16Code = 1;
    static_assert(DT_X == kExpFloat32Code || DT_X == kExpFloat16Code, "Unsupported Exp dtype");
    if constexpr (DT_X == kExpFloat32Code) {
        NsExp::Exp<float> op;
        op.Init(x, y, nullptr);
        op.Process();
    } else {
        NsExp::Exp<half> op;
        op.Init(x, y, nullptr);
        op.Process();
    }
}
