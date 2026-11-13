/*!
 * \file abs.cpp
 * \brief Abs 算子 kernel 入口
 *
 * 按 tilingKey（schMode）分发到 float16 / float32 两个实例：
 *   schMode == 0 -> half
 *   schMode == 1 -> float
 * ★ 入口必须是**非类型**模板参数：官方单测 tests/ut/op_kernel/test_abs.cpp 中
 *   以 ICPU_RUN_KF((abs<1>), ...) 的形式实例化，字面量 1 只能绑定到 uint 形参，
 *   模板自带的 template <typename DT_X> 接不住字面量，会直接编译失败。
 *   schMode 的取值与 op_kernel/abs_tiling_key.h 中的 ABS_TPL_SCH_MODE_0/1 一一对应。
 */

#include "abs.h"

enum class AbsTilingKey : uint32_t
{
    TILING_KEY_ABS_MODE_0 = 0, // float16
    TILING_KEY_ABS_MODE_1 = 1, // float32
};

template <uint32_t schMode>
__global__ __aicore__ void abs(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AbsTilingData);
    GET_TILING_DATA_WITH_STRUCT(AbsTilingData, tilingData, tiling);

    if constexpr (schMode == static_cast<uint32_t>(AbsTilingKey::TILING_KEY_ABS_MODE_0)) {
        NsAbs::Abs<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(AbsTilingKey::TILING_KEY_ABS_MODE_1)) {
        NsAbs::Abs<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
