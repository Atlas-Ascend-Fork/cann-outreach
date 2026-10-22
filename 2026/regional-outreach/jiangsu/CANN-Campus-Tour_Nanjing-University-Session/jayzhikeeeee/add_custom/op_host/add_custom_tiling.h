/**
 * add_custom tiling 结构定义
 * 与核函数侧 get_tiling_ptr 解析的布局保持一致：
 *   totalLength (uint32_t) + tileLength (uint32_t)
 */
#pragma once
#include "register/tilingdata_base.h"

namespace ops {
BEGIN_TILING_DATA_DEF(AddCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 2 维张量展平后的总元素个数 N2*N1
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);  // 单次循环处理长度（half 元素数）
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddCustom, AddCustomTilingData)
} // namespace ops
