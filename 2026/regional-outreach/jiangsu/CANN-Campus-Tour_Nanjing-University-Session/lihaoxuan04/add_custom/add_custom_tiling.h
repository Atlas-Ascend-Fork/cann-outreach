/*
 * add_custom 算子 tiling 数据结构定义
 * host 侧与 kernel 侧共用
 */
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(AddCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 张量元素总数 = N2 * N1
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);     // 每个 AI Core 内部循环次数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddCustom, AddCustomTilingData)

} // namespace optiling

#endif // ADD_CUSTOM_TILING_H
