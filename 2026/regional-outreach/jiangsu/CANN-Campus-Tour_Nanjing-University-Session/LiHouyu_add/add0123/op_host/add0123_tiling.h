/*
 * @file add0123_tiling.h
 * host 与 kernel 共用的 tiling 数据结构定义
 */
#ifndef ADD0123_TILING_H
#define ADD0123_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 展平后的总元素数 N2 * N1
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Add0123, TilingData)
} // namespace optiling

#endif // ADD0123_TILING_H
