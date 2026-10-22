/*!
 * \file add_custom_tiling_data.h
 * \brief add_custom 算子 tiling 数据结构（host 侧填充，kernel 侧读取）
 */

#ifndef _ADD_CUSTOM_TILING_DATA_H_
#define _ADD_CUSTOM_TILING_DATA_H_

#include <cstdint>

struct AddCustomTilingData {
    uint32_t totalLength; // 总元素数（x/y/z 形状一致，逐维相乘）
    uint32_t tileLength;  // 单核内每块元素数（固定 tile 大小）
};

#endif // _ADD_CUSTOM_TILING_DATA_H_
