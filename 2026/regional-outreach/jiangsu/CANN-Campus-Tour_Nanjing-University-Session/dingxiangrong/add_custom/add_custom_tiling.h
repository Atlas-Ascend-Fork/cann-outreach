/**
 * add_custom 算子 tiling 结构与核函数侧宏
 */
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H

// tiling 数据结构：host 侧填充，核函数侧解析
struct AddCustomTilingData {
    uint32_t totalLength; // 总元素个数 = N2 * N1
    uint32_t tileLength;  // 每次处理的数据长度
};

#endif // ADD_CUSTOM_TILING_H
