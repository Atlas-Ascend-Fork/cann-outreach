/**
 * @file add_custom_tiling.h
 * @brief add_custom 算子的 Tiling 参数定义（Host / Device 共享）
 */
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H

struct AddCustomTilingData {
    uint32_t totalLength; // 张量元素总数 = N2 * N1
    uint32_t tileLength;  // 每次核内迭代处理的元素数
};

#endif // ADD_CUSTOM_TILING_H
