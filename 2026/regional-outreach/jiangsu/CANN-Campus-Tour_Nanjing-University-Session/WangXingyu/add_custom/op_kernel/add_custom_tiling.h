/**
 * add_custom_tiling.h
 *
 * add_custom 算子的 TilingData 结构体定义。
 * 该结构体同时被 Host 侧（TilingFunc 写入）与核函数侧（GET_TILING_DATA 读取）使用，
 * 因此字段必须为编译期大小确定、可直接复制的简单类型，且禁止成员函数、指针与引用字段。
 */
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H

#include <cstdint>

struct AddCustomTilingData {
    uint32_t totalLength;  // 待处理数据的总元素个数（2 维 shape [N2, N1] => N2 * N1）
    uint32_t tileNum;      // 每个核上数据切分出的块数（用于单核内流水分块）
};

#endif  // ADD_CUSTOM_TILING_H
