// Tiling结构体定义的头文件
//
// Host 侧 TilingFunc 计算好的切分信息通过该结构体下发给 Kernel 侧的 mul 核函数。
// 注意：Host 侧与 Kernel 侧共用本头文件，字段顺序/类型必须完全一致，
//       且必须保持为 POD（可平凡拷贝），否则 GET_TILING_DATA_WITH_STRUCT 解析会出错。
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength;    // 输入/输出张量的总元素个数，例如 (8, 2048) -> 16384
    uint32_t coreNum;        // 实际参与计算的核（Vector Core）个数
    uint32_t perCoreLength;  // 每个核负责的元素个数（按 32B 对齐）
    uint32_t tileLength;     // 单次 DMA/Vector 计算的元素个数（按 32B 对齐）
    uint32_t tileNum;        // 每个核的 tile 个数（按 perCoreLength 计算，仅供调试参考）
    uint32_t alignElements;  // 32B 对应的元素个数：float32 为 8，float16 为 16
};
