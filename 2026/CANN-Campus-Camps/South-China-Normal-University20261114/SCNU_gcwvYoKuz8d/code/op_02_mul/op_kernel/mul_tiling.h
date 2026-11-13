// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// 修改结构体名为 MulTilingData
struct MulTilingData {
    uint32_t totalLength; // 对齐后的总元素长度 (32B 对齐)
    uint32_t tileNum;     // 单核内将数据进一步切分的块数
};