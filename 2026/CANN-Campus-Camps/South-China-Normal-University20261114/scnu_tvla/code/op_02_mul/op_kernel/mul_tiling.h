// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength; // 输入/输出总元素个数（x、y、z 等长）
    uint32_t tileNum;     // 每个核内的分块数
};
