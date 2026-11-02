#pragma once
#include <cstdint>

struct ReluTilingData {
    uint32_t totalLength; // 输入张量总元素数。
    uint32_t tileNum;     // 每核缓冲组数，实际循环次数为tileNum * 2。
};