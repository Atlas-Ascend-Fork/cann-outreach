#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength = 0;   // 总元素个数（x/y/z 一致）
    uint32_t tileNum = 0;       // 切分出的 tile 总数
};
