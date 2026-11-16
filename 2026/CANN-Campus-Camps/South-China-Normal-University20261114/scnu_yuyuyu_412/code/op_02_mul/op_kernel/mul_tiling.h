#ifndef MUL_TILING_H
#define MUL_TILING_H
#include <cstdint>

class MulTilingData {
public:
    uint32_t totalLength;  // 总元素数
    uint32_t tileNum;      // 单核内分块数
};
#endif
