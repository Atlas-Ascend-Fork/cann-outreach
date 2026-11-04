#ifndef SUB_CUSTOM_TEMPLATE_TILING_H
#define SUB_CUSTOM_TEMPLATE_TILING_H
#include <cstdint>

struct SubCustomTemplateTilingData {
    uint32_t totalLength;  // 总元素个数
    uint32_t tileNum;      // 数据切分的总块数
};

#endif
