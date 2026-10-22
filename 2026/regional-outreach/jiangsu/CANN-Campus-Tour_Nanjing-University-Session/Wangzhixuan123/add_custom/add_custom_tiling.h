/*
 * add_custom host 侧 tiling：按 N2 x N1 与核数计算切分参数
 * 与 add_custom.cpp 中 AddCustomTilingData 保持一致
 */
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H

#include <cstdint>
#include <algorithm>

struct AddCustomTilingData {
    uint32_t totalLength;   // N2 * N1，总元素个数
    uint32_t tileLength;    // 每次搬运/计算的长度（32 对齐）
    uint32_t tilingKey;     // 切分策略编号
};

constexpr uint32_t ADD_CUSTOM_TILE_ALIGN = 32;   // 与核函数侧一致
constexpr uint32_t ADD_CUSTOM_TILE_MAX = 8192;   // 单 tile 上限，需小于统一缓冲区大小（half 个数）

/*
 * 计算 tiling：
 *   totalLength = N2 * N1
 *   tileLength  = min(totalLength / coreNum, ADD_CUSTOM_TILE_MAX)，按 32 对齐（向下）
 *   tilingKey   = 1
 * 返回 false 表示输入不合法
 */
inline bool AddCustomTiling(uint32_t n2, uint32_t n1, uint32_t coreNum,
                            AddCustomTilingData& tiling)
{
    if (n2 == 0 || n1 == 0 || coreNum == 0) {
        return false;
    }
    uint64_t total = static_cast<uint64_t>(n2) * n1;
    if (total > UINT32_MAX) {
        return false;   // 单算子不支持超过 4G 元素
    }

    uint32_t perCore = static_cast<uint32_t>(total / coreNum);
    uint32_t tile = std::min(perCore, ADD_CUSTOM_TILE_MAX);
    tile = (tile / ADD_CUSTOM_TILE_ALIGN) * ADD_CUSTOM_TILE_ALIGN;  // 向下对齐
    if (tile == 0) {
        tile = ADD_CUSTOM_TILE_ALIGN;
    }

    tiling.totalLength = static_cast<uint32_t>(total);
    tiling.tileLength = tile;
    tiling.tilingKey = 1;
    return true;
}

#endif  // ADD_CUSTOM_TILING_H
