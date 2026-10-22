/**
 * @file add_custom_host.cpp
 * @brief Host 侧：调用核函数的封装（aclrt 启动 Kernel）
 */
#include "acl/acl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "add_custom_tiling.h"

#define ACL_CHECK(ret)                                                          \
    do {                                                                        \
        if ((ret) != ACL_SUCCESS) {                                             \
            printf("acl call failed, ret = %d, file = %s, line = %d\n",         \
                   (int)(ret), __FILE__, __LINE__);                             \
            return ret;                                                         \
        }                                                                       \
    } while (0)

/**
 * 调用 add_custom 核函数：z = x + y
 * @param x/y/z device 上的输入/输出指针（half，即 float16）
 * @param totalLength 元素总数 = N2 * N1
 * @param blockDim 使用的 AI Core 个数，要求 totalLength 能被 blockDim * tileLength 整除
 * @param tileLength 每次 Core 内迭代的元素数
 */
extern "C" aclrtError add_custom_do(uint32_t blockDim, void *l2ctrl, void *stream,
                                    const void *x, const void *y, void *z,
                                    uint32_t totalLength, uint32_t tileLength)
{
    AddCustomTilingData tiling;
    tiling.totalLength = totalLength;
    tiling.tileLength = tileLength;

    // Tiling 数据拷贝到 device
    void *tilingDevice = nullptr;
    ACL_CHECK(aclrtMalloc(&tilingDevice, sizeof(AddCustomTilingData), ACL_MEM_MALLOC_HIGHEST_FIRST));
    ACL_CHECK(aclrtMemcpy(tilingDevice, sizeof(AddCustomTilingData),
                          &tiling, sizeof(AddCustomTilingData), ACL_MEMCPY_HOST_TO_DEVICE));

    // 启动核函数（__aicore__ 入口 add_custom）
    add_custom(blockDim, l2ctrl, stream,
               const_cast<void *>(x), const_cast<void *>(y), z,
               tilingDevice, tiling.totalLength, tiling.tileLength);

    ACL_CHECK(aclrtFree(tilingDevice));
    return ACL_SUCCESS;
}
