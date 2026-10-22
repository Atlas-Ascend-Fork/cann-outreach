#include "acl/acl.h"
#include <iostream>
#include <cstdint>
#include <vector>
#include <fstream>
#include <cstring>

extern void add_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *y, uint8_t *z,
                          uint32_t totalLength, uint32_t tileLength);

// shape [N2, N1] 2D, float16, ND
constexpr uint32_t N2 = 8;
constexpr uint32_t N1 = 128;
constexpr uint32_t TOTAL_LENGTH = N2 * N1;              // 元素个数
constexpr uint32_t TILE_LENGTH = 128;                    // 每次计算的分片长度
constexpr uint32_t BLOCK_DIM = 8;                        // AI Core 数量

#define ACL_CHECK(ret) do { \
    if ((ret) != ACL_SUCCESS) { \
        std::cerr << "acl api error at line " << __LINE__ << std::endl; \
        return 1; \
    } \
} while (0)

static bool ReadFile(const std::string &filePath, uint32_t fileSize, void *buffer)
{
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "open file failed: " << filePath << std::endl;
        return false;
    }
    file.read(reinterpret_cast<char *>(buffer), fileSize);
    file.close();
    return true;
}

int32_t main(int32_t argc, char *argv[])
{
    uint32_t fileSize = TOTAL_LENGTH * sizeof(uint16_t); // half = 2 bytes

    // 1. AscendCL 初始化
    ACL_CHECK(aclInit(nullptr));
    int32_t deviceId = 0;
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    // 2. 申请 Host 内存并读入数据
    uint16_t *xHost = nullptr, *yHost = nullptr, *zHost = nullptr;
    ACL_CHECK(aclrtMallocHost((void **)&xHost, fileSize));
    ACL_CHECK(aclrtMallocHost((void **)&yHost, fileSize));
    ACL_CHECK(aclrtMallocHost((void **)&zHost, fileSize));
    if (!ReadFile("input/x_gm.bin", fileSize, xHost) || !ReadFile("input/y_gm.bin", fileSize, yHost)) {
        return 1;
    }

    // 3. 申请 Device 内存并拷入
    uint8_t *xDevice = nullptr, *yDevice = nullptr, *zDevice = nullptr;
    ACL_CHECK(aclrtMalloc((void **)&xDevice, fileSize, ACL_MEM_MALLOC_HIGHEST_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&yDevice, fileSize, ACL_MEM_MALLOC_HIGHEST_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&zDevice, fileSize, ACL_MEM_MALLOC_HIGHEST_FIRST));
    ACL_CHECK(aclrtMemcpy(xDevice, fileSize, xHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(yDevice, fileSize, yHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // 4. 启动 kernel
    add_custom_do(BLOCK_DIM, nullptr, stream, xDevice, yDevice, zDevice, TOTAL_LENGTH, TILE_LENGTH);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // 5. 结果拷回并写出
    ACL_CHECK(aclrtMemcpy(zHost, fileSize, zDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    std::ofstream out("output/z_gm.bin", std::ios::binary);
    out.write(reinterpret_cast<char *>(zHost), fileSize);
    out.close();

    // 6. 释放资源
    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(zDevice);
    aclrtFreeHost(xHost);
    aclrtFreeHost(yHost);
    aclrtFreeHost(zHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    std::cout << "add_custom run success, result written to output/z_gm.bin" << std::endl;
    return 0;
}
