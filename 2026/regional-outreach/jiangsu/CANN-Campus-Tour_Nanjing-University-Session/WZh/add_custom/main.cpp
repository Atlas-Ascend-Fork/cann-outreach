/*
 * @file main.cpp
 * @brief host 侧调用程序：生成数据 -> 拷贝到 device -> 启动 add_custom kernel -> 校验结果
 */
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include "acl/acl.h"
#include "aclrtlaunch_add_custom.h"

// 与 kernel 侧保持一致的 tiling 结构
struct AddCustomTilingData {
    uint32_t totalLength;
    uint32_t tileLength;
    uint32_t blockLength;
    uint32_t blockIdx;
};

static bool ReadFile(const std::string &filePath, uint32_t &fileSize, void *buffer) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Failed to open file: " << filePath << std::endl;
        return false;
    }
    file.seekg(0, std::ios::end);
    fileSize = static_cast<uint32_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char *>(buffer), fileSize);
    file.close();
    return true;
}

static bool WriteFile(const std::string &filePath, uint32_t fileSize, const void *buffer) {
    std::ofstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Failed to open file: " << filePath << std::endl;
        return false;
    }
    file.write(reinterpret_cast<const char *>(buffer), fileSize);
    file.close();
    return true;
}

static bool GenerateTiling(uint32_t totalLength, AddCustomTilingData &tiling) {
    tiling.totalLength = totalLength;
    tiling.blockIdx = 0;
    tiling.blockLength = totalLength;

    constexpr uint32_t ALIGN_NUM = 32;
    uint32_t tileLength = ALIGN_NUM;
    while (tileLength * 2 <= totalLength) {
        tileLength *= 2;
    }
    tiling.tileLength = tileLength < totalLength ? tileLength : totalLength;
    return true;
}

int32_t main(int32_t argc, char *argv[]) {
    // 形状：2 维 [N2, N1]
    const uint32_t N2 = 4, N1 = 2048;
    const uint32_t totalLength = N2 * N1;
    const uint32_t dataSize = totalLength * sizeof(uint16_t); // float16
    const uint32_t tilingSize = sizeof(AddCustomTilingData);
    const uint32_t blockDim = 1;

    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::cout << "acl init failed" << std::endl;
        return -1;
    }
    aclrtContext context;
    if (aclrtCreateContext(&context, 0) != ACL_SUCCESS) {
        std::cout << "create context failed" << std::endl;
        aclFinalize();
        return -1;
    }
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // host 侧数据
    std::vector<uint16_t> xHost(totalLength), yHost(totalLength), zHost(totalLength);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (uint32_t i = 0; i < totalLength; i++) {
        half x = static_cast<half>(dist(gen));
        half y = static_cast<half>(dist(gen));
        xHost[i] = *reinterpret_cast<uint16_t *>(&x);
        yHost[i] = *reinterpret_cast<uint16_t *>(&y);
    }

    // device 侧内存
    void *xDevice = nullptr, *yDevice = nullptr, *zDevice = nullptr, *tilingDevice = nullptr;
    aclrtMalloc(&xDevice, dataSize, ACL_MEM_MALLOC_HIGHEST_FIRST);
    aclrtMalloc(&yDevice, dataSize, ACL_MEM_MALLOC_HIGHEST_FIRST);
    aclrtMalloc(&zDevice, dataSize, ACL_MEM_MALLOC_HIGHEST_FIRST);
    aclrtMalloc(&tilingDevice, tilingSize, ACL_MEM_MALLOC_HIGHEST_FIRST);

    aclrtMemcpy(xDevice, dataSize, xHost.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(yDevice, dataSize, yHost.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);

    AddCustomTilingData tiling;
    GenerateTiling(totalLength, tiling);
    aclrtMemcpy(tilingDevice, tilingSize, &tiling, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // 启动 kernel（由 CANN 编译产物 aclrtlaunch_add_custom.h 提供）
    ACLRT_LAUNCH_KERNEL(add_custom)(blockDim, stream, xDevice, yDevice, zDevice, tilingDevice);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(zHost.data(), dataSize, zDevice, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // 结果校验
    bool pass = true;
    for (uint32_t i = 0; i < totalLength; i++) {
        half x = *reinterpret_cast<half *>(&xHost[i]);
        half y = *reinterpret_cast<half *>(&yHost[i]);
        half z = *reinterpret_cast<half *>(&zHost[i]);
        float golden = static_cast<float>(x) + static_cast<float>(y);
        if (std::abs(static_cast<float>(z) - golden) > 1e-3f) {
            std::cout << "Mismatch at " << i << ": got " << static_cast<float>(z)
                      << ", expect " << golden << std::endl;
            pass = false;
            break;
        }
    }
    std::cout << (pass ? "add_custom PASS" : "add_custom FAILED") << std::endl;

    WriteFile("output/z.bin", dataSize, zHost.data());

    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(zDevice);
    aclrtFree(tilingDevice);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclFinalize();
    return pass ? 0 : -1;
}
