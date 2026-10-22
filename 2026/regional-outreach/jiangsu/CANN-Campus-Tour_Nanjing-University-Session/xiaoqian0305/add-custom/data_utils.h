#include <string>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#define CHECK_ACL(x) \
    do { \
        aclError __ret = (x); \
        if (__ret != ACL_SUCCESS) { \
            std::printf("%s:%d aclError %d\n", __FILE__, __LINE__, __ret); \
        } \
    } while (0)
#else
#define CHECK_ACL(x) (x)
#endif

inline bool ReadFile(const std::string& filePath, size_t size, void* buffer, size_t expectedSize)
{
    if (buffer == nullptr || size != expectedSize) {
        std::cerr << "[ReadFile] invalid buffer or size" << std::endl;
        return false;
    }
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ReadFile] cannot open file: " << filePath << std::endl;
        return false;
    }
    file.read(reinterpret_cast<char*>(buffer), size);
    file.close();
    return true;
}

inline bool WriteFile(const std::string& filePath, const void* buffer, size_t size)
{
    if (buffer == nullptr) {
        std::cerr << "[WriteFile] invalid buffer" << std::endl;
        return false;
    }
    std::ofstream file(filePath, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "[WriteFile] cannot open file: " << filePath << std::endl;
        return false;
    }
    file.write(reinterpret_cast<const char*>(buffer), size);
    file.close();
    return true;
}

inline uint8_t* GenerateTiling(uint32_t totalLength, uint32_t tileLength)
{
    static uint32_t tiling[2] = {0, 0};
    tiling[0] = totalLength;
    tiling[1] = tileLength;
    return reinterpret_cast<uint8_t*>(tiling);
}
