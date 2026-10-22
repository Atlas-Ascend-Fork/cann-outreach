/*
 * add_custom 算子 ACL 单算子调用示例
 * 用法：./main <N2> <N1>
 * 读取当前目录下的 x.bin、y.bin（float16 二进制），执行 z = x + y，结果写入 z_out.bin
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"

#define ACL_CHECK(expr)                                                        \
    do {                                                                       \
        aclError __ret = (expr);                                               \
        if (__ret != ACL_SUCCESS) {                                            \
            std::cerr << "ACL error at " << #expr << ", code=" << __ret        \
                      << " (line " << __LINE__ << ")" << std::endl;            \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

namespace {

bool ReadBinFile(const std::string &path, std::vector<char> &data)
{
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        std::cerr << "cannot open file: " << path << std::endl;
        return false;
    }
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(static_cast<size_t>(size));
    ifs.read(data.data(), size);
    return !ifs.fail();
}

bool WriteBinFile(const std::string &path, const void *data, size_t size)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "cannot open file: " << path << std::endl;
        return false;
    }
    ofs.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    return ofs.good();
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::cerr << "usage: ./main <N2> <N1>" << std::endl;
        return 1;
    }
    const int64_t n2 = std::atoll(argv[1]);
    const int64_t n1 = std::atoll(argv[2]);
    const size_t elemNum = static_cast<size_t>(n2) * static_cast<size_t>(n1);
    const size_t byteSize = elemNum * sizeof(uint16_t); // float16 = 2 字节

    // 1. 初始化 ACL 资源
    ACL_CHECK(aclInit(nullptr));
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    const int32_t deviceId = 0;
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclrtCreateContext(&context, deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    // 2. 读取输入数据并拷入 device
    std::vector<char> xHost, yHost;
    if (!ReadBinFile("x.bin", xHost) || !ReadBinFile("y.bin", yHost)) {
        return 1;
    }
    if (xHost.size() != byteSize || yHost.size() != byteSize) {
        std::cerr << "input size mismatch: expect " << byteSize << " bytes" << std::endl;
        return 1;
    }

    void *xDev = nullptr;
    void *yDev = nullptr;
    void *zDev = nullptr;
    ACL_CHECK(aclrtMalloc(&xDev, byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&yDev, byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&zDev, byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(xDev, byteSize, xHost.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(yDev, byteSize, yHost.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // 3. 构造输入输出描述与数据缓冲，shape [N2, N1]，float16，ND
    const int64_t dims[2] = {n2, n1};
    aclTensorDesc *inputDesc[2] = {
        aclCreateTensorDesc(ACL_FLOAT16, 2, dims, ACL_FORMAT_ND),
        aclCreateTensorDesc(ACL_FLOAT16, 2, dims, ACL_FORMAT_ND),
    };
    aclTensorDesc *outputDesc[1] = {
        aclCreateTensorDesc(ACL_FLOAT16, 2, dims, ACL_FORMAT_ND),
    };
    aclDataBuffer *inputBufs[2] = {
        aclCreateDataBuffer(xDev, byteSize),
        aclCreateDataBuffer(yDev, byteSize),
    };
    aclDataBuffer *outputBufs[1] = {
        aclCreateDataBuffer(zDev, byteSize),
    };

    // 4. 编译并执行算子（op 名与 host 侧注册一致：AddCustom）
    aclopAttrs *attr = aclopCreateAttrs();
    ACL_CHECK(aclopCompileAndExecute("AddCustom", 2, inputDesc, inputBufs,
                                     1, outputDesc, outputBufs, attr,
                                     ACL_ENGINE_AICORE, ACL_HIGH_PRECISION, stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // 5. 拷回结果并写入 z_out.bin
    std::vector<char> zHost(byteSize);
    ACL_CHECK(aclrtMemcpy(zHost.data(), byteSize, zDev, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    if (!WriteBinFile("z_out.bin", zHost.data(), byteSize)) {
        return 1;
    }
    std::cout << "add_custom done, output -> z_out.bin (" << elemNum << " fp16 elems)" << std::endl;

    // 6. 释放资源
    aclopDestroyAttrs(attr);
    for (auto *d : inputDesc) aclDestroyTensorDesc(d);
    for (auto *d : outputDesc) aclDestroyTensorDesc(d);
    for (auto *b : inputBufs) aclDestroyDataBuffer(b);
    for (auto *b : outputBufs) aclDestroyDataBuffer(b);
    ACL_CHECK(aclrtFree(xDev));
    ACL_CHECK(aclrtFree(yDev));
    ACL_CHECK(aclrtFree(zDev));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtDestroyContext(context));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
    return 0;
}
