#include "inplace_rsqrt_api.h"
#include "shape_utils.h"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {
void CheckAcl(aclError code, const char *operation)
{
    if (code == ACL_SUCCESS) return;
    const char *detail = aclGetRecentErrMsg();
    throw std::runtime_error(std::string(operation) + " failed, ACL=" +
        std::to_string(code) + (detail ? std::string("; ") + detail : ""));
}

class AclResources {
public:
    bool initialized = false;
    bool deviceSet = false;
    int32_t device = 0;
    aclrtStream stream = nullptr;
    void *self = nullptr;
    AclResources() = default;
    AclResources(const AclResources &) = delete;
    AclResources &operator=(const AclResources &) = delete;
    aclError Release() noexcept
    {
        aclError first = ACL_SUCCESS;
        const auto remember = [&first](aclError code) {
            if (first == ACL_SUCCESS && code != ACL_SUCCESS) first = code;
        };
        if (stream) remember(aclrtSynchronizeStream(stream));
        if (self) { remember(aclrtFree(self)); self = nullptr; }
        if (stream) { remember(aclrtDestroyStream(stream)); stream = nullptr; }
        if (deviceSet) { remember(aclrtResetDevice(device)); deviceSet = false; }
        if (initialized) { remember(aclFinalize()); initialized = false; }
        return first;
    }
    ~AclResources()
    {
        const aclError code = Release();
        if (code != ACL_SUCCESS) std::fprintf(stderr, "Cleanup failed: ACL=%d\n", code);
    }
};

uint64_t ParseUnsigned(const char *text)
{
    if (!text || *text == '\0') throw std::runtime_error("Empty integer argument");
    for (const char *p = text; *p; ++p) {
        if (*p < '0' || *p > '9') throw std::runtime_error("Expected a nonnegative decimal integer");
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || *end != '\0' || value > std::numeric_limits<uint64_t>::max()) {
        throw std::runtime_error("Integer argument overflows uint64");
    }
    return static_cast<uint64_t>(value);
}

std::vector<uint16_t> ReadInput(const char *path, size_t bytes)
{
    if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Input is too large for stream I/O");
    }
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open input file");
    if (file.tellg() != static_cast<std::streamoff>(bytes)) {
        throw std::runtime_error("Input byte length does not match the 4D FP16 shape");
    }
    file.seekg(0);
    std::vector<uint16_t> values(bytes / sizeof(uint16_t));
    if (bytes && !file.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(bytes))) {
        throw std::runtime_error("Input read failed");
    }
    // Host-side validation only. Device arithmetic stays vectorized.
    for (uint16_t bits : values) {
        if ((bits & 0x8000u) || (bits & 0x7fffu) == 0 || (bits & 0x7c00u) == 0x7c00u) {
            throw std::runtime_error("Native FP16 Rsqrt requires finite, strictly positive input");
        }
    }
    return values;
}

void CheckSeparateFiles(const char *input, const char *output)
{
    struct stat inputInfo{}, outputInfo{};
    if (stat(input, &inputInfo) != 0) throw std::runtime_error("Cannot stat input file");
    if (stat(output, &outputInfo) == 0) {
        // stat resolves relative paths and symbolic links; device/inode also catches hard links.
        if (inputInfo.st_dev == outputInfo.st_dev && inputInfo.st_ino == outputInfo.st_ino) {
            throw std::runtime_error("Use separate input/output files (including path/link aliases)");
        }
    } else if (errno != ENOENT) {
        throw std::runtime_error("Cannot stat output file");
    }
}

void WriteOutput(const char *path, const std::vector<uint16_t> &values)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("Cannot open output file (create its parent directory first)");
    if (!values.empty()) {
        file.write(reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(uint16_t)));
    }
    file.close();
    if (!file) throw std::runtime_error("Output write failed");
}
} // namespace

int main(int argc, char **argv)
{
    if (argc != 7 && argc != 8) {
        std::fprintf(stderr, "Usage: %s N4 N3 N2 N1 input.bin output.bin [device_id]\n", argv[0]);
        return 2;
    }
    try {
        const uint64_t shape[4] = {ParseUnsigned(argv[1]), ParseUnsigned(argv[2]),
                                  ParseUnsigned(argv[3]), ParseUnsigned(argv[4])};
        uint64_t elements = 0;
        size_t bytes = 0;
        if (!inplace_rsqrt_host::ShapeSize(shape, elements, bytes)) {
            throw std::runtime_error("4D shape product/FP16 byte length overflows size_t");
        }
        const uint64_t requestedDevice = argc == 8 ? ParseUnsigned(argv[7]) : 0;
        if (requestedDevice > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("Device ID exceeds int32");
        }
        CheckSeparateFiles(argv[5], argv[6]);
        auto values = ReadInput(argv[5], bytes);
        AclResources resources;
        if (elements) {
            CheckAcl(aclInit(nullptr), "aclInit");
            resources.initialized = true;
            resources.device = static_cast<int32_t>(requestedDevice);
            CheckAcl(aclrtSetDevice(resources.device), "aclrtSetDevice");
            resources.deviceSet = true;
            CheckAcl(aclrtCreateStream(&resources.stream), "aclrtCreateStream");
            CheckAcl(aclrtMalloc(&resources.self, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
            CheckAcl(aclrtMemcpy(resources.self, bytes, values.data(), bytes,
                ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
        }
        CheckAcl(InplaceRsqrtFp16(resources.self, shape[0], shape[1], shape[2], shape[3],
            resources.stream), "InplaceRsqrtFp16 + stream synchronization");
        if (elements) {
            CheckAcl(aclrtMemcpy(values.data(), bytes, resources.self, bytes,
                ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
        }
        CheckAcl(resources.Release(), "ACL cleanup");
        CheckSeparateFiles(argv[5], argv[6]);
        WriteOutput(argv[6], values);
        std::printf("Completed in-place FP16 Rsqrt: %llu elements.\n",
            static_cast<unsigned long long>(elements));
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
