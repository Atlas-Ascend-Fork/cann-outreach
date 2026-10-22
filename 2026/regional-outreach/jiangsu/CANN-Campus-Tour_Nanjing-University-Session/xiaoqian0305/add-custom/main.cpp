#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern void add_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* y,
    uint8_t* z, uint8_t* workspace, uint8_t* tiling);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling);
#endif

extern uint8_t* GenerateTiling(uint32_t totalLength, uint32_t tileLength);

int32_t main(int32_t argc, char* argv[])
{
    // input shape [N2, N1], flattened totalLength = N2 * N1
    uint32_t N2 = 8, N1 = 128;
    uint32_t totalLength = N2 * N1;
    uint32_t tileLength = 256;              // each tile: 256 half elements
    uint32_t blockDim = 8;                  // multi-core parallelism
    size_t inputByteSize = totalLength * sizeof(uint16_t);
    size_t outputByteSize = totalLength * sizeof(uint16_t);
    size_t tilingFileSize = sizeof(uint32_t) * 2;

#ifdef ASCENDC_CPU_DEBUG
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    ReadFile("./input/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile("./input/input_y.bin", inputByteSize, y, inputByteSize);

    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingFileSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);

    add_custom(blockDim, nullptr, tiling);
    // kernel runs on CPU debug mode
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_custom, blockDim, x, y, z, workspace, tiling);

    WriteFile("./output/output_z.bin", z, outputByteSize);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)z);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
#else
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *xHost, *yHost, *zHost, *tilingHost, *workspaceHost;
    uint8_t *xDevice, *yDevice, *zDevice, *tilingDevice, *workspaceDevice;

    CHECK_ACL(aclrtMallocHost((void**)(&xHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&yHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&zHost), outputByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&tilingHost), tilingFileSize));
    CHECK_ACL(aclrtMallocHost((void**)(&workspaceHost), 16 * 1024 * 1024));

    CHECK_ACL(aclrtMalloc((void**)&xDevice, inputByteSize, ACL_MEM_MALLOC_HIGHEST_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&yDevice, inputByteSize, ACL_MEM_MALLOC_HIGHEST_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&zDevice, outputByteSize, ACL_MEM_MALLOC_HIGHEST_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&tilingDevice, tilingFileSize, ACL_MEM_MALLOC_HIGHEST_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&workspaceDevice, 16 * 1024 * 1024, ACL_MEM_MALLOC_HIGHEST_FIRST));

    ReadFile("./input/input_x.bin", inputByteSize, xHost, inputByteSize);
    ReadFile("./input/input_y.bin", inputByteSize, yHost, inputByteSize);

    uint32_t* tilingData = GenerateTiling(totalLength, tileLength);
    memcpy(tilingHost, tilingData, tilingFileSize);

    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice, inputByteSize, yHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    add_custom_do(blockDim, nullptr, stream, xDevice, yDevice, zDevice, workspaceDevice, tilingDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_z.bin", zHost, outputByteSize);

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFree(tilingDevice));
    CHECK_ACL(aclrtFree(workspaceDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));
    CHECK_ACL(aclrtFreeHost(zHost));
    CHECK_ACL(aclrtFreeHost(tilingHost));
    CHECK_ACL(aclrtFreeHost(workspaceHost));
#endif
    return 0;
}
