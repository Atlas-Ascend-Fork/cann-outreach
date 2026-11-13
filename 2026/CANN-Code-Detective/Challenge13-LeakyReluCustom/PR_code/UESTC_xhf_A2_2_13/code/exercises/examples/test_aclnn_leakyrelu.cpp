#include "sample_support.h"
#include "aclnn_leakyrelu.h"

int main(int argc, char** argv) {
    try {
        std::string root = argc > 1 ? argv[1] : "../samples/build";
        int device = argc > 2 ? std::stoi(argv[2]) : 0;
        check(aclInit(nullptr));
        check(aclrtSetDevice(device));
        aclrtStream stream;
        check(aclrtCreateStream(&stream));
        struct SampleCase {
            int64_t rows;
            int64_t columns;
            double slope;
            const char* inputFile;
            const char* outputFile;
        };
        const SampleCase cases[] = {
            {8, 2048, 0.01, "x.bin", "y.bin"},
            {1, 1031, 0.125, "x_tail.bin", "y_tail.bin"},
        };
        for (const auto& test : cases) {
            const std::vector<int64_t> shape{test.rows, test.columns};
            const size_t bytes = test.rows * test.columns * sizeof(float);
            Tensor input(shape, ACL_FLOAT, bytes, test.inputFile, true, false, root);
            Tensor output(shape, ACL_FLOAT, bytes, test.outputFile, false, false, root);
            uint64_t size = 0;
            aclOpExecutor* executor = nullptr;
            check(aclnnLeakyreluGetWorkspaceSize(input.tensor, test.slope, output.tensor, &size, &executor));
            void* workspace = nullptr;
            if (size)
                check(aclrtMalloc(&workspace, size, ACL_MEM_MALLOC_HUGE_FIRST));
            check(aclnnLeakyrelu(workspace, size, executor, stream));
            check(aclrtSynchronizeStream(stream));
            output.save(root);
            if (size)
                check(aclrtFree(workspace));
        }
        check(aclrtDestroyStream(stream));
        check(aclrtResetDevice(device));
        check(aclFinalize());
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 2;
    }
}
