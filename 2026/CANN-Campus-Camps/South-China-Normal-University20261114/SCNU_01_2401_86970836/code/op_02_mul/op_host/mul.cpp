// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    // 固定 8 核：8×2048 数据每核 2048 个元素；不依赖平台查询，避免核数高报导致
    // block 分多波次调度（实测这是最大的性能陷阱）。想试验其他核数只改这一个常量。
    const uint32_t BLOCK_DIM = 8;
    const uint32_t TILE_NUM = 8; // 兜底分块数（与原始模板一致）
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取算子输入数组信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);
        uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();

        // 1. 获取第 0 路输入 x 的存储形状，累乘各维度得到总元素个数（x/y/z 形状一致）
        const gert::StorageShape* x1_shape = context->GetInputShape(0);
        uint32_t data_sz = 1;
        for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
            data_sz *= (uint32_t)x1_shape->GetStorageShape().GetDim(i);
        }

        // 2. tile 选择：优先每核单趟（循环 1 次、同步事件最少——实测最快的形态）；
        //    不满足约束时再从 n=2 起扫描切小块。约束：整除每核元素数、单块 32B 对齐、
        //    单块 ≤ 32KB（单缓冲 3 块缓冲 ≤ 96KB，远小于 910b 每核 256KB UB）
        const uint32_t elemSize = (DT_X == static_cast<uint32_t>(ge::DT_FLOAT16)) ? 2u : 4u;
        const uint32_t elemPerCore = data_sz / BLOCK_DIM;
        uint32_t tileNum = TILE_NUM;                 // 兜底值
        uint32_t tileLength = elemPerCore / tileNum; // 兜底值
        if (elemPerCore > 0 && (elemPerCore * elemSize) % 32u == 0u && elemPerCore * elemSize <= 32768u) {
            // 单趟：一次搬入 x、y，一次 Mul，一次搬出 z
            tileNum = 1;
            tileLength = elemPerCore;
        } else {
            for (uint32_t n = 2; n <= elemPerCore; n++) {
                uint32_t tl = elemPerCore / n;
                if (tl == 0) {
                    break; // 已切到最小块
                }
                if (tl * n != elemPerCore) {
                    continue; // 不能整除
                }
                if ((tl * elemSize) % 32u != 0u) {
                    continue; // 不满足 32B 对齐
                }
                if (tl * elemSize > 32768u) {
                    continue; // 单块超 32KB
                }
                tileNum = n;
                tileLength = tl;
                break;
            }
        }

        // 3. tiling 全部预计算后填充（核内 Init 直接使用，零除法）
        tiling->blockLength = elemPerCore;
        tiling->tileNum = tileNum;
        tiling->tileLength = tileLength;
        // 配置启动核数（固定 8 核）
        context->SetBlockDim(BLOCK_DIM);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // 获取第 0 号输入 x 的 shape（静态/动态维度信息）
        const gert::Shape* x1_shape = context->GetInputShape(0);
        // 获取第 0 号输出 z 的 shape 可写指针，用于设置输出 shape
        gert::Shape* z_shape = context->GetOutputShape(0);
        // Mul 为逐元素算子：输出 shape 与输入 shape 完全一致
        *z_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // 逐元素算子：输出数据类型与第 0 号输入一致（fp32 / fp16 由输入决定）
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Mul : public OpDef {
    public:
        explicit Mul(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("z")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Mul);
}  // namespace ops