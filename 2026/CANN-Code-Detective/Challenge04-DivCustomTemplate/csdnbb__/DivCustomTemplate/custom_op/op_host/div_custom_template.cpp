
#include "../op_kernel/div_custom_template_tiling.h"
#include "register/op_def_registry.h"
#include <limits>


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  constexpr uint32_t blockDim = 8;
  constexpr uint32_t tileNum = 8;
  constexpr uint32_t bufferNum = 2;
  DivCustomTemplateTilingData *tiling = context->GetTilingData<DivCustomTemplateTilingData>();
  const gert::StorageShape* x_shape = context->GetInputShape(0);
  const gert::StorageShape* y_shape = context->GetInputShape(1);
  const gert::Shape& x_storage = x_shape->GetStorageShape();
  const gert::Shape& y_storage = y_shape->GetStorageShape();

  if (x_storage.GetDimNum() != y_storage.GetDimNum()) {
    return ge::GRAPH_FAILED;
  }

  uint64_t data_sz = 1;
  for (int i = 0; i < x_storage.GetDimNum(); i++) {
    if (x_storage.GetDim(i) != y_storage.GetDim(i)) {
      return ge::GRAPH_FAILED;
    }
    data_sz *= static_cast<uint64_t>(x_storage.GetDim(i));
    if (data_sz > std::numeric_limits<uint32_t>::max()) {
      return ge::GRAPH_FAILED;
    }
  }

  const uint32_t totalLength = static_cast<uint32_t>(data_sz);
  const uint32_t alignCount = blockDim * tileNum * bufferNum;
  if (totalLength == 0 || totalLength % alignCount != 0) {
    return ge::GRAPH_FAILED;
  }

  tiling->totalLength = totalLength;
  tiling->tileNum = tileNum;
  tiling->blockLength = totalLength / blockDim;
  tiling->tileLength = tiling->blockLength / tileNum / bufferNum;
  tiling->loopCount = tileNum * bufferNum;
  context->SetBlockDim(blockDim);
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;
  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* x2_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (x1_shape->GetDimNum() != x2_shape->GetDimNum()) {
        return GRAPH_FAILED;
    }
    for (int i = 0; i < x1_shape->GetDimNum(); i++) {
        if (x1_shape->GetDim(i) != x2_shape->GetDim(i)) {
            return GRAPH_FAILED;
        }
    }
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class DivCustomTemplate : public OpDef {
public:
    explicit DivCustomTemplate(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(DivCustomTemplate);
}
