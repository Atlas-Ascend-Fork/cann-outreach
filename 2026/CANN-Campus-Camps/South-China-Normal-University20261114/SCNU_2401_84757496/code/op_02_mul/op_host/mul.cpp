// Host侧Tiling实现
  #include "register/op_def_registry.h"
  #include "tiling/platform/platform_ascendc.h"

  #include "../op_kernel/mul_tiling.h"
  #include "../op_kernel/tiling_key_mul.h"

  namespace optiling {
      constexpr uint32_t TILE_LENGTH_FP32 = 4096;
      constexpr uint32_t TILE_LENGTH_FP16 = 8192;
      constexpr uint32_t MIN_PER_CORE = 1024;

      static ge::graphStatus TilingFunc(gert::TilingContext *context) {
          // 获取算子输入张量信息
          const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
          if (tensor_x == nullptr) {
              return ge::GRAPH_FAILED;
          }
          uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
          ASCENDC_TPL_SEL_PARAM(context, DT_X);

          MulTilingData *tiling = context->GetTilingData<MulTilingData>();
          if (tiling == nullptr) {
              return ge::GRAPH_FAILED;
          }
          const gert::StorageShape* x1_shape = context->GetInputShape(0);
          if (x1_shape == nullptr) {
              return ge::GRAPH_FAILED;
          }

          int64_t data_sz = 1;
          for (size_t i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
              data_sz *= x1_shape->GetStorageShape().GetDim(i);
          }
          if (data_sz <= 0 || data_sz > 0xFFFFFFFFLL) {
              return ge::GRAPH_FAILED;
          }

          tiling->totalLength = static_cast<uint32_t>(data_sz);
          tiling->tileNum = 0;
          tiling->tileLength = (tensor_x->GetDataType() == ge::DT_FLOAT16)
                                   ? TILE_LENGTH_FP16
                                   : TILE_LENGTH_FP32;

          // 配置启动核数：优先取 Vector 核（AIV）
          uint32_t coreNum = 8;
          auto platformInfo = context->GetPlatformInfo();
          if (platformInfo != nullptr) {
              auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
              uint32_t availableCoreNum = ascendcPlatform.GetCoreNumAiv();
              if (availableCoreNum == 0) {
                  availableCoreNum = ascendcPlatform.GetCoreNum();
              }
              if (availableCoreNum > 0) {
                  coreNum = availableCoreNum;
              }
          }

          uint64_t needCore64 =
              (static_cast<uint64_t>(data_sz) + MIN_PER_CORE - 1) / MIN_PER_CORE;
          uint32_t needCore = needCore64 == 0 ? 1 : static_cast<uint32_t>(needCore64);
          uint32_t blockDim = coreNum < needCore ? coreNum : needCore;
          if (blockDim == 0) {
              blockDim = 1;
          }
          context->SetBlockDim(blockDim);

          auto rawTilingData = context->GetRawTilingData();
          if (rawTilingData != nullptr) {
              rawTilingData->SetDataSize(sizeof(MulTilingData));
          }

          // 配置 workspace 大小
          size_t *currentWorkspace = context->GetWorkspaceSizes(1);
          if (currentWorkspace != nullptr) {
              currentWorkspace[0] = 0;
          }
          return ge::GRAPH_SUCCESS;
      }
  }  // namespace optiling

  namespace ge {
      static graphStatus InferShape(gert::InferShapeContext *context) {
          // 获取第0号输入 x 的 shape（静态/动态维度信息）
          const gert::Shape* x1_shape = context->GetInputShape(0);
          // 获取第0号输出 z 的 shape 可写指针，用于设置输出 shape
          gert::Shape* z_shape = context->GetOutputShape(0);
          // 逐元素算子：输出 shape 与输入 shape 完全一致
          *z_shape = *x1_shape;
          return GRAPH_SUCCESS;
      }
      static graphStatus InferDataType(gert::InferDataTypeContext *context) {
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
  }  // namespace op