/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
  #include "op_common/log/log.h"
  #include "op_common/op_host/util/math_util.h"
  #include "op_common/op_host/util/platform_util.h"
  #include "tiling/platform/platform_ascendc.h"
  #include "../op_kernel/relu_tiling_data.h"
  #include "../op_kernel/relu_tiling_key.h"

  namespace optiling {

  constexpr uint32_t TILE_LENGTH_FP32 = 9216;
  constexpr uint32_t TILE_LENGTH_FP16 = 18432;
  constexpr uint32_t MIN_PER_CORE = 1024;

  static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
  {
      auto inputDesc = context->GetInputDesc(0);
      if (inputDesc == nullptr) {
          return ge::GRAPH_FAILED;
      }

      ge::DataType dtype = inputDesc->GetDataType();
      uint32_t DT_X = static_cast<uint32_t>(dtype);
      ASCENDC_TPL_SEL_PARAM(context, DT_X);

      ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
      if (tiling == nullptr) {
          return ge::GRAPH_FAILED;
      }

      const gert::StorageShape* x1_shape = context->GetInputShape(0);
      if (x1_shape == nullptr) {
          return ge::GRAPH_FAILED;
      }

      int64_t totalLength = 1;
      const auto& shape = x1_shape->GetStorageShape();
      for (size_t i = 0; i < shape.GetDimNum(); ++i) {
          totalLength *= shape.GetDim(i);
      }
      if (totalLength <= 0 || totalLength > 0xFFFFFFFFLL) {
          return ge::GRAPH_FAILED;
      }

      tiling->totalLength = static_cast<uint32_t>(totalLength);
      tiling->tileNum = 0;
      tiling->tileLength =
          (dtype == ge::DT_FLOAT16) ? TILE_LENGTH_FP16 : TILE_LENGTH_FP32;

      uint32_t coreNum = 8;
      auto platformInfo = context->GetPlatformInfo();
      if (platformInfo != nullptr) {
          auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
          uint32_t availableCoreNum = ascendcPlatform.GetCoreNum();
          if (availableCoreNum > 0) {
              coreNum = availableCoreNum;
          }
      }

      uint64_t needCore64 =
          (static_cast<uint64_t>(totalLength) + MIN_PER_CORE - 1) / MIN_PER_CORE;
      uint32_t needCore = needCore64 == 0 ? 1 : static_cast<uint32_t>(needCore64);
      uint32_t blockDim = coreNum < needCore ? coreNum : needCore;
      if (blockDim == 0) {
          blockDim = 1;
      }
      context->SetBlockDim(blockDim);

      auto rawTilingData = context->GetRawTilingData();
      if (rawTilingData != nullptr) {
          rawTilingData->SetDataSize(sizeof(ReluTilingData));
      }

      size_t* currentWorkspace = context->GetWorkspaceSizes(1);
      if (currentWorkspace != nullptr) {
          currentWorkspace[0] = 0;
      }
      return ge::GRAPH_SUCCESS;
  }

  static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
  {
      return ge::GRAPH_SUCCESS;
  }

  struct ReluCompileInfo {};

  IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

  } // namespace optiling