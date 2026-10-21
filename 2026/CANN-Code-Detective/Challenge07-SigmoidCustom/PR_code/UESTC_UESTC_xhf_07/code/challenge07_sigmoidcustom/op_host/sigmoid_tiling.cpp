/*!
 * \file sigmoid_tiling.cpp
  * \brief Sigmoid 算子 Tiling 实现
   */

   #include "register/op_def_registry.h"
   #include "op_common/log/log.h"
   #include "op_common/op_host/util/math_util.h"
   #include "op_common/op_host/util/platform_util.h"
   #include "../op_kernel/sigmoid_tiling_data.h"
   #include "../op_kernel/sigmoid_tiling_key.h"

   namespace optiling {

   constexpr uint32_t TILE_NUM = 16;

   static ge::graphStatus SigmoidTilingFunc(gert::TilingContext* context)
   {
       auto inputDesc = context->GetInputDesc(0);
           OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
               uint32_t dataType = static_cast<uint32_t>(inputDesc->GetDataType());
                   ASCENDC_TPL_SEL_PARAM(context, dataType);

                       SigmoidTilingData* tiling = context->GetTilingData<SigmoidTilingData>();
                           OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
                               const gert::StorageShape* xShape = context->GetInputShape(0);
                                   OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

                                       uint32_t totalLength = 1;
                                           const gert::Shape& storageShape = xShape->GetStorageShape();
                                               for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
                                                       totalLength *= storageShape.GetDim(i);
                                                           }
                                                               tiling->totalLength = totalLength;
                                                                   tiling->tileNum = TILE_NUM;

                                                                       context->SetBlockDim(8);
                                                                           size_t* currentWorkspace = context->GetWorkspaceSizes(1);
                                                                               OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
                                                                                   currentWorkspace[0] = 0;
                                                                                       return ge::GRAPH_SUCCESS;
                                                                                       }

                                                                                       static ge::graphStatus TilingParseForSigmoid([[maybe_unused]] gert::TilingParseContext* context)
                                                                                       {
                                                                                           return ge::GRAPH_SUCCESS;
                                                                                           }

                                                                                           struct SigmoidCompileInfo {};

                                                                                           IMPL_OP_OPTILING(Sigmoid).Tiling(SigmoidTilingFunc).TilingParse<SigmoidCompileInfo>(TilingParseForSigmoid);

                                                                                           } // namespace optiling

                                                                                           