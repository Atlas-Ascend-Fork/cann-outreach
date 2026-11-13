/*!
 * \file relu_tiling.cpp
  * \brief Relu 算子 Tiling 实现
   */

   #include "register/op_def_registry.h"
   #include "op_common/log/log.h"
   #include "op_common/op_host/util/math_util.h"
   #include "op_common/op_host/util/platform_util.h"
   #include "../op_kernel/relu_tiling_data.h"
   #include "../op_kernel/relu_tiling_key.h"

   namespace optiling {

   constexpr uint32_t WS_SYS_SIZE = 0U;
   constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
   constexpr int64_t MAX_TILE_BYTES = 65536;

   static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
   {
       fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
           OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
               auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
                   coreNum = ascendcPlatform.GetCoreNumAiv();
                       OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is invalid"), return ge::GRAPH_FAILED);
                           ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
                               OP_CHECK_IF(ubSize <= 8192, OP_LOGE(context, "ubSize is invalid"), return ge::GRAPH_FAILED);
                                   return ge::GRAPH_SUCCESS;
                                   }

                                   static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
                                   {
                                       uint64_t ubSize;
                                           int64_t coreNum;
                                               if (GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS) {
                                                       return ge::GRAPH_FAILED;
                                                           }
                                                               const auto* inputShape = context->GetInputShape(0);
                                                                   const auto* inputDesc = context->GetInputDesc(0);
                                                                       OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
                                                                           OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
                                                                               ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
                                                                                   OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

                                                                                       const bool isHalf = inputDesc->GetDataType() == ge::DT_FLOAT16;
                                                                                           const int64_t typeSize = isHalf ? 2 : 4;
                                                                                               const int64_t alignNum = 32 / typeSize;
                                                                                                   const int64_t totalNum = inputShape->GetStorageShape().GetShapeSize();
                                                                                                       if (totalNum < 0) {
                                                                                                               return ge::GRAPH_FAILED;
                                                                                                                   }
                                                                                                                       int64_t blockDim = (totalNum + MIN_SPLIT_THRESHOLD - 1) / MIN_SPLIT_THRESHOLD;
                                                                                                                           if (blockDim > coreNum) {
                                                                                                                                   blockDim = coreNum;
                                                                                                                                       }
                                                                                                                                           if (blockDim < 1) {
                                                                                                                                                   blockDim = 1;
                                                                                                                                                       }
                                                                                                                                                           int64_t blockFactor = (totalNum + blockDim - 1) / blockDim;
                                                                                                                                                               blockFactor = (blockFactor + alignNum - 1) / alignNum * alignNum;
                                                                                                                                                                   if (blockFactor < alignNum) {
                                                                                                                                                                           blockFactor = alignNum;
                                                                                                                                                                               }
                                                                                                                                                                                   blockDim = totalNum > 0 ? (totalNum + blockFactor - 1) / blockFactor : 1;

                                                                                                                                                                                       // 两块原地计算缓冲，单块最多 64 KB，预留 8 KB。
                                                                                                                                                                                           int64_t ubFactor = (ubSize - 8192) / (2 * typeSize);
                                                                                                                                                                                               ubFactor = ubFactor / alignNum * alignNum;
                                                                                                                                                                                                   const int64_t maxTileNum = MAX_TILE_BYTES / typeSize;
                                                                                                                                                                                                       if (ubFactor > maxTileNum) {
                                                                                                                                                                                                               ubFactor = maxTileNum;
                                                                                                                                                                                                                   }
                                                                                                                                                                                                                       if (ubFactor > blockFactor) {
                                                                                                                                                                                                                               ubFactor = blockFactor;
                                                                                                                                                                                                                                   }
                                                                                                                                                                                                                                       if (ubFactor < alignNum) {
                                                                                                                                                                                                                                               return ge::GRAPH_FAILED;
                                                                                                                                                                                                                                                   }
                                                                                                                                                                                                                                                       tiling->totalNum = totalNum;
                                                                                                                                                                                                                                                           tiling->blockFactor = blockFactor;
                                                                                                                                                                                                                                                               tiling->ubFactor = ubFactor;
                                                                                                                                                                                                                                                                   context->SetBlockDim(static_cast<uint32_t>(blockDim));
                                                                                                                                                                                                                                                                       context->SetTilingKey(isHalf ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0)
                                                                                                                                                                                                                                                                                                      : GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));

                                                                                                                                                                                                                                                                                                          size_t* currentWorkspace = context->GetWorkspaceSizes(1);
                                                                                                                                                                                                                                                                                                              OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
                                                                                                                                                                                                                                                                                                                  currentWorkspace[0] = WS_SYS_SIZE;
                                                                                                                                                                                                                                                                                                                      return ge::GRAPH_SUCCESS;
                                                                                                                                                                                                                                                                                                                      }

                                                                                                                                                                                                                                                                                                                      static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
                                                                                                                                                                                                                                                                                                                      {
                                                                                                                                                                                                                                                                                                                          return ge::GRAPH_SUCCESS;
                                                                                                                                                                                                                                                                                                                          }

                                                                                                                                                                                                                                                                                                                          struct ReluCompileInfo {};
                                                                                                                                                                                                                                                                                                                          IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

                                                                                                                                                                                                                                                                                                                          } // namespace optiling
                                                                                                                                                                                                                                                                                                                          