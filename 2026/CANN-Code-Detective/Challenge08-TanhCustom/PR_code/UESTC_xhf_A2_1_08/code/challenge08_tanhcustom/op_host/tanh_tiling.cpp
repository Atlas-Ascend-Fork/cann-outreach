/*!
 * \file tanh_tiling.cpp
  * \brief Tanh 算子 Tiling 实现
   */
   #include "register/op_def_registry.h"
   #include "op_common/log/log.h"
   #include "op_common/op_host/util/math_util.h"
   #include "op_common/op_host/util/platform_util.h"
   #include "../op_kernel/tanh_tiling_data.h"
   #include "../op_kernel/tanh_tiling_key.h"

   namespace optiling {
   using Ops::Base::CeilDiv;
   using Ops::Base::FloorAlign;
   constexpr uint32_t WS_SYS_SIZE = 0U;
   constexpr int64_t TYPE_SIZE = 4;
   constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
   static const gert::Shape g_vec_1_shape = {1};

   static inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape)
   {
       if (inShape.GetDimNum() == 0) {
               return g_vec_1_shape;
                   }
                       return inShape;
                       }

                       static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
                       {
                           fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
                               OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
                                   auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
                                       coreNum = ascendcPlatform.GetCoreNumAiv();
                                           OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
                                               ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
                                                   OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
                                                       return ge::GRAPH_SUCCESS;
                                                       }

                                                       static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
                                                       {
                                                           uint64_t ubSize;
                                                               int64_t coreNum;
                                                                   OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                                                                           OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
                                                                               size_t* workspace = context->GetWorkspaceSizes(1);
                                                                                   OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
                                                                                       workspace[0] = WS_SYS_SIZE;
                                                                                           TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
                                                                                               OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
                                                                                                   const gert::StorageShape* inputShape = context->GetInputShape(0);
                                                                                                       OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
                                                                                                           const gert::Shape storageShape = EnsureNotScalar(inputShape->GetStorageShape());
                                                                                                               int64_t totalNum = 1;
                                                                                                                   for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
                                                                                                                           totalNum *= storageShape.GetDim(i);
                                                                                                                               }
                                                                                                                                   OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum must be positive"), return ge::GRAPH_FAILED);
                                                                                                                                       int64_t blockDim = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
                                                                                                                                           blockDim = blockDim > coreNum ? coreNum : blockDim;
                                                                                                                                               blockDim = blockDim < 1 ? 1 : blockDim;
                                                                                                                                                   while (blockDim > 1 && totalNum % blockDim != 0) {
                                                                                                                                                           --blockDim;
                                                                                                                                                               }
                                                                                                                                                                   const int64_t blockFactor = totalNum / blockDim;
                                                                                                                                                                       constexpr int64_t QUEUE_BUFFER_COUNT = 4;
                                                                                                                                                                           constexpr int64_t ALIGN_ELEMS = 8;
                                                                                                                                                                               int64_t maxUbFactor = static_cast<int64_t>(ubSize) / (TYPE_SIZE * QUEUE_BUFFER_COUNT);
                                                                                                                                                                                   maxUbFactor = FloorAlign(maxUbFactor, ALIGN_ELEMS);
                                                                                                                                                                                       OP_CHECK_IF(maxUbFactor <= 0, OP_LOGE(context, "UB is too small"), return ge::GRAPH_FAILED);
                                                                                                                                                                                           const int64_t ubFactor = blockFactor < maxUbFactor ? blockFactor : maxUbFactor;
                                                                                                                                                                                               tiling->totalNum = totalNum;
                                                                                                                                                                                                   tiling->blockFactor = blockFactor;
                                                                                                                                                                                                       tiling->ubFactor = ubFactor;
                                                                                                                                                                                                           context->SetBlockDim(static_cast<uint32_t>(blockDim));
                                                                                                                                                                                                               const auto inputDesc = context->GetInputDesc(0);
                                                                                                                                                                                                                   const uint64_t tilingKey = inputDesc != nullptr && inputDesc->GetDataType() == ge::DT_FLOAT16
                                                                                                                                                                                                                           ? GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0)
                                                                                                                                                                                                                                   : GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
                                                                                                                                                                                                                                       context->SetTilingKey(tilingKey);
                                                                                                                                                                                                                                           return ge::GRAPH_SUCCESS;
                                                                                                                                                                                                                                           }

                                                                                                                                                                                                                                           static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
                                                                                                                                                                                                                                           {
                                                                                                                                                                                                                                               return ge::GRAPH_SUCCESS;
                                                                                                                                                                                                                                               }
                                                                                                                                                                                                                                               struct TanhCompileInfo {};
                                                                                                                                                                                                                                               IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);
                                                                                                                                                                                                                                               } // namespace optiling
                                                                                                                                                                                                                                               