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
   using Ops::Base::CeilAlign;
   using Ops::Base::FloorDiv;
   using Ops::Base::FloorAlign;
   using Ops::Base::GetUbBlockSize;

   constexpr uint32_t WS_SYS_SIZE = 0U;
   constexpr int64_t TYPE_SIZE = 4;
   constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

   static const gert::Shape g_vec_1_shape = {1};

   static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
       if (in_shape.GetDimNum() == 0) {
               return g_vec_1_shape;
                   }
                       return in_shape;
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

                                                       static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
                                                       {
                                                           size_t* currentWorkspace = context->GetWorkspaceSizes(1);
                                                               OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
                                                                   currentWorkspace[0] = WS_SYS_SIZE;
                                                                       return ge::GRAPH_SUCCESS;
                                                                       }

                                                                       static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
                                                                       {
                                                                           uint64_t ubSize;
                                                                               int64_t coreNum;
                                                                                   OP_CHECK_IF(
                                                                                           GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                                                                                                   OP_LOGE(context, "GetPlatformInfo error"),
                                                                                                           return ge::GRAPH_FAILED);

                                                                                                               OP_CHECK_IF(
                                                                                                                       GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                                                                                                                               OP_LOGE(context, "GetWorkspaceSize error"),
                                                                                                                                       return ge::GRAPH_FAILED);

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

                                                                                                                                                                                       // 每核至少处理 MIN_SPLIT_THRESHOLD 个元素，并选择能够整除总长度的核数，
                                                                                                                                                                                           // 避免固定长度分核时最后一个核访问输入范围之外的数据。
                                                                                                                                                                                               int64_t blockDim = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
                                                                                                                                                                                                   blockDim = blockDim > coreNum ? coreNum : blockDim;
                                                                                                                                                                                                       blockDim = blockDim < 1 ? 1 : blockDim;
                                                                                                                                                                                                           while (blockDim > 1 && totalNum % blockDim != 0) {
                                                                                                                                                                                                                   --blockDim;
                                                                                                                                                                                                                       }
                                                                                                                                                                                                                           const int64_t blockFactor = totalNum / blockDim;

                                                                                                                                                                                                                               // 输入、输出队列均为双缓冲。按 float 的 4 字节大小保守估算，
                                                                                                                                                                                                                                   // 可同时容纳 2 个输入块和 2 个输出块，并保持 32 字节对齐。
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

                                                                                                                                                                                                                                                                               // 根据输入 dtype 选择 tilingKey
                                                                                                                                                                                                                                                                                   uint64_t tilingKey;
                                                                                                                                                                                                                                                                                       auto inputDesc = context->GetInputDesc(0);
                                                                                                                                                                                                                                                                                           if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
                                                                                                                                                                                                                                                                                                   tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0);
                                                                                                                                                                                                                                                                                                       } else {
                                                                                                                                                                                                                                                                                                               tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
                                                                                                                                                                                                                                                                                                                   }
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

                                                                                                                                                                                                                                                                                                                               