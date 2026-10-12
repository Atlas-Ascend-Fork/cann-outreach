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

   // 矢量系统缓冲区大小为 0（本算子无需系统 workspace）
   constexpr uint32_t WS_SYS_SIZE = 0U;
   // 标量形状常量 {1}
   static const gert::Shape g_vec_1_shape = {1};

   using Ops::Base::CeilDiv;
   using Ops::Base::CeilAlign;
   using Ops::Base::FloorDiv;
   using Ops::Base::FloorAlign;
   using Ops::Base::GetUbBlockSize;

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

                                                                       const int64_t BLOCK_DIM = 8;    // 启动核数
                                                                       const int64_t TILE_NUM = 1;     // 每核分块数

                                                                       static ge::graphStatus SigmoidTilingFunc(gert::TilingContext* context)
                                                                       {
                                                                           // 平台信息（ubSize / coreNum）
                                                                               uint64_t ubSize = 0;
                                                                                   int64_t coreNum = 0;
                                                                                       auto ret = GetPlatformInfo(context, ubSize, coreNum);
                                                                                           if (ret != ge::GRAPH_SUCCESS) {
                                                                                                   return ret;
                                                                                                       }

                                                                                                           // 输入 shape 逐维相乘得到总元素数
                                                                                                               SigmoidTilingData* tiling = context->GetTilingData<SigmoidTilingData>();
                                                                                                                   OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
                                                                                                                       const gert::StorageShape* x_shape = context->GetInputShape(0);
                                                                                                                           OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
                                                                                                                               int64_t data_sz = 1;
                                                                                                                                   for (int i = 0; i < x_shape->GetStorageShape().GetDimNum(); i++) {
                                                                                                                                           data_sz *= x_shape->GetStorageShape().GetDim(i);
                                                                                                                                               }

                                                                                                                                                   // 按输入 dtype 设置 tilingKey（框架编码：0=float, 1=float16，与 ASCENDC_TPL_DATATYPE_DECL 顺序一致）
                                                                                                                                                       auto inputDesc = context->GetInputDesc(0);
                                                                                                                                                           OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
                                                                                                                                                               uint32_t dtype = static_cast<uint32_t>(inputDesc->GetDataType());
                                                                                                                                                                   context->SetTilingKey((dtype == static_cast<uint32_t>(ge::DT_FLOAT16)) ? 1 : 0);

                                                                                                                                                                       tiling->totalNum = data_sz;
                                                                                                                                                                           // 每核元素数：向上取整，非整除 shape 无尾部丢弃
                                                                                                                                                                               tiling->blockFactor = CeilDiv(data_sz, BLOCK_DIM);
                                                                                                                                                                                   if (tiling->blockFactor < 1) {
                                                                                                                                                                                           tiling->blockFactor = 1;
                                                                                                                                                                                               }
                                                                                                                                                                                                   tiling->ubFactor = tiling->blockFactor / TILE_NUM;
                                                                                                                                                                                                       if (tiling->ubFactor < 1) {
                                                                                                                                                                                                               tiling->ubFactor = tiling->blockFactor;
                                                                                                                                                                                                                   }
                                                                                                                                                                                                                       // UB 容量校验：单核 UB 需求（输入 fp16/32 + 双 fp32 计算缓冲 + ones），按最坏 4 份 fp32 + 1 份 fp16 估算
                                                                                                                                                                                                                           const uint64_t ubNeed = (uint64_t)tiling->ubFactor * (4 * 5); // 5份fp32缓冲（input/output/calcA/calcB/ones）——修正原18B估算偏小
                                                                                                                                                                                                                               if (ubNeed > ubSize && tiling->ubFactor > 0) {
                                                                                                                                                                                                                                       tiling->ubFactor = (int64_t)(ubSize / (4 * 5));
                                                                                                                                                                                                                                               if (tiling->ubFactor < 1) {
                                                                                                                                                                                                                                                           tiling->ubFactor = 1;
                                                                                                                                                                                                                                                                   }
                                                                                                                                                                                                                                                                       }

                                                                                                                                                                                                                                                                           // 配置启动核数
                                                                                                                                                                                                                                                                               context->SetBlockDim(BLOCK_DIM);

                                                                                                                                                                                                                                                                                   return ge::GRAPH_SUCCESS;
                                                                                                                                                                                                                                                                                   }

                                                                                                                                                                                                                                                                                   static ge::graphStatus TilingParseForSigmoid([[maybe_unused]] gert::TilingParseContext* context)
                                                                                                                                                                                                                                                                                   {
                                                                                                                                                                                                                                                                                       return ge::GRAPH_SUCCESS;
                                                                                                                                                                                                                                                                                       }

                                                                                                                                                                                                                                                                                       struct SigmoidCompileInfo {};

                                                                                                                                                                                                                                                                                       IMPL_OP_OPTILING(Sigmoid).Tiling(SigmoidTilingFunc).TilingParse<SigmoidCompileInfo>(TilingParseForSigmoid);

                                                                                                                                                                                                                                                                                       } // namespace optiling
                                                                                                                                                                                                                                                                                       