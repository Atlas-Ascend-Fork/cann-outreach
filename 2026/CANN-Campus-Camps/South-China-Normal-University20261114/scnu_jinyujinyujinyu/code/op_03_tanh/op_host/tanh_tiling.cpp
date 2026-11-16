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
 	 
 	 // 队列块数（与 kernel 侧的 BUFFER_NUM 一致）：输入输出各两块做双缓冲
 	 constexpr int64_t QUEUE_BUFFER_NUM = 2;
 	 // 核内临时表张数：a、b、常量 2，三张都按 float32 算
 	 constexpr int64_t TEMP_BUF_NUM = 3;
 	 // 每个元素在 UB 上占的字节：输入输出队列 + 临时表
 	 constexpr int64_t BYTES_PER_ELEM = QUEUE_BUFFER_NUM * 2 * TYPE_SIZE + TEMP_BUF_NUM * TYPE_SIZE;
 	 // 按 32 字节对齐，取 float16 的 16 个元素，两种 dtype 都能对齐
 	 constexpr int64_t ALIGN_NUM = 16;
 	 
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
 	 
 	     // 输入形状连乘得到元素总数
 	     const gert::StorageShape* xStorageShape = context->GetInputShape(0);
 	     OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
 	     const gert::Shape& inShape = EnsureNotScalar(xStorageShape->GetStorageShape());
 	     int64_t totalNum = 1;
 	     for (size_t i = 0; i < inShape.GetDimNum(); i++) {
 	         totalNum *= inShape.GetDim(i);
 	     }
 	     OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "input element num is invalid"), return ge::GRAPH_FAILED);
 	 
 	     // 核数：数据量不大的时候不必铺满所有核，每核至少分到 MIN_SPLIT_THRESHOLD 个元素
 	     int64_t coreNumNeed = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
 	     int64_t usedCoreNum = coreNumNeed < coreNum ? coreNumNeed : coreNum;
 	     if (usedCoreNum < 1) {
 	         usedCoreNum = 1;
 	     }
 	 
 	     // 每核元素数量向上对齐，尾核不足的部分由 kernel 侧截断
 	     int64_t blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), ALIGN_NUM);
 	 
 	     // UB 一次能放下多少元素，再按 blockFactor 封顶
 	     int64_t ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize), BYTES_PER_ELEM), ALIGN_NUM);
 	     if (ubFactor < ALIGN_NUM) {
 	         ubFactor = ALIGN_NUM;
 	     }
 	     if (ubFactor > blockFactor) {
 	         ubFactor = blockFactor;
 	     }
 	 
 	     tiling->totalNum = totalNum;
 	     tiling->blockFactor = blockFactor;
 	     tiling->ubFactor = ubFactor;
 	 
 	     context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
 	 
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