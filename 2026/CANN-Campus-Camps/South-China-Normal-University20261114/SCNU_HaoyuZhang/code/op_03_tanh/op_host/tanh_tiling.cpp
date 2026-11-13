#include <algorithm>
 	 #include <cstring>
 	 
 	 #include "register/op_def_registry.h"
 	 #include "register/op_impl_registry.h"
 	 #include "tiling/platform/platform_ascendc.h"
 	 
 	 namespace optiling {
 	 
 	 // host / kernel 共享的 tiling 布局（kernel 侧按 uint32 下标读）
 	 struct TanhTilingBlob {
 	     uint32_t totalLength;   // 0
 	     uint32_t coreDataNum;   // 1
 	     uint32_t tileNum;       // 2
 	     uint32_t tileLength;    // 3
 	     uint32_t alignCount;    // 4
 	     uint32_t blockDim;      // 5
 	 };
 	 
 	 namespace {
 	 constexpr uint32_t TILE_NUM = 8;         // 每核内部搬运块数
 	 constexpr uint32_t ALIGN_BYTES = 32;     // UB 搬运 32B 对齐
 	 constexpr uint32_t TEMP_BYTES = 5 * 4;   // 5 个 fp32 临时量
 	 constexpr uint32_t UB_SAFE_RATIO = 9;    // UB 只用 90%
 	 
 	 inline uint32_t AlignUp(uint32_t value, uint32_t align)
 	 {
 	     return (value + align - 1U) / align * align;
 	 }
 	 
 	 inline uint32_t AlignDown(uint32_t value, uint32_t align)
 	 {
 	     return value / align * align;
 	 }
 	 }  // namespace
 	 
 	 // Tanh 是静态 shape 逐元素算子，无额外编译期信息
 	 struct TanhCompileInfo {
 	 };
 	 
 	 static ge::graphStatus TilingFunc(gert::TilingContext* context)
 	 {
 	     if (context == nullptr) {
 	         return ge::GRAPH_FAILED;
 	     }
 	 
 	     // ---------- 1. shape / dtype ----------
 	     const gert::StorageShape* xStorageShape = context->GetInputShape(0);
 	     if (xStorageShape == nullptr) {
 	         return ge::GRAPH_FAILED;
 	     }
 	     const gert::Shape& xShape = xStorageShape->GetOriginShape();
 	     int64_t totalLength = xShape.GetShapeSize();
 	     if (totalLength < 0) {
 	         return ge::GRAPH_FAILED;
 	     }
 	 
 	     auto xDesc = context->GetInputDesc(0);
 	     if (xDesc == nullptr) {
 	         return ge::GRAPH_FAILED;
 	     }
 	     ge::DataType dtype = xDesc->GetDataType();
 	     if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT) {
 	         return ge::GRAPH_FAILED;
 	     }
 	     uint32_t dtypeSize = (dtype == ge::DT_FLOAT) ? 4U : 2U;
 	     uint32_t alignCount = ALIGN_BYTES / dtypeSize;   // fp16=16, fp32=8
 	 
 	     // ---------- 2. 硬件信息 ----------
 	     platform_ascendc::PlatformAscendC ascendcPlatform(context->GetPlatformInfo());
 	     uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
 	     if (coreNum == 0) {
 	         return ge::GRAPH_FAILED;
 	     }
 	     uint64_t ubSize = 0;
 	     ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
 	 
 	     // ---------- 3. 默认值（空 tensor 走这里） ----------
 	     TanhTilingBlob blob;
 	     blob.totalLength = static_cast<uint32_t>(totalLength);
 	     blob.alignCount = alignCount;
 	     blob.coreDataNum = 0;
 	     blob.tileNum = 0;
 	     blob.tileLength = alignCount;
 	     blob.blockDim = 1;
 	 
 	     // ---------- 4. 正常切分 ----------
 	     if (totalLength > 0) {
 	         uint32_t alignTotal = AlignUp(static_cast<uint32_t>(totalLength), alignCount);
 	 
 	         uint32_t usefulCoreNum = std::min(coreNum, alignTotal / alignCount);
 	         if (usefulCoreNum == 0) {
 	             usefulCoreNum = 1;
 	         }
 	         uint32_t coreDataNum = AlignUp(alignTotal / usefulCoreNum, alignCount);
 	         // 收紧可能整核空闲的末核
 	         while (usefulCoreNum > 1 &&
 	                static_cast<uint64_t>(coreDataNum) * (usefulCoreNum - 1) >= static_cast<uint64_t>(totalLength)) {
 	             --usefulCoreNum;
 	             coreDataNum = AlignUp(alignTotal / usefulCoreNum, alignCount);
 	         }
 	 
 	         // 每元素占用：x 双缓冲 + y 双缓冲 + 5 个 fp32 临时量
 	         uint32_t bytesPerElem = 2 * dtypeSize * 2 + TEMP_BYTES;
 	         uint64_t usableUb = ubSize * UB_SAFE_RATIO / 10;
 	         uint32_t maxTileByUb = AlignDown(static_cast<uint32_t>(usableUb / bytesPerElem), alignCount);
 	         if (maxTileByUb < alignCount) {
 	             maxTileByUb = alignCount;
 	         }
 	 
 	         uint32_t tileLength = AlignDown(coreDataNum / TILE_NUM, alignCount);
 	         if (tileLength < alignCount) {
 	             tileLength = alignCount;
 	         }
 	         if (tileLength > maxTileByUb) {
 	             tileLength = maxTileByUb;
 	         }
 	         uint32_t tileNum = (coreDataNum + tileLength - 1) / tileLength;
 	 
 	         blob.coreDataNum = coreDataNum;
 	         blob.tileNum = tileNum;
 	         blob.tileLength = tileLength;
 	         blob.blockDim = usefulCoreNum;
 	     }
 	 
 	     // ---------- 5. 写回 ----------
 	     if (context->GetRawTilingData() == nullptr ||
 	         context->GetRawTilingData()->GetCapacity() < sizeof(blob)) {
 	         return ge::GRAPH_FAILED;
 	     }
 	     (void)memcpy(context->GetRawTilingData()->GetData(), &blob, sizeof(blob));
 	     context->GetRawTilingData()->SetDataSize(sizeof(blob));
 	 
 	     context->SetBlockDim(blob.blockDim);
 	     context->SetTilingKey(0UL);   // 单策略 kernel，key 恒为 0
 	 
 	     size_t* workspaces = context->GetWorkspaceSizes(1);
 	     if (workspaces == nullptr) {
 	         return ge::GRAPH_FAILED;
 	     }
 	     workspaces[0] = 0;   // 逐元素算子不需要 workspace
 	 
 	     return ge::GRAPH_SUCCESS;
 	 }
 	 
 	 static ge::graphStatus TilingPrepare4Tanh(gert::TilingParseContext* context)
 	 {
 	     (void)context;
 	     return ge::GRAPH_SUCCESS;
 	 }
 	 
 	 }  // namespace optiling
 	 
 	 IMPL_OP_OPTILING(Tanh)
 	     .Tiling(optiling::TilingFunc)
 	     .TilingParse<optiling::TanhCompileInfo>(optiling::TilingPrepare4Tanh);