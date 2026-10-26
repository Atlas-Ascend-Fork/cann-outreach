# expcustom

提交团队: 北京科技大学
提交者: gcw_JQJBklQH


## 算子实现介绍

本 Exp 算子基于 Ascend C 开源仓结构实现逐元素自然指数 y=e^x：Host 侧 ExpTilingFunc 读取输入 dtype 并通过 ASCENDC_TPL_SEL_PARAM(schMode) 选定模板实例（float16=0、float32=1），将 shape 求积得到的 totalLength 与分块数 tileNum 下发，并 SetBlockDim(8) 启动 8 核；Kernel 侧入口用非类型模板参数 template &lt;uint32_t schMode&gt; 配合 if constexpr 编译期分发到 Exp&lt;half&gt;/Exp&lt;float&gt;，类内按 blockLength=totalLength/GetBlockNum() 做多核切分、tileLength=blockLength/tileNum/BUFFER_NUM 做核内分块，通过 BUFFER_NUM=2 的 TQue 双缓冲实现 DataCopy(GM→UB) → AscendC::Exp → DataCopy(UB→GM) 三段流水；针对判题机旧版 CANN 会把 DATATYPE 模板参数以整数 exp&lt;0&gt; 展开导致的 typename DT_X 编译失败，改用版本无关的 UINT schMode 型 TilingKey 予以解决，最终 op_host/kernel UT、精度比对、完整编译打包及上板 fp32（相对误差 1.19e-07）、fp16（4.8e-04）验证全部通过。
