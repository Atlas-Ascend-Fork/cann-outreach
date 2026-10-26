# expcustom

提交团队: personal
提交者: gcw_93rlw6ed


## 算子实现介绍

逐元素 Exp 算子（y=e^x）。Tiling：totalNum 以 int64 累乘计算，小张量按 MIN_SPLIT_THRESHOLD=1024 缩核，blockFactor 按 16 元素对齐，ubFactor 按 UB 预算限幅（half 12B/元素、float 16B/元素）。Kernel：BUFFER_NUM=2 双缓冲，DataCopyPad 处理非对齐尾块；half 路径 Cast 到 fp32 再调用 AscendC::Exp 后 CAST_RINT 回 half，规避 half 域溢出与精度损失；float 路径直接计算。尾核长度由 totalNum - blockIdx*blockFactor 现场推导（TilingData 无 lastBlockLength 字段）。平台验证 precision=1，2/2 用例 PASS。
