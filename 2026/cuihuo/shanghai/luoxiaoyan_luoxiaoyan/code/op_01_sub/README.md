# op_01_sub

提交团队: luoxiaoyan
提交者: luoxiaoyan2024


## 算子实现介绍

实现结构
  - 核心计算封装为 template &lt;typename T&gt; class KernelSub，同时对 float（fp32）与 half（fp16）实例化；run_kernel 根据 info_x.tensors[0].dtype 分派到 sub_custom_fp32 / sub_custom_fp16 两个
  __global__ __vector__ 入口。
  - 采用 TPipe + TQue&lt;VECIN/VECOUT, 2&gt; 双缓冲，单 tile 2048 元素（fp32 8 KB、fp16 4 KB），流水为 CopyIn → Compute → CopyOut，计算调用 AscendC::Sub。
  - 多核切分：tileCount = ceil(total / 2048)，blockNum = min(tileCount, availableCoreNum)，各核按 GetBlockIdx() 轮转领取 tile。
  - 搬运：整 tile 使用 DataCopy；尾部非 32 B 对齐时使用 DataCopyPad，字节长度与 UB 缓冲区大小均按 sizeof(T) 计算。
  - Host 侧由 shape 连乘得到元素总数（0 维按 1 元素处理），核数非正时兜底为 1，不使用不可靠的 UB 运行时属性查询（UB 预算静态固定为 fp32 48 KB / fp16 24 KB）。

验证结果：910B3 真机编译通过；fp32、fp16 在 (8,2048) 上均与 numpy 结果逐比特一致（16384/16384，max_abs_err=0），fp16 尾块 (8,2047) 亦正确。
