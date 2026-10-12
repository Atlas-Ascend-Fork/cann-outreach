# challenge08_tanhcustom

提交团队: personal
提交者: gcw_93rlw6ed


## 算子实现介绍

Tiling策略：多核切分blockFactor按16对齐，小张量缩核；ubFactor按UB 24B/元素预算限幅。Kernel：CopyIn→Compute→CopyOut三段流水，fp32直接计算，fp16先Cast到fp32再计算，用Duplicate(1)+Div替代Reciprocal保证精度，双dtype（float/float16）分支覆盖。
