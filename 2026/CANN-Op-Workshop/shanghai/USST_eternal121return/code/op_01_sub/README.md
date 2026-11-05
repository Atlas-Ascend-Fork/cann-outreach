# op_01_sub

提交团队: 上海理工大学
提交者: eternal121return


## 算子实现介绍

基于 Ascend C 原生直调实现逐元素矢量减法 z = x − y。输入 x、y 与输出 z 均为 shape (8, 2048)、ND 格式，支持 float32 与 float16 两种数据类型。

实现采用「多核切分 + 核内分块」两级划分：host 侧从 TensorGroupInfo 取总元素数（16384），按 dtype 分发核函数实例——fp16 走 half，其余走 float；device 侧启动 8 个核平分数据，每核 2048 个元素，核内再按 tileNum=8 切块并启用双缓冲（BUFFER_NUM=2），每块 128 个元素。数据流为 CopyIn（GM→UB）→ Compute（AscendC::Sub）→ CopyOut（UB→GM），tiling 参数经 SubCustomTilingData 按值传入。
