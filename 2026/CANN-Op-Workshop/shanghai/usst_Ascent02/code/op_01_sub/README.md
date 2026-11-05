# op_01_sub

提交团队: 上海理工大学
提交者: Ascent02


## 算子实现介绍

该Ascend‑C减法算子通过多核均分全局数据，使用双缓冲完成GM与片上VECIN/VECOUT的数据搬移，调用向量Sub接口分块计算z=x‑y，支持float与half数据类型。
