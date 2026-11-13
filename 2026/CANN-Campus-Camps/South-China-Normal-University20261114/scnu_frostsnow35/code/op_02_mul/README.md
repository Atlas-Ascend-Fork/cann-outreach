# op_02_mul

提交团队: 华南师范大学
提交者: frost35_snow


## 算子实现介绍

实现 z = x * y（(8,2048) ND，float32/float16）。
kernel 按三级流水组织：TPipe 为 x/y/z 各建双缓冲队列，Process 以 2048 元素 tile 分块循环，每轮搬入、Mul 计算、搬出交叠执行。数据类型以模板参数 DT_X 区分，host 侧 tiling key 选择实例，InferShape/InferDataType 使输出 shape、dtype 跟随输入。
多核与对齐策略：段长按 32/sizeof(T) 元素粒度向上对齐（float 8 元素、half 16 元素），段起点为 GetBlockIdx×段长，DataCopy 的 GM 地址与长度恒为 32B 整数倍，因此不使用 DataCopyPad、不特判尾块；核数与分段由 kernel 用 GetBlockNum/GetBlockIdx 自适应，tiling 只传总长度。
