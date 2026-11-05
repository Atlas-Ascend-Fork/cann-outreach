# op_01_sub

提交团队: 上海理工大学
提交者: gcw_YYivh7Ti


## 算子实现介绍

矢量减法算子 z = x - y 的简要说明：

任务规模：输入 x、y 与输出 z 均为 (8, 2048)，支持 fp32 与 fp16，共 16384 个元素。
多核并行：启动 8 个 AI Core，每个核负责 2048 个连续元素，并按核号偏移 Global Memory 指针，各核区间互不重叠。
核内切分：每个核再按 tileNum=8、双缓冲 BUFFER_NUM=2 切块，单块 128 个元素，循环 16 次。
三级流水：CopyIn 用 DataCopy 把数据搬入 UB，Compute 调 AscendC::Sub(z, x, y, 128) 完成减法，CopyOut 再搬回 GM；队列的 EnQue/DeQue 自动同步。
资源管理：TPipe 分配三个队列，FP32 下总 UB 占用仅 3 KB；单块 512 B 满足 32 B 对齐，无尾块。
Host 侧：run_kernel 从张量形状累乘出总长、构造 tiling，按 dtype 分发到 half 或 float 模板实例。
与参考实现区别：框架与 kernel_add.asc 一致，仅把 Add 换成 Sub，并相应改类名。
