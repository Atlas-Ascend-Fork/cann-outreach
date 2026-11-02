# challenge11_sqrtcustom

提交团队: 电子科技大学
提交者: Student_ZZX


## 算子实现介绍

基于 Ascend C 实现 float16/float32 逐元素 Sqrt。Host 按输入长度、AI Vector 核数与 UB 容量动态切分并选择模板 tilingKey；Kernel 使用双缓冲队列、DataCopyPad 处理尾块，逐元素执行 AscendC::Sqrt。已通过 CANNJudge 992957 的 4/4 测试点。
