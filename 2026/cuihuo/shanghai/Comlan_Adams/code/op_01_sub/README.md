# op_01_sub

提交团队: 上海昆联数码科技有限公司
提交者: CatherinaAda


## 算子实现介绍

使用 Ascend C 原生直调实现逐元素矢量减法 z = x - y，输入输出 shape=(8,2048)、ND，支持 float32 和 float16。按输入 dtype 分发核函数，根据可用 Vector 核数划分数据，每块最多 2048 元素，使用双缓冲队列、DataCopyPad 搬运及 AscendC::Sub 计算。两种 dtype 均在 Ascend 910B4 真机完成编译运行，16384/16384 元素与参考结果一致，最大绝对误差为 0。CANNJudge 提交 ID 926557，2/2 测试点 Pass。
