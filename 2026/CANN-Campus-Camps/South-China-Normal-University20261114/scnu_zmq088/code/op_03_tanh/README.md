# op_03_tanh

提交团队: 华南师范大学
提交者: zmq088


## 算子实现介绍

使用多核并行处理，Host侧根据总数据长度配置BlockDim为8，单核内按tileNum分块。
Kernel侧采用CopyIn-Compute-CopyOut流水线，使用TQue双缓冲。
由于Tanh公式含指数运算，针对float16输入采用升精度至float32计算，最后使用CAST_RINT截断回half，避免精度丢失和溢出，完成 z = tanh(x) 计算。
