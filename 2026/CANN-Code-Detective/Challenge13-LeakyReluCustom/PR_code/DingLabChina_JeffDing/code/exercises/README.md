# exercises

提交团队: DingLabChina
提交者: JeffDing


## 算子实现介绍

PyTorch 原生算子：torch.nn.functional.leaky_relu，即 y = max(x, 0) + negative_slope * min(x, 0)，负半轴斜率 negative_slope 默认 0.01
参考文档：https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.leaky_relu.html
