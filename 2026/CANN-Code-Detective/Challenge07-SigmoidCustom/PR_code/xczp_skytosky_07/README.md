# 个人开发者

## 团队信息

- 提交者: skytosky
- 身份: 个人开发者
- 单位: 无（个人开发者）

## 成员

- skytosky (skytosky): 提交者

## 算子: challenge07_sigmoidcustom

多核切分：blockFactor按核数均分并向上16元素对齐（保证32B对齐搬运），尾核处理剩余元素；MIN_SPLIT_THRESHOLD=2048使本例(8,2048)启用8核（实测最优——更多核的launch开销反而拖慢）；单核内按ubFactor(<=2048元素)分块，BUFFER_NUM=2双缓冲CopyIn/Compute/CopyOut流水。Compute采用AscendC Sigmoid高阶API（lib/activation/sigmoid.h），内部自动走高精度路径并对half做精度提升，一次调用完成y=1/(1+exp(-x))，平台实测2.62us/2.53us（测试点2为全场最优）。Tiling：totalNum为shape各维相乘；float->SCH_MODE_1，half->SCH_MODE_0。
