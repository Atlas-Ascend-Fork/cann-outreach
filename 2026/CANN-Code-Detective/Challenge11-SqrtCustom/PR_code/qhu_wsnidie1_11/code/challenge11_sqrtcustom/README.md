# challenge11_sqrtcustom

提交团队: 青海大学
提交者: wsnidie1


## 算子实现介绍

多核切分（8 核）+ UB 双缓冲分块，CopyIn/Compute/CopyOut 三段式流水；Tiling 按输入 dtype 选择模板实例（float16/float32），配置 totalNum、blockFactor、ubFactor，UB 分块长度按容量自适应。
