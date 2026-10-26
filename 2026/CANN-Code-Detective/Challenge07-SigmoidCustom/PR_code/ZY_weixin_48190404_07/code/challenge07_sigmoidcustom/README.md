# challenge07_sigmoidcustom

提交团队: 个人开发者
提交者: weixin_48190404


## 算子实现介绍

使用多核实现（根据数据量自适应核数），UB切分按32B对齐
Tiling策略：totalLength对齐到BLOCK_SIZE的整数倍，计算blockFactor均分给各核；核内UB循环按ubFactor（32B对齐）切分，启用双缓冲提高流水线效率。
