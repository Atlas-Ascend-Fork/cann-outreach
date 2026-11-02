# sub

提交团队: 山东大学
提交者: gcw_ha9CzSW7


## 算子实现介绍

Tiling 策略： 采用 8 核并行，将 totalLength 按核均分；单核内设置 tileNum = 8，结合双缓冲配置划分数据块，按 Tile 循环完成数据搬入、向量减法和结果搬出。
