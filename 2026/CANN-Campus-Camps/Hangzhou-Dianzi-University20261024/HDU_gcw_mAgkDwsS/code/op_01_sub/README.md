# op_01_sub

提交团队: 杭州电子科技大学
提交者: 王宇腾


## 算子实现介绍

8 核均分：每核 2048 元素，核基址 = blockLength × GetBlockIdx()；核内再切 8 tile × 2 buffer，共 16 次迭代、每次 128 元素（fp32 512B、fp16 256B，均为 32B 整数倍）。用 TPipe + TQue(VECIN/VECOUT) 完成「搬入 → Sub → 搬出」三段流水，3 个队列 × 2 buffer 共占 UB 3KB。dtype 分流：fp16 走 half 实例，其余走 float。
