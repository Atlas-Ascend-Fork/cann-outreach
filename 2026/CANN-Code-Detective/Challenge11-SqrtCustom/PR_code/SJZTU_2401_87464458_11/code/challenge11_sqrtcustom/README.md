# challenge11_sqrtcustom

提交团队: 石家庄铁道大学
提交者: LRJ


## 算子实现介绍

【代码侦探Challenge11】实现逐元素 Sqrt 算子。Host 按输入 dtype 选择 half/float 模板，结合 AIV 核数及 UB 容量做 32 字节对齐切分；Kernel 使用双缓冲队列完成 CopyIn → Sqrt → CopyOut，并处理尾块。基于官方模板独立实现，使用 OpenAI Codex（GPT-6 系列）辅助开发与代码复核，Prompt：处理 https://gitcode.com/cann/cann-learning-hub/issues/894 并提交 PR。

## 验证说明

- 算子实现已通过 [CANNJudge 提交 992134](https://cannjudge.cn/cann_code_detective/challenge11-sqrtcustom/challenge11-sqrtcustom/submission/6ae1ce351dac5d235b511e97) 的 4/4 测试点。后续评审修正没有修改 `op_host` 或 `op_kernel`。
- Host UT 校验 float32/float16 对应的 TilingKey 和实际 Tiling 字段；Kernel UT 保留原有二进制比对，并增加非全 1 输入的多核、多 UB 分块精度断言。
- ACLNN 示例回读实际输出，与 `std::sqrt` 逐元素比对，验证失败返回非零状态。

在安装 CANN 的 Ascend910B 环境中执行：

```bash
bash build.sh -u
bash build.sh -e
```

新增 UT、示例断言及构建辅助修正已完成静态复核，但尚未在该硬件环境中编译运行；上述历史平台 Pass 不代表新增 UT 已执行。
