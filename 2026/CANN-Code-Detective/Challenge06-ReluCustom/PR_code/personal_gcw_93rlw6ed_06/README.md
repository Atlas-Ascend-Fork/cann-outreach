# 个人

## 团队信息

- 提交者: gcw_93rlw6ed
- 身份: 企业员工
- 单位: 个人

## 成员

- gcw_93rlw6ed (gcw_93rlw6ed): 提交者

## 算子: challenge06_relucustom

基于 Ascend C 实现 ReLU 逐元素激活算子：op_host 侧 Tiling 按输入 dtype 选择模板实例（float16-&gt;mode0, float-&gt;mode1），多核切分 blockFactor 并核内按 ubFactor 分块；op_kernel 侧采用 CopyIn-&gt;Compute(AscendC::Relu)-&gt;CopyOut 三段式流水线，支持 float/float16。
