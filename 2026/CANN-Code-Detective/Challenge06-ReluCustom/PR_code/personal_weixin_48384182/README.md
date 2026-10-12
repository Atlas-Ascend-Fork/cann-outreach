# 个人

## 团队信息

- 提交者: weixin_48384182
- 身份: 学生
- 单位: 个人

## 成员

- weixin_48384182 (weixin_48384182): 提交者

## 算子: challenge06_relucustom

多核实现，按硬件核数均分数据块，UB 切分按 32B 对齐，TPipe+TQue 双缓冲，CopyIn(DataCopy)→Compute(AscendC::Relu)→CopyOut 三段式流水线，尾块按实际长度搬运，支持 float16/float32。
