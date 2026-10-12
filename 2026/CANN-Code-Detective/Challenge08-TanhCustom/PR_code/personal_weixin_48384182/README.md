# 个人

## 团队信息

- 提交者: weixin_48384182
- 身份: 学生
- 单位: 个人

## 成员

- weixin_48384182 (weixin_48384182): 提交者

## 算子: challenge08_tanhcustom

使用单核实现，UB切分按32B对齐，UB按BUFFER_NUM=2双缓冲切分，掩盖搬运与计算耗时 计算式：tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)，仅需一次 Exp，UB内float32计算后Cast回输出类型 Tiling策略：totalLength对齐到BLOCK_SIZE的整数倍
