# 深圳市科思科技有限公司

## 团队信息

- 提交者: 杨卉
- 身份: 企业员工
- 单位: 深圳市科思科技有限公司

## 成员

- 杨卉 (vigneyyoung): 提交者

## 算子: challenge06_relucustom

每核处理 blockFactor = totalNum / 8 个元素，核内再按 ubFactor = blockFactor / 8 切成 8 片搬入 UB 处理，tiling 结构体为 {totalNum, blockFactor, ubFactor} 三个 uint32_t，无需 workspace
