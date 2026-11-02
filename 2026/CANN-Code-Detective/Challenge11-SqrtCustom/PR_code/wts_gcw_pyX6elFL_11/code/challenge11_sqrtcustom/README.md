# challenge11_sqrtcustom

提交团队: 梧桐森智能科技（东莞）有限公司
提交者: sws168


## 算子实现介绍

完成第 11 期 Sqrt 算子，支援 ND 格式 `(8, 2048)` 的 float32 與 float16 非負輸入，逐元素計算平方根並保持輸入形狀及型別。

Host Tiling 依資料型別選擇官方 schedule key，依 AIV 核數與 UB 容量計算 blockFactor、ubFactor，按 32B 對齊並配置零 workspace。Kernel 使用雙緩衝 queue 完成 CopyIn → AscendC::Sqrt → CopyOut，裁切最後一核及最後一塊的有效資料範圍，並以 DataCopyPad 處理非對齊尾塊。

僅補齊 `op_host/sqrt_tiling.cpp` 與 `op_kernel/sqrt.h`，保留官方工程結構及 Tiling ABI。

驗證：CANNJudge 提交 **993440**，**Pass，4 / 4**，各測試點輸出錯誤占比均為 **0.00%**。

- [CANNJudge 通過紀錄](https://cannjudge.cn/cann_code_detective/challenge11-sqrtcustom/challenge11-sqrtcustom/submission/6ae1e9071dac5d235b65618a)
- [活動 Issue #894](https://gitcode.com/cann/cann-learning-hub/issues/894)

本 PR 為參與者補齊官方範本的獨立實作。平台未公開各驗證點的 dtype/shape 與執行環境；本機未執行 SDK UT。
