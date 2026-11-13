# AbsCustom（Challenge 12）

逐元素绝对值算子 `y[i] = |x[i]|`，Ascend C 实现。

## 实现要点

- **多核切分（op_host）**：`totalNum / AIV 核数` 向上取到 32B 对齐得到 `blockFactor`，
  `blockDim = ceil(totalNum / blockFactor)`，尾核只处理剩余元素。
- **核内切分（op_host）**：默认 16 块，同时受 UB 容量约束
  （`UB_RESERVE_BYTES = 1024`，双缓冲共 4 块 buffer），保证单块不撑爆 UB 且不小于 32B。
- **核函数（op_kernel）**：`CopyIn -> Compute -> CopyOut` 三段式，输入/输出队列各 2 块（`BUFFER_NUM = 2`）
  做 Double Buffer，使搬运与计算重叠；块内偏移全部落在本核连续段内。
- **搬运**：统一使用 `DataCopyPad`，尾块长度不要求 32B 整数倍。
- **计算**：`AscendC::Abs` 逐元素计算，float / float16 由 tilingKey 选择模板实例。
- **形状推导**：输出与输入同形同 dtype。

## 构建与验证

```bash
bash build.sh            # 编译算子包
bash build.sh -u         # 运行 UT（tiling 校验）
bash build.sh -e         # 运行 aclnn 调用示例（需 NPU）
```

`examples/test_aclnn_abs.cpp` 会构造正负混合输入，并回读结果与 `|x|` 逐元素比对，打印最大误差。
