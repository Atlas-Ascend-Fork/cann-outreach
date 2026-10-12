# challenge06_relucustom

提交团队: 个人开发者
提交者: Code_demo111111


## 算子实现介绍

基于Ascend C实现ReLU激活算子。Host侧实现Tiling切分（按核心数均分并做32B对齐）；Kernel侧完成数据搬运到UB，使用Maxs指令实现y=max(0,x)计算，并写回Global Memory。支持float32和float16数据类型。
