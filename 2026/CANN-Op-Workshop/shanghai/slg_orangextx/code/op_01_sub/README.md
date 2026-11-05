# op_01_sub

提交团队: 上海理工大学
提交者: orangextx


## 算子实现介绍

基于Ascend C实现z=x-y：Host按形状算总长、设tileNum=8并按dtype分派float/half；KernelSub多核切分GM，双buffer队列循环CopyIn/Compute/CopyOut，Compute用AscendC::Sub逐元素减法。
