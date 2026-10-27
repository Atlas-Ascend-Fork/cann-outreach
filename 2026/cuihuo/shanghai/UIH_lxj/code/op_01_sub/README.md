# op_01_sub

提交团队: 联影医疗
提交者: gcw_4MyztPAz


## 算子实现介绍

Tiling切分(核间，核内划分)
算子类结构实现：Init(), Process(),CopyIn(), Compute(), CopyOut()
核函数实现: sub_custom()，包含算子类对象实例化，对象初始化，对象执行
算子调用run_kernel（）
