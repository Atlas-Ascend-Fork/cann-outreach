# challenge08_tanhcustom

提交团队: 农科院华东中心
提交者: kandice


## 算子实现介绍

Host侧按输入shape完成tiling切分Device侧采用三段流水，双缓冲重叠搬运，精度满足阈值要求。
