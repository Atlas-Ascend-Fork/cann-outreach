# DivCustomTemplate 算子工程说明

## 目录结构与各文件职责

```
DivCustomTemplate/
├── div_custom_template.json     # 算子原型定义：声明算子名、输入(x,y)/输出(z)的 dtype 与 format
├── run.sh                       # 一键编译运行脚本：msopgen 生成→编译→安装→编译测试→运行测试
├── test/
│   ├── main.cpp                 # ACLNN 接口调用测试：构造 x=1.0/y=2.0，调用 aclnnDivCustomTemplate，验证 z=0.5
│   └── CMakeLists.txt           # 测试构建脚本（可选，run.sh 默认用 g++ 直接编译）
└── custom_op/                   # msopgen 生成的算子工程
    ├── CMakeLists.txt           # 顶层构建脚本，组织子目录编译
    ├── CMakePresets.json        # CMake 预设（ASCEND_CANN_PACKAGE_PATH、compute unit 等）
    ├── build.sh                 # 编译打包脚本，调用 cmake+make 生成 .run 安装包
    ├── framework/tf_plugin/     # 框架适配插件（TensorFlow 图解析），本题不涉及修改
    ├── op_host/                 # Host 侧
    │   ├── CMakeLists.txt
    │   └── div_custom_template.cpp  # 算子注册（OpDef）、InferShape/InferDataType、TilingFunc 计算 tile 参数
    └── op_kernel/               # Kernel 侧
        ├── CMakeLists.txt
        ├── div_custom_template_tiling.h  # Tiling 数据结构定义（totalLength/tileNum/tileLength）
        └── div_custom_template.cpp       # Ascend C 核函数：Double Buffer 流水线，Div 逐元素除法
```

## 三层职责划分

| 层 | 目录 | 职责 |
|---|---|---|
| 原型定义 | `div_custom_template.json` | 声明算子 IO 接口（名称、类型、格式），msopgen 的输入 |
| Host 侧 | `op_host/` | 算子注册到框架、Shape 推导、Tiling 计算（分块参数下发给 kernel） |
| Kernel 侧 | `op_kernel/` | Ascend C 核函数实现，GM↔UB 数据搬运 + Div 计算 |

## 数据流

`GM(x,y) → CopyIn → UB → Div 计算 → UB → CopyOut → GM(z)`

通过 Double Buffer（BUFFER_NUM=2）实现搬运与计算的流水重叠。
