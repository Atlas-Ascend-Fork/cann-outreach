InplaceRsqrt FP16 — 独立完整工程（CANN 9.0.0 / AscendC）

功能：对连续ND四维输入[N4,N3,N2,N1]计算1/sqrt(x)，覆盖原设备输入。
数据类型：float16；输入域：有限、严格正数。无epsilon、无裁剪。
当前验证状态：仅Windows本地CPU工具测试/静态检查；未进行CANN编译和NPU验证。
工程文件齐全不代表目标工具链已编译通过，真实结果须在CANN/NPU环境运行确认。

一、文件
inplace_rsqrt.h       通用LocalTensor<half>原位函数
inplace_rsqrt.asc     任意长度分块Vector Kernel + 同步Host启动API
inplace_rsqrt_api.h   Host C ABI接口
shape_utils.h        shape乘积/字节数溢出检查、4096元素tile常量
main.cpp             ACL初始化、文件输入、单GM分配、启动、回读、完整资源释放
CMakeLists.txt       生成libinplace_rsqrt.a和inplace_rsqrt_demo
build.sh / run.sh    独立编译 / 单用例生成-运行-校验
rsqrt_tools.py       FP16输入和FP64参考生成、校验、33组NPU测试
requirements.txt     仅依赖NumPy，不依赖PyTorch/torch_npu
test_tools.py        CPU工具测试及任务/尾块地址模型，不执行NPU

二、编译和运行（安装CANN9.0.0的Linux/NPU环境）
从本目录执行，CANN路径及NPU_ARCH根据实际环境调整：
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 -m pip install -r requirements.txt
  NPU_ARCH=dav-2201 bash run.sh 1 2 3 17

run.sh默认device0。指定其他设备：
  DEVICE_ID=1 NPU_ARCH=dav-2201 bash run.sh 2 3 7 17
无shape参数时默认[1,2,3,17]。成功退出0，编译、ACL或精度校验失败均非0退出。

仅编译：
  bash build.sh
或直接：
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNPU_ARCH=dav-2201
  cmake --build build --parallel 4

产物：build/libinplace_rsqrt.a、build/inplace_rsqrt_demo。
若find_package(ASC)失败，请配置目标CANN安装所提供的ASC CMake包搜索路径，
可通过bash build.sh -DASC_DIR=<实际ASC包目录>传入；不需要修改父目录工程。

手动分步运行一个用例：
  python3 rsqrt_tools.py generate --shape 2 3 7 17 --out data
  ./build/inplace_rsqrt_demo 2 3 7 17 data/input.bin data/output.bin 0
  python3 rsqrt_tools.py verify --data data

输入和输出为little-endian FP16裸二进制，文件长度必须精确等于元素数*2。
参考为FP64文件golden_fp64.bin，由实际FP16存储值先转FP64再计算1/sqrt生成。
meta.json保存shape、格式和输入/参考SHA256。校验时检查hash、长度并重算参考。
input.bin和output.bin请使用独立文件，以保留原输入用于校验；GM上仍是同一地址。
Host会检查已有文件的device/inode，拒绝相对路径或链接别名指向同一文件。

三、完整33用例NPU验证
  python3 rsqrt_tools.py suite --exe ./build/inplace_rsqrt_demo --out suite_results
  # 其他设备：追加 --device 1

覆盖1/15/16/17/127/128/129等尾块、4095/4096/4097、8191/8192/8193，
多轴四维shape、零元素、跨257块以上的大张量。每次运行前删除该用例旧输出，
避免失败后误校验旧结果。输出suite_results/npu_results.json及每例run.log。
运行失败、超时、输出缺失/异常值/精度超限都视为失败，不自动放宽阈值。
另可生成边界数值（含最小FP16正subnormal和最大有限值）：
  python3 rsqrt_tools.py generate --shape 1 2 3 17 --out edge_data --pattern edges
  ./build/inplace_rsqrt_demo 1 2 3 17 edge_data/input.bin edge_data/output.bin
  python3 rsqrt_tools.py verify --data edge_data

本独立工程默认示例阈值rtol=1e-3、atol=1e-3。原生Rsqrt为近似指令，
官方文档不保证严格误差界，因此不能提前承诺全部用例通过。失败应检查输出，
不要把CPU参考测试当成NPU精度通过。若业务明确要求其他阈值，verify/suite可显式
传--rtol/--atol；脚本不会自动修改它们。原BatchMatmulMaxSum竞赛阈值完全未改。

四、两个集成接口
1. Device LocalTensor接口（仅需inplace_rsqrt.h）
  bool ok = local_ops::InplaceRsqrt(self, N4, N3, N2, N1);
self为half、VECIN/VECCALC/VECOUT，起点32字节对齐。shape元素数<=min(GetSize(),INT32_MAX)，
物理UB分配字节向上对齐32字节。仅允许连续布局，不能每行有stride padding。
函数通过Rsqrt<half>(self,self,count)直接改写，无额外UB和输出tensor。
空shape返回true；shape/容量不合法返回false且未写入。只处理有效count，padding不参与。
调用方需提前保证输入就绪（如GM->UB后的MTE2_V），搬出前需建立V_MTE3依赖。

2. Host接口（链接libinplace_rsqrt.a和AscendCL）
  #include "inplace_rsqrt_api.h"
  aclError status = InplaceRsqrtFp16(deviceSelf, N4, N3, N2, N1, stream);
调用前由上层初始化ACL并设置当前设备，deviceSelf指向足够大小的FP16设备内存。
该接口是同步的：启动后等待stream完成并返回同步状态；不接管stream或设备内存所有权。
零元素无需设备/stream/指针，直接成功；非零输入校验指针非空及half对齐。
API只校验shape/指针，不把GM逐元素读回检查输入域；输入有限正数由调用方保证。
main.cpp会在H2D之前对二进制FP16位模式做输入域检查。

五、实现和资源
将4D连续ND展平成uint64元素数，Host逐步检查乘法与size_t字节容量。
查询实际Vector核数，blocks=min(Vector核数,ceil(elements/4096))。
每个Vector核处理tile=blockIdx,blockIdx+blocks,...，各自负责不重叠的GM字节区间。
每核只申请一段4096*2=8192字节VECIN UB；不是把整个大张量放入UB。
DataCopyPad只搬有效字节，不越读/越写尾块；LocalTensor中直接调用FP16 Rsqrt，
只有tile调度循环，没有设备逐元素标量GetValue/SetValue计算。
MTE2_V保证搬入后可计算，V_MTE3保证计算后可搬出，MTE3_S保证搬出完成再复用UB。
单GM allocation原址写回，无独立输出设备buffer、Workspace、AtomicAdd或核间同步。
Host检查ACL初始化/设备/stream/分配/拷贝/启动同步错误，RAII异常路径也尝试清理。
同步API返回错误后调用方不得将部分写入的输入视为完整结果；原位失败无法自动还原输入。

六、Windows本地可做的检查
  python -m pip install -r requirements.txt
  python -B test_tools.py
这些测试覆盖生成器/参考/校验器、错误退出码、33组工具数据往返、shape溢出、
63组核数/尾块地址模型。CPU模拟输出文件明确命名cpu_rounded_reference.bin，
从未把它标为NPU结果；C++/ASC未在本地编译，NPU suite未运行。

原BatchMatmulMaxSum源码、Golden、官方脚本、评分及best-known引用均不修改。

官方API资料：
Rsqrt（native half、原位重叠约束、正输入/近似误差）
https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0030.html
aclrtGetDeviceInfo / Vector core查询
https://www.hiascend.com/doc_center/zh/CANNCommunityEdition/900/API/runtimeapi/aclcppdevg_03_1876.html
