# op_01_sub

提交团队: 四川省明厚天信息技术股份有限公司
提交者: flowerElapse


## 算子实现介绍

sub_custom&lt;T&gt; 模板（float / half），按 numel 展平为一维；numBlocks = min(availableCoreNum, ceil(16384/1024))，每核 1024 元素；TPipe + │
  │ 实现方式 │  双缓冲 TQue&lt;VECIN/VECOUT&gt;，DataCopyPad 搬入、AscendC::Sub、DataCopyPad 搬出，含尾块处理；run_kernel 按 info_x.tensors[0].dtype
