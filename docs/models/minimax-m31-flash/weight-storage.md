Document-Kind: model-knowledge
Authoring: agent-generated, source-grounded
Model: minimax-m31-flash
Source-Revisions: HF 262ae6830934f3512211081b8a3f16425b93a239; vLLM ced6857afa0ea7b2e3f0846a62e1394e90f15607
Content-Status: draft

# 权重与存储：各层口径

页面入口为“权重与存储总结”。按模块、运算、tensor 用途与格式列出全部权重及量化辅助数据；另有主模型和 DSpark 两个统计范围。

## 统计方法与范围

使用 HTTP Range 只读取 48 个主模型 safetensors 文件和 1 个 DSpark 文件的长度字段/JSON header，没有下载 tensor payload。主模型 89,632 个 key 与原索引一一对应，DSpark 97 个 key 与独立索引一致。每个 tensor 的 bytes=data_offsets[1]−data_offsets[0]；shape×dtype 字节宽度也逐项核对。每个 key 只计一次。

主模型数据区总量 250,103,411,200 B（250.1034112 GB，约 232.926953 GiB）；其中非 scale 张量 223,953,094,144 B，scale 辅助数据 26,150,317,056 B。独立 DSpark 数据区 2,301,852,674 B，含 52,101,120 B scale 数据。文件头、JSON 索引未计入。GB=10^9 B，GiB=2^30 B，MiB=2^20 B。

`public/flash-data/tensor-shapes.json`、`dspark-tensors.json` 包含完整存储元数据；`weight-summary.json` 为汇总，`header-provenance.json` 为固定来源/header hash。`node scripts/build-weight-summary.mjs` 从已提交元数据重建汇总，无需联网。

这些数字是编码权重及辅助张量原样驻留的字节基数，不是 GPU 显存峰值。实际加载可能分片、复制、重排、padding、转换，此外有主 KV/Indexer cache、激活、通信和工作区。不能简单按总量除卡数。Top-4 是每 token 的专家激活路径；权重驻留范围由部署方式决定，不能理解为模型只需四个专家的权重。

## 形状分类

- **逻辑符号**：用于数学，如 W2:[H,I]，线性权重按 [输出轴,输入轴] 排列，计算为 xWᵀ。
- **逻辑尺寸**：H=6144、I=3072，得到 [6144,3072]；尚未指定存储位宽，也未拆 TP/EP。
- **checkpoint 存储尺寸**：文件头真实 shape，NVFP4 Down 为 U8 [6144,1536]。该末轴数的是字节容器，不是逻辑输入特征。
- **运行时布局**：QKV/W13 融合、TP/EP、kernel shuffle、page layout 等，单独描述；不将 /TP 混入 checkpoint 文件头尺寸。
- **激活的维度表达/配置代入**：B/S/TP 等仍是变量，不能称为某次运行“实际 shape”。vLLM 可以展平为 Nq。主 KV cache 物理页 [N_pages,N_kv_rank,B_block,2D_h] 中 128 是 block token 数，256 是 K/V 两半；Index K 的 [T,128] 只表示逻辑历史视图，不冒充 backend 页布局。
- 视觉 ViT 独立使用 D_v=1280、I_v=5120；其 Q/K/V/Out 权重为 [D_v,D_v]，不使用语言模型的 H/N_h/N_kv。共享专家 I_shared=I×n_shared_experts=3072。每个权重/示例逐项列出所用符号及数值。
- d0、d1 等是某张量依序的逻辑轴，数值由旁边逻辑尺寸逐轴给出；不按碰巧相同的数字替换为 H、D_h 等模型符号。

## NVFP4 两值打包

FP4 E2M1 的一个编码占 4 bit。U8 是 8-bit 容器，保存两个编码；每个编码分别解码为 FP4 值，结合 block/global scale 参与量化 GEMM。U8 容器不表示一个整数权重，操作也不是将 U8 数值除以 2。这里描述量化后的近似权重，不声称能恢复训练前的原始 BF16 数值。

沿逻辑输入轴 K 打包：[O,K] → U8 [O,K/2]。仅相对于“每值 1 字节”的数组，主 payload 字节数为一半；相对 BF16 2 字节/值则为四分之一，均未计 scale。逻辑通道数 K 不变。后端可能在内核中解码，不保证产生完整解包矩阵。

单个主模型路由专家的 Down：

| 对象 | 符号 | 尺寸 / dtype | 精确数据量 |
| --- | --- | --- | --- |
| 逻辑 W2 | [H,I] | [6144,3072]，18,874,368 个量化权重值 | 形状不单独指定字节数 |
| weight_packed | [H,I/2] | [6144,1536] U8 | 9,437,184 B = 9 MiB |
| weight_scale | [H,I/16] | [6144,192] FP8 E4M3 | 1,179,648 B = 1.125 MiB |
| weight_global_scale | 标量 | [] F32 | 4 B |
| input_global_scale | 标量，激活量化辅助 | [] F32 | 4 B |
| 四个 checkpoint 张量合计 | — | packed + scale | 10,616,840 B |

若物化为 BF16，单矩阵的条件大小是 36 MiB，不代表运行时一定这样做。SwiGLU 将 [Gate;Up] 的 2I 个激活通道逐元素合为 I 个通道，与存储两个 FP4 编码/U8 是不同操作。W13 在模型加载时把 Gate/Up 合并成 [2I,H]，又是另一层运行时融合。

Gate W1 与 Up W3 的逻辑形状均 [I,H]=[3072,6144]，各自 checkpoint 存储为 U8 [I,H/2]=[3072,3072]；Down 才是 [H,I/2]=[6144,1536]。每 16 个逻辑权重一个 block scale；不把 block scale 的形状当作神经网络中间宽度。

## MXFP8、BF16、F32

MXFP8 主矩阵每个 FP8 E4M3 值 1 B，逻辑与 checkpoint 矩阵轴长相同。每 32 个输入权重另有 scale，checkpoint 的 weight_scale_inv 为 U8 编码的 E8M0 指数，shape [O,K/32]。U8 dtype 在这里不表示 FP4 打包。Dense Down 的矩阵 [6144,12288] 与其 scale [6144,384] 属于不同 tensor，用途和大小分列。

BF16 为 2 B/值，F32 为 4 B/值。Router 的 logits 权重及 correction bias 使用 F32；视觉、embedding、LM Head、Norm 与共享专家等的精度以对应完整文件头为准，不将所有模块泛称为 FP4。Attention/Indexer/Dense/Shared 的 MXFP8 路径与未量化 Norm 分开统计。存储 dtype 不独自确定 GPU 算术/累加精度。

DSpark 的文件头已核实，执行内核/置信度/Markov 的具体运行语义仍未独立验证。页面以 key 标识这些张量，不把旧 MTP 逻辑替代它。

## 各模块数据区汇总

| 位置 / 用途 | tensor 数 | 精确 bytes | GiB |
| --- | --- | --- | --- |
| L3–59 路由专家 · Top-4 | 87,552 | 232,381,393,920 | 216.422038 |
| L0–59 主 Attention · Q/K/V/O | 600 | 6,617,856,000 | 6.163359 |
| LM Head · 词表 logits | 1 | 2,458,386,432 | 2.289551 |
| Token Embedding · 输入查表 | 1 | 2,458,386,432 | 2.289551 |
| L3–59 共享专家 · 始终参与 | 342 | 3,328,376,832 | 3.099792 |
| 视觉 ViT · 图像/视频编码 | 515 | 1,265,382,400 | 1.178479 |
| DSpark 草稿 · 独立 checkpoint | 97 | 2,301,852,674 | 2.143767 |
| Patch Merger · 合并空间 patch | 4 | 377,511,936 | 0.351585 |
| L0–2 Dense FFN · 非路由 MLP | 18 | 700,710,912 | 0.652588 |
| L0–59 Indexer · 选 KV blocks | 360 | 243,333,120 | 0.226622 |
| L3–59 Router · 专家选择 | 114 | 179,335,680 | 0.167019 |
| 视觉 Projector · 视觉维度映射 | 4 | 91,250,688 | 0.084984 |
| Decoder Norm · 残差归一化 | 121 | 1,486,848 | 0.001385 |

## 固定源码

- [NVFP4 MoE 权重、block/global/input scale 分配及后端转换](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py#L86-L245)。global scale 在转换处可取倒数，页面不简化为未经核实的统一乘法规则。
- [MXFP8 主矩阵与 block scale 分配](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_mxfp8.py#L44-L81)。
- [官方模型的固定权重目录](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/tree/262ae6830934f3512211081b8a3f16425b93a239)。
