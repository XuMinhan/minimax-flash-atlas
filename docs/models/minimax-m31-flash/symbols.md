Document-Kind: model-knowledge
Authoring: agent-generated, source-grounded
Model: MiniMax-M3.1-Flash
Source-Revisions: HF 262ae6830934f3512211081b8a3f16425b93a239; vLLM ced6857afa0ea7b2e3f0846a62e1394e90f15607
Content-Status: draft

# 符号、轴与数值

| 符号 | 含义 | Flash 值 / 单位 |
|---|---|---|
| B | batch 内序列数 | 随请求变化 |
| S | 当前 query/token 序列长度 | prefill 可多 token，普通 decode 常为 1 |
| T | 含历史的 KV 总长度 | 包括已缓存上下文 |
| Nq | 运行时 packed query token 总数 | Σ 当前 query lengths；不是 Q 头数 |
| H | 残差流宽度 | 6144 |
| Nₕ | Query heads | 64 |
| Nₖᵥ | KV heads | 4 |
| G | 每 KV 组的 Query heads | 16 |
| Dₕ | 主 head dim | 128 |
| N_idx | Index Query heads | 4 |
| D_idx | 共享 Index Key 向量 / 索引头宽 | 128 |
| Dᵣ | 旋转维数 | 64；余下64直通 |
| θ_base | RoPE 频率基数 | 10000000 |
| ε | RMSNorm epsilon | 1e-6 |
| γ | Gemma Norm 存储参数 | 实际缩放为 1+γ |
| TP / r | 张量并行数 / TP rank | r 只用于TP rank；rot表示旋转 |
| b / g / h / i / j / u | 请求 / 本地Index组 / 本地Q头 / 当前query / 历史token位置 / 逻辑块id | 不能混用 |
| R / b_corr | Router logits / expert correction bias | 与请求编号b不同 |
| Nₖᵥ,rank / N_idx,rank | 每 rank 的 KV / 索引 Query 头数 | max(1,4/TP) |
| B_block | 每逻辑注意力块的 token 数 | 128，不能与 Dₕ 混用 |
| K_block | 每 query/group 所选块上限 | 16，包含 local |
| K_sel | 候选块内可读 token 数 | 上界2048，仍受因果和有效长度约束 |
| E / K | routed 专家总数 / 每 token 选择数 | 128 / 4 |
| E_shared | 始终执行的共享专家 | 1 |
| H_dense | Dense FFN 宽 | 12288 |
| H_expert / H_shared | 专家 FFN 宽 | 3072 / 3072 |
| α / β / c | SwiGLU-OAI 门控参数 | 1.702 / 1 / 7 |
| s_route | routed scaling factor | 2 |
| V | 词表大小 | 200064 |
| Hᵥ / Nᵥ | 视觉 hidden / heads | 1280 / 16 |
| W | 权重矩阵 | PyTorch [out,in]；逻辑与存储 shape 分别标注 |
| ⊙ / σ | 逐元素乘 / sigmoid | 不含跨 token 混合 |
| 𝓑 | 有效选中逻辑块id集合 | Top-16输出剔除-1 |
| 𝓒 | 𝓑展开的历史token位置集合 | floor(j/128)∈𝓑且0≤j<T_b |
| 𝓥 | 因果有效token子集 | j∈𝓒且j≤p_bi |
| 𝒦idx / 𝒱_cache | Index-K cache / 主V cache | 与token集合𝓥不同 |
| G_rank | 本地Q/KV分组比 | (64/TP)/max(1,4/TP) |
| C / Sidx / p_bi / T_b | 块分数 / Index token分数 / query位置 / 请求KV总长 | Index点积无缩放 |
| 𝓛_i / 10²⁹ | query的local块 / 有限优先值 | local=1, init=0 |

图示按语义定义符号，不将数值都为 128 的 head dim、index dim、block size 当成同一概念。输入/权重绑定同时显示符号和实际值。量化 packed shape 应先解包再进行逻辑矩阵乘法分析。

## 视觉与草稿模块符号

| 符号 | 含义与范围 |
|---|---|
| N_patch / H_img / W_img | patch数 / 图像高宽；空间公式按一个时间patch组写，视频还乘时间组数 |
| pixels / X_v | 预处理的像素patch / 1280维视觉特征 |
| P / P_merged | 6144维Projector特征 / 同时间组2×2压缩后特征 |
| E_text / X_0 | text embedding / 融合后的语言输入 |
| W_LM / logits | [200064,6144]输出权重 / 词表分数 |
| 𝓗 / h⁽l⁾ | 官方配置选择的目标层特征集合 / 层id为l的特征，具体采集时序未验证 |
| B_d | DSpark block size=7，不是batch数 |
| r_Markov | Markov head rank=256，不是TP rank |
| Draft / target verify / accept-reject | 草稿、目标验证、接受拒绝的算法概念；本页仅配置证据，具体执行未验证 |

## TP与当前/历史长度对照

主/Index当前张量的token轴为S；历史cache的逻辑token轴为T。真实runner以N_q打包不同请求，图中的[B,S,…]是逻辑视图。

| TP | Q头/rank | KV头/rank | Index Q头/rank | Index K头 | packed宽 | G_rank |
|---|---|---|---|---|---|---|
| 1 | 64 | 4 | 4 | 1 | 9856 | 16 |
| 2 | 32 | 2 | 2 | 1 | 4992 | 16 |
| 4 | 16 | 1 | 1 | 1 | 2560 | 16 |
| 8 | 8 | 1 | 1 | 1 | 1536 | 8 |

例如已有1024个token的chunked prefill当前S=32、总T=1056；后续普通decode当前S=1、总T=1057。TP=8时current K为[B,1,S,128]，历史K的逻辑视图为[B,1,T,128]，不能把Norm输入的S换成T。通用Triton主cache物理布局为[Npages,1,128,256]，K/V共用末维前/后128；它不是两套独立分配的[Npages,128,1,128]张量。Index side cache独立分配且只有K。

## MoE 专家内部符号

- **x / Û**：一个 token 的 Post-attn RMSNorm 输出，[6144]；逐 token 应用
- **R / s / b_corr**：128 个 FP32 logits / sigmoid 路由分数 / 仅用于选择的 correction bias
- **𝓔 / e / j**：每 token 选中的 4 个专家集合 / 集合内专家索引
- **ŵ_e**：原 sigmoid 分数归一化再乘 2 得到的混合权重
- **W1 / W3 / W2**：专家 Gate / Up / Down 逻辑矩阵；NVFP4 存储另见权重详情
- **g / u / z**：Gate 投影 / Up 投影 / SwiGLU-OAI 激活特征，各 [3072]
- **c / α / β**：截断阈值 7 / sigmoid 斜率 1.702 / Up 常量偏移 1
- **σ / ⊙**：逐元素 sigmoid / 逐元素乘，不是跨专家归一化
- **y_e / Y_routed**：单个路由专家输出 / 四个专家加权和，各 [6144]
- **下标 s / Y_shared**：共享专家 / 无需 Top-4 与路由权重的共享输出
