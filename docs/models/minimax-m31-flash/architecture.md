Document-Kind: model-knowledge
Authoring: agent-generated, source-grounded
Model: MiniMax-M3.1-Flash
Source-Revisions: HF 262ae6830934f3512211081b8a3f16425b93a239; vLLM ced6857afa0ea7b2e3f0846a62e1394e90f15607
Content-Status: draft

# 架构与数据依赖

## 主链

输入文本 token 经 200064×6144 的 embedding；图像/视频的视觉 embedding 替换对应占位位置。二者以 [B,S,6144] 的残差流进入 Decoder。60 层之后为 Final Gemma RMSNorm 和独立 LM Head；tie_word_embeddings=false，输出 logits [B,S,200064]。

官方模型卡标称约 428B 总参数，约 23B/token 激活参数和最多 1M 上下文。精确 config.max_position_embeddings=1048576。checkpoint 主权重采用混合量化，250.10 GB 不是推理峰值显存。

## 两种 FFN，一个注意力族

| 层族 | 注意力 | FFN |
|---|---|---|
| L0–2 | MSA + per-head Norm + Partial RoPE | Dense SwiGLU-OAI，宽 12288 |
| L3–59 | 相同 MSA | 128 routed experts，Top-4 + 1 shared，宽 3072 |

moe_layer_freq 与 sparse_attention_freq 是独立配置。后者 60 项全为 1，前者前三项 0、后 57 项 1。

## MSA 完整算子序列

1. 输入 Gemma RMSNorm，保留残差旁路。
2. 五路融合投影 [Q,K,V,Qidx,Kidx]。TP=1 时宽度分别 8192、512、512、512、128，共 9856。
3. 主 Q/K 逐头 Gemma RMSNorm、前 64/128 维 Partial NeoX RoPE；V 不旋转。rope_theta=10000000。
4. Index Q 为四头，Index K 为一份共享 128 维向量，同样做 Norm/RoPE；写入独立 Index-K side cache。
5. Index Q 与完整历史 Index K 点积，不做 1/√D 缩放；先排除未来 token，再对 128-token 块取 max。
6. 每 query、每 KV 组选择最多 16 个逻辑块；local_blocks=1、init_blocks=0。local 块占用总预算，无效槽为 -1。
7. 主 K/V 写入 paged cache。block_indices 决定逻辑块，block_table 映射物理页，slot_mapping 决定当前 token 写槽。
8. 主稀疏 kernel 直接读取页：QK/√128 → token 级 causal/pad bounds → online softmax → PV；不物化整幅 scores 或 selected K/V。
9. 每 16 个 Q 头共享一个 KV 组。拼接 64×128=8192 后 O projection 映回 6144。
10. Attention residual merge → post-attention Norm → 对应 FFN → 最终 residual merge。

两套缓存均保留历史：稀疏的是每 query 的读取范围，并非仅保存所选块。TP>4 时主 KV / Index Q heads 按 vLLM 复制规则处理；Index K 仍为每 rank 一份向量。

## MoE

Router 使用 FP32 logits。sigmoid 分数加 correction bias 只用于选择 Top-4；混合权重从未加 bias 的分数 gather，归一化后乘 2。共享专家始终执行。三个专家矩阵按 w1=gate、w3=up、w2=down 装载。FFN 激活为 gate 单边裁剪、up 双边裁剪，再 g·sigmoid(1.702g)·(u+1)。

每个入选专家独立执行 Gate/Up 两路投影（6144→3072）、SwiGLU-OAI 非线性门控、Down（3072→6144），然后四个专家输出分别乘路由权重求和。SwiGLU-OAI 先计算 ḡ=min(g,7)、ū=clip(u,−7,7)，再计算 z=ḡ⊙σ(1.702ḡ)⊙(ū+1)；Gate 只截上界。α 控制 sigmoid 斜率，β=1 是常量而非可训练 bias。共享专家使用独立权重与同一激活，不参与 Top-4，不乘路由混合权重。

新版原生展开图把这些步骤分别设为可点击节点。公式中的形状是单 token 全局逻辑形状；TP/EP 本地布局与融合方式由后端决定。NVFP4 packed 的 U8 存储形状不等于上述逻辑矩阵形状。激活内部的 sigmoid 调制特征通道，Router 的 sigmoid 则用于专家选路，二者不能混同。

本地实现将 shared_experts 交给 FusedMoEFactory，图中的独立共享分支和 Add 是数学表达，不宣称 Python 单独调用或分配相同中间张量。

## 视觉

动态像素 → Conv3D kernel/stride (2,14,14) → 32 层 ViT（hidden1280、16heads、FFN5120、3D RoPE）→ 两层 projector 1280→6144→6144 → 2×2 patch merger 拼接 24576→6144→6144 → embedding fusion。参考 672² 图像空间上产生 48² patches，merge 后 576 token；视频/动态分辨率依预处理变化。

## DSpark

独立 DSparkMiniMaxDraftModel：5 层、hidden6144、dense FFN12288、sliding window4096；目标层 ids [3,17,31,45,59]，block size7，Markov rank256，置信度头启用，use_mtp=false。图只解释配置可核实的结构及一般候选验证关系，不推断未阅读的具体采样实现，也不声称 7 个候选必定全接受。
