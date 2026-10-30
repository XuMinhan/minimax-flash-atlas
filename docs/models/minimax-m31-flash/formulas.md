Document-Kind: model-knowledge
Authoring: agent-generated, source-grounded
Model: MiniMax-M3.1-Flash
Source-Revisions: HF 262ae6830934f3512211081b8a3f16425b93a239; vLLM ced6857afa0ea7b2e3f0846a62e1394e90f15607
Content-Status: draft

# 算子公式与维度

图中 QK、scale、mask、softmax、PV 是融合内核的数学分解；不表示独立分配这些中间张量。权重以 PyTorch [out,in] 记号显示。

## s-input · Hidden states

本层逻辑 hidden/residual 输入；首层来自 embedding，后续层由上一层 residual 与 FFN partial 流共同表示。

输入：`logical [B,S,6144]; runtime hidden + residual streams` → 输出：`2 × [B,S,6144]`。

$$
X_l=\operatorname{LogicalMerge}(\mathrm{hidden\_states},\mathrm{residual})\in\mathbb R^{B\times S\times H}
$$

X_l 是逻辑全局输入。PP=1 的延迟归并路径由下一 input norm 先 all-reduce FFN partial，再加 residual；这里不执行投影。

- **X_l**：逻辑本层输入
- **hidden_states / residual**：运行时两条流；若FFN延迟归并则先all-reduce
- **B/S/H**：请求数 / 当前token长度 / hidden6144

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L754-L790)。

## s-position · Build Position IDs

由请求已计算长度与本轮 token 偏移构造位置，供主与 Index 分支 RoPE 及因果判断使用。

输入：`[B] + [Nq]` → 输出：`[Nq]`。

$$
\begin{aligned}q_b&=\mathrm{num\_scheduled\_tokens}[b]\\p_{b,i}&=\mathrm{num\_computed\_tokens}[b]+i,\quad 0\le i<q_b\\\mathbf p&=\operatorname{concat}_{b=1}^{B}(p_{b,0},\ldots,p_{b,q_b-1})\in\mathbb Z^{N_q}\end{aligned}
$$

b 表示请求编号；本轮 q_b 个 token 被打包为 N_q=Σq_b；S 是单请求当前长度，T 是含历史的总长度。

- **q_b**：请求 b 本轮调度的 query 数
- **p_b,i**：请求 b 的第 i 个 token position
- **N_q**：本轮 query token 总数
- **B**：batch size

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu_model_runner.py)。

## s-attnmeta · Build Attention Metadata

保存每个请求的 query 起点、总序列长度及因果约束，供选块和主 attention 使用。

输入：`[B+1] · [B] · bool` → 输出：`backend metadata; 非稠密 [S,T]`。

$$
\begin{aligned}q_b&=\mathrm{query\_start\_loc}_{b+1}-\mathrm{query\_start\_loc}_b\\c_b&=\mathrm{seq\_len}_b-q_b\\M_{b,i,j}&=\begin{cases}0,&0\le j\le c_b+i\\-\infty,&\text{otherwise}\end{cases}\end{aligned}
$$

公式表示数学可见性，不表示运行时分配 [S,T] 稠密 mask；c_b=T_b-q_b。

- **q_b**：请求 b 的 query 数
- **c_b**：请求 b 已有 context 长度
- **M**：causal / padding mask
- **−∞**：不可见位置的加性 mask 值

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu_model_runner.py)。

## s-slots · Resolve KV Slots

将请求的逻辑位置转换为缓存物理槽；读缓存时同样通过 block_table 解析物理页。

输入：`[Nq] + [B,Nblocks]` → 输出：`[Nq] + [B,Nblocks]`。

$$
\begin{aligned}\ell&=\lfloor p/B_{block}\rfloor,\quad o=p\bmod B_{block}\\b_{phys}&=\mathrm{block\_table}[b,\ell]\\\mathrm{slot}(b,p)&=b_{phys}B_{block}+o\end{aligned}
$$

b 是请求编号，ℓ 是逻辑块，b_phys 是物理页；主缓存与 side cache 使用各自 metadata/slot mapping。

- **p**：token position
- **B_block**：KV cache block size · runtime
- **b_phys**：物理 block id
- **block_table**：逻辑块到物理页映射
- **slot**：KV cache 物理槽位

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu_model_runner.py)。

## s-norm · Gemma RMSNorm

对本层逻辑 residual 输入执行 Gemma RMSNorm，输出供五路投影使用。

输入：`[B,S,6144]` → 输出：`[B,S,6144]`。

$$
\begin{aligned}\operatorname{RMS}(x)&=\sqrt{\frac1H\sum_{j=1}^{H}x_j^2+\varepsilon}\\y_i&=\frac{x_i}{\operatorname{RMS}(x)}(1+\gamma_i)\end{aligned}
$$

缩放为 1+γ，ε=1e-6。若上一层 FFN 延迟归并，实际 input norm 调用同时完成 all-reduce 与 residual add。

- **x**：hidden_states · [B,S,H]
- **y**：normalized hidden_states
- **H**：hidden_size = 6144
- **γ**：input_layernorm.weight
- **ε**：rms_norm_eps = 10⁻⁶
- **RMS(x)**：√(Σxⱼ²/H + ε)

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L130-L142)。

## s-packed · QKV + Index Projection

一次 column-parallel GEMM 产生本 rank 的 Q、K、V、Index Q 与共享 Index K。

输入：`[B,S,6144]` → 输出：`[B,S,8192/TP+3·max(1,4/TP)·128+128]`。

$$
Z=\hat X[W_Q^\top\mid W_K^\top\mid W_V^\top\mid W_{Q_i}^\top\mid W_{K_i}^\top]
$$

局部宽度为 8192/TP + 2·max(1,4/TP)·128 + max(1,4/TP)·128 +128。TP=1 时总宽9856；此节点不做 Norm/RoPE。

- **Nₕ**：num_attention_heads = 64
- **Nₖᵥ**：num_key_value_heads = 4
- **N_idx**：sparse_num_index_heads = 4
- **Dₕ**：head_dim = 128
- **D_idx**：sparse_index_dim = 128
- **TP**：tensor parallel size

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/layers/linear.py#L1396-L1431)。

## s-split · Split 5 outputs

将 packed projection 解释为五段本地张量；这里只描述切分视图。

输入：`[B,S,8192/TP+3·max(1,4/TP)·128+128]` → 输出：`Q:[B,64/TP,S,128]; K,V:[B,max(1,4/TP),S,128]; Qidx:[B,N_idx,rank,S,D_idx]; Kidx:[B,S,128]`。

$$
Z\longrightarrow(Q_{N_hD_h/TP},K_{N_{kv,\mathrm{rank}}D_h},V_{N_{kv,\mathrm{rank}}D_h},Q^{\mathrm{idx}}_{N_{idx,\mathrm{rank}}D_{idx}},K^{\mathrm{idx}}_{D_{idx}})
$$

主 Q 为64/TP头，KV与Index Q各max(1,4/TP)头；Index K始终一个128维向量。全局TP=1宽度8192/512/512/512/128。

- **Nₕ/TP**：每 rank query heads = 64/TP
- **Nₖᵥ,rank**：max(1,4/TP)
- **N_idx,rank**：max(1,4/TP)
- **Dₕ**：head_dim = 128
- **D_idx**：sparse_index_dim = 128

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/layers/linear.py#L1396-L1431)。

## s-idxnorm · Index Q/K Gemma RMSNorm + RoPE

对当前 Index Q/K 逐头执行 Gemma RMSNorm 和前64维 Partial RoPE。

输入：`[B,N_idx,rank,S,D_idx] · [B,S,D_idx]` → 输出：`[B,N_idx,rank,S,D_idx] · [B,S,D_idx]`。

$$
\begin{aligned}\tilde Q^{\mathrm{idx}},\tilde K^{\mathrm{idx}}&=\operatorname{RMSNorm}(Q^{\mathrm{idx}}),\operatorname{RMSNorm}(K^{\mathrm{idx}})\\Q^{\mathrm{idx},rot},K^{\mathrm{idx},rot}&=\operatorname{RoPE}(\tilde Q^{\mathrm{idx}},\tilde K^{\mathrm{idx}};\mathbf p)\end{aligned}
$$

Index Q 为本 rank 的 max(1,4/TP) 头；Index K 每 token 单份。两者当前长度均为 S，历史长度 T 只出现在 cache 中。

- **Qidx/Kidx**：当前本地Index Q与单份Index K
- **rot**：前64维Partial RoPE后的标记
- **p**：当前token位置
- **N_idx,rank**：max(1,4/TP)

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L615-L640)。

## s-idxcache · Index K Cache · key-only

增量保存当前旋转后的 Index K，供所有本地索引头读取完整历史。

输入：`[B,S,128] + [Nq]` → 输出：`key-only pages · [T,128]`。

$$
\mathcal K_{\mathrm{idx}}[\mathrm{slot}(b,p)]\leftarrow K^{\mathrm{idx},rot}_{b,p}\in\mathbb R^{D_{idx}}
$$

独立 MLAAttentionSpec：num_kv_heads=1，每 token 128维；bind 将[Npages,1,128,128]压为[Npages,128,128]。这是 key-only cache，无 Index V。

- **Kidxᵣ**：当前 token 的旋转后 Index K
- **slot**：index slot_mapping 指定的 side-cache 位置
- **𝒦idx**：独立的 key-only Index K cache
- **D_idx**：每 token 一个 128 维 Index K 向量

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/common/indexer.py#L139-L153)。

## s-idxscore · Index Q × cached Kᵀ

将本轮 Index Q 与 side cache 中的历史 Index K 做无缩放点积。

输入：`[B,max(1,4/TP),S,128] · [B,T,128]` → 输出：`[B,max(1,4/TP),S,T]`。

$$
S^{idx}_{b,g,i,j}=\langle Q^{idx,rot}_{b,g,i,:},\mathcal K_{idx}[b,j,:]\rangle
$$

g 是本 rank 的 Index/KV group，非 TP rank；Sidx 是评分张量，j 是历史 token 位置。评分在融合 kernel 内形成。

- **Qidxᵣ**：当前 rank 的 Index Q query
- **𝒦idx**：独立 side cache 中的完整 Index K history
- **N_idx,rank**：max(1,4/TP)
- **D_idx**：index_dim = 128
- **Sidx**：尚未 mask 的 Index token scores
- **b / g / i / j / u**：请求 / 本地索引组 / 当前query / 历史token位置 / 逻辑块id

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/common/ops/index_topk.py#L145-L163)。

## s-idxmask · Mask Future Index Keys

在块内 max 之前将未来 key 与无效补齐槽设为 −∞。

输入：`[B,max(1,4/TP),S,T] + [B,S]` → 输出：`[B,max(1,4/TP),S,T]`。

$$
\tilde S^{idx}_{b,g,i,j}=\begin{cases}S^{idx}_{b,g,i,j},&0\le j<T_b\ \land\ j\le p_{b,i}\\-\infty,&\text{otherwise}\end{cases}
$$

因果位置 p_bi 和有效长度约束必须先于 block max；不使用主 attention 的1/√128缩放。

- **Sidx**：Index Q/K 点积分数
- **p**：当前 query position
- **j**：key position
- **S̃idx**：未来 key 被置为 −∞ 后的分数
- **b / g / i / j / u**：请求 / 本地索引组 / 当前query / 历史token位置 / 逻辑块id

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/common/ops/index_topk.py#L145-L163)。

## s-blockmax · Block Max · B_block keys

对每128个连续历史 key 的有效 Index 分数取最大值，形成块分数。

输入：`[B,max(1,4/TP),S,T]` → 输出：`[B,max(1,4/TP),S,⌈T/128⌉]`。

$$
C_{b,g,i,u}=\max_{B_{block}u\le j<B_{block}(u+1)}\tilde S^{idx}_{b,g,i,j}
$$

C 是块分数；u 为逻辑块 id，g 为索引组。B_block=128，与向量维D_idx=128是独立概念。

- **B_block**：sparse_block_size = 128
- **S̃idx**：已排除未来 key 的 Index scores
- **C**：每128 keys取max的块分数
- **b / g / i / j / u**：请求 / 本地索引组 / 当前query / 历史token位置 / 逻辑块id

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/common/ops/index_topk.py#L145-L163)。

## s-topk · Top-K_block Blocks · per group

每个当前 query、每个本地 Index/KV group 选择至多16个逻辑块。

输入：`[B,max(1,4/TP),S,Nblocks]` → 输出：`[B,max(1,4/TP),S,16]`。

$$
\begin{aligned}C_u&\leftarrow10^{29},\quad u\in\mathcal L_i\\\mathcal B_{b,g,i}&=\{u\in\operatorname{TopK}_{K_{block}}(C):u\ne-1\}\\\mathcal C_{b,g,i}&=\{j:0\le j<T_b,\lfloor j/B_{block}\rfloor\in\mathcal B_{b,g,i}\}\\\mathcal V_{b,g,i}&=\{j\in\mathcal C_{b,g,i}:j\le p_{b,i}\}\end{aligned}
$$

local=1、init=0；local块以有限1e29分数优先保留在16槽中。无效槽写-1，-1不属于选中块集合。

- **C**：block scores
- **𝓛ᵢ**：当前 query 的 local block 集合
- **10²⁹**：local块的有限优先分数
- **K_block**：sparse_topk_blocks = 16
- **𝓑 / block_indices**：有效选中块id，不包含-1
- **𝓒**：选中块展开且未越过T的token位置
- **𝓥**：𝓒内满足因果约束的token位置
- **b / g / i / j / u**：请求 / 本地索引组 / 当前query / 历史token位置 / 逻辑块id

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/common/ops/index_topk.py#L229-L245)。

## s-mainnorm · Main Q/K Gemma RMSNorm

分别归一化当前主 Q 与当前主 K 的每个 head；不重新归一化已缓存历史 K。

输入：`[B,64/TP,S,128] · [B,max(1,4/TP),S,128]` → 输出：`[B,64/TP,S,128] · [B,max(1,4/TP),S,128]`。

$$
\tilde Q=\operatorname{RMSNorm}(Q),\qquad\tilde K=\operatorname{RMSNorm}(K)
$$

Q 当前形状[B,64/TP,S,128]，K 当前形状[B,max(1,4/TP),S,128]。两者以1+γ缩放，长度都是S。

- **Q/K**：当前S个token的主Q/K，非历史cache
- **RMSNorm**：逐128维head的Gemma Norm，ε=1e-6
- **Q̃/K̃**：归一化输出

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L130-L142)。

## s-rope · Partial RoPE

使用位置向量旋转当前 Q/K 每个 head 的前64维，其余64维直通。

输入：`Q/K + [S]` → 输出：`Q/K unchanged`。

$$
\begin{aligned}(Q^{rot}_{:D_r},K^{rot}_{:D_r})&=\operatorname{RoPE}(\tilde Q_{:D_r},\tilde K_{:D_r};\mathbf p)\\(Q^{rot}_{D_r:},K^{rot}_{D_r:})&=(\tilde Q_{D_r:},\tilde K_{D_r:})\end{aligned}
$$

旋转标记 rot 与 TP rank r 不同。NeoX配对；θ_base=10000000；V不旋转。

- **Dᵣ**：rotary_dim = 64
- **Dₕ**：head_dim = 128
- **θbase**：rope_theta = 10000000
- **p**：token position

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L615-L640)。

## s-cache · Paged KV Cache

把本轮旋转 K 和未旋转 V 写入主 paged cache，历史缓存保留。

输入：`2 × [B,max(1,4/TP),S,128] + runtime metadata` → 输出：`[Npages,max(1,4/TP),128,256]`。

$$
\mathcal K[\mathrm{slot}(b,p)]\leftarrow K^{rot}_{b,p},\qquad\mathcal V_{cache}[\mathrm{slot}(b,p)]\leftarrow V_{b,p}
$$

通用Triton物理布局[Npages,N_kv,rank,128,256]，末维前128为K、后128为V；逻辑视图为K/V各[B,N_kv,rank,T,128]。后端专用布局不在此泛化。

- **Kᵣ / V**：写入 sparse cache 的 Key / Value
- **slot**：物理 cache 位置
- **block_table**：逻辑块到物理页映射

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L556-L565)。

## s-qk · Q × paged Kᵀ · Top-K_block

按所选逻辑块与 block_table 直接读主 K，计算当前 Q 的候选 token 点积。

输入：`[B,Nₕ/TP,S,Dₕ] + KV pages + [B,max(1,4/TP),S,16]` → 输出：`[B,Nₕ/TP,S,K_sel]`。

$$
A_{b,h,i,j}=\sum_{m=1}^{D_h}Q^{rot}_{b,h,i,m}K^{rot}_{b,\lfloor h/G_{rank}\rfloor,j,m},\quad j\in\mathcal C_{b,\lfloor h/G_{rank}\rfloor,i}
$$

𝓑是选中块id集合；𝓒由这些块展开为有效历史token位置；𝓥再施加因果约束。此处j∈𝓒，不把块id当token id。

- **Qᵣ**：当前 rank 的 rotated Q
- **paged K**：主 Paged KV Cache 中的 K pages
- **block_indices**：Indexer 输出的 Top-16 逻辑块索引
- **Ksel**：kernel 最多读取 16×128 个候选 token
- **A**：当前 rank 的 sparse scores
- **𝓑 / 𝓒 / 𝓥**：选中块id / 展开的有效历史token / 因果有效子集
- **G_rank**：(64/TP) / max(1,4/TP)，本地Q/KV分组比
- **b / h / i / j / m**：请求 / 本地Q头 / query位置 / 历史token位置 / head通道

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/common/ops/sparse_attn.py#L140-L155)。

## s-scale · Scale 1/√Dₕ

将主 attention 的点积乘1/√128，保持候选 token 的对应关系。

输入：`[B,Nₕ/TP,S,K_sel]` → 输出：`same`。

$$
\bar A_{b,h,i,j}=A_{b,h,i,j}/\sqrt{D_h}
$$

这是主 attention 的缩放，Indexer dot 不执行该缩放。实际在融合kernel内计算。

- **A**：未缩放 sparse scores
- **Ā**：缩放后的 sparse scores
- **Dₕ**：head_dim = 128

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/common/ops/sparse_attn.py#L140-L155)。

## s-mask · Apply Token Causal / Pad Mask

在选中块展开的token候选中继续排除未来位置和padding。

输入：`[B,64/TP,S,Ksel] + runtime metadata` → 输出：`[B,64/TP,S,Ksel]`。

$$
\tilde A_{b,h,i,j}=\begin{cases}\bar A_{b,h,i,j},&j\in\mathcal V_{b,\lfloor h/G_{rank}\rfloor,i}\\-\infty,&\text{otherwise}\end{cases},\quad j\in\mathcal C_{b,\lfloor h/G_{rank}\rfloor,i}
$$

𝓥={j∈𝓒 | j≤p_bi 且j<T_b}；候选块选择不替代token mask。数学分解不声明独立mask张量。

- **Ā**：selected K 上的缩放 scores
- **valid_token**：候选 blocks 内的 causal / padding 判定
- **Ã**：token mask 后的 selected scores
- **𝓒 / 𝓥**：候选token集合 / 因果有效子集
- **𝓑 / 𝓒 / 𝓥**：选中块id / 展开的有效历史token / 因果有效子集
- **G_rank**：(64/TP) / max(1,4/TP)，本地Q/KV分组比
- **b / h / i / j / m**：请求 / 本地Q头 / query位置 / 历史token位置 / head通道

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/common/ops/sparse_attn.py#L140-L155)。

## s-softmax · Softmax

在当前 query/group 的有效候选token集合上归一化主分数。

输入：`[B,64/TP,S,≤2048]` → 输出：`same`。

$$
\begin{aligned}\mathcal V_i&=\mathcal V_{b,\lfloor h/G_{rank}\rfloor,i},\quad m_i=\max_{t\in\mathcal V_i}\tilde A_{b,h,i,t}\\P_{b,h,i,j}&=\frac{\exp(\tilde A_{b,h,i,j}-m_i)}{\sum_{t\in\mathcal V_i}\exp(\tilde A_{b,h,i,t}-m_i)},\quad j\in\mathcal V_i\end{aligned}
$$

分母对token位置𝓥求和，非对块id求和；代码采用online softmax，包含max稳定化。

- **Ã**：mask 后的 sparse scores
- **P**：selected KV 上的概率
- **𝓥**：当前query/group有效候选token集合
- **𝓑 / 𝓒 / 𝓥**：选中块id / 展开的有效历史token / 因果有效子集
- **G_rank**：(64/TP) / max(1,4/TP)，本地Q/KV分组比
- **b / h / i / j / m**：请求 / 本地Q头 / query位置 / 历史token位置 / head通道

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/common/ops/sparse_attn.py#L140-L155)。

## s-pv · P × paged V · same Top-K_block

用有效候选token的概率加权同一块顺序读取的V，得到本地Q heads。

输入：`[B,Nₕ/TP,S,K_sel] + KV pages` → 输出：`[B,S,NₕDₕ/TP]`。

$$
O_{b,h,i,m}=\sum_{j\in\mathcal V_{b,\lfloor h/G_{rank}\rfloor,i}}P_{b,h,i,j}V_{b,\lfloor h/G_{rank}\rfloor,j,m}
$$

求和索引j∈𝓥；实际内核直接读取V页，不独立物化selected V；输出头宽8192/TP。

- **P**：当前 rank 的 sparse probability
- **paged V**：主 Paged KV Cache 中的 V pages
- **block_indices**：与 Q×K 阶段相同的 Top-16 块顺序
- **O**：当前 rank 的 context heads · 8192/TP
- **𝓑 / 𝓒 / 𝓥**：选中块id / 展开的有效历史token / 因果有效子集
- **G_rank**：(64/TP) / max(1,4/TP)，本地Q/KV分组比
- **b / h / i / j / m**：请求 / 本地Q头 / query位置 / 历史token位置 / head通道

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/common/ops/sparse_attn.py#L140-L155)。

## s-oproj · O Projection

本 rank 的拼接heads经RowParallelLinear产生H=6144宽的partial结果。

输入：`[B,S,8192/TP]` → 输出：`[B,S,6144]`。

$$
\begin{aligned}Y_{attn}^{(r)}&=\operatorname{Concat}(O^{(r)})(W_O^{(r)})^\top\\Y_{attn}&=\sum_rY_{attn}^{(r)}\quad\text{(next post-attention norm)}\end{aligned}
$$

reduce_results=False；跨TP rank归并实际融合到随后post-attention norm。Σ_r只是逻辑全局Yattn，不是此projection内执行的all-reduce。

- **Nₕ**：num_attention_heads = 64
- **Dₕ**：head_dim = 128
- **H**：hidden_size = 6144
- **TP**：tensor parallel size
- **W_O**：o_proj.weight
- **r / Yattn⁽ʳ⁾ / Yattn**：TP rank / 本地partial / 在post-attn norm归并的逻辑结果

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L452-L462)。

## s-addattn · Attention Residual Merge

逻辑上把全局attention结果与本层residual相加得到U。

输入：`2 × [B,S,6144]` → 输出：`[B,S,6144]`。

$$
U=X_l+Y_{\mathrm{attn}}
$$

实际fused_allreduce_gemma_rms_norm先归并O projection的partial，再add并做post-norm；图将逻辑Add与Norm拆开说明。

- **Xₗ**：进入本层的 residual stream · [B,S,H]
- **Yattn**：Sparse Attention 输出 · [B,S,H]
- **U**：更新后的 residual stream
- **H**：hidden_size = 6144

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L754-L790)。

## s-postnorm · Post-attn Gemma RMSNorm

对逻辑U执行Gemma RMSNorm，输出Û供Dense FFN或MoE使用。

输入：`[B,S,6144]` → 输出：`[B,S,6144]`。

$$
\begin{aligned}\operatorname{RMS}(U)&=\sqrt{\frac1H\sum_{j=1}^{H}U_j^2+\varepsilon}\\\hat U_i&=\frac{U_i}{\operatorname{RMS}(U)}(1+\gamma_{\mathrm{post},i})\end{aligned}
$$

数学式仅表示norm；实际调用与attention partial all-reduce和residual add融合，不能将图中分解当作独立kernel。

- **U**：上游 Add 节点的输出
- **Û**：Gemma RMSNorm(U)
- **H**：hidden_size = 6144
- **γpost**：post_attention_layernorm.weight
- **ε**：rms_norm_eps = 10⁻⁶
- **RMS(U)**：√(ΣUⱼ²/H + ε)

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L130-L142)。

## d-gateup · Gate + Up Projection

每个TP rank读取完整Û，分别计算局部gate/up两路线性投影。

输入：`[B,S,6144]` → 输出：`[B,S,24576/TP]`。

$$
\begin{aligned}G^{(r)}&=\hat U\left(W_{\mathrm{gate}}^{(r)}\right)^\top\\U^{(r)}&=\hat U\left(W_{\mathrm{up}}^{(r)}\right)^\top\\G^{(r)},U^{(r)}&\in\mathbb R^{B\times S\times(H_{\mathrm{dense}}/TP)}\end{aligned}
$$

W_gate/up沿输出维切分；每支宽12288/TP。此处不执行split、clamp或逐元素乘。

- **Û**：归一化输入 · [B,S,H]
- **Wgate⁽ʳ⁾**：当前 TP rank 的 gate 权重
- **Wup⁽ʳ⁾**：当前 TP rank 的 up 权重
- **G⁽ʳ⁾**：当前 rank 的 gate 投影
- **U⁽ʳ⁾**：当前 rank 的 up 投影
- **H**：hidden_size = 6144
- **H_dense**：dense_intermediate_size = 12288
- **TP**：tensor parallel size

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L156-L170)。

## d-gatesplit · Split Gate / Up

把当前rank的packed gate_up沿最后一维等分为G与U。

输入：`[B,S,24576/TP]` → 输出：`2 × [B,S,12288/TP]`。

$$
\begin{aligned}X^{(r)}&\in\mathbb R^{B\times S\times(2H_{\mathrm{dense}}/TP)}\\G^{(r)}&=X^{(r)}_{:,:,\,0:H_{\mathrm{dense}}/TP}\\U^{(r)}&=X^{(r)}_{:,:,\,H_{\mathrm{dense}}/TP:2H_{\mathrm{dense}}/TP}\end{aligned}
$$

前12288/TP通道为gate，后12288/TP为up；只切分视图，无权重、无数值计算。

- **X⁽ʳ⁾**：当前 rank 的 packed gate_up
- **G⁽ʳ⁾**：前 H_dense/TP 个通道
- **U⁽ʳ⁾**：后 H_dense/TP 个通道
- **H_dense**：dense_intermediate_size = 12288
- **TP**：tensor parallel size

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/layers/activation.py#L240-L244)。

## d-swiglu · SwiGLU-OAI

对gate做上界裁剪、up做双边裁剪，然后sigmoid、+β及逐元素乘。

输入：`2 × [B,S,12288/TP]` → 输出：`[B,S,12288/TP]`。

$$
\begin{aligned}\bar G^{(r)}&=\min(G^{(r)},c)\\\bar U^{(r)}&=\operatorname{clip}(U^{(r)},-c,c)\\Z^{(r)}&=\bar G^{(r)}\odot\sigma(\alpha\bar G^{(r)})\odot(\bar U^{(r)}+\beta)\end{aligned}
$$

ḡ=min(g,7)，ū=clip(u,-7,7)，sigmoid内部同样使用ḡ；α=1.702，β=1。

- **G⁽ʳ⁾**：当前 rank 的 gate 分片
- **U⁽ʳ⁾**：当前 rank 的 up 分片
- **Ḡ⁽ʳ⁾ / Ū⁽ʳ⁾**：clamp 后的两个分片
- **α**：swiglu_alpha = 1.702
- **β**：swiglu_beta = 1.0
- **c**：swiglu_limit = 7.0
- **Z⁽ʳ⁾**：当前 rank 的激活输出

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/layers/activation.py#L240-L244)。

## d-down · Down Projection

将本rank的12288/TP维激活投回H，输出本rank的partial。

输入：`[B,S,12288/TP]` → 输出：`[B,S,6144]`。

$$
\begin{aligned}Y_{ffn}^{(r)}&=Z^{(r)}(W_{down}^{(r)})^\top\in\mathbb R^{B\times S\times H}\\Y_{ffn}&=\sum_rY_{ffn}^{(r)}\quad\text{(deferred if PP=1)}\end{aligned}
$$

主模型PP=1设置reduce_results=False，归并延迟到下一层input norm或final norm；PP>1或旧MTP设置reduce_results=True。逻辑Yffn=Σ_r partial。

- **Z⁽ʳ⁾**：当前 TP rank 的激活输出
- **Wdown⁽ʳ⁾**：当前 rank 的 down 权重
- **H_dense**：dense_intermediate_size = 12288
- **H**：hidden_size = 6144
- **TP**：tensor parallel size
- **r / Yffn⁽ʳ⁾ / Yffn**：TP rank / 本地partial / 跨rank和
- **PP**：pipeline parallel size，决定FFN归并是否延迟

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L156-L170)。

## d-add2 · Decoder Layer Residual Merge

逻辑上将全局Dense FFN结果加入U，得到下一层输入。

输入：`2 × [B,S,6144]` → 输出：`[B,S,6144]`。

$$
X_{l+1}=U+Y_{\mathrm{ffn}}
$$

PP=1实际返回FFN partial与residual两条流，下个input norm或final norm负责all-reduce/add；PP>1时FFN已归并。

- **U**：Attention 后的 residual stream · [B,S,H]
- **Yffn**：Dense FFN 输出 · [B,S,H]
- **Xₗ₊₁**：逻辑上的下一层输入
- **H**：hidden_size = 6144

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L754-L790)。

## s-router · FP32 Router Logits

把每个token的Û投影为128个FP32 router logits。

输入：`[B,S,6144]` → 输出：`[B,S,128]`。

$$
R=\hat U W_{router}^\top\in\mathbb R^{B\times S\times E}
$$

本节点不做sigmoid或top-k。Router logits在本节记为R，避免与TP rank r混淆。

- **Û**：Post-attn RMSNorm 输出
- **Wrouter**：block_sparse_moe.gate.weight
- **E**：num_local_experts = 128
- **R**：router_logits · [B,S,E]

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L225-L234)。

## s-experts · Fused Top-4 Routing + Experts

Top-4 选路后，每个入选专家执行 Gate/Up → SwiGLU-OAI → Down，最后按路由权重加权求和。展开图逐步展示这条完整链路。

输入：`[B,S,6144] + [B,S,128]` → 输出：`[B,S,6144]`。

$$
\begin{aligned}s&=\sigma(R),\quad\mathcal E=\operatorname{TopK}_4(s+b_{corr})\\\hat w_e&=2s_e/\sum_{j\in\mathcal E}s_j\\g_e&=xW_{1,e}^{\top},\quad u_e=xW_{3,e}^{\top}\\\bar g_e&=\min(g_e,7),\quad\bar u_e=\operatorname{clip}(u_e,-7,7)\\z_e&=\bar g_e\odot\sigma(1.702\bar g_e)\odot(\bar u_e+1)\\y_e&=z_eW_{2,e}^{\top},\quad Y_{routed}=\sum_{e\in\mathcal E}\hat w_e y_e\end{aligned}
$$

SwiGLU-OAI 位于每个专家两次线性投影之间；不负责选专家。图示为逻辑拆解，实际由 FusedMoEFactory 调度。

### 1 · Sigmoid 打分与 Top-4 选路

$$
s=\sigma(R),\qquad\mathcal E=\operatorname{TopK}_{4}(s+b_{corr})
$$

对当前 token 的 128 个 FP32 logits R 做 sigmoid 得 s。correction bias 只用于 Top-4 选择，得到专家集合 𝓔；保留未加 bias 的原分数 s。

### 2 · 计算四个专家的混合权重

$$
\hat w_e=2\frac{s_e}{\sum_{j\in\mathcal E}s_j},\quad e\in\mathcal E,\quad\sum_e\hat w_e=2
$$

只取 4 个入选专家的原始分数，归一化后乘 routed_scaling_factor=2。这四个 ŵ 的和为 2；它们控制专家输出的贡献，不替代专家内部的 SwiGLU-OAI。

### 3 · 每个入选专家分别投影 Gate / Up

$$
\begin{aligned}g_e&=xW_{1,e}^{\top},\quad u_e=xW_{3,e}^{\top}\\x&\in\mathbb R^{6144},\quad g_e,u_e\in\mathbb R^{3072}\end{aligned}
$$

x=Û 是一个 token 的归一化输入。每个 e∈𝓔 有自己的 W1（Gate）与 W3（Up），各把 6144 维映射到 3072 维。逻辑形状 W1,W3=[3072,6144]；运行时 w13 按 [全部 Gate; 全部 Up] 打包，uninterleave 指两半布局。

### 4 · 先截断两个分支

$$
\bar g=\min(g,c),\quad\bar u=\operatorname{clip}(u,-c,c),\quad c=7
$$

g 来自 Gate 投影，u 来自 Up 投影。Gate 只限制上界，负值不会被截到 −7；Up 同时限制上下界。先截断，再将截断后的 g 送入 sigmoid。

### 5 · Gate 控制每个特征通道

$$
a=\bar g\odot\sigma(\alpha\bar g),\quad\alpha=1.702
$$

σ 是逐元素 sigmoid；α 调整 sigmoid 的斜率。a 是带符号的特征门控值，并非概率分布。这里的 Gate 分支位于单个专家内部，与产生 128 个选路分数的 Router 是两套计算。

### 6 · 调制 Up 内容，再交给 Down

$$
z=a\odot(\bar u+\beta),\quad\beta=1
$$

Up 提供内容特征；+1 是配置中的常量 β，不是可训练 bias。⊙ 表示逐元素乘，输出仍为 3072 维。SwiGLU-OAI 自身没有权重，负责非线性门控；之后 Down 才把特征投回 6144 维。普通 SiLU(g)⊙u 不包含这里的 α=1.702、β=1 与截断。

### 7 · 各专家 Down 投影

$$
y_e=z_eW_{2,e}^{\top}\in\mathbb R^{6144},\quad W_{2,e}\in\mathbb R^{6144\times3072}
$$

每个专家用自己的 W2 把激活后的 z_e 从 3072 维投回 6144 维，得到 y_e。这里的矩阵是逻辑权重；checkpoint 中路由专家是 NVFP4 packed，不能把 U8 存储形状当作逻辑矩阵形状。

### 8 · 按路由权重归并

$$
Y_{routed}=\sum_{e\in\mathcal E}\hat w_e y_e\in\mathbb R^{6144}
$$

每个 y_e 乘对应 ŵ_e，再对 4 个专家求和；共享专家输出另行相加。公式展示全局逻辑结果，实际 TP/EP 的分片、重排和通信由后端决定；PP=1 的跨 TP 归并可以延迟至下一 norm。

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

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L247-L267)。


## s-shared · Shared Expert ×1

所有 token 都执行一套 Gate/Up → SwiGLU-OAI → Down，共享专家不参与 Top-4，不乘 routed 权重。

输入：`[B,S,6144]` → 输出：`[B,S,6144]`。

$$
\begin{aligned}g_s&=xW_{gate,s}^{\top},\quad u_s=xW_{up,s}^{\top}\\\bar g_s&=\min(g_s,7),\quad\bar u_s=\operatorname{clip}(u_s,-7,7)\\z_s&=\bar g_s\odot\sigma(1.702\bar g_s)\odot(\bar u_s+1)\\Y_{shared}&=z_sW_{down,s}^{\top}\end{aligned}
$$

共享宽3072，使用相同clamp/α/β。模块传给FusedMoEFactory统一调度，不声称Python forward单独调用它。

### 1 · 独立的共享投影

$$
g_s=xW_{gate,s}^{\top},\quad u_s=xW_{up,s}^{\top}\in\mathbb R^{3072}
$$

同一个 token 输入 x=Û；共享专家始终执行，Gate/Up 各生成 3072 维特征，权重独立于 routed experts。

### 2 · 先截断两个分支

$$
\bar g=\min(g,c),\quad\bar u=\operatorname{clip}(u,-c,c),\quad c=7
$$

g 来自 Gate 投影，u 来自 Up 投影。Gate 只限制上界，负值不会被截到 −7；Up 同时限制上下界。先截断，再将截断后的 g 送入 sigmoid。

### 3 · Gate 控制每个特征通道

$$
a=\bar g\odot\sigma(\alpha\bar g),\quad\alpha=1.702
$$

σ 是逐元素 sigmoid；α 调整 sigmoid 的斜率。a 是带符号的特征门控值，并非概率分布。这里的 Gate 分支位于单个专家内部，与产生 128 个选路分数的 Router 是两套计算。

### 4 · 调制 Up 内容，再交给 Down

$$
z=a\odot(\bar u+\beta),\quad\beta=1
$$

Up 提供内容特征；+1 是配置中的常量 β，不是可训练 bias。⊙ 表示逐元素乘，输出仍为 3072 维。SwiGLU-OAI 自身没有权重，负责非线性门控；之后 Down 才把特征投回 6144 维。普通 SiLU(g)⊙u 不包含这里的 α=1.702、β=1 与截断。

### 最后 · Shared Down

$$
Y_{shared}=z_sW_{down,s}^{\top}\in\mathbb R^{6144}
$$

投回 6144 维，不乘 routed 权重；逻辑上与四路加权结果相加。TP 的局部计算与通信由运行时管理。

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

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L236-L267)。


## s-sum · Add Routed + Shared

逻辑上逐元素合并路由专家与共享专家输出。

输入：`2 × [B,S,6144]` → 输出：`[B,S,6144]`。

$$
Y_{\mathrm{moe}}=Y_{\mathrm{routed}}+E_{\mathrm{shared}}(\hat U)
$$

图中为数学拆分，实际由FusedMoEFactory管理；本地partial与跨rank归并边界由reduce_results决定。

- **Y_routed**：FusedMoE 已完成加权归并的输出
- **Y_shared**：Shared Expert 输出
- **Y_moe**：两路逐元素相加后的 MoE 输出

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L247-L267)。

## s-addout · Decoder Layer Residual Merge

逻辑上将全局MoE结果加入U，得到下一层输入。

输入：`2 × [B,S,6144]` → 输出：`[B,S,6144]`。

$$
X_{l+1}=U+Y_{\mathrm{moe}}
$$

PP=1可返回未归并partial与residual；all-reduce/add在下一input norm或final norm融合。PP>1时FFN归并后跨stage传递。

- **U**：Attention 后的 residual stream · [B,S,H]
- **Ymoe**：MoE 输出 · [B,S,H]
- **Xₗ₊₁**：逻辑上的下一层输入
- **H**：hidden_size = 6144

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L754-L790)。

## flash-image · Image / video input

动态分辨率图像和视频先转换为视觉 patch。672×672 是配置中的参考尺寸；实际 token 数由预处理和分辨率决定。

输入：`动态分辨率；依预处理配置` → 输出：`[N_patch,1176] + [N_items,3]（模型层入口）`。

$$
N_{patch}=\frac{H_{img}}{14}\frac{W_{img}}{14}
$$

逻辑数学解释；实际代码和量化布局以固定版本来源为准。 空间patch公式按单个时间patch组书写；视频具有多个时间组，总token数须再乘预处理产生的时间组数。

- **N_patch**：一个时间patch组内的空间patch数
- **H_img/W_img**：预处理后的图像高/宽；均按14切块
- **14**：空间patch边长；Conv3D另有时间核2

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/common/vision_tower.py#L40-L78)。

## flash-patch · Conv3D patch embedding

视觉塔使用 Conv3D，空间 kernel/stride 为 14，时间 patch 为 2。静态图像由预处理适配时间维度。

输入：`[N_patch,1176]，1176=3×2×14×14` → 输出：`[N_patch,1280]`。

$$
X_v=\mathrm{Conv3D}_{(2,14,14)}(pixels)
$$

逻辑数学解释；实际代码和量化布局以固定版本来源为准。

- **pixels**：预处理patch输入，含时间核2和空间14×14
- **X_v**：Conv3D后1280维视觉embedding

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/common/vision_tower.py#L40-L78)。

## flash-vit · Vision Transformer × 32

视觉塔宽度 1280、16 个注意力头、FFN 5120；使用 3D RoPE。它与语言 Decoder 的 6144 维残差流不同。

输入：`[N_patch,1280] + 边界与位置元数据` → 输出：`[N_patch,1280]`。

$$
X_v\in\mathbb{R}^{N_{patch}\times1280}
$$

逻辑数学解释；实际代码和量化布局以固定版本来源为准。

- **X_v**：ViT输出视觉token
- **N_patch**：包含各时间组的patch token数
- **1280**：视觉hidden维；32层、16头

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/common/vision_tower.py#L294-L339)。

## flash-projector · Multimodal projector

ViT 输出首先通过两层 MLP projector，从视觉宽度 1280 映射到文本宽度 6144，GELU 位于两次线性层之间。

输入：`[N_patch,1280]` → 输出：`[N_patch,6144]`。

$$
P=\mathrm{Linear}_{6144}(\mathrm{GELU}(\mathrm{Linear}_{1280\to6144}(X_v)))
$$

逻辑数学解释；实际代码和量化布局以固定版本来源为准。

- **X_v**：1280维ViT输出
- **P**：6144维投影后特征
- **GELU**：两层线性之间的激活

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/common/vision_tower.py#L593-L631)。

## flash-merge · 2×2 Patch merger

Projector 之后再做空间 2×2 patch merge。每 4 个相邻 patch 拼接后通过 MLP 压回 6144 维；时间 patch 属于更早的 Conv3D 阶段。

输入：`[N_patch,6144]` → 输出：`[N_patch/4,6144]`。

$$
P_{merged}=\mathrm{MLP}(\mathrm{Concat}_{2\times2}(P))
$$

逻辑数学解释；实际代码和量化布局以固定版本来源为准。

- **P**：投影后的6144维空间patch
- **Concat₂×₂**：同一时间组的4个相邻空间patch拼接为24576维
- **P_merged**：MLP压回6144维，空间token数减少4倍

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/common/vision_tower.py#L635-L673)。

## flash-fusion · Multimodal embedding fusion

视觉 embedding 替换输入序列中对应的图像／视频占位 token，与文本 embedding 一起进入语言 Decoder。参考图像为 576 token，动态输入数量会变化。

输入：`[Nq,6144] + [N_vision,6144] + [Nq]` → 输出：`[Nq,6144]，逻辑 [B,S,6144]`。

$$
X_0=\mathrm{Scatter}(E_{text},P_{merged},\mathrm{vision\ positions})
$$

逻辑数学解释；实际代码和量化布局以固定版本来源为准。

- **E_text**：text token embedding
- **P_merged**：视觉merge后特征
- **vision positions**：多模态占位token位置
- **X_0**：供语言decoder输入的融合embedding

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/models/interfaces.py#L493-L528)。

## flash-final · Final Norm → LM Head

60 层 Decoder 后执行 Final Gemma RMSNorm，LM Head 生成 200064 维 logits。Embedding 与 LM Head 不共享权重。

输入：`[B,S,6144]（逻辑）` → 输出：`[B,S,200064]（逻辑；运行时按所需 token 计算）`。

$$
\mathrm{logits}=\mathrm{Norm}(X_{60})W_{LM}^\top
$$

公式中的X_60是逻辑合并后的输出；PP=1的真实final norm可先归并最后一层FFN partial并加residual。此阶段仅消费norm与lm_head权重；embedding位于输入阶段。

- **X_60**：第60层后的逻辑残差输出；实际final norm可先归并并add
- **Norm**：最终Gemma RMSNorm
- **W_LM**：lm_head.weight [200064,6144]
- **logits**：词表维输出；不额外消费embed_tokens权重

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L891-L901)。

## flash-target · Target hidden states

DSpark 配置从主模型的 5 个目标层取上下文特征。图示仅表达配置可验证的数据依赖；不把旧 MTP 当作 DSpark 实现。

输入：`5 × [...,6144]；目标 ids 见配置` → 输出：`具体运行时组织未核实`。

$$
\mathcal H=\{h^{(3)},h^{(17)},h^{(31)},h^{(45)},h^{(59)}\}
$$

h^(l) 表示配置 target_layer_ids 指定的层输出特征；不是残差输入 X_l 的编号约定。仅配置结构已核实。

- **𝓗**：官方配置指定的目标层特征集合
- **h⁽l⁾**：目标层id l 的特征；不推定具体采集时序

[实现定位](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/262ae6830934f3512211081b8a3f16425b93a239/dspark/config.json)。

## flash-draft · DSpark draft × 5

独立 DSparkMiniMaxDraftModel，5 个 dense FFN 层。滑动注意力窗口为 4096，block size 7，use_mtp=false。本地 vLLM 的 minimax_m3/mtp.py 不能作为其执行依据。

输入：`配置 hidden=6144，block size=7` → 输出：`具体运行时组织未核实`。

$$
\mathrm{Draft}(\mathcal H,\mathrm{tokens};\ B_d=7)
$$

仅由独立草稿配置可核实的结构；未验证具体 DSpark 内核及接受采样细节。

- **𝓗**：target_layer_ids=[3,17,31,45,59]的配置特征
- **tokens**：草稿配置所需token上下文；具体forward未验证
- **B_d**：dspark_block_size=7，非batch数

[实现定位](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/262ae6830934f3512211081b8a3f16425b93a239/dspark/config.json)。

## flash-markov · Markov + confidence heads

官方 DSpark 配置启用置信度头，Markov rank 为 256。该页面不从配置推断未验证的采样细节或实际加速比。

输入：`hidden=6144；Markov rank=256（配置）` → 输出：`具体运行时 shape 未核实`。

$$
r_{Markov}=256
$$

仅由独立草稿配置可核实的结构；未验证具体 DSpark 内核及接受采样细节。

- **r_Markov**：markov_head_rank=256，非TP rank
- **Markov**：仅配置事实；头部计算未验证

[实现定位](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/262ae6830934f3512211081b8a3f16425b93a239/dspark/config.json)。

## flash-verify · Target verification

投机解码先生成候选，再由主模型验证。接受多少 token 取决于运行时和输入；block size 不等于保证的每步输出数。

输入：`投机解码关系示意；实现未核实` → 输出：`接受长度由运行时决定`。

$$
\mathrm{draft}\longrightarrow\mathrm{target\ verify}\longrightarrow\mathrm{accept/reject}
$$

仅由独立草稿配置可核实的结构；未验证具体 DSpark 内核及接受采样细节。

- **draft**：配置描述的草稿候选
- **target verify**：目标模型验证概念
- **accept/reject**：算法概念，具体接受规则未验证

[实现定位](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/262ae6830934f3512211081b8a3f16425b93a239/dspark/config.json)。

## s-moe-choice · Sigmoid + Top-4

s=σ(R)；按 s+b_corr 选 4 个专家。bias 只改变选谁，不进入混合权重。

输入：`[128]` → 输出：`[4] + [128]`。

$$
s=\sigma(R),\qquad\mathcal E=\operatorname{TopK}_{4}(s+b_{corr})
$$

s=σ(R)；按 s+b_corr 选 4 个专家。bias 只改变选谁，不进入混合权重。

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

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L247-L267)。


## s-moe-weights · Renormalize × 2

gather 选中专家的原始分数，除以这 4 个分数之和，再乘 2；四个混合权重之和是 2。

输入：`[4] + [128]` → 输出：`[4]`。

$$
\hat w_e=2\frac{s_e}{\sum_{j\in\mathcal E}s_j},\quad e\in\mathcal E,\quad\sum_e\hat w_e=2
$$

gather 选中专家的原始分数，除以这 4 个分数之和，再乘 2；四个混合权重之和是 2。

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

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L247-L267)。


## s-moe-gateup · Expert Gate + Up

每个入选专家各用自己的 W1 / W3，将同一个 6144 维 x 投影成两个 3072 维分支。逻辑 W1,W3=[3072,6144]；w13 按全部 Gate 在前、全部 Up 在后打包。NVFP4 U8 存储形状见权重，不能当逻辑形状。

输入：`[6144] + [4]` → 输出：`2 × [3072] / expert`。

$$
\begin{aligned}g_e&=xW_{1,e}^{\top},\quad u_e=xW_{3,e}^{\top}\\x&\in\mathbb R^{6144},\quad g_e,u_e\in\mathbb R^{3072}\end{aligned}
$$

每个入选专家各用自己的 W1 / W3，将同一个 6144 维 x 投影成两个 3072 维分支。逻辑 W1,W3=[3072,6144]；w13 按全部 Gate 在前、全部 Up 在后打包。NVFP4 U8 存储形状见权重，不能当逻辑形状。

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

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L247-L267)。


## s-moe-swiglu · SwiGLU-OAI · per expert

对 Gate 先做上界 min(g,7)，对 Up 做 clip(u,−7,7)，再计算 ḡ·sigmoid(1.702ḡ)·(ū+1)。这是专家内部的逐通道非线性门控，激活本身没有可训练权重；输出宽度仍为 3072。

输入：`2 × [3072] / expert` → 输出：`[3072] / expert`。

$$
\begin{aligned}\bar g_e&=\min(g_e,7)\\\bar u_e&=\operatorname{clip}(u_e,-7,7)\\z_e&=\bar g_e\odot\sigma(1.702\bar g_e)\odot(\bar u_e+1)\end{aligned}
$$

对 Gate 先做上界 min(g,7)，对 Up 做 clip(u,−7,7)，再计算 ḡ·sigmoid(1.702ḡ)·(ū+1)。这是专家内部的逐通道非线性门控，激活本身没有可训练权重；输出宽度仍为 3072。

### 1 · 先截断两个分支

$$
\bar g=\min(g,c),\quad\bar u=\operatorname{clip}(u,-c,c),\quad c=7
$$

g 来自 Gate 投影，u 来自 Up 投影。Gate 只限制上界，负值不会被截到 −7；Up 同时限制上下界。先截断，再将截断后的 g 送入 sigmoid。

### 2 · Gate 控制每个特征通道

$$
a=\bar g\odot\sigma(\alpha\bar g),\quad\alpha=1.702
$$

σ 是逐元素 sigmoid；α 调整 sigmoid 的斜率。a 是带符号的特征门控值，并非概率分布。这里的 Gate 分支位于单个专家内部，与产生 128 个选路分数的 Router 是两套计算。

### 3 · 调制 Up 内容，再交给 Down

$$
z=a\odot(\bar u+\beta),\quad\beta=1
$$

Up 提供内容特征；+1 是配置中的常量 β，不是可训练 bias。⊙ 表示逐元素乘，输出仍为 3072 维。SwiGLU-OAI 自身没有权重，负责非线性门控；之后 Down 才把特征投回 6144 维。普通 SiLU(g)⊙u 不包含这里的 α=1.702、β=1 与截断。

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

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/layers/fused_moe/activation.py#L381-L389)。


## s-moe-down · Expert Down Projection

各专家用独立 W2=[6144,3072] 把激活特征投回 hidden size。此处尚未与其他专家求和；后端可融合投影、加权和归并。

输入：`[3072] / expert` → 输出：`[6144] / expert`。

$$
y_e=z_eW_{2,e}^{\top}\in\mathbb R^{6144},\quad W_{2,e}\in\mathbb R^{6144\times3072}
$$

各专家用独立 W2=[6144,3072] 把激活特征投回 hidden size。此处尚未与其他专家求和；后端可融合投影、加权和归并。

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

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L247-L267)。


## s-moe-combine · Weighted Sum · 4 experts

每个专家的 6144 维输出乘自己的路由权重，再沿 4 个专家求和。全局公式不规定 TP/EP buffer；PP=1 的 TP 归并可延迟到下一 norm。

输入：`[4,6144] + [4]（逻辑）` → 输出：`[6144]`。

$$
Y_{routed}=\sum_{e\in\mathcal E}\hat w_e y_e\in\mathbb R^{6144}
$$

每个专家的 6144 维输出乘自己的路由权重，再沿 4 个专家求和。全局公式不规定 TP/EP buffer；PP=1 的 TP 归并可延迟到下一 norm。

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

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L247-L267)。


## s-moe-sharedgateup · Shared Gate + Up

共享专家使用自己的 Gate / Up 权重，同样从 6144 维投影到两路 3072 维。所有 token 都执行，无需 Top-4 ids。TP 下每 rank 可只持有 3072/TP 通道。

输入：`[6144]` → 输出：`2 × [3072]（逻辑）`。

$$
g_s=xW_{gate,s}^{\top},\quad u_s=xW_{up,s}^{\top}\in\mathbb R^{3072}
$$

共享专家使用自己的 Gate / Up 权重，同样从 6144 维投影到两路 3072 维。所有 token 都执行，无需 Top-4 ids。TP 下每 rank 可只持有 3072/TP 通道。

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

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L236-L267)。


## s-moe-sharedswiglu · SwiGLU-OAI · shared

对 Gate 先做上界 min(g,7)，对 Up 做 clip(u,−7,7)，再计算 ḡ·sigmoid(1.702ḡ)·(ū+1)。这是专家内部的逐通道非线性门控，激活本身没有可训练权重；输出宽度仍为 3072。共享分支使用相同 c、α、β；TP 时按本地通道逐元素计算。

输入：`2 × [3072]（逻辑）` → 输出：`[3072]（逻辑）`。

$$
\begin{aligned}\bar g_s&=\min(g_s,7),\quad\bar u_s=\operatorname{clip}(u_s,-7,7)\\z_s&=\bar g_s\odot\sigma(1.702\bar g_s)\odot(\bar u_s+1)\end{aligned}
$$

对 Gate 先做上界 min(g,7)，对 Up 做 clip(u,−7,7)，再计算 ḡ·sigmoid(1.702ḡ)·(ū+1)。这是专家内部的逐通道非线性门控，激活本身没有可训练权重；输出宽度仍为 3072。共享分支使用相同 c、α、β；TP 时按本地通道逐元素计算。

### 1 · 先截断两个分支

$$
\bar g=\min(g,c),\quad\bar u=\operatorname{clip}(u,-c,c),\quad c=7
$$

g 来自 Gate 投影，u 来自 Up 投影。Gate 只限制上界，负值不会被截到 −7；Up 同时限制上下界。先截断，再将截断后的 g 送入 sigmoid。

### 2 · Gate 控制每个特征通道

$$
a=\bar g\odot\sigma(\alpha\bar g),\quad\alpha=1.702
$$

σ 是逐元素 sigmoid；α 调整 sigmoid 的斜率。a 是带符号的特征门控值，并非概率分布。这里的 Gate 分支位于单个专家内部，与产生 128 个选路分数的 Router 是两套计算。

### 3 · 调制 Up 内容，再交给 Down

$$
z=a\odot(\bar u+\beta),\quad\beta=1
$$

Up 提供内容特征；+1 是配置中的常量 β，不是可训练 bias。⊙ 表示逐元素乘，输出仍为 3072 维。SwiGLU-OAI 自身没有权重，负责非线性门控；之后 Down 才把特征投回 6144 维。普通 SiLU(g)⊙u 不包含这里的 α=1.702、β=1 与截断。

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

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/layers/activation.py#L240-L244)。


## s-moe-shareddown · Shared Down Projection

共享 Down 将 3072 维投回 6144 维；不乘 Top-4 路由权重，随后与 Y_routed 相加。共享 MLP 以 reduce_results=False 交给 FusedMoEFactory，图中是全局逻辑输出。

输入：`[3072]（逻辑）` → 输出：`[6144]`。

$$
Y_{shared}=z_sW_{down,s}^{\top}\in\mathbb R^{6144}
$$

共享 Down 将 3072 维投回 6144 维；不乘 Top-4 路由权重，随后与 Y_routed 相加。共享 MLP 以 reduce_results=False 交给 FusedMoEFactory，图中是全局逻辑输出。

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

[实现定位](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/minimax_m3/nvidia/model.py#L236-L267)。
