Document-Kind: model-knowledge
Authoring: agent-generated, source-grounded
Model: MiniMax-M3.1-Flash
Source-Revisions: HF 262ae6830934f3512211081b8a3f16425b93a239; vLLM ced6857afa0ea7b2e3f0846a62e1394e90f15607
Content-Status: draft

# 实现、融合边界与权重装载

## 原项目复用

保留原 Model Atlas 的 `app/graph/`、tokens/ui 样式、模型注册与原 MiniMax-M3 模块。Flash 模块独立放在 `app/models/minimax-m31-flash/`，其 `diagrams.tsx` 由原生图谱改造。`flashGraph` 分别选择 MSA 注意力与 Dense/MoE FFN，不再将 Dense FFN 和全注意力绑定。原模型可从 MODEL 下拉切回。

## 真实代码链

`MiniMaxM3DecoderLayer.forward` → input Gemma RMSNorm → `MiniMaxM3SparseAttention.forward` → `MinimaxM3QKVParallelLinearWithIndexer` → `fused_minimax_m3_qknorm_rope_kv_insert`（四套 Norm、Partial RoPE、两套 cache 写入）→ Indexer → `impl.forward` → `o_proj` → post-attention fused residual/Norm → Dense MLP 或 FusedMoEFactory。

原项目的一些展示片段来自更早版本；Flash 的 `source-sections.json` 重新从用户本地固定提交提取，逐段带真实路径、起止行和完整 URL。五路 projection 绑定只展示投影和装载证据；下游 math 节点展示其所属融合 kernel。

## 两套缓存与两种 mask

主缓存保存 K/V，Indexer side cache 只保存共享 Index K。位置 ids 控制 RoPE 和因果边界，slot mapping 控制写入位置，block table 控制逻辑到物理页。Index future mask 发生在 block max 前；主 token causal/pad mask 发生在选中块内。两者不能互相替代。

## Flash 权重格式

主 checkpoint 使用 MXFP8 可量化 Dense/Attention 路径与 NVFP4 routed experts，Router/Norm/视觉等按 config ignored_layers 例外处理。实际文件头区分 F8_E4M3、U8、BF16、F32。

- q/k/v/index_q/index_k 的分离权重装载进一个 qkv_proj。
- Dense 和 shared 的 gate_proj/up_proj 装载为 gate_up_proj。
- routed w1=gate、w3=up 装载为 fused w13，w2 为 down。
- routed `weight_packed` 的 U8 末维为逻辑末维一半；scales 单独存储。
- 分片号从原始索引精确读取，不能用层号猜文件名。
- 对每个展示 weight，note 区分文件头直接读取、按同配置层推导和未采样。

## 独立 DSpark

当前用户源码 checkout 的 MiniMax 路径包含旧 MTP，但不能证明 DSpark 执行。DSpark 标签只呈现官方草稿配置可核实的结构；具体 forward、Markov 头计算和接受算法明确未验证。部署请参照官方框架文档，不将普通 MTP 启动参数套用到 DSpark。

## 图示工具与页面

七张补充图来自 Archify 3.2 typed JSON，原样复制成功交付的 HTML。iframe 与原生详情隔离，不承诺不存在的跨 iframe 节点回调。原生算子图仍提供详细的多输入连线、权重、逐步公式和源码定位。主网站无服务器和凭据，完整权重目录仅加载小型元数据。

## TP partial与延迟all-reduce

算子图将逻辑全局数学输出与真实融合调用边界分开。`o_proj.reduce_results=False`，所以O projection只产生当前TP rank的H维partial；Decoder的post-attention `fused_allreduce_gemma_rms_norm`才执行归并、residual add及norm。不能仅凭RowParallelLinear类名认定projection已经all-reduce。

主模型PP=1的Dense Down和MoE同样可不立即归并，返回FFN partial与residual两条流。`ffn_all_reduce_deferred`通知主模型，将归并融合到下一层input norm或最后的final norm。PP>1或旧MTP块设置`reduce_results=True`，FFN归并后才传递；这不作为DSpark执行证据。具体flag见固定model.py L720–744、L782–790、L853–860；Attention flag见L452–462。

主cache用FullAttentionSpec，通用Triton布局[Npages,N_kv,rank,128,256]，末维包含K与V；逻辑读视图才写成各[B,N_kv,rank,T,128]。Index side cache使用独立MLAAttentionSpec且num_kv_heads=1。当前投影、Norm和RoPE处理S个token，cache读历史为T；TP与prefill/decode示例见symbols.md。
