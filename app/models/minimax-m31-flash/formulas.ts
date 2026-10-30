import {MOE_MATH,MOE_STEPS,SWIGLU_STEPS,MOE_TERMS} from "./moe-math";
import type { OpNode, OpKind } from "../types";

export const SIMPLE_FORMULA: Partial<Record<OpKind,string>> = {
  norm:String.raw`y=\operatorname{Norm}(x)`,linear:String.raw`y=xW^{\mathsf T}`,split:String.raw`(a,b,\ldots)=\operatorname{Split}(x)`,rope:String.raw`q'=\operatorname{RoPE}(q,\mathrm{position})`,matmul:String.raw`y=a\,b^{\mathsf T}`,scale:String.raw`y=x/\sqrt{d_h}`,mask:String.raw`y=x+\mathrm{mask}`,softmax:String.raw`p=\operatorname{softmax}(x)`,activation:String.raw`y=\bar g\,\sigma(\alpha\bar g)\,(\bar u+\beta)`,route:String.raw`I=\operatorname{TopK}(\mathrm{score}(x))`,cache:String.raw`\mathrm{KV}[\mathrm{slot}]\leftarrow(K,V)`,add:String.raw`y=x+f(x)`,io:String.raw`y=x`,
};

export const FORMULA_NOTE: Partial<Record<OpKind,string>> = {
  norm:"把每个 token 的向量缩放到稳定范围；shape 不变。",linear:"W 是当前模块绑定的权重；最后一维由 W 的输出维决定。",split:"只切分最后一维，不做数值计算，也没有权重。",rope:"position 决定旋转角度；这里只旋转每个 head 的前 64 维。",matmul:"沿共同的 head_dim 相乘并求和。",scale:"dₕ=128；缩放避免 score 随维度增大。",mask:"不可见位置加 −∞，softmax 后概率变为 0。",softmax:"把每行 score 转为和为 1 的概率。",activation:"g 是 gate，u 是 up；实际实现还包含 limit=7 的截断。",route:"只选择去哪里计算；Top-K 本身不生成 expert 输出。",cache:"slot 与 block table 由 runtime 提供，权重不参与。",add:"残差支路与计算支路逐元素相加，shape 必须一致。",io:"这是数据入口或运行时元数据，不执行可训练计算。",
};

export type FormulaTerm = readonly [symbol:string,meaning:string];
export type FormulaStep = { title:string; formula:string; explanation:string };

export const FORMULA_TERMS_BY_KIND: Record<OpKind,readonly FormulaTerm[]> = {
  io:[["x","输入"],["y","输出"]],
  norm:[["x","输入向量"],["y","归一化输出"],["H","归一化维度"],["γ","可训练缩放权重"],["ε","数值稳定项"],["RMS(x)","均方根"]],
  linear:[["x","输入张量"],["W","投影权重"],["y","线性投影输出"]],
  split:[["x","待切分张量"],["a,b,…","沿最后一维得到的输出"]],
  rope:[["q / k","Q 或 K 向量"],["p","token position"],["dᵣ","参与旋转的维度"],["θ","旋转角度"]],
  matmul:[["a","左输入张量"],["b","右输入张量"],["y","矩阵乘输出"]],
  scale:[["x","未缩放分数"],["dₕ","head_dim = 128"],["y","缩放后分数"]],
  mask:[["x","原始 attention score"],["M","causal / padding mask"],["y","mask 后 score"]],
  softmax:[["x","输入 score"],["p","归一化概率"]],
  activation:[["g","gate 分支"],["u","up 分支"],["α","sigmoid scale = 1.702"],["c","clamp limit = 7"],["y","SwiGLU-OAI 输出"]],
  route:[["s","路由分数"],["K","选择数量"],["𝓔 / I","选中的 expert 或 block id"]],
  cache:[["K / V","写入 cache 的张量"],["slot","物理 cache 位置"],["block_table","逻辑块到物理页映射"]],
  add:[["x","residual 分支"],["f(x)","当前计算分支"],["y","逐元素相加结果"]],
};

export const FORMULA_TERMS_BY_ID: Partial<Record<string,readonly FormulaTerm[]>> = {
  "d-norm":[["x","hidden_states · [B,S,H]"],["y","normalized hidden_states"],["H","hidden_size = 6144"],["γ","input_layernorm.weight"],["ε","rms_norm_eps = 10⁻⁶"],["RMS(x)","√(Σxⱼ²/H + ε)"]],
  "s-norm":[["x","hidden_states · [B,S,H]"],["y","normalized hidden_states"],["H","hidden_size = 6144"],["γ","input_layernorm.weight"],["ε","rms_norm_eps = 10⁻⁶"],["RMS(x)","√(Σxⱼ²/H + ε)"]],
  "d-postnorm":[["U","上游 Add 节点的输出"],["Û","Gemma RMSNorm(U)"],["H","hidden_size = 6144"],["γpost","post_attention_layernorm.weight"],["ε","rms_norm_eps = 10⁻⁶"],["RMS(U)","√(ΣUⱼ²/H + ε)"]],
  "s-postnorm":[["U","上游 Add 节点的输出"],["Û","Gemma RMSNorm(U)"],["H","hidden_size = 6144"],["γpost","post_attention_layernorm.weight"],["ε","rms_norm_eps = 10⁻⁶"],["RMS(U)","√(ΣUⱼ²/H + ε)"]],
  "d-qnorm":[["Q","Query heads"],["Q̃","归一化后的 Q"],["Dₕ","head_dim = 128"],["γQ","q_norm.weight"],["ε","10⁻⁶"]],
  "d-knorm":[["K","Key heads"],["K̃","归一化后的 K"],["Dₕ","head_dim = 128"],["γK","k_norm.weight"],["ε","10⁻⁶"]],
  "s-mainnorm":[["Q / K","主 Attention 的 Q/K"],["Q̃ / K̃","归一化后的 Q/K"],["Dₕ","head_dim = 128"],["γQ / γK","Q/K norm weights"],["ε","10⁻⁶"]],
  "s-idxnorm":[["Qidx / Kidx","Indexer 的投影输入"],["Q̃idx / K̃idx","RMSNorm 后的 Index Q/K"],["Qidxᵣ / Kidxᵣ","加入 position 后的旋转结果"],["Didx","index_dim = 128"],["p","query/key position"]],
  "d-gateup":[["Û","归一化输入 · [B,S,H]"],["Wgate⁽ʳ⁾","当前 TP rank 的 gate 权重"],["Wup⁽ʳ⁾","当前 TP rank 的 up 权重"],["G⁽ʳ⁾","当前 rank 的 gate 投影"],["U⁽ʳ⁾","当前 rank 的 up 投影"],["H","hidden_size = 6144"],["H_dense","dense_intermediate_size = 12288"],["TP","tensor parallel size"]],
  "d-gatesplit":[["X⁽ʳ⁾","当前 rank 的 packed gate_up"],["G⁽ʳ⁾","前 H_dense/TP 个通道"],["U⁽ʳ⁾","后 H_dense/TP 个通道"],["H_dense","dense_intermediate_size = 12288"],["TP","tensor parallel size"]],
  "d-swiglu":[["G⁽ʳ⁾","当前 rank 的 gate 分片"],["U⁽ʳ⁾","当前 rank 的 up 分片"],["Ḡ⁽ʳ⁾ / Ū⁽ʳ⁾","clamp 后的两个分片"],["α","swiglu_alpha = 1.702"],["β","swiglu_beta = 1.0"],["c","swiglu_limit = 7.0"],["Z⁽ʳ⁾","当前 rank 的激活输出"]],
  "d-qk":[["Qᵣ","当前 rank 的 rotated Q"],["Kᵣ","当前 rank 的 rotated / visible K"],["Nₕ/TP","每 rank 的 query heads = 64/TP"],["Nₖᵥ,rank","每 rank KV heads = max(1,4/TP)"],["Dₕ","head_dim = 128"],["A","当前 rank 的 attention scores"]],
  "d-pv":[["P","当前 rank 的 attention probability"],["V","当前 rank 的 visible V"],["Nₕ/TP","每 rank 的 query heads = 64/TP"],["Nₖᵥ,rank","每 rank KV heads = max(1,4/TP)"],["O","当前 rank 的 context heads"]],
  "s-idxscore":[["Qidxᵣ","当前 rank 的 Index Q query"],["𝒦idx","独立 side cache 中的完整 Index K history"],["N_idx,rank","max(1,4/TP)"],["D_idx","index_dim = 128"],["Sidx","尚未 mask 的 Index token scores"]],
  "s-idxcache":[["Kidxᵣ","当前 token 的旋转后 Index K"],["slot","index slot_mapping 指定的 side-cache 位置"],["𝒦idx","独立的 key-only Index K cache"],["D_idx","每 token 一个 128 维 Index K 向量"]],
  "s-idxmask":[["Sidx","Index Q/K 点积分数"],["p","当前 query position"],["j","key position"],["S̃idx","未来 key 被置为 −∞ 后的分数"]],
  "s-qk":[["Qᵣ","当前 rank 的 rotated Q"],["paged K","主 Paged KV Cache 中的 K pages"],["block_indices","Indexer 输出的 Top-16 逻辑块索引"],["Ksel","kernel 最多读取 16×128 个候选 token"],["A","当前 rank 的 sparse scores"]],
  "s-pv":[["P","当前 rank 的 sparse probability"],["paged V","主 Paged KV Cache 中的 V pages"],["block_indices","与 Q×K 阶段相同的 Top-16 块顺序"],["O","当前 rank 的 context heads · 8192/TP"]],
  "s-topk":[["C","block scores"],["𝓛ᵢ","当前 query 的 local block 集合"],["10²⁹","local块的有限优先分数"],["K_block","sparse_topk_blocks = 16"],["𝓑 / block_indices","有效选中块id，不包含-1"],["𝓒","选中块展开且未越过T的token位置"],["𝓥","𝓒内满足因果约束的token位置"]],
  "s-router":[["Û","Post-attn RMSNorm 输出"],["Wrouter","block_sparse_moe.gate.weight"],["E","num_local_experts = 128"],["R","router_logits · [B,S,E]"]],
  "s-experts":MOE_TERMS,
  "d-qkv":[["Nₕ","num_attention_heads = 64"],["Nₖᵥ","num_key_value_heads = 4"],["Dₕ","head_dim = 128"],["TP","tensor parallel size"],["Z","当前 rank 的 packed QKV"]],
  "d-split":[["Nₕ","num_attention_heads = 64"],["Nₖᵥ,rank","max(1,4/TP)"],["Dₕ","head_dim = 128"],["Q / K / V","当前 rank 的三个输出"]],
  "d-ropeq":[["Dᵣ","rotary_dim = 64"],["Dₕ","head_dim = 128"],["θbase","rope_theta = 10000000"],["p","token position"],["Qᵣ","旋转后的 Q"]],
  "d-ropek":[["Dᵣ","rotary_dim = 64"],["Dₕ","head_dim = 128"],["θbase","rope_theta = 10000000"],["p","token position"],["Kᵣ","旋转后的 K"]],
  "d-oproj":[["Nₕ","num_attention_heads = 64"],["Dₕ","head_dim = 128"],["H","hidden_size = 6144"],["TP","tensor parallel size"],["W_O","o_proj.weight"]],
  "d-down":[["Z⁽ʳ⁾","当前 TP rank 的激活输出"],["Wdown⁽ʳ⁾","当前 rank 的 down 权重"],["H_dense","dense_intermediate_size = 12288"],["H","hidden_size = 6144"],["TP","tensor parallel size"]],
  "s-packed":[["Nₕ","num_attention_heads = 64"],["Nₖᵥ","num_key_value_heads = 4"],["N_idx","sparse_num_index_heads = 4"],["Dₕ","head_dim = 128"],["D_idx","sparse_index_dim = 128"],["TP","tensor parallel size"]],
  "s-split":[["Nₕ/TP","每 rank query heads = 64/TP"],["Nₖᵥ,rank","max(1,4/TP)"],["N_idx,rank","max(1,4/TP)"],["Dₕ","head_dim = 128"],["D_idx","sparse_index_dim = 128"]],
  "s-blockmax":[["B_block","sparse_block_size = 128"],["S̃idx","已排除未来 key 的 Index scores"],["C","每128 keys取max的块分数"]],
  "s-rope":[["Dᵣ","rotary_dim = 64"],["Dₕ","head_dim = 128"],["θbase","rope_theta = 10000000"],["p","token position"]],
  "s-oproj":[["Nₕ","num_attention_heads = 64"],["Dₕ","head_dim = 128"],["H","hidden_size = 6144"],["TP","tensor parallel size"],["W_O","o_proj.weight"]],
  "d-add2":[["U","Attention 后的 residual stream · [B,S,H]"],["Yffn","Dense FFN 输出 · [B,S,H]"],["Xₗ₊₁","逻辑上的下一层输入"],["H","hidden_size = 6144"]],
  "s-addout":[["U","Attention 后的 residual stream · [B,S,H]"],["Ymoe","MoE 输出 · [B,S,H]"],["Xₗ₊₁","逻辑上的下一层输入"],["H","hidden_size = 6144"]],
  "d-add1":[["Xₗ","进入本层的 residual stream · [B,S,H]"],["Yattn","Attention 输出 · [B,S,H]"],["U","更新后的 residual stream"],["H","hidden_size = 6144"]],
  "s-addattn":[["Xₗ","进入本层的 residual stream · [B,S,H]"],["Yattn","Sparse Attention 输出 · [B,S,H]"],["U","更新后的 residual stream"],["H","hidden_size = 6144"]],
  "d-position":[["q_b","请求 b 本轮调度的 query 数"],["p_b,i","请求 b 的第 i 个 token position"],["N_q","本轮 query token 总数"],["B","batch size"]],
  "s-position":[["q_b","请求 b 本轮调度的 query 数"],["p_b,i","请求 b 的第 i 个 token position"],["N_q","本轮 query token 总数"],["B","batch size"]],
  "d-attnmeta":[["q_b","请求 b 的 query 数"],["c_b","请求 b 已有 context 长度"],["M","causal / padding mask"],["−∞","不可见位置的加性 mask 值"]],
  "s-attnmeta":[["q_b","请求 b 的 query 数"],["c_b","请求 b 已有 context 长度"],["M","causal / padding mask"],["−∞","不可见位置的加性 mask 值"]],
  "d-slots":[["p","token position"],["B_block","KV cache block size · runtime"],["b_phys","物理 block id"],["block_table","逻辑块到物理页映射"],["slot","KV cache 物理槽位"]],
  "s-slots":[["p","token position"],["B_block","KV cache block size · runtime"],["b_phys","物理 block id"],["block_table","逻辑块到物理页映射"],["slot","KV cache 物理槽位"]],
  "d-cache":[["Kᵣ / V","写入 cache 的 Key / Value"],["slot","物理 cache 位置"],["block_table","逻辑块到物理页映射"],["K≤p / V≤p","当前 token 可见的 KV"]],
  "d-scale":[["A","未缩放 attention scores"],["Ā","缩放后的 scores"],["Dₕ","head_dim = 128"]],
  "d-mask":[["Ā","缩放后的 scores"],["M","causal / padding mask"],["Ã","mask 后的 scores"],["c_b","请求 b 的 context 长度"]],
  "d-softmax":[["Ã","mask 后的 scores"],["P","attention probability"],["T","可见 KV token 数"],["m","每行最大值，用于数值稳定"]],
  "s-cache":[["Kᵣ / V","写入 sparse cache 的 Key / Value"],["slot","物理 cache 位置"],["block_table","逻辑块到物理页映射"]],
  "s-scale":[["A","未缩放 sparse scores"],["Ā","缩放后的 sparse scores"],["Dₕ","head_dim = 128"]],
  "s-mask":[["Ā","selected K 上的缩放 scores"],["valid_token","候选 blocks 内的 causal / padding 判定"],["Ã","token mask 后的 selected scores"],["𝓒 / 𝓥","候选token集合 / 因果有效子集"]],
  "s-softmax":[["Ã","mask 后的 sparse scores"],["P","selected KV 上的概率"],["𝓥","当前query/group有效候选token集合"]],
  "s-shared":MOE_TERMS,
  "s-sum":[["Y_routed","FusedMoE 已完成加权归并的输出"],["Y_shared","Shared Expert 输出"],["Y_moe","两路逐元素相加后的 MoE 输出"]],
};


FORMULA_TERMS_BY_ID["s-input"]=[["X_l", "逻辑本层输入"], ["hidden_states / residual", "运行时两条流；若FFN延迟归并则先all-reduce"], ["B/S/H", "请求数 / 当前token长度 / hidden6144"]];
FORMULA_TERMS_BY_ID["s-mainnorm"]=[["Q/K", "当前S个token的主Q/K，非历史cache"], ["RMSNorm", "逐128维head的Gemma Norm，ε=1e-6"], ["Q̃/K̃", "归一化输出"]];
FORMULA_TERMS_BY_ID["s-idxnorm"]=[["Qidx/Kidx", "当前本地Index Q与单份Index K"], ["rot", "前64维Partial RoPE后的标记"], ["p", "当前token位置"], ["N_idx,rank", "max(1,4/TP)"]];
FORMULA_TERMS_BY_ID["flash-image"]=[["N_patch", "一个时间patch组内的空间patch数"], ["H_img/W_img", "预处理后的图像高/宽；均按14切块"], ["14", "空间patch边长；Conv3D另有时间核2"]];
FORMULA_TERMS_BY_ID["flash-patch"]=[["pixels", "预处理patch输入，含时间核2和空间14×14"], ["X_v", "Conv3D后1280维视觉embedding"]];
FORMULA_TERMS_BY_ID["flash-vit"]=[["X_v", "ViT输出视觉token"], ["N_patch", "包含各时间组的patch token数"], ["1280", "视觉hidden维；32层、16头"]];
FORMULA_TERMS_BY_ID["flash-projector"]=[["X_v", "1280维ViT输出"], ["P", "6144维投影后特征"], ["GELU", "两层线性之间的激活"]];
FORMULA_TERMS_BY_ID["flash-merge"]=[["P", "投影后的6144维空间patch"], ["Concat₂×₂", "同一时间组的4个相邻空间patch拼接为24576维"], ["P_merged", "MLP压回6144维，空间token数减少4倍"]];
FORMULA_TERMS_BY_ID["flash-fusion"]=[["E_text", "text token embedding"], ["P_merged", "视觉merge后特征"], ["vision positions", "多模态占位token位置"], ["X_0", "供语言decoder输入的融合embedding"]];
FORMULA_TERMS_BY_ID["flash-final"]=[["X_60", "第60层后的逻辑残差输出；实际final norm可先归并并add"], ["Norm", "最终Gemma RMSNorm"], ["W_LM", "lm_head.weight [200064,6144]"], ["logits", "词表维输出；不额外消费embed_tokens权重"]];
FORMULA_TERMS_BY_ID["flash-target"]=[["𝓗", "官方配置指定的目标层特征集合"], ["h⁽l⁾", "目标层id l 的特征；不推定具体采集时序"]];
FORMULA_TERMS_BY_ID["flash-draft"]=[["𝓗", "target_layer_ids=[3,17,31,45,59]的配置特征"], ["tokens", "草稿配置所需token上下文；具体forward未验证"], ["B_d", "dspark_block_size=7，非batch数"]];
FORMULA_TERMS_BY_ID["flash-markov"]=[["r_Markov", "markov_head_rank=256，非TP rank"], ["Markov", "仅配置事实；头部计算未验证"]];
FORMULA_TERMS_BY_ID["flash-verify"]=[["draft", "配置描述的草稿候选"], ["target verify", "目标模型验证概念"], ["accept/reject", "算法概念，具体接受规则未验证"]];

for(const id of ["s-idxscore","s-idxmask","s-blockmax","s-topk"]){
 FORMULA_TERMS_BY_ID[id]=[...(FORMULA_TERMS_BY_ID[id]??[]),["b / g / i / j / u","请求 / 本地索引组 / 当前query / 历史token位置 / 逻辑块id"]];
}
for(const id of ["s-qk","s-mask","s-softmax","s-pv"]){
 FORMULA_TERMS_BY_ID[id]=[...(FORMULA_TERMS_BY_ID[id]??[]),["𝓑 / 𝓒 / 𝓥","选中块id / 展开的有效历史token / 因果有效子集"],["G_rank","(64/TP) / max(1,4/TP)，本地Q/KV分组比"],["b / h / i / j / m","请求 / 本地Q头 / query位置 / 历史token位置 / head通道"]];
}

FORMULA_TERMS_BY_ID["d-down"]=[...(FORMULA_TERMS_BY_ID["d-down"]??[]),["r / Yffn⁽ʳ⁾ / Yffn","TP rank / 本地partial / 跨rank和"],["PP","pipeline parallel size，决定FFN归并是否延迟"]];
FORMULA_TERMS_BY_ID["s-oproj"]=[...(FORMULA_TERMS_BY_ID["s-oproj"]??[]),["r / Yattn⁽ʳ⁾ / Yattn","TP rank / 本地partial / 在post-attn norm归并的逻辑结果"]];

export const FORMULA_STEPS_BY_ID: Partial<Record<string,readonly FormulaStep[]>> = {
  "s-experts":MOE_STEPS,
  "s-shared":[
    {title:"1 · 独立的共享投影",formula:MOE_MATH.sharedgateup,explanation:"同一个 token 输入 x=Û；共享专家始终执行，Gate/Up 各生成 3072 维特征，权重独立于 routed experts。"},
    ...SWIGLU_STEPS.map((step,i)=>({...step,title:`${i+2} · ${step.title.slice(4)}`})),
    {title:"最后 · Shared Down",formula:MOE_MATH.shareddown,explanation:"投回 6144 维，不乘 routed 权重；逻辑上与四路加权结果相加。TP 的局部计算与通信由运行时管理。"},
  ],
  "s-moe-swiglu":SWIGLU_STEPS,
  "s-moe-sharedswiglu":SWIGLU_STEPS,
};

export function formulaTerms(node:OpNode){
  return FORMULA_TERMS_BY_ID[node.id]??(node.latex?[]:FORMULA_TERMS_BY_KIND[node.kind]);
}

export const LATEX_BY_ID: Record<string,string> = {
  "s-input":String.raw`X_l=\operatorname{LogicalMerge}(\mathrm{hidden\_states},\mathrm{residual})\in\mathbb R^{B\times S\times H}`,
  "d-position":String.raw`\begin{aligned}q_b&=\mathrm{num\_scheduled\_tokens}[b]\\p_{b,i}&=\mathrm{num\_computed\_tokens}[b]+i,\quad 0\le i<q_b\\\mathbf p&=\operatorname{concat}_{b=1}^{B}(p_{b,0},\ldots,p_{b,q_b-1})\in\mathbb Z^{N_q}\end{aligned}`,
  "s-position":String.raw`\begin{aligned}q_b&=\mathrm{num\_scheduled\_tokens}[b]\\p_{b,i}&=\mathrm{num\_computed\_tokens}[b]+i,\quad 0\le i<q_b\\\mathbf p&=\operatorname{concat}_{b=1}^{B}(p_{b,0},\ldots,p_{b,q_b-1})\in\mathbb Z^{N_q}\end{aligned}`,
  "d-attnmeta":String.raw`\begin{aligned}q_b&=\mathrm{query\_start\_loc}_{b+1}-\mathrm{query\_start\_loc}_b\\c_b&=\mathrm{seq\_len}_b-q_b\\M_{b,i,j}&=\begin{cases}0,&0\le j\le c_b+i\\-\infty,&\text{otherwise}\end{cases}\end{aligned}`,
  "s-attnmeta":String.raw`\begin{aligned}q_b&=\mathrm{query\_start\_loc}_{b+1}-\mathrm{query\_start\_loc}_b\\c_b&=\mathrm{seq\_len}_b-q_b\\M_{b,i,j}&=\begin{cases}0,&0\le j\le c_b+i\\-\infty,&\text{otherwise}\end{cases}\end{aligned}`,
  "d-slots":String.raw`\begin{aligned}\ell&=\left\lfloor p/B_{block}\right\rfloor,\quad o=p\bmod B_{block}\\b_{\mathrm{phys}}&=\mathrm{block\_table}[r,\ell]\\\mathrm{slot}(r,p)&=b_{\mathrm{phys}}\cdot B_{block}+o\end{aligned}`,
  "s-slots":String.raw`\begin{aligned}\ell&=\lfloor p/B_{block}\rfloor,\quad o=p\bmod B_{block}\\b_{phys}&=\mathrm{block\_table}[b,\ell]\\\mathrm{slot}(b,p)&=b_{phys}B_{block}+o\end{aligned}`,
  "d-norm":String.raw`\begin{aligned}\operatorname{RMS}(x)&=\sqrt{\frac1H\sum_{j=1}^{H}x_j^2+\varepsilon}\\y_i&=\frac{x_i}{\operatorname{RMS}(x)}(1+\gamma_i)\end{aligned}`,
  "d-qkv":String.raw`\begin{aligned}Z&=\hat X\,[W_Q^\top\mid W_K^\top\mid W_V^\top]\\Z&\in\mathbb R^{B\times S\times((N_h+2N_{kv})D_h/TP)}\end{aligned}`,
  "d-split":String.raw`(Q,K,V)=\operatorname{Split}\!\left(Z;\frac{N_hD_h}{TP},N_{kv,\mathrm{rank}}D_h,N_{kv,\mathrm{rank}}D_h\right)`,
  "d-qnorm":String.raw`\begin{aligned}\operatorname{RMS}(Q_{b,h,s})&=\sqrt{\frac1{D_h}\sum_{j=1}^{D_h}Q_{b,h,s,j}^2+\varepsilon}\\\tilde Q_{b,h,s,i}&=\frac{Q_{b,h,s,i}}{\operatorname{RMS}(Q_{b,h,s})}(1+\gamma_{Q,i})\end{aligned}`,
  "d-knorm":String.raw`\begin{aligned}\operatorname{RMS}(K_{b,g,s})&=\sqrt{\frac1{D_h}\sum_{j=1}^{D_h}K_{b,g,s,j}^2+\varepsilon}\\\tilde K_{b,g,s,i}&=\frac{K_{b,g,s,i}}{\operatorname{RMS}(K_{b,g,s})}(1+\gamma_{K,i})\end{aligned}`,
  "d-ropeq":String.raw`\begin{aligned}(Q_{\mathrm{rot}},Q_{\mathrm{pass}})&=\operatorname{Split}(\tilde Q;D_r,D_h-D_r)\\Q^{rot}&=\operatorname{Concat}(\operatorname{RoPE}(Q_{\mathrm{rot}},p),Q_{\mathrm{pass}})\end{aligned}`,
  "d-ropek":String.raw`\begin{aligned}(K_{\mathrm{rot}},K_{\mathrm{pass}})&=\operatorname{Split}(\tilde K;D_r,D_h-D_r)\\K^{rot}&=\operatorname{Concat}(\operatorname{RoPE}(K_{\mathrm{rot}},p),K_{\mathrm{pass}})\end{aligned}`,
  "d-cache":String.raw`\begin{aligned}\mathcal K[\mathrm{slot}(r,p)]&\leftarrow K^{rot}_{r,p}\\\mathcal V[\mathrm{slot}(r,p)]&\leftarrow V_{r,p}\\K_{\le p},V_{\le p}&\leftarrow\operatorname{gather}(\mathcal K,\mathcal V,\mathrm{block\_table}_r)\end{aligned}`,
  "d-qk":String.raw`A_{b,h,i,j}=\sum_{m=1}^{D_h}Q^{rot}_{b,h,i,m}\,K^{rot}_{b,\lfloor h/G\rfloor,j,m}`,
  "d-scale":String.raw`\bar A_{b,h,i,j}=\frac{A_{b,h,i,j}}{\sqrt{D_h}}`,
  "d-mask":String.raw`\tilde A_{b,h,i,j}=\bar A_{b,h,i,j}+M_{b,i,j}=\begin{cases}\bar A_{b,h,i,j},&j\le c_b+i\\-\infty,&j>c_b+i\end{cases}`,
  "d-softmax":String.raw`P_{b,h,i,j}=\frac{\exp(\tilde A_{b,h,i,j}-m_{b,h,i})}{\sum_{t=0}^{T-1}\exp(\tilde A_{b,h,i,t}-m_{b,h,i})},\quad m_{b,h,i}=\max_t\tilde A_{b,h,i,t}`,
  "d-pv":String.raw`O_{b,h,i,m}=\sum_{j=0}^{T-1}P_{b,h,i,j}\,V_{b,\lfloor h/G\rfloor,j,m}`,
  "d-oproj":String.raw`Y_{\mathrm{attn}}=\operatorname{RowParallel}\!\left(\operatorname{Concat}_{h=1}^{N_h/TP}(O_h),W_O\right)\in\mathbb R^{B\times S\times H}`,
  "d-add1":String.raw`U=X_l+Y_{\mathrm{attn}}`,
  "d-postnorm":String.raw`\begin{aligned}\operatorname{RMS}(U)&=\sqrt{\frac1H\sum_{j=1}^{H}U_j^2+\varepsilon}\\\hat U_i&=\frac{U_i}{\operatorname{RMS}(U)}(1+\gamma_{\mathrm{post},i})\end{aligned}`,
  "d-gateup":String.raw`\begin{aligned}G^{(r)}&=\hat U\left(W_{\mathrm{gate}}^{(r)}\right)^\top\\U^{(r)}&=\hat U\left(W_{\mathrm{up}}^{(r)}\right)^\top\\G^{(r)},U^{(r)}&\in\mathbb R^{B\times S\times(H_{\mathrm{dense}}/TP)}\end{aligned}`,
  "d-gatesplit":String.raw`\begin{aligned}X^{(r)}&\in\mathbb R^{B\times S\times(2H_{\mathrm{dense}}/TP)}\\G^{(r)}&=X^{(r)}_{:,:,\,0:H_{\mathrm{dense}}/TP}\\U^{(r)}&=X^{(r)}_{:,:,\,H_{\mathrm{dense}}/TP:2H_{\mathrm{dense}}/TP}\end{aligned}`,
  "d-swiglu":String.raw`\begin{aligned}\bar G^{(r)}&=\min(G^{(r)},c)\\\bar U^{(r)}&=\operatorname{clip}(U^{(r)},-c,c)\\Z^{(r)}&=\bar G^{(r)}\odot\sigma(\alpha\bar G^{(r)})\odot(\bar U^{(r)}+\beta)\end{aligned}`,
  "d-down":String.raw`\begin{aligned}Y_{ffn}^{(r)}&=Z^{(r)}(W_{down}^{(r)})^\top\in\mathbb R^{B\times S\times H}\\Y_{ffn}&=\sum_rY_{ffn}^{(r)}\quad\text{(deferred if PP=1)}\end{aligned}`,
  "d-add2":String.raw`X_{l+1}=U+Y_{\mathrm{ffn}}`,
  "s-norm":String.raw`\begin{aligned}\operatorname{RMS}(x)&=\sqrt{\frac1H\sum_{j=1}^{H}x_j^2+\varepsilon}\\y_i&=\frac{x_i}{\operatorname{RMS}(x)}(1+\gamma_i)\end{aligned}`,
  "s-postnorm":String.raw`\begin{aligned}\operatorname{RMS}(U)&=\sqrt{\frac1H\sum_{j=1}^{H}U_j^2+\varepsilon}\\\hat U_i&=\frac{U_i}{\operatorname{RMS}(U)}(1+\gamma_{\mathrm{post},i})\end{aligned}`,
  "s-packed":String.raw`Z=\hat X[W_Q^\top\mid W_K^\top\mid W_V^\top\mid W_{Q_i}^\top\mid W_{K_i}^\top]`,
  "s-split":String.raw`Z\longrightarrow(Q_{N_hD_h/TP},K_{N_{kv,\mathrm{rank}}D_h},V_{N_{kv,\mathrm{rank}}D_h},Q^{\mathrm{idx}}_{N_{idx,\mathrm{rank}}D_{idx}},K^{\mathrm{idx}}_{D_{idx}})`,
  "s-idxnorm":String.raw`\begin{aligned}\tilde Q^{\mathrm{idx}},\tilde K^{\mathrm{idx}}&=\operatorname{RMSNorm}(Q^{\mathrm{idx}}),\operatorname{RMSNorm}(K^{\mathrm{idx}})\\Q^{\mathrm{idx},rot},K^{\mathrm{idx},rot}&=\operatorname{RoPE}(\tilde Q^{\mathrm{idx}},\tilde K^{\mathrm{idx}};\mathbf p)\end{aligned}`,
  "s-idxcache":String.raw`\mathcal K_{\mathrm{idx}}[\mathrm{slot}(b,p)]\leftarrow K^{\mathrm{idx},rot}_{b,p}\in\mathbb R^{D_{idx}}`,
  "s-idxscore":String.raw`S^{idx}_{b,g,i,j}=\langle Q^{idx,rot}_{b,g,i,:},\mathcal K_{idx}[b,j,:]\rangle`,
  "s-idxmask":String.raw`\tilde S^{idx}_{b,g,i,j}=\begin{cases}S^{idx}_{b,g,i,j},&0\le j<T_b\ \land\ j\le p_{b,i}\\-\infty,&\text{otherwise}\end{cases}`,
  "s-blockmax":String.raw`C_{b,g,i,u}=\max_{B_{block}u\le j<B_{block}(u+1)}\tilde S^{idx}_{b,g,i,j}`,
  "s-topk":String.raw`\begin{aligned}C_u&\leftarrow10^{29},\quad u\in\mathcal L_i\\\mathcal B_{b,g,i}&=\{u\in\operatorname{TopK}_{K_{block}}(C):u\ne-1\}\\\mathcal C_{b,g,i}&=\{j:0\le j<T_b,\lfloor j/B_{block}\rfloor\in\mathcal B_{b,g,i}\}\\\mathcal V_{b,g,i}&=\{j\in\mathcal C_{b,g,i}:j\le p_{b,i}\}\end{aligned}`,
  "s-mainnorm":String.raw`\tilde Q=\operatorname{RMSNorm}(Q),\qquad\tilde K=\operatorname{RMSNorm}(K)`,
  "s-rope":String.raw`\begin{aligned}(Q^{rot}_{:D_r},K^{rot}_{:D_r})&=\operatorname{RoPE}(\tilde Q_{:D_r},\tilde K_{:D_r};\mathbf p)\\(Q^{rot}_{D_r:},K^{rot}_{D_r:})&=(\tilde Q_{D_r:},\tilde K_{D_r:})\end{aligned}`,
  "s-cache":String.raw`\mathcal K[\mathrm{slot}(b,p)]\leftarrow K^{rot}_{b,p},\qquad\mathcal V_{cache}[\mathrm{slot}(b,p)]\leftarrow V_{b,p}`,
  "s-qk":String.raw`A_{b,h,i,j}=\sum_{m=1}^{D_h}Q^{rot}_{b,h,i,m}K^{rot}_{b,\lfloor h/G_{rank}\rfloor,j,m},\quad j\in\mathcal C_{b,\lfloor h/G_{rank}\rfloor,i}`,
  "s-scale":String.raw`\bar A_{b,h,i,j}=A_{b,h,i,j}/\sqrt{D_h}`,
  "s-mask":String.raw`\tilde A_{b,h,i,j}=\begin{cases}\bar A_{b,h,i,j},&j\in\mathcal V_{b,\lfloor h/G_{rank}\rfloor,i}\\-\infty,&\text{otherwise}\end{cases},\quad j\in\mathcal C_{b,\lfloor h/G_{rank}\rfloor,i}`,
  "s-softmax":String.raw`\begin{aligned}\mathcal V_i&=\mathcal V_{b,\lfloor h/G_{rank}\rfloor,i},\quad m_i=\max_{t\in\mathcal V_i}\tilde A_{b,h,i,t}\\P_{b,h,i,j}&=\frac{\exp(\tilde A_{b,h,i,j}-m_i)}{\sum_{t\in\mathcal V_i}\exp(\tilde A_{b,h,i,t}-m_i)},\quad j\in\mathcal V_i\end{aligned}`,
  "s-pv":String.raw`O_{b,h,i,m}=\sum_{j\in\mathcal V_{b,\lfloor h/G_{rank}\rfloor,i}}P_{b,h,i,j}V_{b,\lfloor h/G_{rank}\rfloor,j,m}`,
  "s-oproj":String.raw`\begin{aligned}Y_{attn}^{(r)}&=\operatorname{Concat}(O^{(r)})(W_O^{(r)})^\top\\Y_{attn}&=\sum_rY_{attn}^{(r)}\quad\text{(next post-attention norm)}\end{aligned}`,
  "s-addattn":String.raw`U=X_l+Y_{\mathrm{attn}}`,
  "s-router":String.raw`R=\hat U W_{router}^\top\in\mathbb R^{B\times S\times E}`,
  "s-experts":String.raw`\begin{aligned}s&=\sigma(R),\quad\mathcal E=\operatorname{TopK}_4(s+b_{corr})\\\hat w_e&=2s_e/\sum_{j\in\mathcal E}s_j\\g_e&=xW_{1,e}^{\top},\quad u_e=xW_{3,e}^{\top}\\\bar g_e&=\min(g_e,7),\quad\bar u_e=\operatorname{clip}(u_e,-7,7)\\z_e&=\bar g_e\odot\sigma(1.702\bar g_e)\odot(\bar u_e+1)\\y_e&=z_eW_{2,e}^{\top},\quad Y_{routed}=\sum_{e\in\mathcal E}\hat w_e y_e\end{aligned}`,
  "s-shared":String.raw`\begin{aligned}g_s&=xW_{gate,s}^{\top},\quad u_s=xW_{up,s}^{\top}\\\bar g_s&=\min(g_s,7),\quad\bar u_s=\operatorname{clip}(u_s,-7,7)\\z_s&=\bar g_s\odot\sigma(1.702\bar g_s)\odot(\bar u_s+1)\\Y_{shared}&=z_sW_{down,s}^{\top}\end{aligned}`,
  "s-sum":String.raw`Y_{\mathrm{moe}}=Y_{\mathrm{routed}}+E_{\mathrm{shared}}(\hat U)`,
  "s-addout":String.raw`X_{l+1}=U+Y_{\mathrm{moe}}`,
};

for(const key of Object.keys(MOE_MATH))FORMULA_TERMS_BY_ID[`s-moe-${key}`]=MOE_TERMS;
