import type {CodeDetail,IoBinding} from "../types";
import sections from "./source-sections.json";
export const QKV_INDEX_PROJECTION_SECTIONS=sections.projection;
export const QKV_INDEX_PROJECTION_SYMBOLS=[];
export const CODE_BY_ID:Record<string,CodeDetail>={};
for(const id of ["s-norm", "s-postnorm", "d-norm", "d-postnorm", "s-mainnorm"])CODE_BY_ID[id]={sections:sections.norm,symbols:[]};
for(const id of ["s-input", "s-addattn", "s-addout", "d-add1", "d-add2"])CODE_BY_ID[id]={sections:sections.residual,symbols:[]};
for(const id of ["s-packed", "s-split"])CODE_BY_ID[id]={sections:sections.projection,symbols:[]};
for(const id of ["s-idxnorm", "s-rope"])CODE_BY_ID[id]={sections:sections.preparation,symbols:[]};
for(const id of ["s-idxcache"])CODE_BY_ID[id]={sections:sections.indexcache,symbols:[]};
for(const id of ["s-idxscore", "s-idxmask", "s-blockmax"])CODE_BY_ID[id]={sections:sections.indexscore,symbols:[]};
for(const id of ["s-topk"])CODE_BY_ID[id]={sections:sections.topk,symbols:[]};
for(const id of ["s-qk", "s-scale", "s-mask", "s-softmax", "s-pv"])CODE_BY_ID[id]={sections:sections.paged,symbols:[]};
for(const id of ["s-oproj"])CODE_BY_ID[id]={sections:sections.oproj,symbols:[]};
for(const id of ["d-gateup", "d-down"])CODE_BY_ID[id]={sections:sections.gateup,symbols:[]};
for(const id of ["d-gatesplit", "d-swiglu"])CODE_BY_ID[id]={sections:sections.swiglu,symbols:[]};
for(const id of ["s-router"])CODE_BY_ID[id]={sections:sections.router,symbols:[]};
for(const id of ["s-experts", "s-sum"])CODE_BY_ID[id]={sections:sections.experts,symbols:[]};
for(const id of ["s-shared"])CODE_BY_ID[id]={sections:sections.shared,symbols:[]};
CODE_BY_ID["s-cache"]={sections:sections.maincache,symbols:[]};
for(const id of ["s-norm","s-postnorm"])CODE_BY_ID[id]={sections:[...sections.norm,...sections.residual],symbols:[]};
export const INPUT_OVERRIDES: Record<string, IoBinding[]> = {
  "d-input":[{kind:"external",label:"Xₗ · hidden_states",shape:"[B,S,6144]",from:"上一 decoder layer；L0 时来自 embedding fusion"}],
  "s-input":[{kind:"external",label:"Xₗ · hidden_states",shape:"[B,S,6144]",from:"上一 decoder layer 输出"}],
  "d-position":[{kind:"external",label:"num_computed_tokens + query offsets",shape:"[B] + [Nq]",from:"vLLM GPUModelRunner 请求调度状态"}],
  "s-position":[{kind:"external",label:"num_computed_tokens + query offsets",shape:"[B] + [Nq]",from:"vLLM GPUModelRunner 请求调度状态"}],
  "d-attnmeta":[{kind:"external",label:"query_start_loc · seq_lens · causal",shape:"[B+1] + [B] + bool",from:"vLLM CommonAttentionMetadata"}],
  "s-attnmeta":[{kind:"external",label:"query_start_loc · seq_lens · causal",shape:"[B+1] + [B] + bool",from:"vLLM CommonAttentionMetadata"}],
  "d-slots":[{kind:"external",label:"positions + block_table",shape:"[Nq] + [B,Nblocks]",from:"runner positions 与 KV cache manager"}],
  "s-slots":[{kind:"external",label:"positions + block_table",shape:"[Nq] + [B,Nblocks]",from:"runner positions 与 KV cache manager"}],
  "d-ropeq":[{kind:"upstream",label:"Q̃",shape:"[B,64,S,128]",from:"Q RMSNorm 输出"},{kind:"external",label:"positions",shape:"[Nq]",from:"Build Position IDs 输出"}],
  "d-ropek":[{kind:"upstream",label:"K̃",shape:"[B,4,S,128]",from:"K RMSNorm 输出"},{kind:"external",label:"positions",shape:"[Nq]",from:"Build Position IDs 输出"}],
  "d-cache":[{kind:"upstream",label:"Kᵣ",shape:"[B,4,S,128]",from:"Partial RoPE (K) 输出"},{kind:"upstream",label:"V",shape:"[B,4,S,128]",from:"Split Q / K / V 输出"},{kind:"external",label:"slot_mapping + block_table",shape:"[Nq] + [B,Nblocks]",from:"Resolve KV Slots 输出"}],
  "d-qk":[{kind:"upstream",label:"Qᵣ (TP-local)",shape:"[B,64/TP,S,128]",from:"Partial RoPE (Q) 输出"},{kind:"upstream",label:"visible K (TP-local / replicated)",shape:"[B,max(1,4/TP),T,128]",from:"Paged KV Cache 输出"}],
  "d-mask":[{kind:"upstream",label:"scaled local scores",shape:"[B,64/TP,S,T]",from:"Scale 1/√128 输出"},{kind:"external",label:"causal / padding bounds",shape:"runtime metadata",from:"Build Attention Metadata 输出"}],
  "d-pv":[{kind:"upstream",label:"local attention probability P",shape:"[B,64/TP,S,T]",from:"Softmax 输出"},{kind:"upstream",label:"visible V (TP-local / replicated)",shape:"[B,max(1,4/TP),T,128]",from:"Paged KV Cache 输出"}],
  "s-rope":[{kind:"upstream",label:"Q̃ · K̃",shape:"Q/K unchanged",from:"Main Q/K Norm 输出"},{kind:"external",label:"positions",shape:"[Nq]",from:"Build Position IDs 输出"}],
  "s-cache":[{kind:"upstream",label:"current Kᵣ · V",shape:"2 × [B,max(1,4/TP),S,128]",from:"Partial RoPE 与 Split 5 outputs"},{kind:"external",label:"slot_mapping + block_table",shape:"[Nq] + [B,Nblocks]",from:"Resolve KV Slots 输出"}],
  "s-idxnorm":[{kind:"upstream",label:"Qidx · Kidx",shape:"[B,N_idx,rank,S,D_idx] · [B,S,D_idx]",from:"Split 5 outputs"},{kind:"external",label:"position embeddings",shape:"cos · sin",from:"Build Position IDs / RoPE cache"}],
  "s-idxcache":[{kind:"upstream",label:"current rotated Index K",shape:"[B,S,128]",from:"Index Q/K Gemma RMSNorm + RoPE 输出"},{kind:"external",label:"index slot_mapping",shape:"[Nq]",from:"Indexer metadata builder"}],
  "s-idxscore":[{kind:"upstream",label:"Index Q query (TP-local)",shape:"[B,N_idx,rank,S,D_idx]",from:"Index Q/K Gemma RMSNorm + RoPE 输出"},{kind:"upstream",label:"cached Index K history",shape:"[B,T,128]",from:"Index K Cache 输出"}],
  "s-idxmask":[{kind:"upstream",label:"Index token scores",shape:"[B,max(1,4/TP),S,T]",from:"Index Q × Kᵀ 输出"},{kind:"external",label:"position_ids",shape:"[B,S]",from:"当前 query/key 的因果位置"}],
  "s-blockmax":[{kind:"upstream",label:"causal Index scores",shape:"[B,max(1,4/TP),S,T]",from:"Mask Future Index Keys 输出"}],
  "s-topk":[{kind:"upstream",label:"local block scores",shape:"[B,max(1,4/TP),S,Nblocks]",from:"Block Max 输出"},{kind:"external",label:"local / init priority",shape:"logical block flags",from:"Indexer 配置：local_blocks=1, init_blocks=0"}],
  "s-qk":[{kind:"upstream",label:"Qᵣ (TP-local)",shape:"[B,64/TP,S,128]",from:"Partial RoPE 输出"},{kind:"upstream",label:"paged K",shape:"KV pages",from:"Paged KV Cache 输出"},{kind:"upstream",label:"block_indices",shape:"[B,max(1,4/TP),S,16]",from:"Top-16 Blocks 输出（当前 TP rank）"}],
  "s-mask":[{kind:"upstream",label:"scaled local selected scores",shape:"[B,64/TP,S,Ksel]",from:"Scale 1/√128 输出"},{kind:"external",label:"causal / padding bounds",shape:"runtime metadata",from:"Build Attention Metadata 输出"}],
  "s-pv":[{kind:"upstream",label:"local selected attention P",shape:"[B,64/TP,S,Ksel]",from:"Softmax 输出"},{kind:"upstream",label:"paged V",shape:"KV pages",from:"Paged KV Cache 输出；按同一 Top-16 顺序读取"}],
  "s-router":[{kind:"upstream",label:"post-attn normalized hidden Û",shape:"[B,S,6144]",from:"Post-attn RMSNorm 输出"}],
  "s-experts":[{kind:"upstream",label:"normalized hidden + router logits",shape:"[B,S,6144] + [B,S,128]",from:"Post-attn RMSNorm 与 FP32 Router 输出"}],
  "s-shared":[{kind:"upstream",label:"all normalized tokens Û",shape:"[B,S,6144]",from:"Post-attn RMSNorm 输出；不经过 Top-K"}],
  "s-sum":[{kind:"upstream",label:"weighted routed output",shape:"[B,S,6144]",from:"Fused Top-4 Routing + Experts 输出"},{kind:"upstream",label:"shared output",shape:"[B,S,6144]",from:"Shared Expert ×1 输出"}],
  "d-add2":[{kind:"upstream",label:"U · residual stream",shape:"[B,S,6144]",from:"Attention Residual 输出"},{kind:"upstream",label:"Yffn partial · FFN branch",shape:"[B,S,6144]",from:"Down Projection 输出"}],
  "s-addout":[{kind:"upstream",label:"U · residual stream",shape:"[B,S,6144]",from:"Attention Residual 输出"},{kind:"upstream",label:"Ymoe · MoE branch",shape:"[B,S,6144]",from:"Add Routed + Shared 输出"}],
  "d-add1":[{kind:"upstream",label:"Xₗ · residual stream",shape:"[B,S,6144]",from:"本层输入旁路"},{kind:"upstream",label:"Yattn partial · attention branch",shape:"[B,S,6144]",from:"O Projection 输出"}],
  "s-addattn":[{kind:"upstream",label:"Xₗ · residual stream",shape:"[B,S,6144]",from:"本层输入旁路"},{kind:"upstream",label:"Yattn partial · attention branch",shape:"[B,S,6144]",from:"O Projection 输出"}],
};

export const NEXT_BY_ID: Record<string,string> = {
  "d-input":"Gemma RMSNorm","d-position":"Partial RoPE (Q/K)","d-attnmeta":"Apply Causal / Pad Bounds","d-slots":"Paged KV Cache","d-norm":"QKV Projection","d-qkv":"Split Q / K / V","d-split":"Q RMSNorm · K RMSNorm · Paged KV Cache","d-qnorm":"Partial RoPE (Q)","d-knorm":"Partial RoPE (K)","d-ropeq":"Q × Kᵀ","d-ropek":"Paged KV Cache","d-cache":"Q × Kᵀ · P × V","d-qk":"Scale 1/√128","d-scale":"Apply Causal / Pad Bounds","d-mask":"Softmax","d-softmax":"P × V","d-pv":"O Projection","d-oproj":"Attention Residual Merge","d-add1":"Post-attn Gemma RMSNorm","d-postnorm":"Gate + Up Projection","d-gateup":"Split Gate / Up","d-gatesplit":"SwiGLU-OAI","d-swiglu":"Down Projection","d-down":"Decoder Layer Residual Merge","d-add2":"下一 decoder layer / Final Norm",
  "s-input":"Gemma RMSNorm","s-position":"Partial RoPE","s-attnmeta":"Indexer 与 Sparse Attention mask","s-slots":"Paged KV Cache","s-norm":"QKV + Index Projection","s-packed":"Split 5 outputs","s-split":"Index Q/K Gemma RMSNorm + RoPE · Main Q/K Gemma RMSNorm · Paged KV Cache","s-idxnorm":"Index Q query · Index K Cache","s-idxcache":"Index Q × cached Kᵀ","s-idxscore":"Mask Future Index Keys","s-idxmask":"Block Max","s-blockmax":"Top-K_block Blocks","s-topk":"Q × paged Kᵀ · Top-K_block","s-mainnorm":"Partial RoPE","s-rope":"Paged KV Cache · Q × paged Kᵀ · Top-K_block","s-cache":"Q × paged Kᵀ · Top-K_block · P × paged V","s-qk":"Scale 1/√Dₕ","s-scale":"Apply Token Causal / Pad Mask","s-mask":"Softmax","s-softmax":"P × paged V · same Top-K_block","s-pv":"O Projection","s-oproj":"Attention Residual Merge","s-addattn":"Post-attn Gemma RMSNorm","s-postnorm":"FP32 Router Logits · Fused Top-4 Routing + Experts · Shared Expert","s-router":"router_logits → Fused Top-4 Routing + Experts","s-experts":"Add Routed + Shared","s-shared":"Add Routed + Shared","s-sum":"Decoder Layer Residual Merge","s-addout":"下一 decoder layer / Final Norm",
};
