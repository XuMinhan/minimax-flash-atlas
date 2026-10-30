import config from "../../../public/flash-data/config.json";
export type TensorMeta={shape:number[];dtype:string;bytes?:number;shard?:number};
const c=config.text_config,v=config.vision_config;
export const SHAPE_SYMBOLS:Record<string,number>={H:c.hidden_size,I:c.intermediate_size,I_shared:c.intermediate_size*c.n_shared_experts,I_dense:c.dense_intermediate_size,V:c.vocab_size,E:c.num_local_experts,N_h:c.num_attention_heads,N_kv:c.num_key_value_heads,D_h:c.head_dim,N_idx:c.sparse_attention_config.sparse_num_index_heads,D_idx:c.sparse_attention_config.sparse_index_dim,D_v:v.hidden_size,I_v:v.intermediate_size};
const product=(shape:number[])=>shape.reduce((a,b)=>a*b,1);
export const shapeText=(shape:number[])=>`[${shape.join(",")}]`;
export const bytesText=(bytes:number)=>`${(bytes/2**30).toFixed(3)} GiB · ${(bytes/1e9).toFixed(3)} GB`;
export const smallBytes=(bytes:number)=>bytes>=2**20?`${(bytes/2**20).toFixed(3)} MiB`:bytes>=1024?`${(bytes/1024).toFixed(3)} KiB`:`${bytes} B`;
const dtypeBytes:Record<string,number>={BF16:2,F16:2,F32:4,F64:8,U8:1,I8:1,I32:4,I64:8,F8_E4M3:1,F8_E4M3FN:1};
export function tensorRole(key:string){
 if(key.endsWith("input_global_scale"))return "激活量化辅助 scale";
 if(/(?:weight_scale_inv|weight_scale|weight_global_scale)$/.test(key))return "权重量化辅助 scale";
 if(key.endsWith("weight_packed"))return "矩阵权重";
 if(key.endsWith("bias")||key.endsWith("correction_bias"))return "bias / 选路修正";
 return key.endsWith("weight")?"权重 / Norm 参数":"其他模型张量";
}
export function moduleUse(key:string,model="main"){
 if(model==="dspark")return "DSpark 草稿 · 独立 checkpoint";
 if(key.startsWith("vision_tower."))return "视觉 ViT · 图像/视频编码";
 if(key.startsWith("multi_modal_projector."))return "视觉 Projector · 视觉维度映射";
 if(key.startsWith("patch_merge_mlp."))return "Patch Merger · 合并空间 patch";
 if(key.includes("embed_tokens"))return "Token Embedding · 输入查表";
 if(key.includes("lm_head"))return "LM Head · 词表 logits";
 if(key.includes(".block_sparse_moe.experts."))return "L3–59 路由专家 · Top-4";
 if(key.includes(".shared_experts."))return "L3–59 共享专家 · 始终参与";
 if(key.includes(".block_sparse_moe.gate.")||key.includes("e_score_correction_bias"))return "L3–59 Router · 专家选择";
 if(key.includes(".mlp."))return "L0–2 Dense FFN · 非路由 MLP";
 if(key.includes(".self_attn.index_"))return "L0–59 Indexer · 选 KV blocks";
 if(key.includes(".self_attn."))return "L0–59 主 Attention · Q/K/V/O";
 if(key.includes("norm"))return "Decoder Norm · 残差归一化";
 return "其他模块";
}
export function operationUse(key:string){
 if(/\.w1\./.test(key)||key.includes("gate_proj"))return "Gate 投影";
 if(/\.w3\./.test(key)||key.includes("up_proj"))return "Up 投影";
 if(/\.w2\./.test(key)||key.includes("down_proj"))return "Down 投影";
 if(key.includes("index_q_proj"))return "Index Q 投影";
 if(key.includes("index_k_proj"))return "Index K 投影";
 for(const p of ["q_proj","k_proj","v_proj","o_proj"])if(key.includes(p))return `${p[0].toUpperCase()} 投影`;
 if(key.includes("norm"))return "归一化";
 if(key.includes("e_score_correction_bias"))return "Top-4 选择修正";
 if(key.includes(".gate."))return "Router logits";
 if(key.includes("confidence_head"))return "草稿置信度头（key 标识）";
 if(key.includes("markov"))return "草稿 Markov 参数（key 标识）";
 if(key.includes("embed_tokens"))return "Embedding 查表";
 if(key.includes("lm_head"))return "词表输出";
 return key.split(".").slice(-3,-1).join(".");
}
function logicalSymbol(key:string,shape:number[]){
 if(key.startsWith("vision_tower.")){
  if(/self_attn\.(?:q_proj|k_proj|v_proj|out_proj|o_proj)\.weight$/.test(key))return "[D_v,D_v]";
  if(/\.mlp\.fc1\.weight$/.test(key))return "[I_v,D_v]";
  if(/\.mlp\.fc2\.weight$/.test(key))return "[D_v,I_v]";
  if(shape.length===1&&shape[0]===v.hidden_size)return "[D_v]";
  return `[${shape.map((_,i)=>`d${i}`).join(",")}]`;
 }
 if(/\.experts\.\d+\.w[13]\.weight_packed$/.test(key))return "[I,H]";
 if(/\.experts\.\d+\.w2\.weight_packed$/.test(key))return "[H,I]";
 if(key.includes("shared_experts.")&&key.endsWith(".weight"))return key.includes("down_proj")?"[H,I_shared]":"[I_shared,H]";
 if(/\.mlp\.(gate_proj|up_proj)\.weight$/.test(key))return "[I_dense,H]";
 if(/\.mlp\.down_proj\.weight$/.test(key))return "[H,I_dense]";
 if(/(embed_tokens|lm_head)\.weight$/.test(key))return "[V,H]";
 const attn=key.startsWith("language_model.")?key.match(/self_attn\.(q_proj|k_proj|v_proj|o_proj|index_q_proj|index_k_proj)\.weight$/)?.[1]:undefined;
 if(attn)return ({q_proj:"[N_h·D_h,H]",k_proj:"[N_kv·D_h,H]",v_proj:"[N_kv·D_h,H]",o_proj:"[H,N_h·D_h]",index_q_proj:"[N_idx·D_idx,H]",index_k_proj:"[D_idx,H]"} as Record<string,string>)[attn];
 if(key.includes("self_attn.index_")&&key.includes("norm"))return "[D_idx]";
 if(key.includes("self_attn.")&&key.includes("norm"))return "[D_h]";
 if(key.includes("block_sparse_moe.gate.weight"))return "[E,H]";
 if(key.includes("e_score_correction_bias"))return "[E]";
 if(shape.length===1&&shape[0]===6144)return "[H]";
 return `[${shape.map((_,i)=>`d${i}`).join(",")}]`;
}
/** Storage axes are never inferred by replacing matching numbers with model symbols. */
export function tensorLayout(key:string,meta:TensorMeta){
 const role=tensorRole(key),aux=role.includes("scale");
 const fp4=/\.experts\.\d+\.w[123]\.weight_packed$/.test(key);
 const scale8=key.endsWith("weight_scale_inv")&&meta.dtype==="U8";
 const shape=[...meta.shape];if(fp4)shape[shape.length-1]*=2;
 const format=fp4?"NVFP4 · FP4 E2M1 codes":scale8?"MXFP8 scale · E8M0（U8 编码）":meta.dtype==="F8_E4M3"&&!aux?"MXFP8 · FP8 E4M3":aux?`${meta.dtype} · scale 元数据`:meta.dtype;
 const bytes=meta.bytes??product(meta.shape)*(dtypeBytes[meta.dtype]??0);
 const symbol=logicalSymbol(key,shape),tokens=new Set(symbol.match(/[A-Za-z_][A-Za-z_0-9]*/g)??[]);
 const candidates={...SHAPE_SYMBOLS,...Object.fromEntries(shape.map((n,i)=>[`d${i}`,n]))};
 const symbolValues=Object.fromEntries(Object.entries(candidates).filter(([name])=>tokens.has(name)));
 const symbolDefinitions=Object.entries(symbolValues).map(([name,n])=>`${name}=${n}`).join(" · ");
 return {role,aux,fp4,format,logicalShape:shape,logicalSymbol:symbol,symbolValues,symbolDefinitions,storageShape:meta.shape,bytes,
  valueCount:product(shape),storedElements:product(meta.shape),
  storageSymbol:fp4?(key.includes(".w2.")?"[H,I/2]":"[I,H/2]"):`[${meta.shape.map((_,i)=>`s${i}`).join(",")}]`,
  note:fp4?"仅沿输入轴打包：2 个 4-bit FP4 编码 / 1 个 U8 字节。相对每值占 1 字节的数组，主权重 payload 字节数为一半（不含 scale）；逻辑输入通道数保持不变；还需 block/global scale。":scale8?"U8 在此保存缩放指数编码；它不是两值 FP4 打包，也不是神经网络矩阵。":aux?"辅助量化张量；其形状表示 scale 的组织方式，不代表矩阵的输入/输出宽度。":meta.dtype==="F8_E4M3"?"每个 FP8 值占 1 字节，矩阵轴长度不减半。另存 block scale；不能仅凭 U8 dtype 判断是否 FP4。":"存储形状与逻辑参数形状相同；加载后的融合、分片、重排另由后端决定。"};
}
