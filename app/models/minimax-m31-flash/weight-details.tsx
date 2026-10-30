import type {Weight} from "../types";
import {checkpointRecord} from "./data";
import {tensorLayout,shapeText,smallBytes,type TensorMeta} from "./storage";
import {VLLM_COMMIT} from "./sources";
const toMeta=(weight:Weight):TensorMeta=>({shape:JSON.parse(weight.shape),dtype:weight.dtype,bytes:(checkpointRecord(weight.key) as {bytes?:number}|undefined)?.bytes});

export function WeightShapeRows({weight}:{weight:Weight}){
 const layout=tensorLayout(weight.key,toMeta(weight));
 return <dl className="weight-shape-rows">
  <div><dt>逻辑符号 · 计算矩阵</dt><dd><code>{layout.logicalSymbol}</code><small>{layout.symbolDefinitions}</small></dd></div>
  <div><dt>逻辑尺寸 · 配置代入</dt><dd><code>{shapeText(layout.logicalShape)}</code></dd></div>
  <div className="checkpoint-shape"><dt>checkpoint 存储尺寸</dt><dd><code>{shapeText(layout.storageShape)}</code>{layout.fp4&&<small>存储符号 {layout.storageSymbol} · 最后一轴按字节计</small>}</dd></div>
  <div><dt>数值格式 / 容器 dtype</dt><dd>{layout.format} / <code>{weight.dtype}</code></dd></div>
  <div><dt>此 tensor 数据区大小</dt><dd>{smallBytes(layout.bytes)} · {layout.bytes.toLocaleString()} bytes<small>单独统计，不含附属 scale</small></dd></div>
 </dl>;
}

export function WeightCard({weight,nodeId}:{weight:Weight;nodeId:string}){
 if(!checkpointRecord(weight.key))return <article className="binding binding-weight"><b>{weight.key}</b><p>逻辑尺寸（源码推导）：{weight.shape}。checkpoint 存储尺寸 / dtype 尚未绑定，不能当作文件头实测。</p></article>;
 const layout=tensorLayout(weight.key,toMeta(weight));
 const prefix=weight.key.replace(/\.(?:weight_packed|weight)$/,""),aux=["weight_scale","weight_scale_inv","weight_global_scale","input_global_scale"].map(s=>({key:`${prefix}.${s}`,meta:checkpointRecord(`${prefix}.${s}`)})).filter(x=>x.meta);
 return <article className="binding binding-weight weight-detail-card"><div><span>权重输入 · 文件头已核实</span></div><b>{weight.key}</b><WeightShapeRows weight={weight}/><p>{layout.note}</p>
  {layout.fp4&&<details className="packing-explanation"><summary>为什么末维减半？展开看打包与计算的区别</summary><p>逻辑矩阵的一个元素对应一个 FP4 编码，占 4 bit。一个 U8 容器有 8 bit，保存两个相邻输入位置的编码；它表示两个 FP4 数值，不是把一个 U8 整数当作权重。</p><code>逻辑输入轴 K 个 FP4 编码 → checkpoint 输入轴 K/2 个 U8 字节</code><p>解码时分别解释两个 4-bit 编码，并结合 block scale / global scale 参与 GEMM。逻辑通道数仍为 K。后端可以在内核中解码，不保证先生成一份完整 BF16 矩阵。FP4 编码的解释与缩放也不是“把 U8 数值除以 2”。</p><p>这与 SwiGLU 将 Gate/Up 两个 I 维分支逐元素合为一个 I 维激活，是两件独立的事。</p></details>}
  {aux.length>0&&<details className="weight-aux"><summary>附属量化张量 · 另占数据空间</summary>{aux.map(({key,meta})=>{const m=meta as NonNullable<typeof meta>&{bytes:number};return <p key={key}><code>{key}</code><br/>{m.dtype} · 存储 <code>{m.shape}</code> · {smallBytes(m.bytes)}<br/>{key.endsWith("input_global_scale")?"用于激活量化，不是矩阵权重":key.endsWith("weight_scale_inv")?"MXFP8 的 block scale 编码，不是 FP4 打包":key.endsWith("weight_scale")&&layout.fp4?"每 16 个逻辑输入权重一个 block scale":"权重缩放辅助值"}</p>})}</details>}
  <p><i>checkpoint 分片</i> {weight.shard}</p><p><i>运行时布局另列</i> {nodeId==="d-gateup"?"每 TP rank 的 Gate/Up 逻辑权重 [I_dense/TP,H]；不改写上方 checkpoint 存储尺寸。":nodeId==="d-down"?"每 TP rank 的 Down 逻辑权重 [H,I_dense/TP]；不改写上方 checkpoint 存储尺寸。":"融合、TP/EP 分片、kernel 重排发生在加载/运行阶段；上方文件头形状不承诺等于 GPU 内部布局。"}</p>
  {weight.runtime&&<small>源码中的模块归属：{weight.runtime}</small>}
  {layout.fp4&&<a className="weight-source" href={`https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py#L86-L184`} target="_blank" rel="noreferrer">固定源码：packed 权重及 scale 分配 ↗</a>}
 </article>;
}
