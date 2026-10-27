import {useState} from "react";
import type {OpNode,Tab} from "../types";
import type {ExpandedStage} from "./types";
import {nextDetailState,type DetailEvent,type DetailState} from "../../detail-selection";
import {flashGraph} from "./operators";
import {DecoderDiagram,LayerNavigator} from "./diagrams";
import {DetailPanel} from "./details";
import {Arrow} from "../../graph/nodes";
import {ResearchPane,GeneratedDiagrams,type Pane} from "./research";
import extraData from "./extra-nodes.json";
import {verifyWeight} from "./data";
export function MiniMaxView({onExpandedChange}:{onExpandedChange:(value:boolean)=>void}){
 const [layer,setLayer]=useState(3),[pane,setPane]=useState<Pane>("decoder"),[expanded,setExpandedState]=useState<ExpandedStage>(null),[tab,setTab]=useState<Tab>("io");
 const [detail,setDetail]=useState<DetailState<OpNode>>({hovered:null,pinned:null});
 const type=layer<3?"dense":"sparse",graph=flashGraph(layer),active=detail.pinned??detail.hovered;
 const update=(event:DetailEvent<OpNode>)=>setDetail(state=>nextDetailState(state,event));
 const expand=(stage:ExpandedStage)=>{setExpandedState(stage);onExpandedChange(stage!==null)};
 const chooseLayer=(n:number)=>{setLayer(n);expand(null);update({type:"clear"})};
 const choosePane=(p:Pane)=>{setPane(p);expand(null);update({type:"clear"})};
 const extra=Object.fromEntries(Object.entries(extraData).map(([id,node])=>[id,{...node,weights:node.weights.map(verifyWeight)} as OpNode]));
 const select=(node:OpNode)=>{update({type:"pin",node});setTab("io")};
 return <><nav className="flash-pane-nav" aria-label="Flash 研究内容">{([["decoder","Decoder 算子图"],["vision","视觉输入 / 输出"],["dspark","DSpark 草稿"],["weights","完整权重目录"],["generated","Archify 验证图"]] as const).map(([id,label])=><button key={id} className={pane===id?"active":""} onClick={()=>choosePane(id)}>{label}</button>)}<span>全部 60 层均为 MSA · 基于 Model Atlas 原生源码适配</span></nav>
 {pane==="weights"?<ResearchPane pane="weights"/>:pane==="generated"?<GeneratedDiagrams/>:<div className="screen-grid"><section className="map-panel">
 {pane==="decoder"?<><div className="model-overview"><button className="model-step" onClick={()=>choosePane("vision")}>Text / Vision Inputs ↗</button><Arrow/><button className="model-step" onClick={()=>select(extra.fusion)}>Embedding Fusion <code>[B,S,H]</code></button><Arrow/><div className="overview-stack"><b>Decoder ×60 · 全层 MSA</b><span><i className="dense"/>L0–2 · MSA + Partial RoPE + SwiGLU-OAI MLP</span><span><i className="sparse"/>L3–59 · MSA + Partial RoPE + Top-4 MoE</span></div><Arrow/><button className="model-step" onClick={()=>select(extra.final)}>Final Gemma RMSNorm <code>[B,S,H]</code></button><Arrow/><button className="model-step" onClick={()=>select(extra.final)}>LM Head <code>[B,S,V]</code></button></div>
 <LayerNavigator type={type} onChange={t=>chooseLayer(t==="dense"?0:3)}/><div className="flash-layer-picker"><label>查看实际层 <select aria-label="选择实际 Decoder 层" value={layer} onChange={e=>chooseLayer(Number(e.target.value))}>{Array.from({length:60},(_,i)=><option key={i} value={i}>L{i} · MSA + {i<3?"Dense FFN":"MoE"}</option>)}</select></label><span>图示按全局逻辑 shape；标有 TP 的 I/O 为每 rank 局部形状。权重分片随层切换。</span></div>
 <section className="layer-canvas"><header><div><span>DECODER LAYER {layer} · MSA + {layer<3?"DENSE FFN":"MOE"}</span><h1>MiniMax Sparse Attention + Partial RoPE + {layer<3?"SwiGLU-OAI MLP":"Top-4 MoE"}</h1></div><div className="node-legend"><span><i className="tensor-swatch"/>TENSOR</span><span><i className="external-swatch"/>EXTERNAL</span><span><i className="weight-swatch"/>WEIGHT</span><span><i className="operator-swatch"/>OPERATOR</span><code>点击模块展开 · 选中算子固定详情</code></div></header><DecoderDiagram type={type} g={graph} active={active?.id??""} expanded={expanded} onExpand={expand} onHover={node=>update({type:"hover",node})} onLeave={()=>update({type:"leave"})} onSelect={select}/></section></>:<ResearchPane pane={pane} extra={extra} onSelect={select}/>}
 </section><DetailPanel node={active} tab={tab} setTab={setTab} pinned={Boolean(detail.pinned)} onClear={()=>update({type:"clear"})} expanded={expanded} layerType={type}/></div>}
 </>;
}
