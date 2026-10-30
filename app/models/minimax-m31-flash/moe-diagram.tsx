import type {OpNode} from "../types";
import type {GraphEdge} from "../../graph/types";
import {GraphSurface} from "../../graph/surface";
import {GraphPan} from "../../graph/pan";
import {Tensor} from "../../graph/nodes";

export const MOE_DETAIL_EDGES:GraphEdge[]=[
 {from:"md-input",to:"md-router",fanout:"md-inputs",departure:24},
 {from:"md-input",to:"md-x",fanout:"md-inputs",departure:24},
 {from:"md-input",to:"md-xs",fanout:"md-inputs",departure:24},
 {from:"md-router",to:"md-choice"},
 {from:"md-choice",to:"md-routeweights"},
 {from:"md-choice",to:"md-expertgateup",fromPort:"right",toPort:"left"},
 {from:"md-x",to:"md-expertgateup"},
 {from:"md-expertgateup",to:"md-expertswiglu"},
 {from:"md-expertswiglu",to:"md-expertdown"},
 {from:"md-expertdown",to:"md-expertcombine"},
 {from:"md-routeweights",to:"md-expertcombine",fromPort:"bottom",toPort:"left",approach:28},
 {from:"md-xs",to:"md-sharedgateup"},
 {from:"md-sharedgateup",to:"md-sharedswiglu"},
 {from:"md-sharedswiglu",to:"md-shareddown"},
 {from:"md-shareddown",to:"md-sharedout"},
 {from:"md-expertcombine",to:"md-sum",toPort:"top-left",approach:26},
 {from:"md-sharedout",to:"md-sum",toPort:"top-right",approach:26},
 {from:"md-sum",to:"md-output"},
];

export function MoEDiagram({g,active,onHover,onLeave,onSelect,onClose}:{g:Record<string,OpNode>;active:string;onHover:(n:OpNode)=>void;onLeave:()=>void;onSelect:(n:OpNode)=>void;onClose:()=>void}){
 const selectProps=(node:OpNode)=>({"aria-pressed":active===node.id,onMouseEnter:()=>onHover(node),onMouseLeave:onLeave,onFocus:()=>onHover(node),onBlur:onLeave,onPointerDown:()=>onSelect(node),onClick:(event:React.MouseEvent)=>{if(event.detail===0)onSelect(node)}});
 const N=({id,lane,caption}:{id:string;lane:string;caption:string})=><button className={`op-node moe-detail-op op-${g[id].kind} ${active===g[id].id?"active":""}`} data-graph-id={`md-${id}`} {...selectProps(g[id])}><small>{lane}</small><b>{g[id].title}</b><code>{caption}</code></button>;
 return <section className="stage-zoom lesson-zoom moe-detail-zoom"><header><span>TOP-4 MOE + SHARED EXPERT · L3–59</span><button onClick={onClose}>收起 ×</button></header>
  <div className="moe-explanation-bar"><button {...selectProps(g.experts)}>Fused Top-4 Routing + Experts · 完整公式 ↗</button><button {...selectProps(g.shared)}>Shared Expert · 完整公式 ↗</button><span>每个 token：128 选 4 + 1 共享 · H=6144 · I=3072</span></div>
  <GraphPan><GraphSurface className="moe-detail-graph" edges={MOE_DETAIL_EDGES}>
   <Tensor name="x = Û · Post-attn RMSNorm" shape="每 token [6144] · 整批 [B,S,6144]" graphId="md-input"/>
   <N id="router" lane="01 · ROUTER · FP32" caption="x Wrouterᵀ → R [128]"/>
   <Tensor name="路由专家输入 x" shape="同一 x → 每个选中专家" graphId="md-x"/>
   <Tensor name="共享专家输入 x" shape="所有 token · 不经过 Top-4" graphId="md-xs"/>
   <N id="choice" lane="ROUTING · 128 → 4" caption="s = σ(R) · Top-4(s + bias)"/>
   <N id="routeweights" lane="ROUTING · 保留原始 s" caption="ŵe = 2se / Σselected sj"/>
   <N id="expertgateup" lane="02 · ROUTED EXPERTS ×4" caption="W1 / W3 · 6144 → 3072 ×2"/>
   <N id="expertswiglu" lane="ACTIVATION · c=7 · α=1.702 · β=1" caption="ḡ ⊙ σ(1.702ḡ) ⊙ (ū + 1)"/>
   <N id="expertdown" lane="ROUTED · 每专家独立 W2" caption="ze W2,eᵀ · 3072 → 6144"/>
   <N id="expertcombine" lane="ROUTED · 跨 4 个专家求和" caption="Yrouted = Σe ŵe ye"/>
   <N id="sharedgateup" lane="03 · SHARED EXPERT ×1" caption="Wgate / Wup · 6144 → 3072 ×2"/>
   <N id="sharedswiglu" lane="ACTIVATION · 相同 c / α / β" caption="ḡs ⊙ σ(1.702ḡs) ⊙ (ūs + 1)"/>
   <N id="shareddown" lane="SHARED · 无路由混合权重" caption="zs Wdown,sᵀ · 3072 → 6144"/>
   <Tensor name="Yshared" shape="[6144] · 始终参与" graphId="md-sharedout"/>
   <N id="sum" lane="MOE OUTPUT · 逐元素相加" caption="Ymoe = Yrouted + Yshared"/>
   <Tensor name="Ymoe" shape="每 token [6144] · 整批 [B,S,6144]" graphId="md-output"/>
   <aside className="moe-activation-note"><b>SwiGLU-OAI 在做什么？</b><p>Gate：ḡ=min(g,7)<br/>Up：ū=clip(u,−7,7)</p><p>先截断，再用 Gate 的非线性值逐通道调制 Up；激活本身没有可训练权重。</p><p>图示为全局逻辑拆解。中间值可在 fused kernel 内完成，TP/EP 的本地形状由后端决定。</p></aside>
  </GraphSurface></GraphPan>
 </section>;
}
