import type {OpNode,OpKind,CodeSection,Weight} from "../types";
import sections from "./source-sections.json";
import {MOE_MATH} from "./moe-math";

/** Logical per-token operations; intermediates need not be materialized by FusedMoE. */
export function withMoeDetails(g:Record<string,OpNode>):Record<string,OpNode>{
 const node=(key:keyof typeof MOE_MATH,kind:OpKind,title:string,input:string,inputShape:string,output:string,outputShape:string,summary:string,weights:Weight[],codeSections:CodeSection[]):OpNode=>({
  ...g.experts,id:`s-moe-${key}`,kind,title,kicker:"MOE · 单 TOKEN 逻辑拆解",input,inputShape,output,outputShape,summary,weights,
  formula:MOE_MATH[key],latex:MOE_MATH[key],formulaNote:summary,
  runtime:"FusedMoE 内部数学步骤 · 全局逻辑形状，非独立 kernel / buffer 保证",
  codeSections,source:codeSections[0].location,sourceUrl:codeSections[0].url!,code:codeSections[0].code,
 });
 const routed= g.experts.weights;
 const shared= g.shared.weights;
 const expertCode=[...sections.experts,...sections.expert_activation];
 const sharedCode=[...sections.shared,...sections.gateup,...sections.swiglu];
 const swigluSummary="对 Gate 先做上界 min(g,7)，对 Up 做 clip(u,−7,7)，再计算 ḡ·sigmoid(1.702ḡ)·(ū+1)。这是专家内部的逐通道非线性门控，激活本身没有可训练权重；输出宽度仍为 3072。";
 return {...g,
  experts:{...g.experts,summary:"Top-4 选路后，每个入选专家执行 Gate/Up → SwiGLU-OAI → Down，最后按路由权重加权求和。展开图逐步展示这条完整链路。",formulaNote:"SwiGLU-OAI 位于每个专家两次线性投影之间；不负责选专家。图示为逻辑拆解，实际由 FusedMoEFactory 调度。",codeSections:expertCode},
  shared:{...g.shared,summary:"所有 token 都执行一套 Gate/Up → SwiGLU-OAI → Down，共享专家不参与 Top-4，不乘 routed 权重。",codeSections:sharedCode},
  choice:node("choice","route","Sigmoid + Top-4","R · router logits","[128]","𝓔 · ids + 原始 scores s","[4] + [128]","s=σ(R)；按 s+b_corr 选 4 个专家。bias 只改变选谁，不进入混合权重。",routed.filter(w=>w.key.includes("correction_bias")),sections.experts),
  routeweights:node("weights","scale","Renormalize × 2","𝓔 + 原始 sigmoid scores s","[4] + [128]","ŵ · 路由混合权重","[4]","gather 选中专家的原始分数，除以这 4 个分数之和，再乘 2；四个混合权重之和是 2。",[],sections.experts),
  expertgateup:node("gateup","linear","Expert Gate + Up","x=Û + 选中专家 ids 𝓔","[6144] + [4]","g_e / u_e · 每个入选专家","2 × [3072] / expert","每个入选专家各用自己的 W1 / W3，将同一个 6144 维 x 投影成两个 3072 维分支。逻辑 W1,W3=[3072,6144]；w13 按全部 Gate 在前、全部 Up 在后打包。NVFP4 U8 存储形状见权重，不能当逻辑形状。",routed.filter(w=>/\.w[13]\./.test(w.key)),expertCode),
  expertswiglu:node("swiglu","activation","SwiGLU-OAI · per expert","g_e + u_e","2 × [3072] / expert","z_e · 门控后的特征","[3072] / expert",swigluSummary,[],sections.expert_activation),
  expertdown:node("down","linear","Expert Down Projection","z_e","[3072] / expert","y_e · 单专家输出","[6144] / expert","各专家用独立 W2=[6144,3072] 把激活特征投回 hidden size。此处尚未与其他专家求和；后端可融合投影、加权和归并。",routed.filter(w=>/\.w2\./.test(w.key)),expertCode),
  expertcombine:node("combine","add","Weighted Sum · 4 experts","4 个 y_e + 对应 ŵ_e","[4,6144] + [4]（逻辑）","Y_routed","[6144]","每个专家的 6144 维输出乘自己的路由权重，再沿 4 个专家求和。全局公式不规定 TP/EP buffer；PP=1 的 TP 归并可延迟到下一 norm。",[],sections.experts),
  sharedgateup:node("sharedgateup","linear","Shared Gate + Up","x=Û · 所有 token","[6144]","g_s / u_s","2 × [3072]（逻辑）","共享专家使用自己的 Gate / Up 权重，同样从 6144 维投影到两路 3072 维。所有 token 都执行，无需 Top-4 ids。TP 下每 rank 可只持有 3072/TP 通道。",shared.filter(w=>!w.key.includes("down_proj")),sharedCode),
  sharedswiglu:node("sharedswiglu","activation","SwiGLU-OAI · shared","g_s + u_s","2 × [3072]（逻辑）","z_s","[3072]（逻辑）",swigluSummary+"共享分支使用相同 c、α、β；TP 时按本地通道逐元素计算。",[],[...sections.swiglu,...sections.shared]),
  shareddown:node("shareddown","linear","Shared Down Projection","z_s","[3072]（逻辑）","Y_shared","[6144]","共享 Down 将 3072 维投回 6144 维；不乘 Top-4 路由权重，随后与 Y_routed 相加。共享 MLP 以 reduce_results=False 交给 FusedMoEFactory，图中是全局逻辑输出。",shared.filter(w=>w.key.includes("down_proj")),sharedCode),
 };
}
