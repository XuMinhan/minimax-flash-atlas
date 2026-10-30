import type {FormulaStep} from "./formulas";

export const MOE_MATH = {
 choice:String.raw`s=\sigma(R),\qquad\mathcal E=\operatorname{TopK}_{4}(s+b_{corr})`,
 weights:String.raw`\hat w_e=2\frac{s_e}{\sum_{j\in\mathcal E}s_j},\quad e\in\mathcal E,\quad\sum_e\hat w_e=2`,
 gateup:String.raw`\begin{aligned}g_e&=xW_{1,e}^{\top},\quad u_e=xW_{3,e}^{\top}\\x&\in\mathbb R^{6144},\quad g_e,u_e\in\mathbb R^{3072}\end{aligned}`,
 swiglu:String.raw`\begin{aligned}\bar g_e&=\min(g_e,7)\\\bar u_e&=\operatorname{clip}(u_e,-7,7)\\z_e&=\bar g_e\odot\sigma(1.702\bar g_e)\odot(\bar u_e+1)\end{aligned}`,
 down:String.raw`y_e=z_eW_{2,e}^{\top}\in\mathbb R^{6144},\quad W_{2,e}\in\mathbb R^{6144\times3072}`,
 combine:String.raw`Y_{routed}=\sum_{e\in\mathcal E}\hat w_e y_e\in\mathbb R^{6144}`,
 sharedgateup:String.raw`g_s=xW_{gate,s}^{\top},\quad u_s=xW_{up,s}^{\top}\in\mathbb R^{3072}`,
 sharedswiglu:String.raw`\begin{aligned}\bar g_s&=\min(g_s,7),\quad\bar u_s=\operatorname{clip}(u_s,-7,7)\\z_s&=\bar g_s\odot\sigma(1.702\bar g_s)\odot(\bar u_s+1)\end{aligned}`,
 shareddown:String.raw`Y_{shared}=z_sW_{down,s}^{\top}\in\mathbb R^{6144}`,
};

export const SWIGLU_STEPS: readonly FormulaStep[] = [
 {title:"1 · 先截断两个分支",formula:String.raw`\bar g=\min(g,c),\quad\bar u=\operatorname{clip}(u,-c,c),\quad c=7`,explanation:"g 来自 Gate 投影，u 来自 Up 投影。Gate 只限制上界，负值不会被截到 −7；Up 同时限制上下界。先截断，再将截断后的 g 送入 sigmoid。"},
 {title:"2 · Gate 控制每个特征通道",formula:String.raw`a=\bar g\odot\sigma(\alpha\bar g),\quad\alpha=1.702`,explanation:"σ 是逐元素 sigmoid；α 调整 sigmoid 的斜率。a 是带符号的特征门控值，并非概率分布。这里的 Gate 分支位于单个专家内部，与产生 128 个选路分数的 Router 是两套计算。"},
 {title:"3 · 调制 Up 内容，再交给 Down",formula:String.raw`z=a\odot(\bar u+\beta),\quad\beta=1`,explanation:"Up 提供内容特征；+1 是配置中的常量 β，不是可训练 bias。⊙ 表示逐元素乘，输出仍为 3072 维。SwiGLU-OAI 自身没有权重，负责非线性门控；之后 Down 才把特征投回 6144 维。普通 SiLU(g)⊙u 不包含这里的 α=1.702、β=1 与截断。"},
];

export const MOE_STEPS: readonly FormulaStep[] = [
 {title:"1 · Sigmoid 打分与 Top-4 选路",formula:MOE_MATH.choice,explanation:"对当前 token 的 128 个 FP32 logits R 做 sigmoid 得 s。correction bias 只用于 Top-4 选择，得到专家集合 𝓔；保留未加 bias 的原分数 s。"},
 {title:"2 · 计算四个专家的混合权重",formula:MOE_MATH.weights,explanation:"只取 4 个入选专家的原始分数，归一化后乘 routed_scaling_factor=2。这四个 ŵ 的和为 2；它们控制专家输出的贡献，不替代专家内部的 SwiGLU-OAI。"},
 {title:"3 · 每个入选专家分别投影 Gate / Up",formula:MOE_MATH.gateup,explanation:"x=Û 是一个 token 的归一化输入。每个 e∈𝓔 有自己的 W1（Gate）与 W3（Up），各把 6144 维映射到 3072 维。逻辑形状 W1,W3=[3072,6144]；运行时 w13 按 [全部 Gate; 全部 Up] 打包，uninterleave 指两半布局。"},
 ...SWIGLU_STEPS.map((step,i)=>({...step,title:`${i+4} · ${step.title.slice(4)}`})),
 {title:"7 · 各专家 Down 投影",formula:MOE_MATH.down,explanation:"每个专家用自己的 W2 把激活后的 z_e 从 3072 维投回 6144 维，得到 y_e。这里的矩阵是逻辑权重；checkpoint 中路由专家是 NVFP4 packed，不能把 U8 存储形状当作逻辑矩阵形状。"},
 {title:"8 · 按路由权重归并",formula:MOE_MATH.combine,explanation:"每个 y_e 乘对应 ŵ_e，再对 4 个专家求和；共享专家输出另行相加。公式展示全局逻辑结果，实际 TP/EP 的分片、重排和通信由后端决定；PP=1 的跨 TP 归并可以延迟至下一 norm。"},
];

export const MOE_TERMS: [string,string][] = [
 ["x / Û","一个 token 的 Post-attn RMSNorm 输出，[6144]；逐 token 应用"],
 ["R / s / b_corr","128 个 FP32 logits / sigmoid 路由分数 / 仅用于选择的 correction bias"],
 ["𝓔 / e / j","每 token 选中的 4 个专家集合 / 集合内专家索引"],
 ["ŵ_e","原 sigmoid 分数归一化再乘 2 得到的混合权重"],
 ["W1 / W3 / W2","专家 Gate / Up / Down 逻辑矩阵；NVFP4 存储另见权重详情"],
 ["g / u / z","Gate 投影 / Up 投影 / SwiGLU-OAI 激活特征，各 [3072]"],
 ["c / α / β","截断阈值 7 / sigmoid 斜率 1.702 / Up 常量偏移 1"],
 ["σ / ⊙","逐元素 sigmoid / 逐元素乘，不是跨专家归一化"],
 ["y_e / Y_routed","单个路由专家输出 / 四个专家加权和，各 [6144]"],
 ["下标 s / Y_shared","共享专家 / 无需 Top-4 与路由权重的共享输出"],
];
