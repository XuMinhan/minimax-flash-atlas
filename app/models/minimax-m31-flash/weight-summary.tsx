import {useEffect,useMemo,useState} from "react";
import {bytesText,smallBytes} from "./storage";
import {HF_REVISION,WEIGHTS_URL} from "./sources";
type ModelStats={id:string;bytes:number;tensors:number;auxBytes:number;parameterBytes:number;parameterValues:number};
type Group={model:string;module:string;operation:string;role:string;format:string;dtype:string;bytes:number;tensors:number;values:number;pattern:string;patterns:string[];examples:{key:string;logicalSymbol:string;symbolDefinitions:string;logicalShape:string|null;storageShape:string;bytes:number;note:string}[]};
type Summary={revision:string;models:ModelStats[];groups:Group[]};
const integer=(n:number)=>n.toLocaleString();

export function StorageGuide(){return <section className="storage-guide"><h2>先分清：计算矩阵、文件里的字节、运行时布局</h2>
 <p><b>逻辑符号</b>用于公式，<b>逻辑尺寸</b>代入模型配置，<b>checkpoint 存储尺寸</b>按文件头原样记录。三者分别标注。语言 H=6144，I=3072，I_shared=3072，I_dense=12288；视觉 D_v=1280，I_v=5120。每个示例和权重卡分别列出它使用的符号定义；d0、d1 等按所列尺寸依序定义该张量的逻辑轴；矩阵约定为 [输出维,输入维]，计算使用 xWᵀ。</p>
 <div className="storage-guide-grid"><article><h3>NVFP4 · 路由专家</h3><p>一个逻辑权重值用一个 4-bit FP4 编码表示。U8 是 8-bit 存储容器，一个容器装两个编码；它不是一个 8-bit 整数权重。</p><code>逻辑 [O,K] → 存储 U8 [O,K/2]</code><p>只沿输入轴打包；输出轴 O 不变。两个编码分别解码为 FP4 值，再结合 scale 参与计算。量化后的逻辑近似权重依然有 K 个输入通道。</p><p>还要存每 16 个逻辑输入值一个 block scale（FP8 E4M3）、global scale（F32）及激活 input_global_scale（F32）。这些字节也计入总量。后端的缩放/倒数约定见固定源码，不把 packed 字节直接当作矩阵乘输入。</p></article>
 <article><h3>MXFP8 · Attention / Dense FFN / Shared</h3><code>逻辑 [O,K] → 存储 FP8 [O,K]</code><p>每个 FP8 值占 1 字节，矩阵轴长度不减半。每 32 个输入值另有一个 block scale；该 checkpoint 的 weight_scale_inv 用 U8 保存指数编码，存储形状为 [O,K/32]。</p><p><b>U8 不一定表示 FP4。</b>必须同时看 tensor key、量化方案和用途。scale 是量化辅助数据，不是另一张 [O,K/32] 的神经网络投影矩阵。</p></article>
 <article><h3>BF16 / F32 · 未打包参数</h3><p>BF16 每元素 2 字节，F32 每元素 4 字节；逻辑形状和 checkpoint 形状相同。Router 使用 F32；视觉、Embedding、LM Head、Norm 等以各文件头记录为准。</p><p>TP/EP 分片、QKV 或 Gate/Up 融合、padding 和 kernel 重排发生在加载/执行阶段。它们另列为运行时布局，不能改写 checkpoint 的文件头形状。W13 融合两路 Gate/Up 与 FP4 两值/字节打包也是两件事。</p></article></div>
 <details className="down-storage-example" open><summary>例：主模型单个路由专家 Down，为什么是 [6144,1536]？</summary><table><thead><tr><th>对象 / 口径</th><th>符号</th><th>配置代入 / 文件头</th><th>数据区大小</th></tr></thead><tbody>
  <tr><th>逻辑计算矩阵 W2</th><td>[H,I]</td><td>[6144,3072] · 18,874,368 个 FP4 近似权重值</td><td>逻辑形状本身不指定内存格式</td></tr>
  <tr><th>weight_packed · U8 容器</th><td>[H,I/2]</td><td>[6144,1536] · 每字节装两个 FP4 编码</td><td>9,437,184 B = 9 MiB</td></tr>
  <tr><th>weight_scale · FP8 E4M3</th><td>[H,I/16]</td><td>[6144,192] · 每 16 个逻辑权重一个 scale</td><td>1,179,648 B = 1.125 MiB</td></tr>
  <tr><th>weight_global_scale + input_global_scale</th><td>两个标量</td><td>[] + [] · 各 F32</td><td>8 B</td></tr>
  <tr><th>这组 checkpoint 张量合计</th><td>不含其他专家</td><td>packed + block scale + 两个 global scale</td><td>10,616,840 B = 10.125 MiB + 8 B</td></tr>
 </tbody></table><p>若把该逻辑矩阵物化为 BF16，单矩阵会占 36 MiB；这只是条件计算，不代表运行时一定生成此副本。SwiGLU-OAI 的两路 I 维特征合成一路 I 维，和此处 FP4 存储字节减半相互独立。</p></details>
 </section>}

export function WeightSummary({initialData}:{initialData?:Summary}){
 const [data,setData]=useState<Summary|undefined>(initialData),[error,setError]=useState(""),[model,setModel]=useState("main"),[module,setModule]=useState("all"),[query,setQuery]=useState("");
 useEffect(()=>{if(initialData)return;fetch('./flash-data/weight-summary.json').then(r=>{if(!r.ok)throw Error(String(r.status));return r.json()}).then(setData).catch(e=>setError(e.message))},[initialData]);
 const modules=useMemo(()=>[...new Set(data?.groups.filter(g=>g.model===model).map(g=>g.module))], [data,model]);
 const groups=data?.groups.filter(g=>g.model===model&&(module==="all"||g.module===module)&&`${g.module} ${g.operation} ${g.role} ${g.format} ${g.patterns.join(" ")}`.toLowerCase().includes(query.toLowerCase()))??[];
 const stats=data?.models.find(m=>m.id===model),filteredBytes=groups.reduce((n,g)=>n+g.bytes,0);
 return <section className="flash-weight-summary"><header><span>WEIGHT / STORAGE SUMMARY</span><h1>权重在哪里使用，以什么格式保存，占多少数据空间</h1><p>主模型 48 个分片 + DSpark 独立 1 个分片，全部文件头已读取。按每个 tensor 的 data_offsets 差值精确求和；每个 key 只计一次。这里统计 checkpoint 的 tensor 数据区，不含文件头/索引，也不是实测 GPU 峰值。</p><a href={`${WEIGHTS_URL}/tree/${HF_REVISION}`} target="_blank" rel="noreferrer">固定 revision · 官方 checkpoint ↗</a></header>
  {!data&&!error&&<p>读取精确统计…</p>}{error&&<p role="alert">读取失败：{error}</p>}
  <nav className="storage-model-tabs" aria-label="选择统计范围"><button className={model==="main"?"active":""} onClick={()=>{setModel("main");setModule("all")}}>主模型 · 48 shards</button><button className={model==="dspark"?"active":""} onClick={()=>{setModel("dspark");setModule("all")}}>DSpark · 独立 checkpoint</button></nav>
  {stats&&<div className="storage-stat-cards"><article><span>全部 tensor 数据区</span><strong>{bytesText(stats.bytes)}</strong><small>{integer(stats.bytes)} bytes · {integer(stats.tensors)} 个 tensor</small></article><article><span>非 scale 张量数据</span><strong>{bytesText(stats.parameterBytes)}</strong><small>矩阵、Norm、bias 等编码后的数据</small></article><article><span>量化辅助 scale 数据</span><strong>{bytesText(stats.auxBytes)}</strong><small>包含 weight scale 与 activation scale，已计入总量</small></article><article><span>非 scale 逻辑元素数</span><strong>{(stats.parameterValues/1e9).toFixed(3)} B</strong><small>FP4 按两个值/U8 还原计数；不是 U8 元素数</small></article></div>}
  <section className="storage-memory-note"><h2>这些数字如何理解为“内存占用”</h2><p>若保留 checkpoint 的编码与所有 scale，全量驻留的原始 tensor 字节基数就是上方总量。实际运行还要考虑后端重排/转换、复制或 padding、KV/Indexer cache、激活、通信及工作区，因此这些数字不能直接当作单卡显存峰值。Top-4 表示每 token 使用四个路由专家，并不表示只需加载四个专家的权重；TP/EP 分配也不能一律按总量除卡数。</p><p>GB=10⁹ bytes，GiB=2³⁰ bytes，MiB=2²⁰ bytes。DSpark 单独统计，不混入主模型 250.10 GB；启用草稿时还需考虑它自己的权重与运行空间。DSpark 此页核实权重文件事实，具体执行内核与峰值未实测。</p></section>
  <StorageGuide/>
  <h2>各模块权重与精确大小</h2>
  <div className="flash-filter"><input aria-label="搜索权重总结" value={query} onChange={e=>setQuery(e.target.value)} placeholder="搜索位置、用途、FP4/FP8、tensor key…"/><select aria-label="筛选权重用途" value={module} onChange={e=>setModule(e.target.value)}><option value="all">所有模块</option>{modules.map(m=><option key={m}>{m}</option>)}</select></div>
  {stats&&<p className="storage-count">当前 {groups.length} 类，共 {bytesText(filteredBytes)}；下表按用途、格式及辅助数据分组。</p>}
  <div className="storage-table-scroll"><table className="storage-summary-table"><thead><tr><th>使用位置 / 运算</th><th>权重与辅助 tensor</th><th>格式 / dtype</th><th>逻辑与存储形状 · 示例</th><th>精确数据量</th></tr></thead><tbody>{groups.map(g=><tr key={`${g.module}-${g.operation}-${g.role}-${g.format}`}><td><b>{g.module}</b><p>{g.operation}</p></td><td><b>{g.role}</b>{g.patterns.map(p=><code key={p}>{p}</code>)}<small>{integer(g.tensors)} 个 tensor，覆盖所有匹配层/专家</small></td><td>{g.format}<small>文件头 dtype：{g.dtype}</small></td><td>{g.examples.map(e=><div className="storage-shape-example" key={e.key}>{e.logicalShape?<><span>逻辑符号 <code>{e.logicalSymbol}</code></span><small>{e.symbolDefinitions}</small><span>逻辑尺寸 <code>{e.logicalShape}</code></span></>:<span>量化辅助数据 · 不作为投影矩阵</span>}<span>存储尺寸 <code>{e.storageShape}</code></span><details><summary>示例来源与编码说明</summary><code>{e.key}</code><p>{e.note}</p><small>此示例 tensor：{smallBytes(e.bytes)} · {integer(e.bytes)} B</small></details></div>)}</td><td><strong>{bytesText(g.bytes)}</strong><small>{integer(g.bytes)} B</small><small>{stats?(100*g.bytes/stats.bytes).toFixed(3):0}% · 当前 checkpoint</small></td></tr>)}</tbody></table></div>
  
 </section>;
}
