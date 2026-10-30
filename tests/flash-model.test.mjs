import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,existsSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {createServer} from 'vite';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import katex from 'katex';
const read=p=>readFileSync(new URL('../'+p,import.meta.url),'utf8');
const json=p=>JSON.parse(read(p));
const config=json('public/flash-data/config.json'),draft=json('public/flash-data/dspark-config.json'),index=json('public/flash-data/weights.json'),shapes=json('public/flash-data/tensor-shapes.json');
const keys=new Map(index.rows.map(([p,s,n])=>[index.prefixes[p]+index.suffixes[s],`model-${String(n).padStart(5,'0')}-of-00048.safetensors`]));
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom'});
try{
 const {flashGraph}=await server.ssrLoadModule('/app/models/minimax-m31-flash/operators.ts');
 const {DetailPanel}=await server.ssrLoadModule('/app/models/minimax-m31-flash/details.tsx');
 const {MiniMaxView}=await server.ssrLoadModule('/app/models/minimax-m31-flash/view.tsx');
 const {CONFIG_GROUPS}=await server.ssrLoadModule('/app/models/minimax-m31-flash/config.ts');
 test('Flash: 60 sparse attention layers with an independent 3/57 FFN schedule',()=>{
  assert.equal(config.text_config.num_hidden_layers,60);assert.deepEqual(config.text_config.sparse_attention_config.sparse_attention_freq,Array(60).fill(1));
  assert.deepEqual(config.text_config.moe_layer_freq,[...Array(3).fill(0),...Array(57).fill(1)]);
  for(let l=0;l<60;l++){const g=flashGraph(l);assert.equal(g.packed.id,'s-packed');assert.equal(g.idxcache.id,'s-idxcache');assert.equal(Boolean(g.gateup),l<3)}
 });
 test('Flash visible weights use exact checkpoint keys, formats and shard mappings',()=>{
  for(let l=0;l<60;l++)for(const n of Object.values(flashGraph(l)))for(const w of n.weights){
   assert.equal(keys.get(w.key),w.shard,`${l}: ${n.id}: ${w.key}`);
   if(w.key.includes('.experts.')){assert.match(w.key,/weight_packed$/);assert.equal(w.dtype,'U8');assert.match(w.note,/NVFP4/)}
   if(shapes[w.key]){assert.equal(w.shape,'['+shapes[w.key].shape.join(',')+']');assert.equal(w.dtype,shapes[w.key].dtype)}
  }
 });
 test('Flash inventory losslessly reconstructs all 89632 tensor keys in 48 shards',()=>{
  assert.equal(index.rows.length,89632);assert.equal(keys.size,89632);assert.equal(new Set(keys.values()).size,48);
  assert.equal(index.totalBytes,250103411200);assert.equal(Object.keys(shapes).length,5277);
  for(const [k,m] of Object.entries(shapes))assert.equal(keys.get(k),`model-${String(m.shard).padStart(5,'0')}-of-00048.safetensors`);
 });
 test('Flash config reference includes quantization and independent DSpark settings',()=>{
  const text=CONFIG_GROUPS.find(g=>g.title==='text_config');assert.equal(text.rows.find(r=>r[0]==='rope_theta')[1],'10000000');
  assert.ok(CONFIG_GROUPS.some(g=>g.title==='quantization_config'));assert.ok(CONFIG_GROUPS.some(g=>g.title==='DSpark text_config'));
  assert.equal(draft.text_config.num_hidden_layers,5);assert.equal(draft.text_config.use_mtp,false);assert.equal(draft.text_config.dspark_block_size,7);
 });
 test('Flash source traces contain pinned exact excerpts, no floating Transformers claims',()=>{
  const sections=json('app/models/minimax-m31-flash/source-sections.json');
  for(const ss of Object.values(sections))for(const s of ss){assert.match(s.url,/\/ced6857afa0ea7b2e3f0846a62e1394e90f15607\//);assert.doesNotMatch(s.url,/\/main\//);const m=s.location.match(/L(\d+)–(\d+)/);assert.equal(s.code.split('\n').length,Number(m[2])-Number(m[1])+1)}
  for(const n of Object.values(flashGraph(3)))if(!['io','mask','route'].includes(n.kind)||n.id.startsWith('s-idx'))assert.ok(n.codeSections?.length,n.id);
 });
 test('Flash operator formulas, extra modalities and symbolic terms compile',()=>{
  const extra=json('app/models/minimax-m31-flash/extra-nodes.json');
  const nodes=[...Object.values(flashGraph(0)),...Object.values(flashGraph(3)),...Object.values(extra)];
  for(const n of nodes)if(n.latex)assert.doesNotThrow(()=>katex.renderToString(n.latex,{throwOnError:true,displayMode:true,strict:false}),n.id);
  assert.match(extra.draft.summary,/不是旧 MTP|不能作为|不能把/);
 });
 test('Flash L0–2 and L3–59 attention stage summaries both describe MSA',()=>{
  const render=(layerType,expanded)=>renderToStaticMarkup(createElement(DetailPanel,{node:null,tab:'io',setTab:()=>{},pinned:false,onClear:()=>{},expanded,layerType}));
  const first=render('dense','attention'),later=render('sparse','attention');
  assert.equal(first,later);assert.match(first,/Indexer/);assert.match(first,/Top-16/);assert.doesNotMatch(first,/DENSE ATTENTION|完整可见 KV/);
  assert.match(render('dense','ffn'),/SwiGLU-OAI MLP/);assert.match(render('sparse','ffn'),/Top-4 MoE/);
 });
 test('Flash native page preserves Model Atlas operator and inspector structure',()=>{
  const html=renderToStaticMarkup(createElement(MiniMaxView,{onExpandedChange:()=>{}}));
  assert.match(html,/decoder-node-graph/);assert.match(html,/input_layernorm.weight/);assert.match(html,/post_attention_layernorm.weight/);
  assert.match(html,/MSA.*Dense|MSA.*SwiGLU/);assert.match(html,/选择实际 Decoder 层/);assert.match(html,/Archify 验证图/);
  assert.doesNotMatch(html,/GQA \+ Partial RoPE \+ SwiGLU|59 shards|869 GB/);
 });
 const {MoEDiagram,MOE_DETAIL_EDGES}=await server.ssrLoadModule('/app/models/minimax-m31-flash/moe-diagram.tsx');
 const {FORMULA_STEPS_BY_ID}=await server.ssrLoadModule('/app/models/minimax-m31-flash/formulas.ts');
 test('Flash MoE exposes routed and shared activations between their own projections',()=>{
  const g=flashGraph(3),html=renderToStaticMarkup(createElement(MoEDiagram,{g,active:'',onHover:()=>{},onLeave:()=>{},onSelect:()=>{},onClose:()=>{}}));
  const connections=new Set(MOE_DETAIL_EDGES.map(e=>`${e.from}>${e.to}`));
  for(const lane of ['expert','shared']){
   assert.ok(connections.has(`md-${lane}gateup>md-${lane}swiglu`));
   assert.ok(connections.has(`md-${lane}swiglu>md-${lane}down`));
   assert.equal(g[`${lane}swiglu`].weights.length,0);
   assert.match(html,new RegExp(`data-graph-id="md-${lane}swiglu"`));
  }
  assert.ok(connections.has('md-choice>md-expertgateup'));
  assert.ok(connections.has('md-routeweights>md-expertcombine'));
  assert.ok(!MOE_DETAIL_EDGES.some(e=>e.from==='md-routeweights'&&e.to.includes('shared')));
  assert.match(html,/SwiGLU-OAI · per expert/);assert.match(html,/SwiGLU-OAI · shared/);
  assert.match(html,/1.702/);assert.match(html,/min\(g,7\)/);assert.match(html,/clip\(u,−7,7\)/);
 });
 test('Flash fused expert inspector explains activation, down projection and weighted output',()=>{
  const g=flashGraph(3);assert.equal(FORMULA_STEPS_BY_ID['s-experts'].length,8);
  for(const id of ['experts','shared','expertswiglu','sharedswiglu']){
   const html=renderToStaticMarkup(createElement(DetailPanel,{node:g[id],tab:'formula',setTab:()=>{},pinned:true,onClear:()=>{},expanded:'ffn',layerType:'sparse'}));
   assert.match(html,/Gate/);assert.match(html,/Up/);assert.match(html,/Down/);
   assert.match(html,/1.702/);assert.match(html,/常量/);assert.match(html,/formula-terms/);
   assert.doesNotMatch(html,/katex-error/);
  }
  assert.ok(g.expertgateup.weights.every(w=>/\.w[13]\./.test(w.key)));
  assert.ok(g.expertdown.weights.every(w=>/\.w2\./.test(w.key)));
  assert.ok(g.choice.weights.every(w=>/correction_bias/.test(w.key)));
  for(const steps of Object.values(FORMULA_STEPS_BY_ID))for(const step of steps)assert.doesNotThrow(()=>katex.renderToString(step.formula,{throwOnError:true,strict:false}));
 });
 test('Seven published Archify documents exactly match their delivered artifact hashes',()=>{
  for(const name of ['overview','indexer','attention','dense-ffn','moe','vision','dspark']){
   const spec=json(`docs/models/minimax-m31-flash/diagrams/${name}.json`);assert.equal(spec.meta.quality_profile,'showcase');
   const file=`public/models/minimax-m31-flash/diagrams/${name}.html`;assert.ok(existsSync(new URL('../'+file,import.meta.url)));
   const registry=json('public/flash-data/diagram-provenance.json');const d=registry[name];
   assert.equal(createHash('sha256').update(read(file)).digest('hex'),d.sha256);assert.equal(d.gates['browser-check'],'pass');
  }
 });
}finally{await server.close()}
