import {createServer} from 'vite';
import {readFileSync,writeFileSync} from 'node:fs';
const read=p=>JSON.parse(readFileSync(p,'utf8'));
const main=read('public/flash-data/tensor-shapes.json'),draft=read('public/flash-data/dspark-tensors.json');
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom'});
try{
 const {tensorLayout,moduleUse,operationUse,shapeText}=await server.ssrLoadModule('/app/models/minimax-m31-flash/storage.ts');
 const groups=new Map(),models=[];
 for(const [model,metadata] of [['main',main],['dspark',draft]]){
  let bytes=0,auxBytes=0,parameterValues=0,parameterBytes=0;
  for(const [key,meta] of Object.entries(metadata)){
   const l=tensorLayout(key,meta),module=moduleUse(key,model),operation=operationUse(key);
   bytes+=meta.bytes;if(l.aux)auxBytes+=meta.bytes;else {parameterBytes+=meta.bytes;parameterValues+=l.valueCount;}
   const id=[model,module,operation,l.role,l.format].join('|');
   if(!groups.has(id))groups.set(id,{model,module,operation,role:l.role,format:l.format,dtype:meta.dtype,bytes:0,tensors:0,values:0,examples:[],patterns:[],pattern:key.replace(/\.layers\.\d+\./g,'.layers.{l}.').replace(/\.experts\.\d+\./g,'.experts.{e}.')});
   const g=groups.get(id);g.bytes+=meta.bytes;g.tensors++;g.values+=l.valueCount;
   const pattern=key.replace(/\.layers\.\d+\./g,'.layers.{l}.').replace(/\.experts\.\d+\./g,'.experts.{e}.');if(!g.patterns.includes(pattern))g.patterns.push(pattern);
   const example={key,logicalSymbol:l.aux?'辅助张量 · 无矩阵输入/输出轴':l.logicalSymbol,symbolDefinitions:l.symbolDefinitions,logicalShape:l.aux?null:shapeText(l.logicalShape),storageShape:shapeText(meta.shape),bytes:meta.bytes,note:l.note};
   if(g.examples.length<3&&!g.examples.some(e=>e.storageShape===example.storageShape))g.examples.push(example);
  }
  models.push({id:model,bytes,tensors:Object.keys(metadata).length,auxBytes,parameterBytes,parameterValues});
 }
 const result={revision:'262ae6830934f3512211081b8a3f16425b93a239',method:'Exact sum of safetensors header data_offsets[end]-data_offsets[start]; each key counted once; no tensor payload downloaded',models,groups:[...groups.values()].sort((a,b)=>b.bytes-a.bytes)};
 if(models[0].bytes!==250103411200||models[0].tensors!==89632)throw Error('Main index reconciliation failed');
 writeFileSync('public/flash-data/weight-summary.json',JSON.stringify(result,null,2)+'\n');
 console.log(JSON.stringify({models,groups:groups.size},null,2));
}finally{await server.close();}
