import type {ModelDefinition} from "../registry";
import "./layout.css";
import "./flash.css";
import {MiniMaxView} from "./view";
import {HelpModal} from "./reference";
import {CODE_URL,CHECKPOINT_URL,WEIGHTS_URL,VLLM_COMMIT} from "./sources";
export const minimaxFlash={id:"minimax-m31-flash",name:"MiniMax-M3.1-Flash",resources:[
 {label:"CODE",description:`vLLM @ ${VLLM_COMMIT.slice(0,7)}`,url:CODE_URL},
 {label:"CONFIG",description:"Flash 官方固定配置",url:CHECKPOINT_URL},
 {label:"WEIGHTS",description:"Hugging Face · 48 shards",url:WEIGHTS_URL},
],facts:[{value:"≈428B",label:"模型总参数量"},{value:"≈23B",label:"每 token 激活参数"},{value:"1M",label:"标称最大上下文"},{value:"250.10 GB",label:"主 checkpoint 数据区"}],View:MiniMaxView,Reference:HelpModal} satisfies ModelDefinition;
