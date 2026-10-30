# MiniMax M3.1 Flash · Model Atlas

基于 [CXY-Katrina/model-atlas](https://github.com/CXY-Katrina/model-atlas) 的完整仓库源码适配，保留原项目的原生算子拓扑、Tensor/Weight/External 分层连线、自动路由、模块展开、hover/pin 详情、输入绑定、多步公式和源码追踪。

默认模型为 **MiniMax-M3.1-Flash**，原 MiniMax-M3 模块保留，可从 MODEL 下拉选择。上游基线提交：`b4eadfe2727bf5a391ba3ac23d52c062ba684fe0`。原项目说明保留在 [docs/upstream-readme.md](docs/upstream-readme.md)。

**在线站点：[xuminhan.github.io/minimax-flash-atlas](https://xuminhan.github.io/minimax-flash-atlas/)**

## Flash 适配内容

- 所有 60 层均为 MSA；前 3 层 Dense FFN，后 57 层 MoE。层选择同步真实权重 key 和分片。
- 注意力展开保留五路投影、Q/K/V 分支、Indexer 评分/选块、两套缓存、两种 mask、QK/scale/softmax/PV 和 O projection。
- Dense FFN 拆解 Gate/Up、Split、非对称裁剪、SwiGLU-OAI、Down；MoE 拆解 Top-4 选路、路由权重、每专家 Gate/Up → SwiGLU-OAI → Down、加权归并及同样的共享专家链路。
- 54 个算子/模块公式和逐节点证据；代码片段重新从用户提供的固定 vLLM 提交提取，非简单替换模型名称。
- 完整 89,632 个 tensor keys / 48 分片的搜索目录；5,277 个实际 Safetensors 文件头条目。
- 视觉输入/输出链、独立 DSpark 配置说明，以及七张由 Archify typed JSON 生成的数据流补充图。
- 原项目类型、构建、语义与路由测试保留，并新增 Flash 专属检查。

## 运行

Node.js >=22.13.0。

```bash
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
npm test
npm run preview
```

Vite 使用相对路径，生产产物在 `dist/`，可部署到 GitHub Pages 的项目子路径。仓库保留原生 Pages 工作流。手动发布也可以运行 `npm run deploy`（需要当前 GitHub 仓库写权限和 gh 登录）。

## 研究成果

[成果索引](docs/models/minimax-m31-flash/README.md)、[架构](docs/models/minimax-m31-flash/architecture.md)、[54 个模块公式](docs/models/minimax-m31-flash/formulas.md)、[符号](docs/models/minimax-m31-flash/symbols.md)、[实现](docs/models/minimax-m31-flash/implementation.md)、[证据矩阵](docs/models/minimax-m31-flash/evidence.md)。

HF checkpoint：`262ae6830934f3512211081b8a3f16425b93a239`。vLLM：`ced6857afa0ea7b2e3f0846a62e1394e90f15607`。主配置以 Flash checkpoint 为准，不采用配置类中的旧 Preview 默认值。没有改动用户的 vLLM checkout。

约 428B/23B 来自模型卡；250,103,411,200 bytes 来自主索引。量化 packed 存储元素与逻辑参数数量不同，权重大小也不等于推理峰值显存。本站仅发布元数据，不包含模型权重或凭据。

DSpark 的结构依据官方独立配置；旧 MTP 代码不等同于 DSpark。未实测模型推理、性能、精度、Ascend 或其他设备兼容性。页面清楚标注直接读取、配置推导和未验证内容。构建、语义/路由自动测试与独立 Archify 图示的自动浏览器检查已执行。集成页面的完整 viewport/theme/pan/详情交互回归受本地浏览器访问权限阻断，尚未完成；不能把独立图示通过当作整站验收。研究文档保持 draft，后续取得完整同版双审后再提升状态。

## 来源与许可

上游 Model Atlas 的源码、历史和作者归属保持不变。MiniMax checkpoint 元数据沿用其 [MiniMax Community License](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/262ae6830934f3512211081b8a3f16425b93a239/LICENSE)。引用的 vLLM 代码为 Apache-2.0；Archify 为 Apache-2.0；React、KaTeX、Vite 等依赖沿用各自许可。这里未为上游源码另行指定许可证。
