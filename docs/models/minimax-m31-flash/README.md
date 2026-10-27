Document-Kind: model-knowledge
Authoring: agent-generated, source-grounded
Model: MiniMax-M3.1-Flash
Source-Revisions: HF 262ae6830934f3512211081b8a3f16425b93a239; vLLM ced6857afa0ea7b2e3f0846a62e1394e90f15607
Content-Status: draft

# MiniMax M3.1 Flash 模型研究

本模块基于 Model Atlas 完整仓库源码适配，保留其原生图元、自动路由、展开图、hover/pin 详情、输入绑定与公式解释。Flash 是默认模型，原 MiniMax-M3 模块保留用于对照。

- [整体架构](architecture.md)
- [45 个算子公式及推导](formulas.md)
- [符号与实际配置](symbols.md)
- [实现调用链及融合边界](implementation.md)
- [逐节点固定来源](evidence.md)
- [七张 Archify typed JSON](diagrams/overview.json)：overview、indexer、attention、dense-ffn、moe、vision、dspark。

范围：Flash text/image/video 架构、60 层 MSA 与两种 FFN、NVIDIA 模型层及通用 Triton 数学参考、量化权重元数据、视觉塔；DSpark 为独立配置结构解释。硬件专用内核、AMD/Ascend 和性能不在已验证范围。DSpark 不以本地旧 MTP 实现充当依据。
