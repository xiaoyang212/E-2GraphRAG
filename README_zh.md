# E²GraphRAG：高效且有效的图结构增强式检索生成框架

E²GraphRAG 是一个轻量级、模块化的框架，旨在提升基于图的增强式检索生成（Retrieval-Augmented Generation, RAG）在 **效率** 和 **效果** 两方面的表现。该框架通过结构化图推理，简化了从文档解析到答案生成的整个流程。

## 🆕 双索引 GraphRAG

本仓库现已包含增强版的**双索引 GraphRAG** 实现，结合了：
- **概念图（细粒度）**：基于 TF-IDF 的细粒度概念提取，支持句子级索引
- **摘要树（宏观全局）**：层次化摘要提供全局上下文
- **双路径检索**：从两个索引并行检索并自适应融合

👉 详细文档请参阅 [DUAL_INDEX_README.md](./DUAL_INDEX_README.md)

## 📁 项目结构

```
.
├── README.md
├── requirements.txt
├── main.py                      # 原始 E²GraphRAG 流程
├── main_dual_index.py          # 🆕 双索引 GraphRAG 流程
├── build_tree.py
├── dataloader.py
├── extract_graph.py
├── dual_index_builder.py       # 🆕 双索引构建
├── dual_retriever.py           # 🆕 双路径检索
├── process_utils.py
├── prompt_dict.py
├── query.py
├── utils.py
├── DUAL_INDEX_README.md        # 🆕 双索引文档
└── configs/
    ├── example_config.yaml
    └── dual_index_config.yaml  # 🆕 双索引配置
```


## 📦 数据集

我们使用了以下两个数据集：

- [📚 NovelQA](https://huggingface.co/datasets/NovelQA/NovelQA)  
  部分开源，若需获取完整数据集，请联系原作者申请。
  
- [🔁 InfiniteBench](https://github.com/OpenBMB/InfiniteBench)  
  完全开源，公开可用。

你可以在 `./data/README.md` 中找到获取数据的具体说明。

> **注意：** 获取数据后，请在初始化 `Dataloader` 类时指定数据路径。

## 🚀 快速开始

### 1. 安装依赖

请先安装项目所需的 Python 库：

```bash
pip install -r requirements.txt
```

### 2. 运行主流程

#### 方案 A：原始 E²GraphRAG

整个流程包括构建文档树、提取图结构并生成答案：

```bash
python main.py --config configs/example_config.yaml
```

#### 方案 B：双索引 GraphRAG 🆕

增强版的双索引流程，支持概念图和双路径检索：

```bash
python main_dual_index.py --config configs/dual_index_config.yaml
```

**主要区别：**
- 使用 TF-IDF 进行概念提取（相对于 NER）
- 句子级索引实现细粒度检索
- 双路径检索（概念图 + 摘要树）
- 自适应融合检索结果

详细文档请参阅 [DUAL_INDEX_README.md](./DUAL_INDEX_README.md)

### 📬 联系我们 & 引用

如果您在研究中使用了本代码，或者觉得本项目对您有帮助，欢迎引用我们的工作。如有问题，或需获取 NovelQA 数据集，请联系原作者。
