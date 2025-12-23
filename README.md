# E²GraphRAG: Streamlining Graph-based RAG for High Efficiency and Effectiveness

<p align="center">
  <a href="https://arxiv.org/abs/2505.24226" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2505.24226-b31b1b?logo=arxiv&logoColor=white&style=for-the-badge" alt="arXiv">
  </a>
  &nbsp;&nbsp;
  <a href="./README_zh.md">
    <img src="https://img.shields.io/badge/文档-中文版-blue?style=for-the-badge&logo=readthedocs&logoColor=white" alt="中文说明">
  </a>
</p>

E²GraphRAG is a lightweight and modular framework designed to enhance both **efficiency** and **effectiveness** in Graph-based Retrieval-Augmented Generation (RAG). It streamlines the pipeline from document parsing to answer generation via structured graph reasoning.

## 🆕 Dual-Index GraphRAG

This repository now includes an enhanced **Dual-Index GraphRAG** implementation that combines:
- **Concept Graph (细粒度)**: TF-IDF-based fine-grained concept extraction with sentence-level indexing
- **Summary Tree (宏观全局)**: Hierarchical summarization for global context
- **Dual-Path Retrieval**: Parallel retrieval from both indices with adaptive fusion

👉 See [DUAL_INDEX_README.md](./DUAL_INDEX_README.md) for detailed documentation.

## 📁 Project Structure

```
.
├── README.md
├── requirements.txt
├── main.py                      # Original E²GraphRAG pipeline
├── main_dual_index.py          # 🆕 Dual-Index GraphRAG pipeline
├── build_tree.py
├── dataloader.py
├── extract_graph.py
├── dual_index_builder.py       # 🆕 Dual-index construction
├── dual_retriever.py           # 🆕 Dual-path retrieval
├── process_utils.py
├── prompt_dict.py
├── query.py
├── utils.py
├── DUAL_INDEX_README.md        # 🆕 Dual-index documentation
└── configs/
    ├── example_config.yaml
    └── dual_index_config.yaml  # 🆕 Dual-index configuration
```

## 📦 Datasets

We use data from:

- [📚 NovelQA](https://huggingface.co/datasets/NovelQA/NovelQA)
  Partly open-source, to obtain the full dataset, please *access via a request to the original authors.*
- [🔁 InfiniteBench](https://github.com/OpenBMB/InfiniteBench)
  *Fully open-source and publicly available.*

You can find how to obtain the data in the `./data/README.md`.

> **Note:** After obtaining the datasets, specify the data path when initializing the `Dataloader` class.

## 🚀 Getting Started

### 1. Install Dependencies

Ensure your environment is set up by installing the required packages:

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

#### Option A: Original E²GraphRAG

The entire pipeline—tree construction, graph extraction, and answer generation—is executed via `main.py`.

```bash
python main.py --config configs/example_config.yaml
```

#### Option B: Dual-Index GraphRAG 🆕

The enhanced dual-index pipeline with concept graph and dual-path retrieval:

```bash
python main_dual_index.py --config configs/dual_index_config.yaml
```

**Key differences:**
- Uses TF-IDF for concept extraction (vs NER)
- Sentence-level indexing for fine-grained retrieval
- Dual-path retrieval (concept graph + summary tree)
- Adaptive fusion of retrieval results

See [DUAL_INDEX_README.md](./DUAL_INDEX_README.md) for detailed documentation.

## 📬 Contact & Citation

If you use this code or find it helpful in your research, please consider citing our work. For questions or dataset access (NovelQA), please contact the original authors.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=YiboZhao624/E-2GraphRAG&type=Date)](https://www.star-history.com/#YiboZhao624/E-2GraphRAG&Date)
