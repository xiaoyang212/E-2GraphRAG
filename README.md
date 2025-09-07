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

## 🆕 New: Fusion Retrieval Strategy

E²GraphRAG now supports a **fusion retrieval strategy** that combines the strengths of both entity graph retrieval and summary tree retrieval:

- **Entity Graph Retrieval**: Excels at capturing specific entities and factual relationships  
- **Summary Tree Retrieval**: Captures broader semantic context through dense embeddings
- **Dynamic Fusion**: Automatically adjusts retrieval weights based on query characteristics
- **Cross-Encoder Re-ranking**: Optional re-ranking for improved precision

### Key Features:
- ✅ **Score Normalization**: Min-max normalization ensures fair score combination
- ✅ **Dynamic Weighting**: Adapts to entity-heavy vs. semantic queries automatically  
- ✅ **Threshold Filtering**: Removes low-relevance chunks to reduce noise
- ✅ **Cross-Encoder Re-ranking**: Optional but recommended for better results
- ✅ **Backward Compatible**: Works alongside existing retrieval methods

See [`FUSION_RETRIEVAL.md`](./FUSION_RETRIEVAL.md) for detailed documentation.

## 📁 Project Structure

```
.
├── README.md
├── requirements.txt
├── main.py
├── build_tree.py
├── dataloader.py
├── extract_graph.py
├── GlobalConfig.py
├── process_utils.py
├── prompt_dict.py
├── query.py                    # Enhanced with fusion retrieval
├── utils.py
├── FUSION_RETRIEVAL.md        # Fusion strategy documentation
├── test_fusion.py             # Fusion functionality tests
└── test_integration.py        # Integration tests
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

The entire pipeline—tree construction, graph extraction, and answer generation—is executed via `main.py`.

Step-by-step:

1. Create a config file

> Prepare a YAML configuration file to define key parameters.

> 👉 Example: `./configs/example_config.yaml`

2. Run the pipeline

> ```
> bash
> python main.py --config <path_to_config_file>
> ```

### 3. Using Fusion Retrieval

To enable the new fusion retrieval strategy, update your config file:

```yaml
retriever:
  kwargs:
    use_fusion: True                    # Enable fusion retrieval
    use_cross_encoder_rerank: True      # Enable re-ranking (optional)
    max_chunk_setting: 25               # Maximum chunks to return
```

Or use it programmatically:

```python
from query import Retriever

# Initialize retriever
retriever = Retriever(cache_tree, G, index, appearance_count, nlp, **kwargs)

# Query with fusion mode  
result = retriever.query(
    "What is the relationship between machine learning and artificial intelligence?",
    use_fusion=True,
    debug=True
)

print(f"Retrieval method: {result['retrieval_type']}")
print(f"Retrieved content: {result['chunks']}")
```

## 📬 Contact & Citation

If you use this code or find it helpful in your research, please consider citing our work. For questions or dataset access (NovelQA), please contact the original authors.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=YiboZhao624/E-2GraphRAG&type=Date)](https://www.star-history.com/#YiboZhao624/E-2GraphRAG&Date)
