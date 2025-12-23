# Dual-Index GraphRAG Implementation

## 算法概述 (Algorithm Overview)

本实现基于**双索引长文本GraphRAG**算法，构建了"概念图（细粒度）"与"摘要树（宏观全局）"的双重索引，并在检索时通过双路并行检索进行自适应融合。

This implementation is based on the **Dual-Index GraphRAG for Long Context** algorithm, which constructs dual indices: a "Concept Graph (fine-grained)" and a "Summary Tree (global)", with adaptive fusion through parallel dual-path retrieval.

## 核心特性 (Key Features)

### 第一阶段：索引构建 (Phase 1: Indexing)

1. **文本分块与预处理 (Text Chunking and Preprocessing)**
   - 重叠分块 (Overlapping chunking)
   - 句子分割与映射 (Sentence segmentation and mapping)

2. **摘要树构建 (Summary Tree Construction)**
   - 递归摘要 (Recursive summarization)
   - 自底向上聚合 (Bottom-up aggregation)
   - 保留原始块和各层级摘要 (Preserving original chunks and hierarchical summaries)

3. **概念图构建 (Concept Graph Construction)**
   - **TF-IDF 概念提取** (TF-IDF-based concept extraction)
   - **概念向量化** (Concept vectorization using sentence embeddings)
   - **边构建与加权** (Edge construction and weighting):
     - 语义相似度阈值 (Semantic similarity threshold)
     - 共现阈值 (Co-occurrence threshold)
     - Dice系数权重 (Dice coefficient weighting)
   - **倒排索引** (Inverted index: concept → sentences)

### 第二阶段：检索与融合 (Phase 2: Retrieval and Fusion)

1. **路径A：基于概念图的细粒度检索 (Path A: Concept Graph Fine-grained Retrieval)**
   - A1: 概念匹配 (Concept matching with cosine similarity)
   - A2: 候选句子召回 (Candidate sentence recall using inverted index)
   - A3: 精确重排序 (Precise re-ranking)
   - A4: 映射回块 (Mapping sentences back to chunks)

2. **路径B：基于摘要树的全局检索 (Path B: Summary Tree Global Retrieval)**
   - B1: 扁平化检索 (Flattened retrieval)
   - B2: 向量匹配 (Vector matching)
   - B3: Top-K选择 (Top-K selection)

3. **融合与生成 (Fusion and Generation)**
   - 集合并集 (Set union with deduplication)
   - 上下文池生成 (Context pool generation)

## 文件结构 (File Structure)

```
├── dual_index_builder.py      # 双索引构建器 (Dual-index builder)
├── dual_retriever.py           # 双路径检索器 (Dual-path retriever)
├── main_dual_index.py          # 主程序入口 (Main entry point)
├── configs/
│   └── dual_index_config.yaml # 配置文件示例 (Configuration example)
```

## 使用方法 (Usage)

### 1. 安装依赖 (Install Dependencies)

```bash
pip install -r requirements.txt
pip install scikit-learn  # For TF-IDF
```

### 2. 配置文件 (Configuration)

编辑 `configs/dual_index_config.yaml`，设置：
- 数据集路径
- LLM 模型路径
- 检索参数

Edit `configs/dual_index_config.yaml` to set:
- Dataset path
- LLM model path
- Retrieval parameters

### 3. 运行 (Run)

```bash
python main_dual_index.py --config configs/dual_index_config.yaml
```

## 关键参数说明 (Key Parameters)

### 索引构建参数 (Indexing Parameters)

- `min_tfidf_score`: TF-IDF最小分数阈值 (Minimum TF-IDF score threshold)
- `semantic_threshold`: 语义相似度阈值，用于边构建 (Semantic similarity threshold for edge construction)
- `cooccurrence_threshold`: 共现阈值 (Co-occurrence threshold)

### 检索参数 (Retrieval Parameters)

- `concept_top_k`: 路径A中检索的概念数量 (Number of concepts to retrieve in Path A)
- `sentence_top_k`: 路径A中检索的句子数量 (Number of sentences to retrieve in Path A)
- `tree_top_k`: 路径B中检索的树节点数量 (Number of tree nodes to retrieve in Path B)
- `concept_threshold`: 概念匹配的相似度阈值 (Similarity threshold for concept matching)

## 算法详细说明 (Detailed Algorithm)

### 概念提取 (Concept Extraction)

使用 TF-IDF 算法从每个文本块中提取关键概念：
- 支持 unigrams 和 bigrams
- 过滤停用词
- 提取 top-k 个高分概念

Using TF-IDF algorithm to extract key concepts from each chunk:
- Supports unigrams and bigrams
- Filters stop words
- Extracts top-k high-scoring concepts

### 概念向量化 (Concept Vectorization)

对每个概念 $w$：
1. 收集包含该概念的所有句子 $S_w$
2. 使用句子编码器编码这些句子
3. 计算平均向量：$Vector(w) = \frac{1}{|S_w|} \sum_{s \in S_w} \phi(s)$

For each concept $w$:
1. Collect all sentences containing the concept $S_w$
2. Encode these sentences using sentence encoder
3. Calculate average vector: $Vector(w) = \frac{1}{|S_w|} \sum_{s \in S_w} \phi(s)$

### 边权重计算 (Edge Weight Calculation)

对于概念对 $(w_i, w_j)$，仅当满足以下条件时建立边：
1. 向量相似度 $\geq \theta_{sem}$ (semantic threshold)
2. 共现次数 $\geq \theta_{co}$ (co-occurrence threshold)

边权重使用 Dice 系数：
$$r(w_i, w_j) = \frac{2 \cdot Co(w_i, w_j)}{|T_{w_i}| + |T_{w_j}|}$$

For concept pair $(w_i, w_j)$, an edge is created only when:
1. Vector similarity $\geq \theta_{sem}$ (semantic threshold)
2. Co-occurrence count $\geq \theta_{co}$ (co-occurrence threshold)

Edge weight uses Dice coefficient:
$$r(w_i, w_j) = \frac{2 \cdot Co(w_i, w_j)}{|T_{w_i}| + |T_{w_j}|}$$

### 双路径检索 (Dual-Path Retrieval)

#### 路径A流程 (Path A Flow)
```
Query → Concept Matching → Candidate Sentences → Re-ranking → Chunks
```

#### 路径B流程 (Path B Flow)
```
Query → Tree Node Matching → Top-K Selection → Nodes
```

#### 融合策略 (Fusion Strategy)
```
C_pool = C_cpt ∪ C_tree (union with deduplication)
```

## 与原始实现的对比 (Comparison with Original Implementation)

| 特性 (Feature) | 原始实现 (Original) | 双索引实现 (Dual-Index) |
|----------------|---------------------|-------------------------|
| 概念提取 (Concept Extraction) | NER (Named Entity Recognition) | TF-IDF |
| 索引粒度 (Index Granularity) | Chunk-level | Sentence-level |
| 检索路径 (Retrieval Paths) | Single path | Dual paths (concept + tree) |
| 边权重 (Edge Weights) | Simple co-occurrence | Dice coefficient + semantic similarity |
| 概念表示 (Concept Representation) | Not vectorized | Sentence-averaged vectors |

## 性能考虑 (Performance Considerations)

1. **缓存机制** (Caching): 所有索引结构都会被缓存以加速后续运行
2. **批处理** (Batching): 句子编码使用批处理以提高效率
3. **并行化** (Parallelization): 双路径检索可以并行执行（当前为顺序实现）

1. **Caching**: All index structures are cached to speed up subsequent runs
2. **Batching**: Sentence encoding uses batching for efficiency
3. **Parallelization**: Dual-path retrieval can be executed in parallel (currently sequential)

## 扩展建议 (Extension Suggestions)

1. 真正的并行执行双路径检索（使用线程或进程）
2. 添加更复杂的融合策略（如加权融合）
3. 支持动态调整检索参数
4. 添加概念重要性评分机制

1. True parallel execution of dual-path retrieval (using threads or processes)
2. Add more sophisticated fusion strategies (e.g., weighted fusion)
3. Support dynamic adjustment of retrieval parameters
4. Add concept importance scoring mechanism

## 引用 (Citation)

如果使用此实现，请引用原始论文和本仓库。

If using this implementation, please cite the original paper and this repository.

## 许可证 (License)

遵循原仓库的许可证。

Follows the license of the original repository.
