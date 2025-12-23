# Improved Dual-Index GraphRAG - 改进方案

## 问题诊断

经过测试发现原始dual-index方案的性能不如entity graph + summary tree方案，主要原因：

### 1. 概念提取质量问题
- **原方案**: 纯TF-IDF关键词提取，可能包含大量不重要的通用词
- **问题**: 缺少语义重要性识别，无法区分人名、地名等关键实体
- **原系统优势**: 使用NER提取命名实体，语义更精确

### 2. 缺少图遍历逻辑
- **原方案**: 简单的概念匹配，没有利用概念间的关系
- **问题**: 无法找到间接相关的概念和chunk
- **原系统优势**: 使用shortest path图遍历，能发现实体间关系

### 3. 融合策略过于简单
- **原方案**: 简单集合并集，所有chunk权重相同
- **问题**: 无法区分重要性，可能引入噪声
- **原系统优势**: 有entity-aware filtering和occurrence ranking

## 改进方案

### 核心改进

#### 1. 混合概念提取 (Hybrid Concept Extraction)
```python
# 同时使用NER和TF-IDF
- NER实体 (人名、地名、组织) → 权重2.0
- TF-IDF关键词 → 权重为TF-IDF分数
```

**优势**:
- 保留原系统的实体识别能力
- 增加TF-IDF的覆盖面
- 概念重要性评分用于后续排序

#### 2. 图遍历检索 (Graph Traversal Retrieval)
```python
# Path A改进:
1. 使用NER提取查询实体 (like original)
2. 在概念图中匹配实体
3. 图遍历找相关概念:
   - Shortest paths between entities (最大路径长度3)
   - 直接邻居节点 (按边权重排序)
4. 映射到chunk时考虑概念重要性
```

**优势**:
- 恢复原系统的图遍历能力
- 能发现间接相关的概念
- 考虑概念权重，提升排序质量

#### 3. 加权融合 (Weighted Fusion)
```python
# 不同路径的chunk赋予不同权重:
- Path A (graph) chunks: 权重 2.0
- Path B (tree) chunks: 权重 1.0  
- 两个路径都选中: 权重 3.0 (boost)
- 按权重排序，取top-K
```

**优势**:
- 优先选择实体相关的chunks
- 同时保留全局语义相关的chunks
- 避免简单并集带来的噪声

#### 4. 参数优化
```yaml
# 调整后的参数:
concept_top_k: 30          # 增加 (20→30)
sentence_top_k: 40         # 增加 (30→40)  
concept_threshold: 0.5     # 降低 (0.6→0.5)
min_tfidf_score: 0.15      # 提高 (0.1→0.15)
semantic_threshold: 0.65   # 降低 (0.7→0.65)
max_path_length: 3         # 新增
use_graph_traversal: True  # 新增
```

**调整理由**:
- 增加top_k: 提高召回率
- 降低concept_threshold: 避免过于严格
- 提高min_tfidf_score: 过滤低质量关键词
- 降低semantic_threshold: 增加图的连通性
- 新增图遍历参数: 恢复原系统能力

## 使用方法

### 安装
```bash
# 无需额外依赖，使用现有requirements.txt
pip install -r requirements.txt
```

### 运行改进版本
```bash
# 使用改进的dual-index配置
python main_dual_index.py --config configs/improved_dual_index_config.yaml
```

### 代码集成

```python
from improved_dual_index_builder import ImprovedDualIndexBuilder
from improved_dual_retriever import create_improved_dual_retriever
from extract_graph import load_nlp

# 1. 加载NLP extractor (用于NER)
nlp_extractor = load_nlp(language="en", method="Spacy")

# 2. 构建改进的dual-index
builder = ImprovedDualIndexBuilder(
    embedder_model="BAAI/bge-m3",
    device="cuda:0",
    language="en",
    nlp_extractor=nlp_extractor,
    use_ner=True,              # 使用NER
    use_tfidf=True,            # 使用TF-IDF
    min_tfidf_score=0.15,
    semantic_threshold=0.65
)

G, concept_to_sentences, sentence_to_chunk, concept_vectors, sentences, concept_importance = \
    builder.build_dual_index(chunks, cache_folder)

# 3. 创建改进的retriever
retriever = create_improved_dual_retriever(
    cache_tree=tree,
    concept_graph=G,
    concept_to_sentences=concept_to_sentences,
    sentence_to_chunk=sentence_to_chunk,
    concept_vectors=concept_vectors,
    sentences=sentences,
    concept_importance=concept_importance,
    nlp_extractor=nlp_extractor,
    embedder_model="BAAI/bge-m3",
    device="cuda:0",
    concept_top_k=30,
    use_graph_traversal=True,
    max_path_length=3
)

# 4. 查询
result = retriever.query(query)
chunks = result["chunks"]
```

## 与原方案对比

| 特性 | 原Dual-Index | 改进版Dual-Index | 原Entity Graph |
|------|--------------|------------------|----------------|
| 概念提取 | TF-IDF only | NER + TF-IDF | NER + Nouns |
| 图遍历 | ❌ | ✅ | ✅ |
| 概念权重 | ❌ | ✅ | ✅ (appearance count) |
| 融合策略 | 简单并集 | 加权融合 | Entity-aware filter |
| 句子级索引 | ✅ | ✅ | ❌ |
| 双路径检索 | ✅ | ✅ | ❌ (单路径) |

## 预期改进

1. **召回率提升**: 
   - NER提取更精确的实体
   - 图遍历发现间接相关内容
   
2. **准确率提升**:
   - 概念重要性评分
   - 加权融合优先选择重要chunks
   
3. **鲁棒性提升**:
   - 混合提取策略（NER失败时用TF-IDF）
   - 双路径互补（graph失败时有tree）

## 下一步优化方向

1. **动态参数调整**: 根据查询类型自动调整权重
2. **学习型融合**: 基于历史效果学习最优权重
3. **并行检索**: 真正并行执行Path A和Path B
4. **缓存优化**: 增量更新索引

## 测试建议

建议在相同数据集上对比测试:
```bash
# 原entity graph方案
python main.py --config configs/example_config.yaml

# 改进的dual-index方案  
python main_dual_index.py --config configs/improved_dual_index_config.yaml
```

对比指标:
- 准确率 (accuracy)
- 召回率 (recall) 
- 查询时间 (query time)
- 检索相关性 (relevance)
