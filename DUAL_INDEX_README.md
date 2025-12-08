# Dual-Index GraphRAG Documentation

## 双索引长文本GraphRAG (Dual-Index GraphRAG for Long Context)

This document describes the implementation of the Dual-Index GraphRAG algorithm, which combines fine-grained concept graph retrieval with global summary tree retrieval for enhanced long-context question answering.

## Overview

The Dual-Index GraphRAG system constructs two complementary indices:

1. **Concept Graph (细粒度)**: Fine-grained, sentence-level index for precise entity and concept matching
2. **Summary Tree (宏观全局)**: Hierarchical, global index for broader context retrieval

During retrieval, the system performs parallel dual-path retrieval and fuses the results through set union.

## Architecture

### Phase 1: Index Construction (索引构建阶段)

#### 1.1 Text Preprocessing
- **Overlapping Chunking**: Split document $D$ into chunks $C = \{c_1, c_2, ..., c_N\}$ with length $L$ and overlap $O$ tokens
- **Sentence Segmentation**: Extract sentences $S = \{s_1, s_2, ..., s_P\}$ from each chunk with mapping $I_{s→c}: s → c(s)$

#### 1.2 Summary Tree Construction
- Initialize leaf nodes with text chunks
- Recursively summarize groups of $g$ adjacent nodes using LLM
- Build hierarchical tree $T$ containing original chunks and multi-level summaries

#### 1.3 Concept Graph Construction
- **Concept Extraction**: Extract keywords from each chunk using **unsupervised algorithms** (RAKE or TF-IDF)
  - **RAKE** (Rapid Automatic Keyword Extraction): Extracts multi-word keyphrases based on word co-occurrence
  - **TF-IDF**: Identifies important terms based on term frequency-inverse document frequency
  - **NLP** (optional): Uses Spacy/NLTK for entity and noun extraction
  
- **Node Vectorization**: For each concept $w$, compute vector by averaging sentence embeddings:
  ```
  Vector(w) = (1/|S_w|) * Σ_{s ∈ S_w} φ(s)
  ```
  where $φ$ is a sentence encoder (default: BAAI/bge-m3)
  
- **Edge Construction**: Build edges between concepts $(w_i, w_j)$ if:
  - Semantic similarity $≥ θ_{sem}$ (default: 0.3)
  - Co-occurrence count $≥ θ_{co}$ (default: 1)
  
- **Edge Weighting**: Use Dice coefficient:
  ```
  r(w_i, w_j) = 2 * Co(w_i, w_j) / (|T_{w_i}| + |T_{w_j}|)
  ```

- **Inverted Index**: Build $I_{c→s}: w → S_w$ (concept to sentences mapping)

### Phase 2: Retrieval & Fusion (检索与融合阶段)

#### 2.1 Path A: Concept Graph Fine-grained Retrieval

**A1. Concept Matching**
- Compute similarity between query $q$ and all concept vectors
- Select concepts $W_{relevant}$ with similarity $> \text{threshold}$

**A2. Candidate Sentence Recall**
- Use index $I_{c→s}$ to retrieve sentences containing $W_{relevant}$
- Merge into candidate set $S_{candidate}$

**A3. Sentence Reranking**
- Compute similarity between $q$ and each sentence in $S_{candidate}$
- Select Top-$K_s$ sentences → $S_{top}$

**A4. Map to Chunks**
- Use index $I_{s→c}$ to map $S_{top}$ to original chunks → $C_{cpt}$

#### 2.2 Path B: Summary Tree Global Retrieval

**B1. Flatten Tree**
- Treat all tree nodes (leaves + summaries) as equal candidates

**B2. Vector Matching**
- Compute similarity $Sim(φ(q), φ(n))$ for all nodes $n$ in tree

**B3. Top-K Selection**
- Select Top-$K_t$ most similar nodes → $C_{tree}$

#### 2.3 Fusion

- **Union**: Compute final pool $C_{pool} = C_{cpt} ∪ C_{tree}$
- Deduplication is automatic (using set operations)

## Usage

### Basic Setup

```python
from dual_index_graphrag import DualIndexBuilder, DualIndexRetriever
from extract_graph import extract_graph_with_dual_index, load_nlp
from query import Retriever

# 1. Extract graph with dual index using RAKE for concept extraction
(G, index, appearance_count, dual_index_data), time_cost = extract_graph_with_dual_index(
    text=text_chunks,
    cache_folder="./cache",
    nlp=None,  # Not required when using RAKE or TF-IDF
    build_dual_index=True,
    embedder_model="BAAI/bge-m3",
    device="cuda:0",
    concept_extraction_method="rake"  # Options: "rake", "tfidf", or "nlp"
)

# Alternative: Use TF-IDF for concept extraction
(G, index, appearance_count, dual_index_data), time_cost = extract_graph_with_dual_index(
    text=text_chunks,
    cache_folder="./cache",
    nlp=None,
    build_dual_index=True,
    concept_extraction_method="tfidf"
)

# 2. Initialize retriever with dual index
retriever = Retriever(
    cache_tree=cache_tree,
    G=G,
    index=index,
    appearance_count=appearance_count,
    nlp=nlp,
    dual_index_data=dual_index_data,
    **retriever_kwargs
)

# 3. Perform dual-path retrieval
result = retriever.query_dual_path(
    query="Your question here",
    K_s=10,  # Top sentences for Path A
    K_t=10,  # Top tree nodes for Path B
    concept_threshold=0.3  # Similarity threshold
)

print(result["chunks"])  # Retrieved context
print(result["metadata"])  # Retrieval statistics
```

### Configuration

Add to your YAML config file:

```yaml
# Dual-Index GraphRAG Configuration
dual_index:
  enabled: True
  build_sentence_index: True
  embedder_model: "BAAI/bge-m3"
  concept_extraction_method: "rake"  # Options: "rake", "tfidf", or "nlp"
  theta_sem: 0.3  # Semantic similarity threshold
  theta_co: 1     # Co-occurrence threshold

retriever:
  mode: "dual_path"  # or "standard"
  kwargs:
    K_s: 10                    # Sentences for Path A
    K_t: 10                    # Tree nodes for Path B
    concept_threshold: 0.3     # Concept matching threshold
```

### Running with Config

```bash
python main.py --config configs/dual_index_config.yaml
```

## API Reference

### DualIndexBuilder

```python
class DualIndexBuilder:
    def __init__(self, embedder_model: str = "BAAI/bge-m3", device: str = "cuda:0")
    
    def build_sentence_index(self, text_chunks: List[str], nlp: Extractor) -> Tuple[Dict, Dict, Dict]
        """Build sentence-level indices"""
        
    def build_concept_vectors(self, I_c_to_s: Dict, sentence_texts: Dict) -> Dict[str, np.ndarray]
        """Build concept vectors from sentence embeddings"""
```

### DualIndexRetriever

```python
class DualIndexRetriever:
    def __init__(self, cache_tree, G, index, I_s_to_c, I_c_to_s, sentence_texts, 
                 concept_vectors, embedder_model="BAAI/bge-m3", device="cuda:0")
    
    def path_a_concept_graph_retrieval(self, query: str, K_s: int = 10, 
                                       similarity_threshold: float = 0.3) -> Set[str]
        """Path A: Fine-grained concept graph retrieval"""
        
    def path_b_summary_tree_retrieval(self, query: str, K_t: int = 10) -> Set[str]
        """Path B: Global summary tree retrieval"""
        
    def dual_path_retrieval(self, query: str, K_s: int = 10, K_t: int = 10,
                           concept_threshold: float = 0.3) -> Tuple[Set[str], Dict]
        """Dual-path retrieval with fusion"""
```

### Extended Retriever Methods

```python
class Retriever:
    def query_dual_path(self, query, **kwargs)
        """
        Dual-path retrieval combining concept graph and summary tree
        
        Args:
            query: User query string
            K_s: Number of top sentences for Path A (default: 10)
            K_t: Number of top tree nodes for Path B (default: 10)
            concept_threshold: Similarity threshold (default: 0.3)
        
        Returns:
            Dictionary with chunks, metadata, and retrieval statistics
        """
```

## Testing

Run the test suite:

```bash
python test_dual_index.py
```

This will test:
1. DualIndexBuilder functionality
2. DualIndexRetriever dual-path retrieval
3. Integration with existing pipeline

## Performance Considerations

### Memory Usage
- Sentence embeddings are cached in concept vectors
- Tree node embeddings are pre-computed during initialization
- Use `device="cpu"` for lower memory footprint

### Retrieval Speed
- Path A and Path B can be executed in parallel (future optimization)
- Sentence-level granularity provides better precision but more computation
- Adjust $K_s$ and $K_t$ to balance precision/recall

### Quality Tuning

**For better recall (broader results):**
- Increase $K_s$ and $K_t$
- Lower `concept_threshold`
- Lower $θ_{sem}$ (semantic similarity threshold)

**For better precision (focused results):**
- Decrease $K_s$ and $K_t$
- Raise `concept_threshold`
- Raise $θ_{sem}$

## Citation

If you use this implementation, please cite:

```bibtex
@article{e2graphrag2024,
  title={E²GraphRAG: Streamlining Graph-based RAG for High Efficiency and Effectiveness},
  author={...},
  journal={arXiv preprint arXiv:2505.24226},
  year={2024}
}
```

## License

This implementation is part of the E²GraphRAG project and follows the same license terms.
