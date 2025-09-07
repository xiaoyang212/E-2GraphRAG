# Fusion Retrieval Strategy in E²GraphRAG

This document describes the fusion retrieval strategy implementation that combines entity graph retrieval and summary tree retrieval for enhanced performance.

## Overview

The fusion strategy implements a 5-step pipeline that leverages the complementary strengths of both retrieval methods:

- **Entity Graph Retrieval**: Excels at capturing specific entities and facts
- **Summary Tree Retrieval**: Captures broader semantic context through dense retrieval

## Pipeline Steps

### Step 1: Independent Retrieval and Score Calculation

#### Entity Graph Scoring
- **Entity Frequency**: Count of query entities in each chunk
- **Entity Co-occurrence**: Bonus scoring for chunks containing multiple query entities
- **Entity Importance**: Optional weighting based on graph algorithms (e.g., PageRank)

Formula: `entity_score(chunk_i) = sum_{e in Q} [freq(e, chunk_i) * weight(e)]`

#### Summary Tree Scoring  
- Uses dense retrieval with sentence transformers (BAAI/bge-m3)
- Computes semantic similarity between query and summary nodes
- Chunks inherit the maximum score from covering summary nodes

### Step 2: Score Normalization

Min-max normalization to [0,1] range:
```
normalized_score = (score - min_score) / (max_score - min_score)
```

Division by zero protection: if all scores equal, set to 0.5

### Step 3: Weighted Fusion with Dynamic Adjustment

#### Dynamic Weight Calculation
Based on entity ratio: `entity_ratio = len(entities) / len(query_tokens)`

- **High entity density** (>0.5): α=0.7, β=0.3 (emphasize entity matching)
- **Low entity density** (<0.2): α=0.3, β=0.7 (emphasize semantic matching)  
- **Balanced**: α=0.5, β=0.5

#### Fusion Formula
```
fused_score = α * entity_score_norm + β * summary_score_norm
```

Threshold filtering (default: 0.2) removes low-relevance chunks.

### Step 4: Cross-Encoder Re-ranking (Optional)

Re-ranks top candidates using cross-encoder for more precise relevance scoring. Currently implements a text overlap heuristic but can be replaced with proper cross-encoder models.

### Step 5: Result Merging and Deduplication

- Ensures unique chunk list
- Groups chunks by contained entities for consistent formatting
- Returns results sorted by final relevance scores

## Configuration

### Basic Usage

```python
# Enable fusion retrieval
retriever = Retriever(cache_tree, G, index, appearance_count, nlp, **kwargs)

# Query with fusion mode
result = retriever.query(query_text, use_fusion=True)
```

### Configuration Options

```yaml
retriever:
  kwargs:
    use_fusion: True                    # Enable fusion retrieval
    use_cross_encoder_rerank: True      # Enable cross-encoder re-ranking
    max_chunk_setting: 25               # Maximum chunks to return
    shortest_path_k: 4                  # For entity graph retrieval
```

### API Parameters

- `use_fusion: bool` - Enable/disable fusion mode (default: False)
- `use_cross_encoder_rerank: bool` - Enable/disable re-ranking (default: True)
- `max_chunk_setting: int` - Maximum number of chunks to return (default: 25)

## Performance Characteristics

### Strengths
- **Complementary Coverage**: Combines precise entity matching with broad semantic understanding
- **Dynamic Adaptation**: Automatically adjusts to query characteristics
- **Noise Reduction**: Multiple filtering stages remove irrelevant content
- **Flexibility**: Optional components can be enabled/disabled

### Considerations
- **Computational Cost**: Higher than single-method retrieval due to multiple scoring phases
- **Memory Usage**: Requires embeddings for summary tree scoring
- **Latency**: Cross-encoder re-ranking adds processing time but improves quality

## Implementation Details

### Core Methods

- `fusion_retrieval()` - Main fusion pipeline
- `query_fusion()` - Integration wrapper
- `_entity_scoring()` - Entity-based scoring
- `_summary_scoring()` - Semantic similarity scoring  
- `_normalize_scores()` - Score normalization
- `_calculate_dynamic_weights()` - Dynamic weight calculation
- `_fuse_scores()` - Score combination with thresholding
- `_cross_encoder_rerank()` - Optional re-ranking

### Backward Compatibility

The fusion implementation maintains full backward compatibility:
- Original retrieval methods unchanged
- Fusion activated only when explicitly requested
- Graceful fallback when embedder unavailable

## Example Usage

```python
import yaml
from query import Retriever

# Load configuration
config = yaml.safe_load(open('config.yaml'))
config['retriever']['kwargs']['use_fusion'] = True

# Initialize retriever
retriever = Retriever(
    cache_tree=cache_tree,
    G=graph, 
    index=index,
    appearance_count=appearance_count,
    nlp=nlp_extractor,
    **config['retriever']['kwargs']
)

# Query with fusion
query = "What is the relationship between artificial intelligence and machine learning?"
result = retriever.query(query, use_fusion=True, debug=True)

print(f"Retrieval type: {result['retrieval_type']}")
print(f"Retrieved chunks: {result['chunks']}")
```

## Future Enhancements

1. **Cross-Encoder Integration**: Replace heuristic re-ranking with proper cross-encoder models
2. **Advanced Entity Weighting**: Implement PageRank or other graph-based entity importance measures  
3. **Adaptive Thresholds**: Dynamic threshold adjustment based on query complexity
4. **Caching Optimizations**: Cache normalized scores and dynamic weights for repeated queries
5. **Performance Profiling**: Add detailed timing and memory usage metrics