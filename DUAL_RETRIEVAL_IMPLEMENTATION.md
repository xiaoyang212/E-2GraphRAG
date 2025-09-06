# Dual Retrieval Strategy Implementation

This document describes the implementation of the new dual retrieval strategy as specified in the problem statement.

## Overview

The new retrieval strategy implements a dual approach that combines:
1. **Primary retrieval using summary tree** (以摘要树为主检索)
2. **Auxiliary retrieval using entity graph** (以实体图为辅助检索)

## Implementation Details

### 1. Primary Retrieval (`tree_based_retrieval`)
- Performs semantic search in the summary tree
- Identifies relevant high-level nodes and leaf nodes
- Uses FAISS index for efficient similarity search
- Returns both summary nodes (high-level context) and leaf nodes (detailed content)

### 2. Auxiliary Retrieval (`graph_based_retrieval`) 
- Searches entity graph for entities mentioned in query
- Limits to top k entities (default k=4) ranked by appearance count
- Uses existing `local_retrieval` method for graph-based search
- Provides precise factual information to supplement tree retrieval

### 3. Result Fusion and Deduplication (`deduplicate_and_merge`)
- **Always preserves high-level summary nodes** - essential background framework
- **Keeps overlapping chunks** between tree and graph results 
- Manages unique chunks from both sources separately
- Implements the core fusion logic as specified

### 4. Supplementary Chunk Ranking (`rank_supplementary_chunks`)
Implements 3-tier priority system for remaining chunk slots:

**Priority 1**: Chunks containing more different entities
**Priority 2**: Chunks with more neighbor nodes
**Priority 3**: Chunks with higher entity appearance frequency

### 5. Main Query Method
The updated `query()` method orchestrates the complete workflow:

1. Extract entities from query
2. Primary retrieval using summary tree  
3. Auxiliary retrieval using entity graph
4. Fusion and deduplication of results
5. Supplementary chunk ranking and addition
6. Format and return results

## Backward Compatibility

- Original `query()` method replaced with new dual strategy
- `query_legacy()` method preserves original implementation
- All existing APIs maintained for seamless integration

## Testing

The implementation includes comprehensive tests:
- Method signature validation
- Integration tests for all new methods
- End-to-end workflow verification
- Compliance verification against problem statement

## Usage

The new strategy is enabled by default when calling `query()`. For legacy behavior, use `query_legacy()`.

```python
# New dual strategy (default)
result = retriever.query("Tell me about science", max_chunk_setting=25)

# Legacy strategy (backward compatibility)  
result = retriever.query_legacy("Tell me about science", max_chunk_setting=25)
```

## Benefits

1. **Improved relevance**: Combines semantic similarity with precise entity matching
2. **Better context**: Preserves high-level summaries for background understanding
3. **Precise facts**: Entity graph provides specific factual information
4. **Intelligent ranking**: 3-tier priority system optimizes supplementary content
5. **Full compatibility**: Maintains existing API contracts