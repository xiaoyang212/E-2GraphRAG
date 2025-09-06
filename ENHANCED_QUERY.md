# Enhanced Query Implementation

This document describes the enhanced query functionality added to the E²GraphRAG system.

## Overview

The enhanced query method implements a sophisticated retrieval strategy that combines semantic search in the summary tree with entity-based graph retrieval, following the Chinese specification:

1. **Semantic search in summary tree** - Identifies relevant high-level and leaf nodes
2. **Entity graph retrieval** - Finds entity relationship networks within shortest path constraints  
3. **Merge and deduplicate** - Combines candidate chunks from both sources
4. **High-level node filtering** - Uses high-level nodes as filters to keep only relevant chunks
5. **Intelligent ranking** - Applies existing entity-aware filtering for supplementary chunks

## New Methods Added

### `semantic_tree_search(query, k=25)`
- Performs semantic search across the entire summary tree
- Separates results into high-level nodes (`summary_*`) and leaf nodes (`leaf_*`)  
- Returns both types separately for different processing

### `find_descendants(node_id, visited=None)`
- Recursively finds all leaf node descendants of a given summary node
- Handles tree traversal with cycle detection
- Essential for filtering candidate chunks by high-level node branches

### `filter_by_high_level_branches(candidate_chunks, high_level_nodes)`
- Filters candidate chunks to keep only those belonging to high-level node branches
- Uses `find_descendants` to determine valid leaf nodes
- Maintains the key-value structure of chunk dictionaries

### `enhanced_query(query, **kwargs)`
- Main method implementing the complete enhanced retrieval strategy
- Combines semantic and entity-based approaches intelligently
- Includes fallback mechanisms for robust operation
- Maintains backward compatibility with existing debug information

## Usage

The enhanced query is enabled by default. To use the original method:

```python
result = retriever.query("What did Alice do?", use_enhanced=False)
```

To use the enhanced method (default):

```python  
result = retriever.query("What did Alice do?")  # use_enhanced=True by default
# or explicitly:
result = retriever.enhanced_query("What did Alice do?")
```

## Benefits

1. **Better precision** - High-level node filtering ensures chunks are contextually related
2. **Improved recall** - Combines semantic and entity-based retrieval
3. **Intelligent ranking** - Uses existing entity-aware filtering when needed
4. **Robust fallbacks** - Graceful degradation when components are unavailable
5. **Backward compatibility** - Original method preserved for comparison

## Parameters

All existing parameters are supported:
- `shortest_path_k`: Entity graph shortest path constraint (default: 4)
- `max_chunk_setting`: Maximum chunks to return (default: 25) 
- `debug`: Include debug information in results (default: True)
- `use_enhanced`: Use enhanced vs original method (default: True)