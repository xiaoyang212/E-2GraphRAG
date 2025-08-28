# Enhanced E-2GraphRAG Configuration Documentation

This document describes the configuration options for the enhanced E-2GraphRAG retrieval system.

## Enhanced Retrieval Configuration

The enhanced retrieval system can be configured through the `retriever.kwargs` section in your YAML configuration file.

### Basic Configuration

```yaml
retriever:
  kwargs:
    # Core settings
    device: cuda                      # Device for embeddings (cuda/cpu)
    debug: True                       # Enable debug information
    tokenizer: path/to/your/tokenizer # Path to tokenizer
    embedder: BAAI/bge-m3            # Embedding model
    
    # Enhanced retrieval settings
    use_enhanced_retrieval: True      # Enable enhanced system (default: True)
    
    # Budget allocation
    budget_N: 10                      # Total chunks to return
    cap_I_ratio: 0.5                  # Max ratio for hard overlap chunks (0.4-0.6)
    max_chunk_setting: 10             # Alias for budget_N (legacy compatibility)
    
    # Parallel recall parameters
    tree_top_k: 15                    # Top-K chunks from tree recall
    graph_top_k: 15                   # Top-K chunks from graph recall
    bm25_top_k: 8                     # Top-K chunks from BM25 recall
    dense_top_k: 8                    # Top-K chunks from dense recall
    graph_max_hops: 2                 # Maximum hops for graph expansion (1-2)
    
    # RRF parameters for soft overlap
    rrf_k: 60                         # RRF constant for ranking fusion
    
    # MMR parameters
    mmr_lambda: 0.75                  # MMR trade-off parameter (relevance vs diversity)
    
    # Diversity constraints
    max_same_subtree_ratio: 0.5       # Max chunks from same subtree
    max_same_entity_cluster_ratio: 0.5 # Max chunks from same entity cluster
    min_query_entities_covered: 2     # Minimum query entities to cover
    
    # Recency parameters
    recency_half_life: 180            # Half-life for recency decay (days)
    recency_min: 0.2                  # Minimum recency score
```

### Advanced Scoring Configuration

You can customize the linear scoring weights for different question types:

```yaml
retriever:
  kwargs:
    # Default scoring weights (baseline)
    scoring_weights:
      sim_emb: 0.35          # Embedding similarity
      sim_lex: 0.20          # Lexical similarity (BM25 + coverage + proximity)
      ent_overlap: 0.20      # Entity overlap (weighted Jaccard)
      path_score: 0.15       # Graph path scoring
      level_boost: 0.05      # Tree level boost
      recency: 0.05          # Recency boost
      authority: 0.03        # Authority/citation score
      overlap_bonus: 0.02    # Multi-channel overlap bonus
    
    # Question type specific weights
    question_type_weights:
      definition:
        sim_emb: 0.35
        sim_lex: 0.25
        level_boost: 0.20
        ent_overlap: 0.10
        recency: 0.05
        path_score: 0.05
      
      relation:
        ent_overlap: 0.25
        path_score: 0.25
        sim_emb: 0.20
        sim_lex: 0.15
        level_boost: 0.10
        recency: 0.05
      
      recent:
        recency: 0.30
        sim_emb: 0.25
        sim_lex: 0.20
        ent_overlap: 0.10
        level_boost: 0.10
        path_score: 0.05
```

## Question Types

The system automatically detects question types and applies appropriate weights:

### Definition Questions
- **Patterns**: "What is", "What are", "Define", "Explain", "Describe", "Who is"
- **Focus**: High embedding similarity and tree level benefits
- **Examples**: "What is artificial intelligence?", "Who is John Smith?"

### Relation Questions  
- **Patterns**: "Compare", "Versus", "Related", "How does", "Connection"
- **Focus**: High entity overlap and graph path scoring
- **Examples**: "How is X related to Y?", "Compare Google and Microsoft"

### Recent Questions
- **Patterns**: "Recent", "Latest", "New", "Current", "Breakthrough"
- **Focus**: High recency weighting and temporal relevance
- **Examples**: "What are recent breakthroughs?", "Latest developments in AI"

### Baseline
- **Default**: Balanced weights for general queries
- **Use**: When no specific pattern is detected

## Retrieval Channels

### Tree Recall
- **Purpose**: Top-down traversal of summary tree to find relevant leaf chunks
- **Algorithm**: Starts from root summaries, traverses down to leaves containing query entities
- **Configuration**: `tree_top_k` controls number of candidates

### Graph Recall  
- **Purpose**: Entity-seeded expansion through knowledge graph
- **Algorithm**: Starts from query entities, expands 1-2 hops to find related entities and their chunks
- **Configuration**: `graph_top_k`, `graph_max_hops`

### BM25 Recall
- **Purpose**: Lexical matching for keyword-based relevance
- **Algorithm**: TF-IDF approximation of BM25 scoring
- **Configuration**: `bm25_top_k`

### Dense Recall
- **Purpose**: Semantic similarity using neural embeddings
- **Algorithm**: FAISS-based similarity search with sentence transformers
- **Configuration**: `dense_top_k`

## Overlap Detection

### Hard Overlap (I = Tree ∩ Graph)
- **Definition**: Chunks that appear in both tree and graph recall results
- **Priority**: Gets priority allocation up to `cap_I_ratio` of budget
- **Boost**: Receives 20% score boost for appearing in multiple structured channels

### Soft Overlap  
- **Definition**: Chunks appearing across any combination of channels
- **Scoring**: RRF-style scoring with configurable `rrf_k` parameter
- **Purpose**: Provides additional confidence signal

## Budget Allocation

1. **Hard Overlap Priority**: Allocate up to `cap_I_ratio * budget_N` slots
2. **Remainder Selection**: Fill remaining slots using MMR selection
3. **Fallback Strategy**: If any channel fails, automatically adjust allocation
4. **Diversity Constraints**: Enforce limits on same-source clustering

## Legacy Compatibility

To use the original retrieval system:

```yaml
retriever:
  kwargs:
    use_enhanced_retrieval: False     # Disable enhanced system
    # ... other legacy parameters
```

The enhanced system maintains full backward compatibility and will fallback gracefully if needed.

## Performance Tuning

### For Speed
- Reduce `tree_top_k`, `graph_top_k`, `bm25_top_k`, `dense_top_k`
- Lower `budget_N`
- Set `graph_max_hops: 1`

### For Quality
- Increase channel top-k values
- Set `graph_max_hops: 2`
- Increase `budget_N`
- Fine-tune scoring weights for your domain

### For Diversity
- Increase `mmr_lambda` (more diversity)
- Decrease `cap_I_ratio` (less hard overlap dominance)
- Adjust `max_same_subtree_ratio` and `max_same_entity_cluster_ratio`