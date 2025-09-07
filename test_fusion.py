#!/usr/bin/env python3
"""
Simple test script to validate the fusion retrieval functionality
"""
import sys
import os
import logging
from unittest.mock import Mock
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_fusion_methods():
    """Test the fusion methods in isolation"""
    try:
        from query import Retriever
        
        # Create mock objects for testing - simplified version
        cache_tree = {
            "leaf_1": {"text": "This is about artificial intelligence and machine learning."},
            "leaf_2": {"text": "Computer vision is a field of AI that trains computers to interpret visual information."},
            "leaf_3": {"text": "Natural language processing enables computers to understand human language."}
        }
        
        G = Mock()  # Mock graph
        index = {
            "artificial": ["leaf_1"],
            "intelligence": ["leaf_1", "leaf_2"],
            "computer": ["leaf_2", "leaf_3"],
            "vision": ["leaf_2"],
            "language": ["leaf_3"],
            "processing": ["leaf_3"]
        }
        
        appearance_count = {
            "leaf_1": {"artificial": 1, "intelligence": 1, "machine": 1, "learning": 1},
            "leaf_2": {"computer": 2, "vision": 1, "intelligence": 1, "artificial": 1},
            "leaf_3": {"natural": 1, "language": 2, "processing": 1, "computer": 1}
        }
        
        nlp = Mock()
        nlp.naive_extract_graph = Mock(return_value={"nouns": ["artificial", "intelligence"]})
        
        # Create a mock tokenizer to avoid network calls
        mock_tokenizer = Mock()
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])  # Mock token IDs
        
        # Manually create retriever instance without calling the constructor
        retriever = object.__new__(Retriever)
        retriever.cache_tree = cache_tree
        retriever.collapse_tree = ["text1", "text2", "text3"]  
        retriever.collapse_tree_ids = ["leaf_1", "leaf_2", "leaf_3"]
        retriever.G = G
        retriever.index = index
        retriever.appearance_count = appearance_count
        retriever.inverse_index = {"leaf_1": ["artificial", "intelligence"], 
                                  "leaf_2": ["intelligence", "computer", "vision"],
                                  "leaf_3": ["language", "processing", "computer"]}
        retriever.nlp = nlp
        retriever.device = "cpu"
        retriever.embedder = None
        retriever.faiss_index = None
        retriever.tokenizer = mock_tokenizer
        
        # Test 1: Score normalization
        print("\n1. Testing score normalization:")
        test_scores = {"chunk1": 5.0, "chunk2": 10.0, "chunk3": 0.0}
        normalized = retriever._normalize_scores(test_scores)
        print(f"   Original: {test_scores}")
        print(f"   Normalized: {normalized}")
        assert all(0 <= score <= 1 for score in normalized.values()), "Scores not in [0,1] range"
        print("   ✓ Normalization works correctly")
        
        # Test 2: Dynamic weight calculation
        print("\n2. Testing dynamic weight calculation:")
        test_queries = [
            ("artificial intelligence machine learning", ["artificial", "intelligence"]),
            ("What is the future of technology?", ["technology"]),
            ("Tell me about computer vision and natural language processing systems", 
             ["computer", "vision", "natural", "language", "processing"])
        ]
        
        for query, entities in test_queries:
            alpha, beta = retriever._calculate_dynamic_weights(query, entities)
            entity_ratio = len(entities) / len(query.split()) if query.split() else 0
            print(f"   Query: '{query[:50]}...'")
            print(f"   Entities: {entities}")
            print(f"   Entity ratio: {entity_ratio:.2f}")
            print(f"   Weights: α={alpha}, β={beta}")
            assert abs(alpha + beta - 1.0) < 0.001, "Weights don't sum to 1"
            print("   ✓ Dynamic weights calculated correctly")
        
        # Test 3: Entity scoring
        print("\n3. Testing entity scoring:")
        test_candidates = {"artificial_intelligence": ["leaf_1", "leaf_2"]}
        test_entities = ["artificial", "intelligence"]
        entity_scores = retriever._entity_scoring(test_entities, test_candidates)
        print(f"   Candidates: {test_candidates}")
        print(f"   Entity scores: {entity_scores}")
        assert len(entity_scores) > 0, "No entity scores calculated"
        print("   ✓ Entity scoring works correctly")
        
        # Test 4: Score fusion
        print("\n4. Testing score fusion:")
        entity_scores = {"leaf_1": 0.8, "leaf_2": 0.6}
        summary_scores = {"leaf_1": 0.3, "leaf_2": 0.9, "leaf_3": 0.7}
        fused_scores = retriever._fuse_scores(entity_scores, summary_scores, 0.6, 0.4, threshold=0.2)
        print(f"   Entity scores: {entity_scores}")
        print(f"   Summary scores: {summary_scores}")
        print(f"   Fused scores: {fused_scores}")
        assert len(fused_scores) > 0, "No fused scores calculated"
        print("   ✓ Score fusion works correctly")
        
        # Test 5: Cross-encoder re-ranking
        print("\n5. Testing cross-encoder re-ranking:")
        test_query = "artificial intelligence and computer vision"
        test_chunks = ["leaf_1", "leaf_2", "leaf_3"]
        reranked = retriever._cross_encoder_rerank(test_query, test_chunks, top_n=2)
        print(f"   Original chunks: {test_chunks}")
        print(f"   Re-ranked: {reranked}")
        assert len(reranked) <= 2, "Re-ranking didn't respect top_n limit"
        print("   ✓ Cross-encoder re-ranking works correctly")
        
        print("\n🎉 All fusion methods tested successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing E²GraphRAG Fusion Retrieval Implementation")
    print("=" * 60)
    
    success = test_fusion_methods()
    
    if success:
        print("\n✅ All tests passed! Fusion retrieval implementation is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)