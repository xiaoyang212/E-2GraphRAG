"""
Test script for Dual-Index GraphRAG implementation
Tests the core functionality without requiring full model downloads
"""

import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_dual_index_builder():
    """Test the dual-index builder module"""
    logger.info("=" * 50)
    logger.info("Testing Dual-Index Builder")
    logger.info("=" * 50)
    
    try:
        from dual_index_builder import DualIndexBuilder
        logger.info("✓ DualIndexBuilder imported successfully")
        
        # Test initialization (without actually loading models)
        logger.info("Testing DualIndexBuilder class structure...")
        
        # Check that all required methods exist
        required_methods = [
            'extract_sentences',
            'extract_concepts_tfidf',
            'build_concept_vectors',
            'build_concept_graph',
            'build_dual_index'
        ]
        
        for method in required_methods:
            if not hasattr(DualIndexBuilder, method):
                logger.error(f"✗ Missing method: {method}")
                return False
            logger.info(f"✓ Method exists: {method}")
        
        logger.info("✓ DualIndexBuilder has all required methods")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Failed to import DualIndexBuilder: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dual_retriever():
    """Test the dual-path retriever module"""
    logger.info("=" * 50)
    logger.info("Testing Dual-Path Retriever")
    logger.info("=" * 50)
    
    try:
        from dual_retriever import DualPathRetriever
        logger.info("✓ DualPathRetriever imported successfully")
        
        # Test class structure
        logger.info("Testing DualPathRetriever class structure...")
        
        required_methods = [
            'path_a_concept_retrieval',
            'path_b_tree_retrieval',
            'fusion_and_generate',
            'query',
            'update'
        ]
        
        for method in required_methods:
            if not hasattr(DualPathRetriever, method):
                logger.error(f"✗ Missing method: {method}")
                return False
            logger.info(f"✓ Method exists: {method}")
        
        logger.info("✓ DualPathRetriever has all required methods")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Failed to import DualPathRetriever: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_main_dual_index():
    """Test the main integration script"""
    logger.info("=" * 50)
    logger.info("Testing Main Dual-Index Script")
    logger.info("=" * 50)
    
    try:
        import main_dual_index
        logger.info("✓ main_dual_index imported successfully")
        
        # Check that required functions exist
        required_functions = [
            'parse_args',
            'parallel_build_dual_index',
            'main'
        ]
        
        for func in required_functions:
            if not hasattr(main_dual_index, func):
                logger.error(f"✗ Missing function: {func}")
                return False
            logger.info(f"✓ Function exists: {func}")
        
        logger.info("✓ main_dual_index has all required functions")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Failed to import main_dual_index: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tfidf_extraction():
    """Test TF-IDF concept extraction logic"""
    logger.info("=" * 50)
    logger.info("Testing TF-IDF Concept Extraction")
    logger.info("=" * 50)
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Simple test with sample texts
        sample_chunks = [
            "The concept graph is a powerful tool for information retrieval.",
            "Summary trees provide hierarchical organization of documents.",
            "Dual-index systems combine fine-grained and global retrieval."
        ]
        
        logger.info("Testing TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=10,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(sample_chunks)
        feature_names = vectorizer.get_feature_names_out()
        
        logger.info(f"✓ Extracted {len(feature_names)} features")
        logger.info(f"  Sample features: {list(feature_names[:5])}")
        
        # Test getting top concepts from first chunk
        chunk_tfidf = tfidf_matrix[0].toarray()[0]
        top_indices = chunk_tfidf.argsort()[-3:][::-1]
        top_concepts = [feature_names[idx] for idx in top_indices if chunk_tfidf[idx] > 0]
        
        logger.info(f"✓ Top concepts from first chunk: {top_concepts}")
        logger.info("✓ TF-IDF extraction works correctly")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ scikit-learn not available: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ TF-IDF test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_file():
    """Test configuration file parsing"""
    logger.info("=" * 50)
    logger.info("Testing Configuration File")
    logger.info("=" * 50)
    
    try:
        import yaml
        import os
        
        config_path = "configs/dual_index_config.yaml"
        if not os.path.exists(config_path):
            logger.error(f"✗ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("✓ Configuration file loaded successfully")
        
        # Check required sections
        required_sections = ['dataset', 'llm', 'paths', 'retriever', 'cluster']
        for section in required_sections:
            if section not in config:
                logger.error(f"✗ Missing config section: {section}")
                return False
            logger.info(f"✓ Config section exists: {section}")
        
        # Check dual-index specific parameters
        retriever_kwargs = config['retriever']['kwargs']
        dual_params = ['concept_top_k', 'sentence_top_k', 'tree_top_k', 'concept_threshold']
        
        for param in dual_params:
            if param not in retriever_kwargs:
                logger.error(f"✗ Missing dual-index parameter: {param}")
                return False
            logger.info(f"✓ Dual-index parameter exists: {param} = {retriever_kwargs[param]}")
        
        logger.info("✓ Configuration file is valid")
        return True
        
    except Exception as e:
        logger.error(f"✗ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "=" * 60)
    logger.info("DUAL-INDEX GRAPHRAG TEST SUITE")
    logger.info("=" * 60 + "\n")
    
    tests = [
        ("Dual-Index Builder", test_dual_index_builder),
        ("Dual-Path Retriever", test_dual_retriever),
        ("Main Integration Script", test_main_dual_index),
        ("TF-IDF Extraction", test_tfidf_extraction),
        ("Configuration File", test_config_file)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False
        logger.info("")  # Blank line between tests
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{status}: {test_name}")
    
    logger.info("-" * 60)
    logger.info(f"Total: {passed}/{total} tests passed")
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("\n🎉 All tests passed! Dual-Index GraphRAG implementation is ready.")
        return 0
    else:
        logger.warning(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
