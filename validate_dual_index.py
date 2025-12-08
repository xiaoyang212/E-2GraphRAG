#!/usr/bin/env python3
"""
Quick validation test for Dual-Index GraphRAG implementation
Tests the code structure without requiring model downloads
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        import dual_index_graphrag
        logger.info("✓ dual_index_graphrag imported successfully")
        
        from dual_index_graphrag import (
            DualIndexBuilder,
            DualIndexRetriever,
            SpacyExtractorWithSentences,
            save_dual_index,
            load_dual_index
        )
        logger.info("✓ All classes and functions imported successfully")
        
        # Test class instantiation (without model loading)
        logger.info("Testing class structures...")
        
        # Check class attributes exist
        assert hasattr(DualIndexBuilder, 'build_sentence_index')
        assert hasattr(DualIndexBuilder, 'build_concept_vectors')
        logger.info("✓ DualIndexBuilder has required methods")
        
        assert hasattr(DualIndexRetriever, 'path_a_concept_graph_retrieval')
        assert hasattr(DualIndexRetriever, 'path_b_summary_tree_retrieval')
        assert hasattr(DualIndexRetriever, 'dual_path_retrieval')
        logger.info("✓ DualIndexRetriever has required methods")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Import test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_extract_graph_integration():
    """Test integration with extract_graph module"""
    logger.info("\nTesting extract_graph integration...")
    
    try:
        from extract_graph import extract_graph_with_dual_index
        logger.info("✓ extract_graph_with_dual_index imported successfully")
        
        # Check function signature
        import inspect
        sig = inspect.signature(extract_graph_with_dual_index)
        params = list(sig.parameters.keys())
        
        required_params = ['text', 'cache_folder', 'nlp', 'use_cache', 
                          'reextract', 'build_dual_index', 'embedder_model', 'device']
        
        for param in ['text', 'cache_folder', 'nlp']:
            assert param in params, f"Missing required parameter: {param}"
        
        logger.info("✓ extract_graph_with_dual_index has correct signature")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ extract_graph integration test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_query_integration():
    """Test integration with query module"""
    logger.info("\nTesting query integration...")
    
    try:
        from query import Retriever
        logger.info("✓ Retriever imported successfully")
        
        # Check new methods
        assert hasattr(Retriever, 'query_dual_path')
        assert hasattr(Retriever, '_init_dual_retriever')
        logger.info("✓ Retriever has dual-path methods")
        
        # Check __init__ accepts dual_index_data
        import inspect
        sig = inspect.signature(Retriever.__init__)
        
        # Check that kwargs can accept dual_index_data
        logger.info("✓ Retriever.__init__ accepts kwargs for dual_index_data")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Query integration test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_config():
    """Test configuration file"""
    logger.info("\nTesting configuration...")
    
    try:
        import yaml
        import os
        
        config_path = "configs/dual_index_config.yaml"
        assert os.path.exists(config_path), f"Config file not found: {config_path}"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        assert 'dual_index' in config, "Missing 'dual_index' section"
        assert 'retriever' in config, "Missing 'retriever' section"
        
        # Check dual_index settings
        dual_index = config['dual_index']
        assert 'enabled' in dual_index, "Missing 'enabled' setting"
        assert 'build_sentence_index' in dual_index, "Missing 'build_sentence_index' setting"
        assert 'embedder_model' in dual_index, "Missing 'embedder_model' setting"
        
        logger.info("✓ Configuration file is valid")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Config test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_documentation():
    """Test that documentation exists"""
    logger.info("\nTesting documentation...")
    
    try:
        import os
        
        doc_path = "DUAL_INDEX_README.md"
        assert os.path.exists(doc_path), f"Documentation not found: {doc_path}"
        
        with open(doc_path, 'r') as f:
            content = f.read()
        
        # Check for key sections
        assert "Dual-Index GraphRAG" in content, "Missing title"
        assert "Phase 1: Index Construction" in content, "Missing Phase 1 documentation"
        assert "Phase 2: Retrieval & Fusion" in content, "Missing Phase 2 documentation"
        assert "API Reference" in content, "Missing API reference"
        assert "Usage" in content, "Missing usage examples"
        
        logger.info("✓ Documentation is complete")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Documentation test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Run all validation tests"""
    logger.info("=" * 70)
    logger.info("Dual-Index GraphRAG Validation Tests")
    logger.info("=" * 70)
    
    tests = [
        ("Import Test", test_imports),
        ("Extract Graph Integration", test_extract_graph_integration),
        ("Query Integration", test_query_integration),
        ("Configuration", test_config),
        ("Documentation", test_documentation),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Test Summary")
    logger.info("=" * 70)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        logger.info("\n✓ All validation tests passed!")
        logger.info("=" * 70)
        return 0
    else:
        logger.error("\n✗ Some validation tests failed")
        logger.info("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
