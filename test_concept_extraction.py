#!/usr/bin/env python3
"""
Quick test for RAKE and TF-IDF concept extraction
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_rake_extraction():
    """Test RAKE-based concept extraction"""
    from dual_index_graphrag import DualIndexBuilder
    
    logger.info("Testing RAKE concept extraction...")
    
    # Sample text chunks
    text_chunks = [
        "Machine learning algorithms are becoming increasingly sophisticated in natural language processing.",
        "Deep neural networks have revolutionized computer vision and image recognition tasks.",
        "Artificial intelligence systems can now perform complex reasoning and decision-making."
    ]
    
    builder = DualIndexBuilder(
        embedder_model="BAAI/bge-m3",
        device="cpu",
        concept_extraction_method="rake"
    )
    
    try:
        I_s_to_c, I_c_to_s, sentence_texts = builder.build_sentence_index(text_chunks, nlp=None)
        
        logger.info(f"✓ RAKE extraction successful")
        logger.info(f"  Sentences: {len(sentence_texts)}")
        logger.info(f"  Concepts: {len(I_c_to_s)}")
        logger.info(f"  Sample concepts: {list(I_c_to_s.keys())[:5]}")
        
        return True
    except Exception as e:
        logger.error(f"✗ RAKE extraction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_tfidf_extraction():
    """Test TF-IDF-based concept extraction"""
    from dual_index_graphrag import DualIndexBuilder
    
    logger.info("\nTesting TF-IDF concept extraction...")
    
    # Sample text chunks
    text_chunks = [
        "Machine learning algorithms are becoming increasingly sophisticated in natural language processing.",
        "Deep neural networks have revolutionized computer vision and image recognition tasks.",
        "Artificial intelligence systems can now perform complex reasoning and decision-making."
    ]
    
    builder = DualIndexBuilder(
        embedder_model="BAAI/bge-m3",
        device="cpu",
        concept_extraction_method="tfidf"
    )
    
    try:
        I_s_to_c, I_c_to_s, sentence_texts = builder.build_sentence_index(text_chunks, nlp=None)
        
        logger.info(f"✓ TF-IDF extraction successful")
        logger.info(f"  Sentences: {len(sentence_texts)}")
        logger.info(f"  Concepts: {len(I_c_to_s)}")
        logger.info(f"  Sample concepts: {list(I_c_to_s.keys())[:5]}")
        
        return True
    except Exception as e:
        logger.error(f"✗ TF-IDF extraction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    logger.info("=" * 70)
    logger.info("Testing Concept Extraction Methods (RAKE & TF-IDF)")
    logger.info("=" * 70)
    
    rake_ok = test_rake_extraction()
    tfidf_ok = test_tfidf_extraction()
    
    logger.info("\n" + "=" * 70)
    if rake_ok and tfidf_ok:
        logger.info("✓ All concept extraction tests passed!")
        return 0
    else:
        logger.error("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
