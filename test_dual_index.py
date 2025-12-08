#!/usr/bin/env python3
"""
Test script for Dual-Index GraphRAG
Demonstrates the dual-path retrieval functionality
"""

import os
import sys
import logging
from typing import List
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dual_index_builder():
    """Test the DualIndexBuilder class"""
    from dual_index_graphrag import DualIndexBuilder, SpacyExtractorWithSentences
    
    logger.info("=" * 50)
    logger.info("Testing DualIndexBuilder")
    logger.info("=" * 50)
    
    # Sample text chunks
    text_chunks = [
        "John Smith works at Google. He is a software engineer.",
        "Mary Johnson is the CEO of Microsoft. She leads the company.",
        "Google and Microsoft are competing in cloud computing. Both companies invest heavily in AI."
    ]
    
    # Initialize extractor
    nlp = SpacyExtractorWithSentences(language="en")
    
    # Initialize dual index builder
    builder = DualIndexBuilder(embedder_model="BAAI/bge-m3", device="cpu")
    
    # Build sentence index
    logger.info("Building sentence index...")
    I_s_to_c, I_c_to_s, sentence_texts = builder.build_sentence_index(text_chunks, nlp)
    
    logger.info(f"Total sentences: {len(sentence_texts)}")
    logger.info(f"Total concepts: {len(I_c_to_s)}")
    
    # Show some examples
    logger.info("\nSample concepts and their sentences:")
    for i, (concept, sent_ids) in enumerate(list(I_c_to_s.items())[:5]):
        logger.info(f"  Concept '{concept}' appears in {len(sent_ids)} sentence(s)")
    
    # Build concept vectors
    logger.info("\nBuilding concept vectors...")
    concept_vectors = builder.build_concept_vectors(I_c_to_s, sentence_texts)
    
    logger.info(f"Built vectors for {len(concept_vectors)} concepts")
    logger.info(f"Vector dimension: {list(concept_vectors.values())[0].shape[0] if concept_vectors else 'N/A'}")
    
    logger.info("\n✓ DualIndexBuilder test passed!\n")
    
    return I_s_to_c, I_c_to_s, sentence_texts, concept_vectors, nlp, text_chunks


def test_dual_index_retriever():
    """Test the DualIndexRetriever class"""
    from dual_index_graphrag import DualIndexRetriever
    from extract_graph import build_graph
    
    logger.info("=" * 50)
    logger.info("Testing DualIndexRetriever")
    logger.info("=" * 50)
    
    # First build the indices
    I_s_to_c, I_c_to_s, sentence_texts, concept_vectors, nlp, text_chunks = test_dual_index_builder()
    
    # Build a simple cache tree (mock)
    cache_tree = {}
    for i, chunk in enumerate(text_chunks):
        cache_tree[f"leaf_{i}"] = {
            "text": chunk,
            "children": None,
            "parent": None
        }
    
    # Add a summary node
    cache_tree["summary_0_0"] = {
        "text": "Summary: Companies and their leaders in tech industry.",
        "children": ["leaf_0", "leaf_1", "leaf_2"],
        "parent": None
    }
    
    # Build concept graph
    result = nlp.naive_extract_graph(" ".join(text_chunks))
    edges = []
    for (n1, n2), weight in result["cooccurrence"].items():
        edges.append([n1, n2, weight])
    
    G = build_graph(edges)
    
    # Build index
    index = {}
    for concept in result["nouns"]:
        index[concept] = []
        for i, chunk in enumerate(text_chunks):
            if concept.lower() in chunk.lower():
                index[concept].append(f"leaf_{i}")
    
    # Initialize retriever
    logger.info("\nInitializing DualIndexRetriever...")
    retriever = DualIndexRetriever(
        cache_tree=cache_tree,
        G=G,
        index=index,
        I_s_to_c=I_s_to_c,
        I_c_to_s=I_c_to_s,
        sentence_texts=sentence_texts,
        concept_vectors=concept_vectors,
        embedder_model="BAAI/bge-m3",
        device="cpu"
    )
    
    # Test queries
    test_queries = [
        "Who works at Google?",
        "Tell me about Microsoft's CEO",
        "What are Google and Microsoft competing in?"
    ]
    
    for query in test_queries:
        logger.info("\n" + "-" * 50)
        logger.info(f"Query: {query}")
        
        # Perform dual-path retrieval
        C_pool, metadata = retriever.dual_path_retrieval(
            query, K_s=5, K_t=3, concept_threshold=0.2
        )
        
        logger.info(f"Results:")
        logger.info(f"  Path A (Concept Graph): {metadata['path_a_chunks']} chunks")
        logger.info(f"  Path B (Summary Tree): {metadata['path_b_chunks']} chunks")
        logger.info(f"  Total (Union): {metadata['total_chunks']} chunks")
        
        # Get chunks text
        chunks_text = retriever.get_chunks_text(C_pool)
        logger.info(f"\nRetrieved content preview:")
        logger.info(f"  {chunks_text[:200]}...")
    
    logger.info("\n✓ DualIndexRetriever test passed!\n")


def test_integration():
    """Test integration with existing code"""
    from extract_graph import extract_graph_with_dual_index, load_nlp
    from build_tree import build_tree
    from transformers import AutoTokenizer, pipeline
    from query import Retriever
    import tempfile
    import shutil
    
    logger.info("=" * 50)
    logger.info("Testing Integration with Main Pipeline")
    logger.info("=" * 50)
    
    # Create temporary cache folder
    cache_folder = tempfile.mkdtemp(prefix="dual_index_test_")
    logger.info(f"Using temporary cache folder: {cache_folder}")
    
    try:
        # Sample text
        text = [
            "The artificial intelligence revolution is transforming industries. Machine learning algorithms are becoming more sophisticated.",
            "Deep learning models require large amounts of data. Neural networks can learn complex patterns from training data.",
            "Natural language processing enables computers to understand human language. Transformers have revolutionized NLP tasks."
        ]
        
        # Load NLP extractor
        logger.info("\nLoading NLP extractor...")
        nlp = load_nlp(language="en", method="Spacy")
        
        # Extract graph with dual index
        logger.info("\nExtracting graph with dual index...")
        (G, index, appearance_count, dual_index_data), time_cost = extract_graph_with_dual_index(
            text=text,
            cache_folder=cache_folder,
            nlp=nlp,
            use_cache=False,
            reextract=False,
            build_dual_index=True,
            embedder_model="BAAI/bge-m3",
            device="cpu"
        )
        
        logger.info(f"Graph extraction completed in {time_cost:.2f}s")
        logger.info(f"  Graph nodes: {G.number_of_nodes()}")
        logger.info(f"  Graph edges: {G.number_of_edges()}")
        logger.info(f"  Concepts indexed: {len(index)}")
        
        if dual_index_data:
            logger.info(f"  Sentences indexed: {len(dual_index_data['sentence_texts'])}")
            logger.info(f"  Concept vectors: {len(dual_index_data['concept_vectors'])}")
        
        # Build a simple tree for testing
        logger.info("\nBuilding mock tree...")
        cache_tree = {}
        for i, chunk in enumerate(text):
            cache_tree[f"leaf_{i}"] = {
                "text": chunk,
                "children": None,
                "parent": None
            }
        
        # Initialize retriever with dual index
        logger.info("\nInitializing retriever...")
        retriever = Retriever(
            cache_tree=cache_tree,
            G=G,
            index=index,
            appearance_count=appearance_count,
            nlp=nlp,
            device="cpu",
            embedder="BAAI/bge-m3",
            tokenizer="bert-base-uncased",  # Use a simple tokenizer for testing
            dual_index_data=dual_index_data
        )
        
        # Test dual-path query
        test_query = "What is deep learning and how does it work?"
        logger.info(f"\nTesting dual-path query: '{test_query}'")
        
        if retriever.dual_retriever:
            result = retriever.query_dual_path(test_query, K_s=5, K_t=3, concept_threshold=0.2)
            logger.info(f"Retrieval type: {result['retrieval_type']}")
            logger.info(f"Total chunks: {result.get('total_chunks', 'N/A')}")
            logger.info(f"Path A chunks: {result.get('path_a_chunks', 'N/A')}")
            logger.info(f"Path B chunks: {result.get('path_b_chunks', 'N/A')}")
        else:
            logger.warning("Dual retriever not initialized, using standard query")
            result = retriever.query(test_query)
        
        logger.info("\n✓ Integration test passed!\n")
        
    finally:
        # Cleanup
        if os.path.exists(cache_folder):
            shutil.rmtree(cache_folder)
            logger.info(f"Cleaned up temporary folder: {cache_folder}")


def main():
    """Run all tests"""
    logger.info("Starting Dual-Index GraphRAG Tests")
    logger.info("=" * 70)
    
    try:
        # Test 1: DualIndexBuilder
        test_dual_index_builder()
        
        # Test 2: DualIndexRetriever
        test_dual_index_retriever()
        
        # Test 3: Integration
        test_integration()
        
        logger.info("=" * 70)
        logger.info("All tests passed! ✓")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
