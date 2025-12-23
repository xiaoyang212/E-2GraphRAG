"""
Dual-Path Retriever for Dual-Index GraphRAG
Implements parallel retrieval from concept graph and summary tree with fusion
"""

import logging
import numpy as np
from typing import List, Dict, Set, Tuple
from sentence_transformers import SentenceTransformer
import networkx as nx
from collections import defaultdict

logger = logging.getLogger(__name__)


class DualPathRetriever:
    """
    Dual-path retriever implementing:
    - Path A: Concept graph-based fine-grained retrieval
    - Path B: Summary tree-based global retrieval
    - Fusion strategy: Union of both paths
    """
    
    def __init__(self,
                 # Summary tree components
                 cache_tree: Dict,
                 # Concept graph components (dual-index)
                 concept_graph: nx.Graph,
                 concept_to_sentences: Dict[str, List[str]],
                 sentence_to_chunk: Dict[str, str],
                 concept_vectors: Dict[str, np.ndarray],
                 sentences: List[str],
                 # Embedder for query encoding
                 embedder: SentenceTransformer,
                 device: str = "cuda:0",
                 # Retrieval parameters
                 concept_top_k: int = 20,
                 sentence_top_k: int = 30,
                 tree_top_k: int = 25,
                 concept_threshold: float = 0.6):
        """
        Initialize dual-path retriever
        
        Args:
            cache_tree: Summary tree structure
            concept_graph: Concept graph from dual-index
            concept_to_sentences: Inverted index (concept -> sentences)
            sentence_to_chunk: Sentence to chunk mapping
            concept_vectors: Concept vectors for similarity matching
            sentences: List of all sentences
            embedder: Sentence encoder for query
            device: Device for embedder
            concept_top_k: Number of top concepts to retrieve
            sentence_top_k: Number of top sentences to retrieve
            tree_top_k: Number of top tree nodes to retrieve
            concept_threshold: Similarity threshold for concept matching
        """
        # Summary tree
        self.cache_tree = cache_tree
        self.collapse_tree, self.collapse_tree_ids = self._collapse_tree(cache_tree)
        
        # Concept graph (dual-index)
        self.concept_graph = concept_graph
        self.concept_to_sentences = concept_to_sentences
        self.sentence_to_chunk = sentence_to_chunk
        self.concept_vectors = concept_vectors
        self.sentences = sentences
        
        # Embedder
        self.embedder = embedder
        self.device = device
        
        # Parameters
        self.concept_top_k = concept_top_k
        self.sentence_top_k = sentence_top_k
        self.tree_top_k = tree_top_k
        self.concept_threshold = concept_threshold
        
        # Pre-encode tree nodes and sentences for efficient retrieval
        logger.info("Pre-encoding tree nodes and sentences...")
        self.tree_embeddings = self.embedder.encode(self.collapse_tree, 
                                                    batch_size=32,
                                                    device=self.device)
        self.sentence_embeddings = self.embedder.encode(self.sentences,
                                                       batch_size=32,
                                                       device=self.device)
        logger.info("Dual-path retriever initialized")
    
    def _collapse_tree(self, cache_tree: Dict) -> Tuple[List[str], List[str]]:
        """Flatten tree for retrieval"""
        collapsed_tree = []
        collapsed_tree_ids = []
        for key, value in cache_tree.items():
            collapsed_tree.append(value["text"])
            collapsed_tree_ids.append(key)
        return collapsed_tree, collapsed_tree_ids
    
    def update(self, cache_tree: Dict, concept_graph: nx.Graph,
              concept_to_sentences: Dict, sentence_to_chunk: Dict,
              concept_vectors: Dict, sentences: List[str]):
        """Update retriever with new document"""
        self.cache_tree = cache_tree
        self.collapse_tree, self.collapse_tree_ids = self._collapse_tree(cache_tree)
        
        self.concept_graph = concept_graph
        self.concept_to_sentences = concept_to_sentences
        self.sentence_to_chunk = sentence_to_chunk
        self.concept_vectors = concept_vectors
        self.sentences = sentences
        
        # Re-encode
        logger.info("Re-encoding tree nodes and sentences...")
        self.tree_embeddings = self.embedder.encode(self.collapse_tree,
                                                    batch_size=32,
                                                    device=self.device)
        self.sentence_embeddings = self.embedder.encode(self.sentences,
                                                       batch_size=32,
                                                       device=self.device)
    
    def path_a_concept_retrieval(self, query: str) -> Set[str]:
        """
        Path A: Concept graph-based fine-grained retrieval
        
        Steps:
        A1: Concept matching - find relevant concepts
        A2: Candidate sentence recall - get sentences for concepts
        A3: Sentence re-ranking - rank sentences by similarity
        A4: Map sentences back to chunks
        
        Args:
            query: User query
            
        Returns:
            Set of chunk IDs from concept-based retrieval
        """
        logger.info("=== Path A: Concept Graph Retrieval ===")
        
        # A1: Concept Matching
        query_embedding = self.embedder.encode(query, device=self.device)
        
        relevant_concepts = []
        concept_similarities = []
        
        for concept, concept_vec in self.concept_vectors.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, concept_vec) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(concept_vec)
            )
            
            if similarity > self.concept_threshold:
                relevant_concepts.append(concept)
                concept_similarities.append(similarity)
        
        # Sort by similarity and take top-k
        if relevant_concepts:
            sorted_indices = np.argsort(concept_similarities)[::-1][:self.concept_top_k]
            relevant_concepts = [relevant_concepts[i] for i in sorted_indices]
        
        logger.info(f"A1: Found {len(relevant_concepts)} relevant concepts")
        logger.debug(f"Relevant concepts: {relevant_concepts[:5]}...")
        
        if not relevant_concepts:
            logger.warning("No relevant concepts found, Path A returns empty")
            return set()
        
        # A2: Candidate Sentence Recall
        candidate_sentences = set()
        for concept in relevant_concepts:
            if concept in self.concept_to_sentences:
                candidate_sentences.update(self.concept_to_sentences[concept])
        
        logger.info(f"A2: Retrieved {len(candidate_sentences)} candidate sentences")
        
        if not candidate_sentences:
            return set()
        
        # A3: Sentence Re-ranking
        candidate_sent_ids = list(candidate_sentences)
        candidate_sent_indices = []
        for sid in candidate_sent_ids:
            try:
                idx = int(sid.split('_')[1])
                candidate_sent_indices.append(idx)
            except (ValueError, IndexError) as e:
                logger.warning(f"Invalid sentence ID format: {sid}, skipping")
        
        # Get embeddings for candidate sentences
        candidate_embeddings = self.sentence_embeddings[candidate_sent_indices]
        
        # Calculate similarities
        similarities = np.dot(candidate_embeddings, query_embedding) / (
            np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k sentences
        top_indices = np.argsort(similarities)[::-1][:self.sentence_top_k]
        top_sentence_ids = [candidate_sent_ids[i] for i in top_indices]
        
        logger.info(f"A3: Re-ranked to top {len(top_sentence_ids)} sentences")
        
        # A4: Map sentences back to chunks
        chunk_ids = set()
        for sent_id in top_sentence_ids:
            if sent_id in self.sentence_to_chunk:
                chunk_ids.add(self.sentence_to_chunk[sent_id])
        
        logger.info(f"A4: Mapped to {len(chunk_ids)} chunks from concept path")
        
        return chunk_ids
    
    def path_b_tree_retrieval(self, query: str) -> Set[str]:
        """
        Path B: Summary tree-based global retrieval
        
        Steps:
        B1: Flatten tree - treat all nodes equally
        B2: Vector matching - compute similarity with query
        B3: Top-K selection - select most similar nodes
        
        Args:
            query: User query
            
        Returns:
            Set of node IDs (chunk or summary) from tree retrieval
        """
        logger.info("=== Path B: Summary Tree Retrieval ===")
        
        # B1 & B2: Already flattened tree, compute similarities
        query_embedding = self.embedder.encode(query, device=self.device)
        
        similarities = np.dot(self.tree_embeddings, query_embedding) / (
            np.linalg.norm(self.tree_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # B3: Top-K selection
        top_indices = np.argsort(similarities)[::-1][:self.tree_top_k]
        top_node_ids = [self.collapse_tree_ids[i] for i in top_indices]
        
        logger.info(f"B3: Selected top {len(top_node_ids)} nodes from tree")
        logger.debug(f"Top tree nodes: {top_node_ids[:5]}...")
        
        return set(top_node_ids)
    
    def fusion_and_generate(self, path_a_chunks: Set[str], 
                           path_b_nodes: Set[str]) -> Dict[str, any]:
        """
        Fuse results from both paths and prepare context for generation
        
        Args:
            path_a_chunks: Chunk IDs from concept path
            path_b_nodes: Node IDs from tree path
            
        Returns:
            Dictionary with fused chunks and metadata
        """
        logger.info("=== Fusion and Context Preparation ===")
        
        # Union of both paths (remove duplicates)
        all_node_ids = path_a_chunks | path_b_nodes
        
        logger.info(f"Path A contributed: {len(path_a_chunks)} chunks")
        logger.info(f"Path B contributed: {len(path_b_nodes)} nodes")
        logger.info(f"Total unique nodes after fusion: {len(all_node_ids)}")
        
        # Prepare context by retrieving text from cache_tree
        context_chunks = []
        chunk_only_ids = []
        
        for node_id in all_node_ids:
            if node_id in self.cache_tree:
                context_chunks.append(self.cache_tree[node_id]["text"])
                # Track which are leaf chunks vs summary nodes
                if node_id.startswith("leaf_"):
                    chunk_only_ids.append(node_id)
        
        # Sort leaf chunks by their index for better coherence
        chunk_only_ids.sort(key=lambda x: int(x.split('_')[1]))
        
        context_text = "\n\n".join(context_chunks)
        
        return {
            "chunks": context_text,
            "chunk_ids": list(all_node_ids),
            "leaf_chunk_ids": chunk_only_ids,
            "path_a_count": len(path_a_chunks),
            "path_b_count": len(path_b_nodes),
            "total_count": len(all_node_ids)
        }
    
    def query(self, query: str, debug: bool = True) -> Dict:
        """
        Execute dual-path retrieval and fusion
        
        Args:
            query: User query
            debug: Whether to include debug information
            
        Returns:
            Dictionary with context and metadata
        """
        logger.info(f"Processing query: {query[:100]}...")
        
        # Execute both paths in parallel (conceptually - could use threading)
        path_a_chunks = self.path_a_concept_retrieval(query)
        path_b_nodes = self.path_b_tree_retrieval(query)
        
        # Fusion
        result = self.fusion_and_generate(path_a_chunks, path_b_nodes)
        
        if debug:
            result["debug_info"] = {
                "path_a_chunks": list(path_a_chunks),
                "path_b_nodes": list(path_b_nodes),
                "retrieval_type": "Dual-Path (Concept Graph + Summary Tree)"
            }
        
        return result


def create_dual_retriever(cache_tree: Dict,
                         concept_graph: nx.Graph,
                         concept_to_sentences: Dict,
                         sentence_to_chunk: Dict,
                         concept_vectors: Dict,
                         sentences: List[str],
                         embedder_model: str = "BAAI/bge-m3",
                         device: str = "cuda:0",
                         **kwargs) -> DualPathRetriever:
    """
    Convenience function to create dual-path retriever
    """
    embedder = SentenceTransformer(embedder_model, device=device)
    
    return DualPathRetriever(
        cache_tree=cache_tree,
        concept_graph=concept_graph,
        concept_to_sentences=concept_to_sentences,
        sentence_to_chunk=sentence_to_chunk,
        concept_vectors=concept_vectors,
        sentences=sentences,
        embedder=embedder,
        device=device,
        **kwargs
    )
