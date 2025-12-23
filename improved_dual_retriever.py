"""
Improved Dual-Path Retriever with Graph Traversal
Combines concept graph traversal with tree-based retrieval
"""

import logging
import numpy as np
from typing import List, Dict, Set, Tuple
from sentence_transformers import SentenceTransformer
import networkx as nx
from collections import defaultdict
from itertools import combinations

logger = logging.getLogger(__name__)


class ImprovedDualPathRetriever:
    """
    Improved retriever that combines:
    - Path A: Graph traversal + concept matching (like original)
    - Path B: Tree-based retrieval  
    - Weighted fusion based on concept importance
    """
    
    def __init__(self,
                 cache_tree: Dict,
                 concept_graph: nx.Graph,
                 concept_to_sentences: Dict[str, List[str]],
                 sentence_to_chunk: Dict[str, str],
                 concept_vectors: Dict[str, np.ndarray],
                 sentences: List[str],
                 concept_importance: Dict[str, float],
                 nlp_extractor,
                 embedder: SentenceTransformer,
                 device: str = "cuda:0",
                 concept_top_k: int = 30,
                 sentence_top_k: int = 40,
                 tree_top_k: int = 25,
                 concept_threshold: float = 0.5,
                 max_path_length: int = 3,
                 use_graph_traversal: bool = True):
        """
        Initialize improved dual-path retriever
        """
        self.cache_tree = cache_tree
        self.collapse_tree, self.collapse_tree_ids = self._collapse_tree(cache_tree)
        
        self.concept_graph = concept_graph
        self.concept_to_sentences = concept_to_sentences
        self.sentence_to_chunk = sentence_to_chunk
        self.concept_vectors = concept_vectors
        self.sentences = sentences
        self.concept_importance = concept_importance
        self.nlp_extractor = nlp_extractor
        
        self.embedder = embedder
        self.device = device
        
        self.concept_top_k = concept_top_k
        self.sentence_top_k = sentence_top_k
        self.tree_top_k = tree_top_k
        self.concept_threshold = concept_threshold
        self.max_path_length = max_path_length
        self.use_graph_traversal = use_graph_traversal
        
        # Pre-encode tree nodes
        logger.info("Pre-encoding tree nodes...")
        self.tree_embeddings = self.embedder.encode(self.collapse_tree, 
                                                    batch_size=32,
                                                    device=self.device)
        
        # Pre-encode sentences
        logger.info("Pre-encoding sentences...")
        self.sentence_embeddings = self.embedder.encode(self.sentences,
                                                       batch_size=32,
                                                       device=self.device)
        logger.info("Improved dual-path retriever initialized")
    
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
              concept_vectors: Dict, sentences: List[str],
              concept_importance: Dict):
        """Update retriever with new document"""
        self.cache_tree = cache_tree
        self.collapse_tree, self.collapse_tree_ids = self._collapse_tree(cache_tree)
        
        self.concept_graph = concept_graph
        self.concept_to_sentences = concept_to_sentences
        self.sentence_to_chunk = sentence_to_chunk
        self.concept_vectors = concept_vectors
        self.sentences = sentences
        self.concept_importance = concept_importance
        
        # Re-encode
        logger.info("Re-encoding tree nodes and sentences...")
        self.tree_embeddings = self.embedder.encode(self.collapse_tree,
                                                    batch_size=32,
                                                    device=self.device)
        self.sentence_embeddings = self.embedder.encode(self.sentences,
                                                       batch_size=32,
                                                       device=self.device)
    
    def path_a_graph_traversal_retrieval(self, query: str) -> Tuple[Set[str], List[str]]:
        """
        Path A: Graph traversal + concept matching
        
        Steps:
        1. Extract query concepts using NER
        2. Find matching concepts in graph
        3. Traverse graph to find related concepts
        4. Map back to chunks with ranking
        """
        logger.info("=== Path A: Graph Traversal Retrieval ===")
        
        # Extract query concepts using NER (like original system)
        query_result = self.nlp_extractor.naive_extract_graph(query)
        query_entities = set(query_result["nouns"])
        
        logger.info(f"A1: Extracted {len(query_entities)} query entities")
        logger.debug(f"Query entities: {list(query_entities)[:5]}...")
        
        if not query_entities:
            logger.warning("No query entities found, using concept matching fallback")
            return self._concept_matching_fallback(query)
        
        # Find matched concepts in graph
        matched_concepts = set()
        for entity in query_entities:
            if entity in self.concept_graph.nodes():
                matched_concepts.add(entity)
        
        logger.info(f"A2: Found {len(matched_concepts)} concepts in graph")
        
        if not matched_concepts:
            logger.warning("No concepts matched in graph, using concept matching fallback")
            return self._concept_matching_fallback(query)
        
        # Graph traversal: find related concepts
        if self.use_graph_traversal and len(matched_concepts) >= 2:
            logger.info("A3: Traversing graph for related concepts...")
            related_concepts = self._find_related_concepts(matched_concepts)
            all_concepts = matched_concepts.union(related_concepts)
            logger.info(f"A3: Found {len(related_concepts)} related concepts via traversal")
        else:
            all_concepts = matched_concepts
        
        # Map to sentences and chunks
        logger.info("A4: Mapping concepts to chunks...")
        chunk_scores = defaultdict(float)
        
        for concept in all_concepts:
            if concept in self.concept_to_sentences:
                sent_ids = self.concept_to_sentences[concept]
                importance = self.concept_importance.get(concept, 1.0)
                
                for sent_id in sent_ids:
                    if sent_id in self.sentence_to_chunk:
                        chunk_id = self.sentence_to_chunk[sent_id]
                        # Weight by concept importance
                        chunk_scores[chunk_id] += importance
        
        # Rank chunks by score
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        top_chunks = set([chunk_id for chunk_id, _ in sorted_chunks[:self.concept_top_k]])
        
        logger.info(f"A4: Retrieved {len(top_chunks)} chunks from graph path")
        
        return top_chunks, list(all_concepts)
    
    def _find_related_concepts(self, seed_concepts: Set[str]) -> Set[str]:
        """Find related concepts via graph traversal"""
        related = set()
        
        # Find shortest paths between seed concepts
        for c1, c2 in combinations(seed_concepts, 2):
            try:
                if nx.has_path(self.concept_graph, c1, c2):
                    path = nx.shortest_path(self.concept_graph, c1, c2)
                    if len(path) <= self.max_path_length:
                        # Add intermediate concepts
                        related.update(path[1:-1])
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        
        # Add direct neighbors of seed concepts
        for concept in seed_concepts:
            if concept in self.concept_graph:
                neighbors = list(self.concept_graph.neighbors(concept))
                # Add top neighbors by edge weight
                neighbor_weights = [(n, self.concept_graph[concept][n].get('weight', 0)) 
                                   for n in neighbors]
                neighbor_weights.sort(key=lambda x: x[1], reverse=True)
                related.update([n for n, w in neighbor_weights[:5]])
        
        return related
    
    def _concept_matching_fallback(self, query: str) -> Tuple[Set[str], List[str]]:
        """Fallback to concept matching when no entities found"""
        logger.info("Using concept vector matching...")
        
        query_embedding = self.embedder.encode(query, device=self.device)
        
        relevant_concepts = []
        concept_similarities = []
        
        for concept, concept_vec in self.concept_vectors.items():
            similarity = np.dot(query_embedding, concept_vec) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(concept_vec)
            )
            
            if similarity > self.concept_threshold:
                relevant_concepts.append(concept)
                concept_similarities.append(similarity)
        
        if relevant_concepts:
            sorted_indices = np.argsort(concept_similarities)[::-1][:self.concept_top_k]
            relevant_concepts = [relevant_concepts[i] for i in sorted_indices]
        
        logger.info(f"Found {len(relevant_concepts)} relevant concepts via matching")
        
        # Map to chunks
        chunk_ids = set()
        for concept in relevant_concepts:
            if concept in self.concept_to_sentences:
                for sent_id in self.concept_to_sentences[concept]:
                    if sent_id in self.sentence_to_chunk:
                        chunk_ids.add(self.sentence_to_chunk[sent_id])
        
        return chunk_ids, relevant_concepts
    
    def path_b_tree_retrieval(self, query: str) -> Set[str]:
        """Path B: Summary tree-based global retrieval"""
        logger.info("=== Path B: Summary Tree Retrieval ===")
        
        query_embedding = self.embedder.encode(query, device=self.device)
        
        similarities = np.dot(self.tree_embeddings, query_embedding) / (
            np.linalg.norm(self.tree_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_indices = np.argsort(similarities)[::-1][:self.tree_top_k]
        top_node_ids = [self.collapse_tree_ids[i] for i in top_indices]
        
        logger.info(f"B3: Selected top {len(top_node_ids)} nodes from tree")
        logger.debug(f"Top tree nodes: {top_node_ids[:5]}...")
        
        return set(top_node_ids)
    
    def fusion_and_generate(self, path_a_chunks: Set[str], 
                           path_b_nodes: Set[str],
                           query_concepts: List[str]) -> Dict[str, any]:
        """Weighted fusion of both paths"""
        logger.info("=== Fusion and Context Preparation ===")
        
        # Score chunks from both paths
        chunk_scores = {}
        
        # Path A chunks get higher weight (entity-based)
        for chunk_id in path_a_chunks:
            chunk_scores[chunk_id] = 2.0  # Higher weight for graph-based
        
        # Path B chunks
        for node_id in path_b_nodes:
            if node_id in chunk_scores:
                chunk_scores[node_id] += 1.0  # Boost if in both paths
            else:
                chunk_scores[node_id] = 1.0
        
        # Sort by score and take top chunks
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        top_chunk_ids = [chunk_id for chunk_id, score in sorted_chunks[:self.tree_top_k + 10]]
        
        logger.info(f"Path A contributed: {len(path_a_chunks)} chunks")
        logger.info(f"Path B contributed: {len(path_b_nodes)} nodes")
        logger.info(f"Final selection: {len(top_chunk_ids)} chunks after weighted fusion")
        
        # Prepare context
        context_chunks = []
        leaf_chunk_ids = []
        
        for node_id in top_chunk_ids:
            if node_id in self.cache_tree:
                context_chunks.append(self.cache_tree[node_id]["text"])
                if node_id.startswith("leaf_"):
                    leaf_chunk_ids.append(node_id)
        
        context_text = "\n\n".join(context_chunks)
        
        return {
            "chunks": context_text,
            "chunk_ids": top_chunk_ids,
            "leaf_chunk_ids": leaf_chunk_ids,
            "path_a_count": len(path_a_chunks),
            "path_b_count": len(path_b_nodes),
            "total_count": len(top_chunk_ids),
            "query_concepts": query_concepts
        }
    
    def query(self, query: str, debug: bool = True) -> Dict:
        """Execute improved dual-path retrieval with weighted fusion"""
        logger.info(f"Processing query: {query[:100]}...")
        
        # Execute both paths
        path_a_chunks, query_concepts = self.path_a_graph_traversal_retrieval(query)
        path_b_nodes = self.path_b_tree_retrieval(query)
        
        # Weighted fusion
        result = self.fusion_and_generate(path_a_chunks, path_b_nodes, query_concepts)
        
        if debug:
            result["debug_info"] = {
                "path_a_chunks": list(path_a_chunks),
                "path_b_nodes": list(path_b_nodes),
                "retrieval_type": "Improved Dual-Path (Graph Traversal + Tree)",
                "query_concepts": query_concepts
            }
        
        return result


def create_improved_dual_retriever(cache_tree: Dict,
                                   concept_graph: nx.Graph,
                                   concept_to_sentences: Dict,
                                   sentence_to_chunk: Dict,
                                   concept_vectors: Dict,
                                   sentences: List[str],
                                   concept_importance: Dict,
                                   nlp_extractor,
                                   embedder_model: str = "BAAI/bge-m3",
                                   device: str = "cuda:0",
                                   **kwargs) -> ImprovedDualPathRetriever:
    """Convenience function to create improved dual-path retriever"""
    embedder = SentenceTransformer(embedder_model, device=device)
    
    return ImprovedDualPathRetriever(
        cache_tree=cache_tree,
        concept_graph=concept_graph,
        concept_to_sentences=concept_to_sentences,
        sentence_to_chunk=sentence_to_chunk,
        concept_vectors=concept_vectors,
        sentences=sentences,
        concept_importance=concept_importance,
        nlp_extractor=nlp_extractor,
        embedder=embedder,
        device=device,
        **kwargs
    )
