"""
Dual-Index GraphRAG Implementation
Based on: 双索引长文本GraphRAG (Dual-Index GraphRAG for Long Context)

This module implements a dual-index retrieval system combining:
1. Concept Graph (fine-grained retrieval)
2. Summary Tree (global retrieval)
"""

import os
import json
import logging
from typing import List, Dict, Set, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from extract_graph import Extractor
import networkx as nx
import spacy
import nltk

logger = logging.getLogger(__name__)


class DualIndexBuilder:
    """
    Builds dual indices: concept graph and summary tree
    Implements the indexing phase from the algorithm
    """
    
    def __init__(self, embedder_model: str = "BAAI/bge-m3", device: str = "cuda:0"):
        self.embedder = SentenceTransformer(embedder_model, device=device)
        self.device = device
    
    def build_sentence_index(self, text_chunks: List[str], nlp: Extractor) -> Tuple[Dict, Dict]:
        """
        Build sentence-level indices for concept graph retrieval
        
        Args:
            text_chunks: List of text chunks (C = {c_1, c_2, ..., c_N})
            nlp: Extractor instance for NLP processing
            
        Returns:
            I_s_to_c: Mapping from sentences to chunks (I_{s→c}: s → c(s))
            I_c_to_s: Mapping from concepts to sentences (I_{c→s}: w → S_w)
        """
        I_s_to_c = {}  # sentence ID to chunk ID mapping
        I_c_to_s = {}  # concept to sentence IDs mapping
        sentence_texts = {}  # sentence ID to text mapping
        
        sentence_id = 0
        
        for chunk_id, chunk_text in enumerate(text_chunks):
            # Extract sentences from the chunk
            if isinstance(nlp, SpacyExtractorWithSentences):
                sentences = nlp.extract_sentences(chunk_text)
            else:
                # Fallback to simple sentence splitting
                sentences = self._simple_sentence_split(chunk_text)
            
            for sentence in sentences:
                # Map sentence to chunk
                I_s_to_c[f"sent_{sentence_id}"] = f"leaf_{chunk_id}"
                sentence_texts[f"sent_{sentence_id}"] = sentence
                
                # Extract concepts from sentence
                result = nlp.naive_extract_graph(sentence)
                concepts = result.get("nouns", [])
                
                # Map concepts to sentences
                for concept in concepts:
                    if concept not in I_c_to_s:
                        I_c_to_s[concept] = []
                    I_c_to_s[concept].append(f"sent_{sentence_id}")
                
                sentence_id += 1
        
        return I_s_to_c, I_c_to_s, sentence_texts
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting fallback"""
        try:
            return nltk.sent_tokenize(text)
        except:
            # Very basic fallback
            return [s.strip() for s in text.split('.') if s.strip()]
    
    def build_concept_vectors(self, I_c_to_s: Dict[str, List[str]], 
                             sentence_texts: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Build concept vectors by averaging sentence embeddings
        
        Vector(w) = (1/|S_w|) * Σ_{s ∈ S_w} φ(s)
        
        Args:
            I_c_to_s: Concept to sentences mapping
            sentence_texts: Sentence ID to text mapping
            
        Returns:
            concept_vectors: Mapping from concept to vector
        """
        concept_vectors = {}
        
        for concept, sentence_ids in I_c_to_s.items():
            # Get all sentences containing this concept
            sentences = [sentence_texts[sid] for sid in sentence_ids if sid in sentence_texts]
            
            if not sentences:
                continue
            
            # Encode sentences
            sentence_embeds = self.embedder.encode(sentences, convert_to_numpy=True)
            
            # Average the embeddings
            concept_vector = np.mean(sentence_embeds, axis=0)
            concept_vectors[concept] = concept_vector
        
        return concept_vectors
    
    def build_graph_with_vectors(self, concept_vectors: Dict[str, np.ndarray],
                                 cooccurrence: Dict[Tuple[str, str], int],
                                 theta_sem: float = 0.3,
                                 theta_co: int = 1) -> nx.Graph:
        """
        Build concept graph with edge weights using Dice coefficient
        
        Edge weight: r(w_i, w_j) = 2 * Co(w_i, w_j) / (|T_{w_i}| + |T_{w_j}|)
        
        Args:
            concept_vectors: Concept to vector mapping
            cooccurrence: Co-occurrence counts between concepts
            theta_sem: Semantic similarity threshold
            theta_co: Co-occurrence threshold
            
        Returns:
            Graph with weighted edges
        """
        G = nx.Graph()
        
        # Add nodes
        for concept in concept_vectors.keys():
            G.add_node(concept)
        
        # Add edges based on similarity and co-occurrence
        concepts = list(concept_vectors.keys())
        
        for i, concept_i in enumerate(concepts):
            for j in range(i + 1, len(concepts)):
                concept_j = concepts[j]
                
                # Check co-occurrence
                pair = tuple(sorted([concept_i, concept_j]))
                co_count = cooccurrence.get(pair, 0)
                
                if co_count < theta_co:
                    continue
                
                # Check semantic similarity
                vec_i = concept_vectors[concept_i]
                vec_j = concept_vectors[concept_j]
                
                # Cosine similarity
                similarity = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j))
                
                if similarity >= theta_sem:
                    # Calculate Dice coefficient for edge weight
                    # For simplicity, using co-occurrence count as weight
                    weight = co_count
                    G.add_edge(concept_i, concept_j, weight=weight)
        
        return G


class SpacyExtractorWithSentences(Extractor):
    """Extended Spacy extractor that preserves sentence information"""
    
    def __init__(self, language: str = "en"):
        super().__init__(language)
        self.nlp = self.load_model(language)
        self.method = "Spacy"
    
    def load_model(self, language):
        if language == "en":
            try:
                nlp = spacy.load("en_core_web_lg")
            except:
                logger.info("Downloading spacy model...")
                spacy.cli.download("en_core_web_lg")
                nlp = spacy.load("en_core_web_lg")
        elif language == "zh":
            try:
                nlp = spacy.load("zh_core_web_lg")
            except:
                logger.info("Downloading spacy model...")
                spacy.cli.download("zh_core_web_lg")
                nlp = spacy.load("zh_core_web_lg")
        return nlp
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]
    
    def naive_extract_graph(self, text: str):
        """Extract graph with sentence-level information"""
        doc = self.nlp(text)
        
        noun_pairs = {}
        all_nouns = set()
        double_nouns = {}
        appearance_count = {}
        
        for sent in doc.sents:
            sentence_terms = []
            ent_positions = set()
            
            for ent in sent.ents:
                if ent.label_ == "PERSON":
                    name_parts = ent.text.split()
                    if len(name_parts) >= 2:
                        for name in name_parts:
                            double_nouns[name] = name_parts
                        sentence_terms.extend(name_parts)
                        for name in name_parts:
                            appearance_count[name] = appearance_count.get(name, 0) + 1
                    else:
                        sentence_terms.append(ent.text)
                        appearance_count[ent.text] = appearance_count.get(ent.text, 0) + 1
                
                elif ent.label_ in ["ORG", "GPE"]:
                    sentence_terms.append(ent.text)
                    appearance_count[ent.text] = appearance_count.get(ent.text, 0) + 1
                
                for token in ent:
                    ent_positions.add(token.i)
            
            for token in sent:
                if token.i in ent_positions:
                    continue
                if token.pos_ == "NOUN" and token.lemma_.strip():
                    sentence_terms.append(token.lemma_.lower())
                    appearance_count[token.lemma_.lower()] = appearance_count.get(token.lemma_.lower(), 0) + 1
                elif token.pos_ == "PROPN" and token.text.strip():
                    sentence_terms.append(token.text)
                    appearance_count[token.text] = appearance_count.get(token.text, 0) + 1
            
            all_nouns.update(sentence_terms)
            
            # Count co-occurrence
            for i in range(len(sentence_terms)):
                for j in range(i + 1, len(sentence_terms)):
                    term1, term2 = sorted([sentence_terms[i], sentence_terms[j]])
                    pair = (term1, term2)
                    noun_pairs[pair] = noun_pairs.get(pair, 0) + 1
        
        return {
            "nouns": list(all_nouns),
            "cooccurrence": noun_pairs,
            "double_nouns": double_nouns,
            "appearance_count": appearance_count
        }


class DualIndexRetriever:
    """
    Dual-path retrieval combining concept graph and summary tree
    Implements the retrieval and fusion phase from the algorithm
    """
    
    def __init__(self, 
                 cache_tree: Dict,
                 G: nx.Graph,
                 index: Dict,
                 I_s_to_c: Dict,
                 I_c_to_s: Dict,
                 sentence_texts: Dict,
                 concept_vectors: Dict,
                 embedder_model: str = "BAAI/bge-m3",
                 device: str = "cuda:0"):
        """
        Initialize dual-index retriever
        
        Args:
            cache_tree: Summary tree structure
            G: Concept graph
            index: Concept to chunks index
            I_s_to_c: Sentence to chunk mapping
            I_c_to_s: Concept to sentences mapping
            sentence_texts: Sentence texts
            concept_vectors: Pre-computed concept vectors
            embedder_model: Sentence encoder model
            device: Device to run embedder
        """
        self.cache_tree = cache_tree
        self.G = G
        self.index = index
        self.I_s_to_c = I_s_to_c
        self.I_c_to_s = I_c_to_s
        self.sentence_texts = sentence_texts
        self.concept_vectors = concept_vectors
        self.embedder = SentenceTransformer(embedder_model, device=device)
        self.device = device
        
        # Build tree node embeddings for Path B
        self._build_tree_embeddings()
    
    def _build_tree_embeddings(self):
        """Build embeddings for all tree nodes (leaves + summaries)"""
        self.tree_nodes = []
        self.tree_node_ids = []
        self.tree_embeddings = None
        
        for node_id, node_data in self.cache_tree.items():
            self.tree_nodes.append(node_data["text"])
            self.tree_node_ids.append(node_id)
        
        if self.tree_nodes:
            self.tree_embeddings = self.embedder.encode(self.tree_nodes, convert_to_numpy=True)
    
    def path_a_concept_graph_retrieval(self, query: str, K_s: int = 10, 
                                      similarity_threshold: float = 0.3) -> Set[str]:
        """
        Path A: Fine-grained retrieval via concept graph
        
        Steps:
        A1: Concept matching - find relevant concepts
        A2: Candidate sentence recall - get sentences for concepts
        A3: Sentence reranking - rank and select top sentences
        A4: Map to chunks - get corresponding chunks
        
        Args:
            query: User query
            K_s: Number of top sentences to select
            similarity_threshold: Threshold for concept matching
            
        Returns:
            Set of chunk IDs (C_{cpt})
        """
        # A1: Concept matching
        query_embed = self.embedder.encode(query, convert_to_numpy=True)
        
        W_relevant = []
        for concept, concept_vec in self.concept_vectors.items():
            # Calculate cosine similarity
            similarity = np.dot(query_embed, concept_vec) / (
                np.linalg.norm(query_embed) * np.linalg.norm(concept_vec)
            )
            if similarity > similarity_threshold:
                W_relevant.append((concept, similarity))
        
        if not W_relevant:
            logger.info("Path A: No relevant concepts found")
            return set()
        
        logger.info(f"Path A: Found {len(W_relevant)} relevant concepts")
        
        # A2: Candidate sentence recall
        S_candidate = set()
        for concept, _ in W_relevant:
            if concept in self.I_c_to_s:
                S_candidate.update(self.I_c_to_s[concept])
        
        if not S_candidate:
            logger.info("Path A: No candidate sentences found")
            return set()
        
        logger.info(f"Path A: Found {len(S_candidate)} candidate sentences")
        
        # A3: Sentence reranking
        sentence_scores = []
        for sent_id in S_candidate:
            if sent_id in self.sentence_texts:
                sent_text = self.sentence_texts[sent_id]
                sent_embed = self.embedder.encode(sent_text, convert_to_numpy=True)
                
                # Calculate similarity with query
                similarity = np.dot(query_embed, sent_embed) / (
                    np.linalg.norm(query_embed) * np.linalg.norm(sent_embed)
                )
                sentence_scores.append((sent_id, similarity))
        
        # Sort by similarity and select top K_s
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        S_top = [sent_id for sent_id, _ in sentence_scores[:K_s]]
        
        logger.info(f"Path A: Selected top {len(S_top)} sentences")
        
        # A4: Map sentences back to chunks
        C_cpt = set()
        for sent_id in S_top:
            if sent_id in self.I_s_to_c:
                chunk_id = self.I_s_to_c[sent_id]
                C_cpt.add(chunk_id)
        
        logger.info(f"Path A: Mapped to {len(C_cpt)} chunks")
        return C_cpt
    
    def path_b_summary_tree_retrieval(self, query: str, K_t: int = 10) -> Set[str]:
        """
        Path B: Global retrieval via summary tree
        
        Steps:
        B1: Flatten tree - treat all nodes equally
        B2: Vector matching - compute similarity with all nodes
        B3: Top-K selection - select most similar nodes
        
        Args:
            query: User query
            K_t: Number of top tree nodes to select
            
        Returns:
            Set of chunk/node IDs (C_{tree})
        """
        if self.tree_embeddings is None or len(self.tree_embeddings) == 0:
            logger.info("Path B: No tree embeddings available")
            return set()
        
        # B1 & B2: Flatten tree and compute similarities
        query_embed = self.embedder.encode(query, convert_to_numpy=True)
        
        # Calculate similarity with all tree nodes
        similarities = []
        for i, node_embed in enumerate(self.tree_embeddings):
            similarity = np.dot(query_embed, node_embed) / (
                np.linalg.norm(query_embed) * np.linalg.norm(node_embed)
            )
            similarities.append((self.tree_node_ids[i], similarity))
        
        # B3: Select top K_t nodes
        similarities.sort(key=lambda x: x[1], reverse=True)
        C_tree = set([node_id for node_id, _ in similarities[:K_t]])
        
        logger.info(f"Path B: Selected {len(C_tree)} tree nodes")
        return C_tree
    
    def dual_path_retrieval(self, query: str, 
                           K_s: int = 10, 
                           K_t: int = 10,
                           concept_threshold: float = 0.3) -> Tuple[Set[str], Dict[str, Any]]:
        """
        Dual-path retrieval with fusion
        
        Args:
            query: User query
            K_s: Number of sentences for Path A
            K_t: Number of tree nodes for Path B
            concept_threshold: Similarity threshold for concept matching
            
        Returns:
            C_pool: Final set of chunks (union of both paths)
            metadata: Retrieval metadata for debugging
        """
        # Path A: Concept graph retrieval
        C_cpt = self.path_a_concept_graph_retrieval(query, K_s, concept_threshold)
        
        # Path B: Summary tree retrieval
        C_tree = self.path_b_summary_tree_retrieval(query, K_t)
        
        # Fusion: Union of both sets (deduplication automatic with sets)
        C_pool = C_cpt.union(C_tree)
        
        metadata = {
            "path_a_chunks": len(C_cpt),
            "path_b_chunks": len(C_tree),
            "total_chunks": len(C_pool),
            "path_a_ids": list(C_cpt),
            "path_b_ids": list(C_tree),
            "final_ids": list(C_pool)
        }
        
        logger.info(f"Dual-path retrieval: A={len(C_cpt)}, B={len(C_tree)}, Total={len(C_pool)}")
        
        return C_pool, metadata
    
    def get_chunks_text(self, chunk_ids: Set[str]) -> str:
        """
        Get text for the given chunk IDs
        
        Args:
            chunk_ids: Set of chunk IDs
            
        Returns:
            Concatenated text of all chunks
        """
        chunks_text = []
        for chunk_id in sorted(chunk_ids):
            if chunk_id in self.cache_tree:
                chunks_text.append(self.cache_tree[chunk_id]["text"])
        
        return "\n\n".join(chunks_text)


def save_dual_index(cache_folder: str, I_s_to_c: Dict, I_c_to_s: Dict, 
                   sentence_texts: Dict, concept_vectors: Dict, method: str = "Spacy"):
    """Save dual index structures to cache"""
    
    # Convert numpy arrays to lists for JSON serialization
    concept_vectors_serializable = {
        k: v.tolist() for k, v in concept_vectors.items()
    }
    
    with open(os.path.join(cache_folder, f"I_s_to_c_{method}.json"), "w") as f:
        json.dump(I_s_to_c, f, indent=2)
    
    with open(os.path.join(cache_folder, f"I_c_to_s_{method}.json"), "w") as f:
        json.dump(I_c_to_s, f, indent=2)
    
    with open(os.path.join(cache_folder, f"sentence_texts_{method}.json"), "w") as f:
        json.dump(sentence_texts, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(cache_folder, f"concept_vectors_{method}.json"), "w") as f:
        json.dump(concept_vectors_serializable, f, indent=2)
    
    logger.info(f"Saved dual index to {cache_folder}")


def load_dual_index(cache_folder: str, method: str = "Spacy") -> Tuple[Dict, Dict, Dict, Dict]:
    """Load dual index structures from cache"""
    
    with open(os.path.join(cache_folder, f"I_s_to_c_{method}.json"), "r") as f:
        I_s_to_c = json.load(f)
    
    with open(os.path.join(cache_folder, f"I_c_to_s_{method}.json"), "r") as f:
        I_c_to_s = json.load(f)
    
    with open(os.path.join(cache_folder, f"sentence_texts_{method}.json"), "r") as f:
        sentence_texts = json.load(f)
    
    with open(os.path.join(cache_folder, f"concept_vectors_{method}.json"), "r") as f:
        concept_vectors_lists = json.load(f)
        # Convert lists back to numpy arrays
        concept_vectors = {
            k: np.array(v) for k, v in concept_vectors_lists.items()
        }
    
    logger.info(f"Loaded dual index from {cache_folder}")
    return I_s_to_c, I_c_to_s, sentence_texts, concept_vectors
