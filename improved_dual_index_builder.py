"""
Improved Dual-Index GraphRAG Builder
Combines NER entities with TF-IDF concepts for better retrieval quality
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import networkx as nx
import spacy
import nltk
from itertools import combinations

logger = logging.getLogger(__name__)


class ImprovedDualIndexBuilder:
    """
    Improved builder that combines:
    1. NER entities (precise, important concepts)
    2. TF-IDF keywords (broader coverage)
    3. Graph relationships from original system
    """
    
    def __init__(self, 
                 embedder_model: str = "BAAI/bge-m3",
                 device: str = "cuda:0",
                 language: str = "en",
                 nlp_extractor = None,
                 use_ner: bool = True,
                 use_tfidf: bool = True,
                 min_tfidf_score: float = 0.15,
                 semantic_threshold: float = 0.65,
                 cooccurrence_threshold: int = 2):
        """
        Initialize the improved dual-index builder
        
        Args:
            embedder_model: Sentence encoder model
            device: Device to run the embedder
            language: Language for text processing
            nlp_extractor: Existing NLP extractor (SpacyExtractor or NLTKExtractor)
            use_ner: Whether to use NER entities
            use_tfidf: Whether to use TF-IDF keywords
            min_tfidf_score: Minimum TF-IDF score (increased from 0.1)
            semantic_threshold: Semantic similarity threshold (decreased from 0.7)
            cooccurrence_threshold: Co-occurrence threshold
        """
        self.embedder = SentenceTransformer(embedder_model, device=device)
        self.device = device
        self.language = language
        self.nlp_extractor = nlp_extractor
        self.use_ner = use_ner
        self.use_tfidf = use_tfidf
        self.min_tfidf_score = min_tfidf_score
        self.semantic_threshold = semantic_threshold
        self.cooccurrence_threshold = cooccurrence_threshold
        
        # Initialize NLP tools for sentence extraction
        if language == "en":
            try:
                self.nlp = spacy.load("en_core_web_lg")
            except (OSError, IOError):
                logger.info("Downloading spacy model...")
                spacy.cli.download("en_core_web_lg")
                self.nlp = spacy.load("en_core_web_lg")
        else:
            self.nlp = None
    
    def extract_sentences(self, chunks: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """Extract sentences from chunks with mapping to chunks"""
        sentences = []
        sentence_to_chunk = {}
        
        for chunk_id, chunk_text in enumerate(chunks):
            chunk_key = f"leaf_{chunk_id}"
            
            if self.language == "en" and self.nlp:
                doc = self.nlp(chunk_text)
                chunk_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            else:
                chunk_sentences = nltk.sent_tokenize(chunk_text)
            
            for sent in chunk_sentences:
                if len(sent.split()) > 3:
                    sent_id = f"sent_{len(sentences)}"
                    sentences.append(sent)
                    sentence_to_chunk[sent_id] = chunk_key
        
        logger.info(f"Extracted {len(sentences)} sentences from {len(chunks)} chunks")
        return sentences, sentence_to_chunk
    
    def extract_hybrid_concepts(self, chunks: List[str], top_k_tfidf: int = 15) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
        """
        Extract concepts using hybrid approach: NER entities + TF-IDF keywords
        
        Returns:
            chunk_concepts: Mapping from chunk ID to set of concepts
            concept_importance: Importance score for each concept
        """
        logger.info("Extracting concepts using hybrid NER + TF-IDF approach...")
        
        chunk_concepts = {}
        concept_importance = defaultdict(float)
        
        # Step 1: Extract NER entities if enabled
        if self.use_ner and self.nlp_extractor:
            logger.info("Extracting NER entities...")
            for chunk_id, chunk_text in enumerate(chunks):
                chunk_key = f"leaf_{chunk_id}"
                
                # Use the original NER extraction
                ner_result = self.nlp_extractor.naive_extract_graph(chunk_text)
                entities = set(ner_result["nouns"])
                
                chunk_concepts[chunk_key] = entities
                
                # Higher importance for entities
                for entity in entities:
                    concept_importance[entity] += 2.0
        
        # Step 2: Add TF-IDF keywords if enabled
        if self.use_tfidf:
            logger.info("Extracting TF-IDF keywords...")
            
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english' if self.language == 'en' else None,
                ngram_range=(1, 2),
                min_df=2,  # Require terms to appear in at least 2 chunks
                max_df=0.7  # Ignore very common terms
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(chunks)
                feature_names = vectorizer.get_feature_names_out()
                
                for chunk_id in range(len(chunks)):
                    chunk_key = f"leaf_{chunk_id}"
                    
                    # Get TF-IDF scores
                    chunk_tfidf = tfidf_matrix[chunk_id].toarray()[0]
                    
                    # Get top-k keywords
                    top_indices = chunk_tfidf.argsort()[-top_k_tfidf:][::-1]
                    
                    keywords = set()
                    for idx in top_indices:
                        if chunk_tfidf[idx] >= self.min_tfidf_score:
                            keyword = feature_names[idx]
                            keywords.add(keyword)
                            # Lower importance for TF-IDF keywords
                            concept_importance[keyword] += chunk_tfidf[idx]
                    
                    # Merge with NER entities
                    if chunk_key in chunk_concepts:
                        chunk_concepts[chunk_key].update(keywords)
                    else:
                        chunk_concepts[chunk_key] = keywords
                        
            except Exception as e:
                logger.error(f"TF-IDF extraction error: {e}")
        
        total_concepts = sum(len(concepts) for concepts in chunk_concepts.values())
        logger.info(f"Extracted {total_concepts} total concepts ({len(concept_importance)} unique)")
        
        return chunk_concepts, dict(concept_importance)
    
    def build_concept_vectors(self, sentences: List[str], sentence_to_chunk: Dict[str, str],
                             chunk_concepts: Dict[str, Set[str]]) -> Dict[str, np.ndarray]:
        """Build concept vectors by averaging sentence embeddings"""
        logger.info("Building concept vectors...")
        
        # Map concept to sentences
        concept_to_sentences = defaultdict(list)
        
        for sent_idx, sent_text in enumerate(sentences):
            sent_id = f"sent_{sent_idx}"
            chunk_id = sentence_to_chunk[sent_id]
            
            if chunk_id in chunk_concepts:
                sent_lower = sent_text.lower()
                for concept in chunk_concepts[chunk_id]:
                    # Check if concept appears in sentence
                    if concept.lower() in sent_lower:
                        concept_to_sentences[concept].append(sent_text)
        
        # Encode sentences
        logger.info("Encoding sentences for concept vectors...")
        sentence_embeddings = self.embedder.encode(sentences, batch_size=32, 
                                                   show_progress_bar=True,
                                                   device=self.device)
        
        # Build concept vectors
        sentence_to_idx = {sent: idx for idx, sent in enumerate(sentences)}
        concept_vectors = {}
        
        for concept, concept_sentences in concept_to_sentences.items():
            if concept_sentences:
                sent_indices = []
                for sent_text in concept_sentences:
                    sent_idx = sentence_to_idx.get(sent_text)
                    if sent_idx is not None:
                        sent_indices.append(sent_idx)
                
                if sent_indices:
                    concept_vector = np.mean(sentence_embeddings[sent_indices], axis=0)
                    concept_vectors[concept] = concept_vector
        
        logger.info(f"Built vectors for {len(concept_vectors)} concepts")
        return concept_vectors
    
    def build_concept_graph(self, chunk_concepts: Dict[str, Set[str]], 
                           concept_vectors: Dict[str, np.ndarray],
                           concept_importance: Dict[str, float],
                           sentences: List[str],
                           sentence_to_chunk: Dict[str, str]) -> Tuple[nx.Graph, Dict[str, List[str]], Dict[str, int]]:
        """Build concept graph with improved edge weighting"""
        logger.info("Building concept graph...")
        
        # Build inverted index
        concept_to_sentences = defaultdict(list)
        appearance_count = defaultdict(int)
        
        for sent_idx, sent_text in enumerate(sentences):
            sent_id = f"sent_{sent_idx}"
            chunk_id = sentence_to_chunk[sent_id]
            
            if chunk_id in chunk_concepts:
                sent_lower = sent_text.lower()
                for concept in chunk_concepts[chunk_id]:
                    if concept.lower() in sent_lower:
                        concept_to_sentences[concept].append(sent_id)
                        appearance_count[concept] += 1
        
        # Calculate co-occurrence
        cooccurrence = defaultdict(int)
        
        for sent_idx, sent_text in enumerate(sentences):
            sent_id = f"sent_{sent_idx}"
            chunk_id = sentence_to_chunk[sent_id]
            
            if chunk_id in chunk_concepts:
                sent_lower = sent_text.lower()
                sent_concepts = [c for c in chunk_concepts[chunk_id] if c.lower() in sent_lower]
                
                for c1, c2 in combinations(sent_concepts, 2):
                    pair = tuple(sorted([c1, c2]))
                    cooccurrence[pair] += 1
        
        # Build graph with improved weighting
        G = nx.Graph()
        
        for (concept1, concept2), co_count in cooccurrence.items():
            if co_count < self.cooccurrence_threshold:
                continue
            
            # Check semantic similarity if vectors available
            if concept1 in concept_vectors and concept2 in concept_vectors:
                vec1 = concept_vectors[concept1]
                vec2 = concept_vectors[concept2]
                
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                
                if similarity >= self.semantic_threshold:
                    # Improved edge weight combining Dice coefficient and importance
                    count1 = appearance_count[concept1]
                    count2 = appearance_count[concept2]
                    dice_weight = (2.0 * co_count) / (count1 + count2)
                    
                    # Factor in concept importance
                    importance_factor = (concept_importance.get(concept1, 1.0) + 
                                       concept_importance.get(concept2, 1.0)) / 2.0
                    
                    final_weight = dice_weight * (1.0 + 0.2 * importance_factor)
                    
                    G.add_edge(concept1, concept2, weight=final_weight)
        
        logger.info(f"Built concept graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G, dict(concept_to_sentences), dict(appearance_count)
    
    def build_dual_index(self, chunks: List[str], cache_folder: str, 
                        use_cache: bool = True) -> Tuple[nx.Graph, Dict, Dict, Dict, List[str], Dict]:
        """Build the improved dual-index structure"""
        cache_file = os.path.join(cache_folder, "improved_dual_index.json")
        graph_file = os.path.join(cache_folder, "improved_dual_index_graph.json")
        vectors_file = os.path.join(cache_folder, "improved_dual_index_vectors.npy")
        
        if use_cache and os.path.exists(cache_file) and os.path.exists(graph_file):
            logger.info("Loading improved dual-index from cache...")
            return self._load_cache(cache_file, graph_file, vectors_file)
        
        # Extract sentences
        sentences, sentence_to_chunk = self.extract_sentences(chunks)
        
        # Extract hybrid concepts
        chunk_concepts, concept_importance = self.extract_hybrid_concepts(chunks)
        
        # Build concept vectors
        concept_vectors = self.build_concept_vectors(sentences, sentence_to_chunk, chunk_concepts)
        
        # Build concept graph
        G, concept_to_sentences, appearance_count = self.build_concept_graph(
            chunk_concepts, concept_vectors, concept_importance, sentences, sentence_to_chunk
        )
        
        # Save to cache
        self._save_cache(cache_file, graph_file, vectors_file,
                        G, concept_to_sentences, sentence_to_chunk, 
                        concept_vectors, appearance_count, sentences,
                        concept_importance)
        
        return G, concept_to_sentences, sentence_to_chunk, concept_vectors, sentences, concept_importance
    
    def _save_cache(self, cache_file: str, graph_file: str, vectors_file: str,
                   G: nx.Graph, concept_to_sentences: Dict, sentence_to_chunk: Dict,
                   concept_vectors: Dict, appearance_count: Dict, sentences: List[str],
                   concept_importance: Dict):
        """Save improved dual-index to cache"""
        logger.info("Saving improved dual-index to cache...")
        
        edges = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
        with open(graph_file, 'w') as f:
            json.dump(edges, f, indent=2)
        
        cache_data = {
            'concept_to_sentences': concept_to_sentences,
            'sentence_to_chunk': sentence_to_chunk,
            'appearance_count': appearance_count,
            'sentences': sentences,
            'concept_list': list(concept_vectors.keys()),
            'concept_importance': concept_importance
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        if concept_vectors:
            vector_array = np.array([concept_vectors[c] for c in cache_data['concept_list']])
            np.save(vectors_file, vector_array)
    
    def _load_cache(self, cache_file: str, graph_file: str, vectors_file: str):
        """Load improved dual-index from cache"""
        logger.info("Loading from cache...")
        
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        with open(graph_file, 'r') as f:
            edges = json.load(f)
        
        G = nx.Graph()
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)
        
        concept_vectors = {}
        if os.path.exists(vectors_file):
            vector_array = np.load(vectors_file)
            for i, concept in enumerate(cache_data['concept_list']):
                concept_vectors[concept] = vector_array[i]
        
        return (G, 
                cache_data['concept_to_sentences'],
                cache_data['sentence_to_chunk'],
                concept_vectors,
                cache_data['sentences'],
                cache_data.get('concept_importance', {}))
