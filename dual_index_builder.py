"""
Dual-Index GraphRAG Builder
Implements the indexing phase for Dual-Index GraphRAG system with:
1. Concept Graph (fine-grained) - TF-IDF based concept extraction
2. Summary Tree (global) - hierarchical summarization
3. Inverted indices for efficient retrieval
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


class DualIndexBuilder:
    """
    Builder for dual-index GraphRAG system
    Constructs concept graph and prepares sentence-level indexing
    """
    
    def __init__(self, 
                 embedder_model: str = "BAAI/bge-m3",
                 device: str = "cuda:0",
                 language: str = "en",
                 min_tfidf_score: float = 0.1,
                 semantic_threshold: float = 0.7,
                 cooccurrence_threshold: int = 2):
        """
        Initialize the dual-index builder
        
        Args:
            embedder_model: Sentence encoder model for concept vectorization
            device: Device to run the embedder
            language: Language for text processing
            min_tfidf_score: Minimum TF-IDF score for concept extraction
            semantic_threshold: Semantic similarity threshold for edge construction
            cooccurrence_threshold: Co-occurrence threshold for edge construction
        """
        self.embedder = SentenceTransformer(embedder_model, device=device)
        self.device = device
        self.language = language
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
            # For Chinese or other languages
            self.nlp = None
    
    def extract_sentences(self, chunks: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """
        Extract sentences from chunks and create sentence-to-chunk mapping
        
        Args:
            chunks: List of text chunks
            
        Returns:
            sentences: List of all sentences
            sentence_to_chunk: Mapping from sentence to chunk ID
        """
        sentences = []
        sentence_to_chunk = {}
        
        for chunk_id, chunk_text in enumerate(chunks):
            chunk_key = f"leaf_{chunk_id}"
            
            if self.language == "en" and self.nlp:
                doc = self.nlp(chunk_text)
                chunk_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            else:
                # Fallback to NLTK for sentence splitting
                chunk_sentences = nltk.sent_tokenize(chunk_text)
            
            for sent in chunk_sentences:
                if len(sent.split()) > 3:  # Filter out very short sentences
                    sent_id = f"sent_{len(sentences)}"
                    sentences.append(sent)
                    sentence_to_chunk[sent_id] = chunk_key
        
        logger.info(f"Extracted {len(sentences)} sentences from {len(chunks)} chunks")
        return sentences, sentence_to_chunk
    
    def extract_concepts_tfidf(self, chunks: List[str], top_k: int = 10) -> Dict[str, Set[str]]:
        """
        Extract key concepts from each chunk using TF-IDF
        
        Args:
            chunks: List of text chunks
            top_k: Number of top concepts to extract per chunk
            
        Returns:
            chunk_concepts: Mapping from chunk ID to set of concepts
        """
        logger.info("Extracting concepts using TF-IDF...")
        
        # Use TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english' if self.language == 'en' else None,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=1,
            max_df=0.8
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(chunks)
            feature_names = vectorizer.get_feature_names_out()
            
            chunk_concepts = {}
            for chunk_id in range(len(chunks)):
                chunk_key = f"leaf_{chunk_id}"
                
                # Get TF-IDF scores for this chunk
                chunk_tfidf = tfidf_matrix[chunk_id].toarray()[0]
                
                # Get top-k concepts
                top_indices = chunk_tfidf.argsort()[-top_k:][::-1]
                concepts = set()
                
                for idx in top_indices:
                    if chunk_tfidf[idx] >= self.min_tfidf_score:
                        concept = feature_names[idx]
                        concepts.add(concept)
                
                chunk_concepts[chunk_key] = concepts
                
            logger.info(f"Extracted concepts for {len(chunk_concepts)} chunks")
            return chunk_concepts
            
        except Exception as e:
            logger.error(f"Error in TF-IDF extraction: {e}")
            # Fallback: use simple word frequency
            return self._fallback_concept_extraction(chunks, top_k)
    
    def _fallback_concept_extraction(self, chunks: List[str], top_k: int = 10) -> Dict[str, Set[str]]:
        """Fallback concept extraction using word frequency"""
        logger.warning("Using fallback concept extraction")
        chunk_concepts = {}
        
        for chunk_id, chunk_text in enumerate(chunks):
            chunk_key = f"leaf_{chunk_id}"
            words = chunk_text.lower().split()
            # Simple frequency counting
            word_freq = Counter(words)
            # Get top-k most frequent words
            top_words = [word for word, _ in word_freq.most_common(top_k) if len(word) > 3]
            chunk_concepts[chunk_key] = set(top_words)
        
        return chunk_concepts
    
    def build_concept_vectors(self, sentences: List[str], sentence_to_chunk: Dict[str, str],
                             chunk_concepts: Dict[str, Set[str]]) -> Dict[str, np.ndarray]:
        """
        Build concept vectors by averaging sentence embeddings where concept appears
        
        Args:
            sentences: List of sentences
            sentence_to_chunk: Mapping from sentence ID to chunk ID
            chunk_concepts: Mapping from chunk ID to concepts
            
        Returns:
            concept_vectors: Mapping from concept to its vector representation
        """
        logger.info("Building concept vectors...")
        
        # Create mapping: concept -> list of sentences containing it
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
        
        # Encode all sentences
        logger.info("Encoding sentences for concept vectors...")
        sentence_embeddings = self.embedder.encode(sentences, batch_size=32, 
                                                   show_progress_bar=True,
                                                   device=self.device)
        
        # Build concept vectors by averaging sentence embeddings
        # Create sentence to index mapping for O(1) lookup
        sentence_to_idx = {sent: idx for idx, sent in enumerate(sentences)}
        
        concept_vectors = {}
        for concept, concept_sentences in concept_to_sentences.items():
            if concept_sentences:
                # Find indices of these sentences
                sent_indices = []
                for sent_text in concept_sentences:
                    sent_idx = sentence_to_idx.get(sent_text)
                    if sent_idx is not None:
                        sent_indices.append(sent_idx)
                
                if sent_indices:
                    # Average the embeddings
                    concept_vector = np.mean(sentence_embeddings[sent_indices], axis=0)
                    concept_vectors[concept] = concept_vector
        
        logger.info(f"Built vectors for {len(concept_vectors)} concepts")
        return concept_vectors
    
    def build_concept_graph(self, chunk_concepts: Dict[str, Set[str]], 
                           concept_vectors: Dict[str, np.ndarray],
                           sentences: List[str],
                           sentence_to_chunk: Dict[str, str]) -> Tuple[nx.Graph, Dict[str, List[str]], Dict[str, int]]:
        """
        Build concept graph with edges based on semantic similarity and co-occurrence
        
        Args:
            chunk_concepts: Mapping from chunk ID to concepts
            concept_vectors: Concept vectors
            sentences: List of sentences
            sentence_to_chunk: Sentence to chunk mapping
            
        Returns:
            G: NetworkX graph
            concept_to_sentences: Inverted index from concept to sentences
            appearance_count: Concept appearance count
        """
        logger.info("Building concept graph...")
        
        # Build inverted index: concept -> sentences
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
        
        # Calculate co-occurrence between concepts
        cooccurrence = defaultdict(int)
        
        for sent_idx, sent_text in enumerate(sentences):
            sent_id = f"sent_{sent_idx}"
            chunk_id = sentence_to_chunk[sent_id]
            
            if chunk_id in chunk_concepts:
                sent_lower = sent_text.lower()
                # Find which concepts appear in this sentence
                sent_concepts = [c for c in chunk_concepts[chunk_id] if c.lower() in sent_lower]
                
                # Count co-occurrence
                for c1, c2 in combinations(sent_concepts, 2):
                    pair = tuple(sorted([c1, c2]))
                    cooccurrence[pair] += 1
        
        # Build graph edges
        G = nx.Graph()
        edges = []
        
        for (concept1, concept2), co_count in cooccurrence.items():
            # Check co-occurrence threshold
            if co_count < self.cooccurrence_threshold:
                continue
            
            # Check semantic similarity threshold
            if concept1 in concept_vectors and concept2 in concept_vectors:
                vec1 = concept_vectors[concept1]
                vec2 = concept_vectors[concept2]
                
                # Cosine similarity
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                
                if similarity >= self.semantic_threshold:
                    # Calculate Dice coefficient
                    count1 = appearance_count[concept1]
                    count2 = appearance_count[concept2]
                    dice_weight = (2.0 * co_count) / (count1 + count2)
                    
                    edges.append((concept1, concept2, dice_weight))
                    G.add_edge(concept1, concept2, weight=dice_weight)
        
        logger.info(f"Built concept graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G, dict(concept_to_sentences), dict(appearance_count)
    
    def build_dual_index(self, chunks: List[str], cache_folder: str, 
                        use_cache: bool = True) -> Tuple[nx.Graph, Dict, Dict, Dict, List[str]]:
        """
        Build the complete dual-index structure
        
        Args:
            chunks: List of text chunks
            cache_folder: Folder to save/load cache
            use_cache: Whether to use cached results
            
        Returns:
            G: Concept graph
            concept_to_sentences: Inverted index (concept -> sentences)
            sentence_to_chunk: Sentence to chunk mapping
            concept_vectors: Concept vectors for retrieval
            sentences: List of all sentences
        """
        cache_file = os.path.join(cache_folder, "dual_index.json")
        graph_file = os.path.join(cache_folder, "dual_index_graph.json")
        vectors_file = os.path.join(cache_folder, "dual_index_vectors.npy")
        
        if use_cache and os.path.exists(cache_file) and os.path.exists(graph_file):
            logger.info("Loading dual-index from cache...")
            return self._load_cache(cache_file, graph_file, vectors_file)
        
        # Step 1: Extract sentences and create mapping
        sentences, sentence_to_chunk = self.extract_sentences(chunks)
        
        # Step 2: Extract concepts using TF-IDF
        chunk_concepts = self.extract_concepts_tfidf(chunks)
        
        # Step 3: Build concept vectors
        concept_vectors = self.build_concept_vectors(sentences, sentence_to_chunk, chunk_concepts)
        
        # Step 4: Build concept graph
        G, concept_to_sentences, appearance_count = self.build_concept_graph(
            chunk_concepts, concept_vectors, sentences, sentence_to_chunk
        )
        
        # Save to cache
        self._save_cache(cache_file, graph_file, vectors_file,
                        G, concept_to_sentences, sentence_to_chunk, 
                        concept_vectors, appearance_count, sentences)
        
        return G, concept_to_sentences, sentence_to_chunk, concept_vectors, sentences
    
    def _save_cache(self, cache_file: str, graph_file: str, vectors_file: str,
                   G: nx.Graph, concept_to_sentences: Dict, sentence_to_chunk: Dict,
                   concept_vectors: Dict, appearance_count: Dict, sentences: List[str]):
        """Save dual-index to cache"""
        logger.info("Saving dual-index to cache...")
        
        # Save graph edges
        edges = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
        with open(graph_file, 'w') as f:
            json.dump(edges, f, indent=2)
        
        # Save indices and mappings
        cache_data = {
            'concept_to_sentences': concept_to_sentences,
            'sentence_to_chunk': sentence_to_chunk,
            'appearance_count': appearance_count,
            'sentences': sentences,
            'concept_list': list(concept_vectors.keys())
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        # Save concept vectors
        if concept_vectors:
            vector_array = np.array([concept_vectors[c] for c in cache_data['concept_list']])
            np.save(vectors_file, vector_array)
    
    def _load_cache(self, cache_file: str, graph_file: str, vectors_file: str):
        """Load dual-index from cache"""
        logger.info("Loading from cache...")
        
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        with open(graph_file, 'r') as f:
            edges = json.load(f)
        
        # Rebuild graph
        G = nx.Graph()
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)
        
        # Load concept vectors
        concept_vectors = {}
        if os.path.exists(vectors_file):
            vector_array = np.load(vectors_file)
            for i, concept in enumerate(cache_data['concept_list']):
                concept_vectors[concept] = vector_array[i]
        
        return (G, 
                cache_data['concept_to_sentences'],
                cache_data['sentence_to_chunk'],
                concept_vectors,
                cache_data['sentences'])


def build_dual_index(chunks: List[str], cache_folder: str, 
                    embedder_model: str = "BAAI/bge-m3",
                    device: str = "cuda:0",
                    language: str = "en",
                    use_cache: bool = True) -> Tuple:
    """
    Convenience function to build dual-index
    """
    builder = DualIndexBuilder(
        embedder_model=embedder_model,
        device=device,
        language=language
    )
    
    return builder.build_dual_index(chunks, cache_folder, use_cache)
