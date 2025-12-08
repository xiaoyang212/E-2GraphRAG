import os
from typing import List
import json
import spacy, nltk
import networkx as nx
from itertools import combinations
from typing import List, Tuple, Literal
import time
import logging
import threading

# Get logger for this module
logger = logging.getLogger(__name__)

def load_nlp(language:str="en", method: Literal["Spacy", "NLTK"]="Spacy"):
    if method == "Spacy":
        nlp = SpacyExtractor(language)
    elif method == "NLTK":
        nlp = NLTKExtractor(language)
    return nlp
        
class Extractor:
    def __init__(self, language):
        self.language = language
        self.nlp = self.load_model(language)
        self.method = "Extractor"
    
    def load_model(self):
        raise NotImplementedError("Subclass must implement the load_model method.")

    def __call__(self, text:str):
        raise NotImplementedError("Subclass must implement __call__ method")
    
    def naive_extract_graph(self, text:str):
        raise NotImplementedError("Subclass must implement the naive_extract_graph method.")

class SpacyExtractor(Extractor):
    def __init__(self, language:str="en"):
        super().__init__(language)
        self.nlp = self.load_model(language)
        self.method = "Spacy"
    
    def load_model(self, language):
        if language == "en":
            try:
                nlp = spacy.load("en_core_web_lg")
            except (OSError, ImportError):
                logger.info("Downloading spacy model...")
                spacy.cli.download("en_core_web_lg")
                nlp = spacy.load("en_core_web_lg")
        elif language == "zh":
            try:
                nlp = spacy.load("zh_core_web_lg")
            except (OSError, ImportError):
                logger.info("Downloading spacy model...")
                spacy.cli.download("zh_core_web_lg")
                nlp = spacy.load("zh_core_web_lg")
        return nlp
    
    def naive_extract_graph(self, text: str):
        doc = self.nlp(text)

        # noun pairs provide the edge.
        noun_pairs = {}

        # all_nouns saving the nodes.
        all_nouns = set()

        # process the name like John Brown
        double_nouns = {}
        appearance_count = {}

        # TODO: 一个chunk里的连通还是一个句子里连通？
        for sent in doc.sents:
            sentence_terms = []

            ent_positions = set()
            for ent in sent.ents:
                if ent.label_ == "PERSON":
                    # handle the name like John Brown, John Brown Smith.
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
                
                # process the organization or country.
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
                    sentence_terms.append(token.lemma_.lower())
                    appearance_count[token.lemma_.lower()] = appearance_count.get(token.lemma_.lower(), 0) + 1
                elif token.pos_ == "PROPN" and token.text.strip():
                    sentence_terms.append(token.text)
                    appearance_count[token.text] = appearance_count.get(token.text, 0) + 1
                    
            all_nouns.update(sentence_terms)
            
            # Count the cooccurrence of terms
            for i in range(len(sentence_terms)):
                for j in range(i+1, len(sentence_terms)):
                    term1, term2 = sorted([sentence_terms[i], sentence_terms[j]])
                    pair = (term1, term2)
                    noun_pairs[pair] = noun_pairs.get(pair, 0) + 1
        
        return {
            "nouns": list(all_nouns),
            "cooccurrence": noun_pairs,
            "double_nouns": double_nouns,
            "appearance_count": appearance_count
        }
    
class NLTKExtractor(Extractor):
    _nltk_initialized = False
    _nltk_init_lock = threading.Lock()
    def __init__(self, language:str="en"):
        super().__init__(language)
        self.nlp = self.load_model(language)
        self.method = "NLTK"

    def load_model(self, language):
        """
        The core logic that performs the one-time, thread-safe initialization.
        This method contains your original code, adapted for this pattern.
        """
        # 1. Fast, lock-free check. If already initialized, do nothing.
        if NLTKExtractor._nltk_initialized:
            return
        # 2. If not initialized, acquire lock to prevent race conditions
        with NLTKExtractor._nltk_init_lock:
            # 3. Double-check after acquiring the lock, in case another thread finished
            #    while this one was waiting.
            if NLTKExtractor._nltk_initialized:
                return
            
            logger.info("="*10)
            logger.info("First-time setup: Running thread-safe NLTK initialization...")
            logger.info("="*10)

            data_dir = "/root/nltk_data"
            if not os.path.exists(data_dir):
                logger.info(f"NLTK dir does not exist, now creating.")
                os.makedirs(data_dir)
            if data_dir not in nltk.data.path:
                logger.info(f"Adding '{data_dir}' to NLTK search path.")
                nltk.data.path.append(data_dir)
            else:
                logger.info(f"NLTK data directory '{data_dir}' is already in the search path.")
            required_packages = {
                'tokenizers/punkt': 'punkt',
                'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger',
                'chunkers/maxent_ne_chunker': 'maxent_ne_chunker',
                'corpora/words': 'words',
                'taggers/averaged_perceptron_tagger_eng': 'averaged_perceptron_tagger_eng',
                'chunkers/maxent_ne_chunker_tab': 'maxent_ne_chunker_tab'
            }
            all_packages_available = True

            for resource_path, package_id in required_packages.items():
                try:
                    nltk.data.find(resource_path)
                except LookupError:
                    all_packages_available = False
                    logger.info(f"Package {package_id} is missing, now downloading...")
                    nltk.download(package_id, download_dir=data_dir)
                    logger.info(f"Package {package_id} downloaded.")

            if all_packages_available:
                logger.info("All required NLTK packages are ready.")
            else:
                logger.info("Some packages are missing, now downloading...")
            NLTKExtractor._nltk_initialized = True
            return None

    def naive_extract_graph(self, text: str):
        sentences = nltk.tokenize.sent_tokenize(text)

        # noun pairs provide the edge.
        noun_pairs = {}

        # all_nouns saving the nodes.
        all_nouns = set()

        # process the name like John Brown
        double_nouns = {}
        appearance_count = {}

        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            tagged_tokens = nltk.pos_tag(tokens)
            
            # Extract named entities using NLTK's NER
            ne_tree = nltk.ne_chunk(tagged_tokens)
            
            sentence_terms = []
            ent_positions = set()
            
            # Process named entities
            for chunk in ne_tree:
                if hasattr(chunk, 'label'):
                    if chunk.label() == 'PERSON':
                        # handle the name like John Brown, John Brown Smith.
                        name_parts = [word for word, pos in chunk.leaves()]
                        if len(name_parts) >= 2:
                            for name in name_parts:
                                double_nouns[name] = name_parts
                            sentence_terms.extend(name_parts)
                            for name in name_parts:
                                appearance_count[name] = appearance_count.get(name, 0) + 1
                        else:
                            sentence_terms.append(' '.join(name_parts))
                            appearance_count[' '.join(name_parts)] = appearance_count.get(' '.join(name_parts), 0) + 1
                    
                    # process the organization or country.
                    elif chunk.label() in ["ORGANIZATION", "GPE"]:
                        entity_text = ' '.join([word for word, pos in chunk.leaves()])
                        sentence_terms.append(entity_text)
                        appearance_count[entity_text] = appearance_count.get(entity_text, 0) + 1
                    
                    # Mark entity positions to avoid double counting
                    for word, pos in chunk.leaves():
                        ent_positions.add(word)
            
            # Process regular nouns and proper nouns
            for word, pos in tagged_tokens:
                if word in ent_positions:
                    continue
                if pos.startswith('NN') and word.strip():
                    # Convert to lowercase for common nouns, keep proper nouns as is
                    if pos == 'NN' or pos == 'NNS':
                        sentence_terms.append(word.lower())
                        appearance_count[word.lower()] = appearance_count.get(word.lower(), 0) + 1
                    elif pos == 'NNP' or pos == 'NNPS':
                        sentence_terms.append(word)
                        appearance_count[word] = appearance_count.get(word, 0) + 1
            
            all_nouns.update(sentence_terms)
            
            # Count the cooccurrence of terms
            for i in range(len(sentence_terms)):
                for j in range(i+1, len(sentence_terms)):
                    term1, term2 = sorted([sentence_terms[i], sentence_terms[j]])
                    pair = (term1, term2)
                    noun_pairs[pair] = noun_pairs.get(pair, 0) + 1
        
        return {
            "nouns": list(all_nouns),
            "cooccurrence": noun_pairs,
            "double_nouns": double_nouns,
            "appearance_count": appearance_count
        }

def build_graph(triplets: List[Tuple[str, str, int]]) -> nx.Graph:
    '''
    build the graph from the triplets, merging weights of duplicate edges
    Args:
        triplets: List of [node1, node2, weight] List
    Returns:
        NetworkX graph with merged weights
    '''
    G = nx.Graph()
    
    # 创建字典来存储边的权重和
    edge_weights = {}
    for n1, n2, weight in triplets:
        # 因为是无向图，所以(a,b)和(b,a)是相同的边
        edge = tuple(sorted([n1, n2]))
        edge_weights[edge] = edge_weights.get(edge, 0) + weight
    
    # 将合并后的边添加到图中
    for (n1, n2), weight in edge_weights.items():
        G.add_edge(n1, n2, weight=weight)
    
    return G

def load_cache(cache_path:str):
    graph_file_path = os.path.join(cache_path, "graph.json")
    index_file_path = os.path.join(cache_path, "index.json")
    appearance_count_file_path = os.path.join(cache_path, "appearance_count.json")
    edges = json.load(open(graph_file_path, "r"))
    index = json.load(open(index_file_path, "r"))
    appearance_count = json.load(open(appearance_count_file_path, "r"))
    graph = build_graph(edges)
    return graph, index, appearance_count

def save_graph(result, cache_path:str):
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=4)

def save_index(result, cache_path:str):
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=4)

def save_appearance_count(result, cache_path:str):
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=4)
    
def extract_graph_with_dual_index(text:List[str], cache_folder:str, nlp:Extractor = None, 
                                  use_cache=True, reextract=False, build_dual_index=True,
                                  embedder_model="BAAI/bge-m3", device="cuda:0",
                                  concept_extraction_method="rake"):
    """
    Extract graph with optional dual-index building
    
    Args:
        text: List of text chunks
        cache_folder: Folder to cache results
        nlp: NLP extractor (optional, required only if concept_extraction_method="nlp")
        use_cache: Whether to use cached results
        reextract: Force re-extraction
        build_dual_index: Whether to build dual index (sentence-level indices)
        embedder_model: Model for sentence embeddings
        device: Device to run embedder
        concept_extraction_method: Method for concept extraction ("rake", "tfidf", or "nlp")
        
    Returns:
        Tuple of (graph, index, appearance_count, dual_index_data)
    """
    from dual_index_graphrag import DualIndexBuilder, save_dual_index, load_dual_index
    
    extract_start_time = time.time()
    
    # Determine cache suffix based on method
    method_suffix = concept_extraction_method if build_dual_index else (nlp.method if nlp else "default")
    
    # Check if dual index cache exists
    dual_index_cache_exists = all([
        os.path.exists(os.path.join(cache_folder, f"I_s_to_c_{method_suffix}.json")),
        os.path.exists(os.path.join(cache_folder, f"I_c_to_s_{method_suffix}.json")),
        os.path.exists(os.path.join(cache_folder, f"sentence_texts_{method_suffix}.json")),
        os.path.exists(os.path.join(cache_folder, f"concept_vectors_{method_suffix}.json"))
    ])
    
    # Load from cache if available
    if use_cache and os.path.exists(os.path.join(cache_folder, f"graph_{method_suffix}.json")) and dual_index_cache_exists:
        graph_data = load_cache(cache_folder)
        if build_dual_index:
            I_s_to_c, I_c_to_s, sentence_texts, concept_vectors = load_dual_index(cache_folder, method_suffix)
            dual_index_data = {
                "I_s_to_c": I_s_to_c,
                "I_c_to_s": I_c_to_s, 
                "sentence_texts": sentence_texts,
                "concept_vectors": concept_vectors
            }
            return graph_data + (dual_index_data,), -1
        else:
            return graph_data + (None,), -1
    
    # Build from scratch - only need nlp for original graph construction
    if nlp is None and concept_extraction_method == "nlp":
        raise ValueError("NLP extractor required when concept_extraction_method='nlp'")
    
    # For backward compatibility, if nlp is provided, use it for graph construction
    if nlp is not None:
        graph_file_path = os.path.join(cache_folder, f"graph_{nlp.method}.json")
        index_file_path = os.path.join(cache_folder, f"index_{nlp.method}.json")
        appearance_count_file_path = os.path.join(cache_folder, f"appearance_count_{nlp.method}.json")
        edges = []
        index = {}
        appearance_count = {}

        for i, chunk in enumerate(text):
            if i % 10 == 1:
                logger.info(f"Now extracting the {i}th chunk...")
            naive_result = nlp.naive_extract_graph(chunk)
            # not merge the entities.
            appearance_count["leaf_{}".format(i)] = naive_result["appearance_count"]

            for noun in naive_result["nouns"]:
                if noun not in index:
                    index[noun] = []
                index[noun].append("leaf_{}".format(i))
            
            for noun, count in naive_result["appearance_count"].items():
                appearance_count[noun] = appearance_count.get(noun, 0) + count

            # add the cooccurrence.
            for pair, weight in naive_result["cooccurrence"].items():
                head, tail = pair
                edges.append([head, tail, weight])

        # build the graph.
        G = build_graph(edges)
    else:
        # No graph construction if nlp not provided
        G = nx.Graph()
        index = {}
        appearance_count = {}
        graph_file_path = os.path.join(cache_folder, f"graph_{method_suffix}.json")
        index_file_path = os.path.join(cache_folder, f"index_{method_suffix}.json")
        appearance_count_file_path = os.path.join(cache_folder, f"appearance_count_{method_suffix}.json")
    
    # Build dual index if requested
    dual_index_data = None
    if build_dual_index:
        logger.info(f"Building dual index (sentence-level indices) using {concept_extraction_method}...")
        dual_builder = DualIndexBuilder(
            embedder_model=embedder_model, 
            device=device,
            concept_extraction_method=concept_extraction_method
        )
        
        # Build sentence indices
        I_s_to_c, I_c_to_s, sentence_texts = dual_builder.build_sentence_index(text, nlp)
        logger.info(f"Built sentence index: {len(sentence_texts)} sentences, {len(I_c_to_s)} concepts")
        
        # Build concept vectors
        concept_vectors = dual_builder.build_concept_vectors(I_c_to_s, sentence_texts)
        logger.info(f"Built concept vectors for {len(concept_vectors)} concepts")
        
        # Save dual index
        save_dual_index(cache_folder, I_s_to_c, I_c_to_s, sentence_texts, concept_vectors, method_suffix)
        
        dual_index_data = {
            "I_s_to_c": I_s_to_c,
            "I_c_to_s": I_c_to_s,
            "sentence_texts": sentence_texts,
            "concept_vectors": concept_vectors
        }
    
    # save the graph and index if nlp was provided
    if nlp is not None:
        save_graph(edges, graph_file_path)
        save_index(index, index_file_path)
        save_appearance_count(appearance_count, appearance_count_file_path)
    
    extract_end_time = time.time()
    return (G, index, appearance_count, dual_index_data), extract_end_time - extract_start_time

def extract_graph(text:List[str], cache_folder:str, nlp:Extractor, use_cache=True, reextract=False):
    """
    Original extract_graph function for backward compatibility
    """
    extract_start_time = time.time()
    if use_cache and os.path.exists(os.path.join(cache_folder, f"graph_{nlp.method}.json")) and os.path.exists(os.path.join(cache_folder, f"index_{nlp.method}.json")) and os.path.exists(os.path.join(cache_folder, f"appearance_count_{nlp.method}.json")):
        return load_cache(cache_folder), -1
    else:
        graph_file_path = os.path.join(cache_folder, f"graph_{nlp.method}.json")
        index_file_path = os.path.join(cache_folder, f"index_{nlp.method}.json")
        appearance_count_file_path = os.path.join(cache_folder, f"appearance_count_{nlp.method}.json")
        edges = []
        index = {}
        appearance_count = {}

        for i, chunk in enumerate(text):
            if i % 10 == 1:
                logger.info(f"Now extracting the {i}th chunk...")
            naive_result = nlp.naive_extract_graph(chunk)
            # not merge the entities.
            appearance_count["leaf_{}".format(i)] = naive_result["appearance_count"]

            for noun in naive_result["nouns"]:
                if noun not in index:
                    index[noun] = []
                index[noun].append("leaf_{}".format(i))
            
            for noun, count in naive_result["appearance_count"].items():
                appearance_count[noun] = appearance_count.get(noun, 0) + count

            # add the cooccurrence.
            for pair, weight in naive_result["cooccurrence"].items():
                head, tail = pair
                edges.append([head, tail, weight])

        # build the graph.
        G = build_graph(edges)
        # save the graph and index.
        save_graph(edges, graph_file_path)
        save_index(index, index_file_path)
        save_appearance_count(appearance_count, appearance_count_file_path)
        extract_end_time = time.time()
        return (G, index, appearance_count), extract_end_time - extract_start_time

if __name__ == "__main__":
    edges = [
        ('a', 'b', 1),
        ('a', 'b', 3),
        ('b', 'a', 2)
    ]
    
    G = build_graph(edges)
    
    # 打印所有边的权重
    for u, v, w in G.edges(data='weight'):
        logger.info(f"Edge ({u}, {v}): weight = {w}")  # 应该输出: Edge (a, b): weight = 6