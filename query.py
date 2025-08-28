from extract_graph import Extractor
from build_tree import sequential_merge
from typing import List, Tuple, Dict, Set, Optional, Union
from itertools import combinations
import networkx as nx
import faiss
import spacy
from collections import defaultdict, Counter
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import random
import logging
import re
import math
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

random.seed(1)
import copy

# Get logger for this module
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, cache_tree, G:nx.Graph, index, appearance_count:Dict[str, int], nlp:Extractor, **kwargs) -> None:
        # cache_tree: summary tree of the document.
        # G: graph of the document.
        # index: noun to chunks index. another index will be built in the get_inverse_index function.
        # appearance_count: appearance count of the entities in the chunks.
        # nlp: Extractor class.
        self.cache_tree = cache_tree
        self.collapse_tree, self.collapse_tree_ids = self._collapse_tree(self.cache_tree)
        self.G = G
        self.index = index
        self.appearance_count = appearance_count
        # get the inverse index, i.e., chunk_id to noun index.
        self.inverse_index = self.get_inverse_index()
        self.nlp = nlp
        # set up the parameters.
        self.device = kwargs.get("device", "cuda:0")
        self.merge_num = kwargs.get("merge_num", 5)
        self.min_count = kwargs.get("min_count", 2)
        self.overlap = kwargs.get("overlap", 100)
        self.tokenizer = kwargs.get("tokenizer","/path/to/your/model")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        if kwargs.get("embedder", "BAAI/bge-m3") is not None:
            self.embedder = SentenceTransformer(kwargs.get("embedder", "BAAI/bge-m3"),device=self.device)
            self.faiss_index = self._build_faiss_index()
        else:
            logger.warning("Warning: the embedder is set to None, dense retrieval is not implemented.")
            self.embedder = None
            self.faiss_index = None
        
        # Enhanced retrieval configuration
        self._init_enhanced_config(kwargs)
        # Initialize BM25 for lexical retrieval
        self._init_bm25_index()

    def _init_enhanced_config(self, kwargs):
        """Initialize enhanced retrieval configuration."""
        # Budget allocation
        self.budget_N = kwargs.get('budget_N', 10)
        self.cap_I_ratio = kwargs.get('cap_I_ratio', 0.5)
        
        # Parallel recall parameters
        self.tree_top_k = kwargs.get('tree_top_k', 15)
        self.graph_top_k = kwargs.get('graph_top_k', 15)
        self.bm25_top_k = kwargs.get('bm25_top_k', 8)
        self.dense_top_k = kwargs.get('dense_top_k', 8)
        self.graph_max_hops = kwargs.get('graph_max_hops', 2)
        
        # RRF parameters for soft overlap
        self.rrf_k = kwargs.get('rrf_k', 60)
        
        # Linear scoring weights (baseline)
        default_weights = {
            'sim_emb': 0.35, 'sim_lex': 0.20, 'ent_overlap': 0.20, 'path_score': 0.15,
            'level_boost': 0.05, 'recency': 0.05, 'authority': 0.03, 'overlap_bonus': 0.02
        }
        self.scoring_weights = kwargs.get('scoring_weights', default_weights)
        
        # Question type specific weights
        self.question_type_weights = kwargs.get('question_type_weights', {
            'definition': {'sim_emb': 0.35, 'sim_lex': 0.25, 'level_boost': 0.20, 'ent_overlap': 0.10, 'recency': 0.05, 'path_score': 0.05},
            'relation': {'ent_overlap': 0.25, 'path_score': 0.25, 'sim_emb': 0.20, 'sim_lex': 0.15, 'level_boost': 0.10, 'recency': 0.05},
            'recent': {'recency': 0.30, 'sim_emb': 0.25, 'sim_lex': 0.20, 'ent_overlap': 0.10, 'level_boost': 0.10, 'path_score': 0.05}
        })
        
        # MMR parameters
        self.mmr_lambda = kwargs.get('mmr_lambda', 0.75)
        
        # Diversity constraints
        self.max_same_subtree_ratio = kwargs.get('max_same_subtree_ratio', 0.5)
        self.max_same_entity_cluster_ratio = kwargs.get('max_same_entity_cluster_ratio', 0.5)
        self.min_query_entities_covered = kwargs.get('min_query_entities_covered', 2)
        
        # Recency parameters
        self.recency_half_life = kwargs.get('recency_half_life', 180)
        self.recency_min = kwargs.get('recency_min', 0.2)
        
        # Enable enhanced retrieval by default
        self.use_enhanced_retrieval = kwargs.get('use_enhanced_retrieval', True)

    def _init_bm25_index(self):
        """Initialize BM25 index for lexical retrieval."""
        try:
            # Prepare documents for BM25
            documents = []
            self.chunk_id_to_doc_idx = {}
            
            for i, chunk_id in enumerate(self.collapse_tree_ids):
                text = self.collapse_tree[i] if isinstance(self.collapse_tree, list) else self.cache_tree.get(chunk_id, {}).get('text', '')
                documents.append(text)
                self.chunk_id_to_doc_idx[chunk_id] = i
            
            # Initialize TF-IDF vectorizer as BM25 approximation
            self.tfidf_vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                max_features=10000,
                ngram_range=(1, 2)
            )
            
            # Fit and transform documents
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            
            logger.info(f"Initialized BM25 index with {len(documents)} documents")
            
        except Exception as e:
            logger.warning(f"Failed to initialize BM25 index: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None

    def __del__(self):
        """Ensure proper cleanup of resources
           avoid the memory leak.
        """
        try:
            if hasattr(self, 'embedder'):
                del self.embedder
            if hasattr(self, 'faiss_index'):
                del self.faiss_index
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error during Retriever cleanup: {e}")

    def update(self, cache_tree, G, index, appearance_count):
        # update the retriever, from a document to another document.
        self.cache_tree = cache_tree
        self.collapse_tree, self.collapse_tree_ids = self._collapse_tree(self.cache_tree)
        self.G = G
        self.index = index
        self.appearance_count = appearance_count
        self.inverse_index = self.get_inverse_index()
        self.docs = self.collapse_tree
        if self.embedder is not None:
            self.faiss_index = self._build_faiss_index()

    def get_inverse_index(self):
        # get the inverse index, i.e., chunk_id to noun index.
        inverse_index = {}
        for key, value in self.index.items():
            for chunk_id in value:
                inverse_index.setdefault(chunk_id, []).append(key)
        return inverse_index

    def _collapse_tree(self, cache_tree:Dict[str, Dict]) -> Dict[str, Dict]:
        # collapse the tree. for dense retrieval
        # return the collapsed tree.
        collapsed_tree = []
        collapsed_tree_ids = []
        for key, value in self.cache_tree.items():
            collapsed_tree.append(value["text"])
            collapsed_tree_ids.append(key)
        return collapsed_tree, collapsed_tree_ids

    def _detect_neighbor_nodes(self, keys:Set[str], chunk_id: str) -> List[str]:
        # detect the neighbor nodes of the chunk_id.
        # return the neighbor nodes.
        int_chunk_id = int(chunk_id.split("_")[-1])
        front = True
        back = True
        neighbor_nodes = [chunk_id]
        front_int_chunk_id = int_chunk_id
        back_int_chunk_id = int_chunk_id
        while front or back:
            if front:
                front_int_chunk_id = front_int_chunk_id - 1
                if front_int_chunk_id < 0 or set(self.inverse_index.get(front_int_chunk_id, [])) & keys != keys:
                    front = False
                str_chunk_id = "leaf_{}".format(front_int_chunk_id)
                append = True
                for key in keys:
                    if str_chunk_id not in self.index.get(key, []):
                        append = False
                        front = False
                        break
                if append:
                    neighbor_nodes.append(str_chunk_id)
            if back:
                back_int_chunk_id += 1
                if set(self.inverse_index.get(back_int_chunk_id, [])) & keys != keys:
                    back = False
                str_chunk_id = "leaf_{}".format(back_int_chunk_id)
                append = True
                for key in keys:
                    if str_chunk_id not in self.index.get(key, []):
                        append = False
                        back = False
                        break
                if append:
                    neighbor_nodes.append(str_chunk_id)
        return neighbor_nodes

    def _build_faiss_index(self):
        # build the faiss index.
        # only used when the dense retrieval is implemented.
        # return the faiss index.
        docs = self.collapse_tree
        if self.embedder is None:
            self.embedder = SentenceTransformer("BAAI/bge-m3",device=self.device)
            self.embedder.eval()
            logger.info("the embedder is not set, using the default embedder BAAI/bge-m3.")
        doc_embeds = self.embedder.encode(docs, batch_size=16, device=self.device)
        # print("doc_embeds examples", doc_embeds[0:5][0:5])
        # print("doc_embeds shape", doc_embeds.shape)
        vector_database = faiss.IndexFlatIP(doc_embeds.shape[1])
        vector_database.add(doc_embeds)
        return vector_database

    def index_mapping(self, entities:list) -> List[str]:
        # get the chunks from the cache tree.
        # for two types:
        # 1. entity is a list of strings, i.e., the entities are not related,
        chunk_ids = {}
        
        for entity in entities:
            if isinstance(entity, str):
                if entity in self.index.keys():
                    chunk_ids[entity] = self.index[entity]
            elif isinstance(entity, tuple):
                chunk_ids_set = set()
                entity_key = "_".join(entity)
                for e in entity:
                    if e in self.index.keys():
                        if chunk_ids_set == set():
                            chunk_ids_set = set(self.index[e])
                        else:
                            chunk_ids_set = chunk_ids_set & set(self.index[e])
                chunk_ids[entity_key] = sorted(list(chunk_ids_set))

        return chunk_ids
    
    def graph_filter(self, entities:List[str], k) -> List[str]:
        # get the shortest path between the entities.
        shortest_path_pairs = []
        for head, tail in combinations(entities, 2):
            if head in self.G.nodes() and tail in self.G.nodes():
                try:
                    shortest_path = nx.shortest_path(self.G, head, tail)
                except nx.NetworkXNoPath:
                    continue
                if len(shortest_path) <= k:
                    shortest_path_pairs.append((head, tail))

        # shortest_path_pairs = self.merge_tuples(shortest_path_pairs)
        return shortest_path_pairs

    def merge_tuples(self, lst):
        graph = defaultdict(set)
        
        for a, b in lst:
            graph[a].add(b)
            graph[b].add(a)
        
        visited = set()
        result = []
        
        def dfs(entity, cluster):
            if entity in visited:
                return
            visited.add(entity)
            cluster.add(entity)
            for neighbor in graph[entity]:
                dfs(neighbor, cluster)
        
        for a, b in lst:
            if a not in visited:
                cluster = set()
                dfs(a, cluster)
                result.append(tuple(sorted(cluster)))
        
        return result
    
    def validate_by_checking_father_chunks(self, init_chunk_ids:Dict[str, List[str]], min_count:int=2) -> Dict[str, List[str]]:
        # w, by input the shortest path pairs, get the father nodes.
        valid_child_ids = {}
        for key, chunk_ids in init_chunk_ids.items():
            father_nodes = {}
            for chunk_id in chunk_ids:
                father_chunk_id = self.cache_tree[chunk_id]["parent"]
                father_nodes.setdefault(father_chunk_id, []).append(chunk_id)
            valid_leaf_nodes = [child_id_list for _, child_id_list in father_nodes.items()
                               if len(child_id_list) >= min_count]
            
            valid_child_ids[key] = []
            if len(valid_leaf_nodes) > 0:
                for leaf_node in valid_leaf_nodes:
                    valid_child_ids[key].extend(leaf_node)
        
        return valid_child_ids

    def merge_keys(self, neighbor_nodes:Dict[str, List[str]]) -> Dict[str, List[str]]:
        # merge the nodes with different keys.
        # return with the same format.
        chunks_to_keys = defaultdict(set)
        for key, chunk_lists in neighbor_nodes.items():
            for chunk in chunk_lists:
                chunks_to_keys[chunk].add(key)

        merged_result = {}
        for chunk, keys in chunks_to_keys.items():
            # get the new key.
            if len(keys) > 1:
                all_entities = set()
                for key in keys:
                    all_entities.update(key.split("_"))
                new_key = "_".join(sorted(all_entities))
            else:
                new_key = keys.pop()
            # add the chunk to the new key.
            if new_key in merged_result.keys():
                merged_result.setdefault(new_key, []).append(chunk)
            else:
                merged_result[new_key] = [chunk]
        return merged_result


    def get_contiguous_chunks(self, leaf_nodes:List[str]) -> str:
        leaf_texts = []
        for leaf_node in leaf_nodes:
            leaf_text = self.cache_tree[leaf_node]["text"]
            leaf_texts.append(leaf_text)
        return sequential_merge(leaf_texts, self.tokenizer, self.overlap)

    def detect_contiguous_chunks(self, chunk_ids:List[str]) -> List[List[str]]:
        # Detect the contiguous chunks from the chunk_ids,
        # if there are contiguous chunks, return the list of the contiguous chunks.
        # otherwise, the only id will be a list.
        res = []
        current_chunk = []
        chunk_ids = sorted(chunk_ids, key=lambda x: int(x.split("_")[1]))
        for chunk_id in chunk_ids:
            # Extract the numeric part of the chunk_id
            id_num = int(chunk_id.split("_")[1])
            if not current_chunk:
                current_chunk.append(chunk_id)
            else:
                # Check if the current id is contiguous with the last one
                last_id_num = int(current_chunk[-1].split("_")[1])
                if id_num == last_id_num + 1:
                    current_chunk.append(chunk_id)
                else:
                    res.append(current_chunk)
                    current_chunk = [chunk_id]

        if current_chunk:
            res.append(current_chunk)

        return res

    def format_res(self, res:Dict[str, List[str]]) -> str:
        res_str = ""
        for key, chunks in res.items():
            chunks = self.detect_contiguous_chunks(chunks)
            for chunk_list in chunks:
                str_of_list = self.get_contiguous_chunks(chunk_list)
                res_str += "{}: {}\n".format(key, str_of_list)
        return res_str
    
    def str_chunkid_2_int_chunkid(self, str_chunk:str) -> int:
        return int(str_chunk.split("_")[-1])

    def local_retrieval(self, entities:List[str], shortest_path_k:int=4)->Dict[str, List[str]]:
        # initialize by shortest path
        shortest_path = self.graph_filter(entities, shortest_path_k) 
        # it returns the list of pairs existing shortest path shorter than k.

        # initialize the chunks.
        init_chunk_ids = self.index_mapping(shortest_path)
        
        neighbor_nodes = self.merge_keys(init_chunk_ids)
        return neighbor_nodes

    def dense_retrieval(self, query,k):
        # using dense retrieval to get the chunks.
        query_embed = self.embedder.encode(query).reshape(1, -1) # need (1, -1) for faiss.
        _, condidate_chunks_indexs = self.faiss_index.search(query_embed, k = k)
        # the normal faiss index return the (1, k) shape. squeeze it to (k,).
        condidate_chunks_indexs = condidate_chunks_indexs[0]
        condidate_chunk_ids = [self.collapse_tree_ids[i] for i in condidate_chunks_indexs]
        res = {"": condidate_chunk_ids}
        return res

    def _count_chunks(self, res:Dict[str, List[str]]) -> int:
        # count the chunks.
        count = 0
        for chunk_ids in res.values():
            count += len(chunk_ids)
        return count

    def entityaware_filter(self, candidate_chunks:Dict[str, List[str]], entities:List[str]) -> Dict[str, List[str]]:
        # filter rules:
        # 1. the chunk includes more different entities, the priority is higher.
        # 2. if the chunk has longer neighbor nodes, the priority is higher.
        # 3. if the chunk includes the same number of entities, the higher the number of appearance of the entities is, the higher the priority is.
        # Initialize result dictionary
        chunks_info = []
        for key, value in candidate_chunks.items():
            for chunk_id in value:
                key_count = len(key.split("_"))
                set_key = set(key.split("_"))
                neighbor_nodes_count = len(self._detect_neighbor_nodes(keys=set_key, chunk_id=chunk_id))
                entity_count = 0
                for key_entity in key.split('_'):
                    entity_count += self.appearance_count[chunk_id].get(key_entity, 0)
                chunk_id_info = {
                    "chunk_id": chunk_id,
                    "key_count": key_count,
                    "neighbor_nodes_count": neighbor_nodes_count,
                    "entity_count": entity_count
                }
                chunks_info.append(chunk_id_info)
        # sort the chunks_info by the key_count, neighbor_nodes_count, and entity_count.
        sorted_chunks_info = sorted(chunks_info, key=lambda x: (x["key_count"], x["neighbor_nodes_count"], x["entity_count"]), reverse=True)
        # get the top 25 chunks.
        top_25_chunks = sorted_chunks_info[:25]
        # get the chunk_ids from the top_25_chunks.
        top_25_chunk_ids = [chunk["chunk_id"] for chunk in top_25_chunks]
        # return the result.
        filtered_res = {}
        for id in top_25_chunk_ids:
            for entity in entities:
                if id in self.index.get(entity, []):
                    filtered_res.setdefault(entity, []).append(id)
        filtered_res = self.merge_keys(filtered_res)
        logger.debug(f"filtered_res: {filtered_res}")
        return filtered_res

    def _check_children(self, chunk_id:str, entities:List[str], visited=None) -> int:
        if visited is None:
            visited = set()
        
        if chunk_id in visited:
            return 0
        
        visited.add(chunk_id)
        
        entity_count = 0
        
        children = self.cache_tree.get(chunk_id, {}).get("children", [])

        for child in children:
            if not child.startswith("leaf_"):
                entity_count += self._check_children(child, entities, visited)
            else:
                chunk_appearance_stat = self.appearance_count.get(child, {})
                for entity in entities:
                    entity_count += chunk_appearance_stat.get(entity, 0)
        
        return entity_count

    def occurrence_ranking(self, candidate_chunk_ids:List[str], entities:List[str]) -> Dict[str, List[str]]:
        # occurrence ranking.
        filtered_res = {}
        filtered_chunk_ids = []
        chunk_count = []
        
        for chunk_id in candidate_chunk_ids:
            if not chunk_id.startswith("leaf_"):
                chunk_count.append(self._check_children(chunk_id, entities))
                continue
            
            chunk_appearance_stat = self.appearance_count.get(chunk_id, {})
            this_chunk_count = 0
            for entity in entities:
                this_chunk_count += chunk_appearance_stat.get(entity, 0)
            chunk_count.append(this_chunk_count)

        chunk_count = np.array(chunk_count)
        nonzero_indices = np.nonzero(chunk_count)[0]
        
        if len(nonzero_indices) == 0:
            return {"": candidate_chunk_ids[:25]}
        
        argsorted_chunk_ids = np.argsort(chunk_count)[::-1]
        filtered_chunk_ids = [candidate_chunk_ids[i] for i in argsorted_chunk_ids if chunk_count[i] > 0]
        
        if len(filtered_chunk_ids) > 25:
            filtered_chunk_ids = filtered_chunk_ids[:25]
        
        for id in filtered_chunk_ids:
            for entity in entities:
                if entity in self.inverse_index.get(id, []):
                    filtered_res.setdefault(entity, []).append(id)
        
        filtered_res = self.merge_keys(filtered_res)
        return filtered_res

    def query(self, query, **kwargs):
        """
        Enhanced query method with parallel recall, overlap detection, and advanced scoring.
        
        Args:
            query: The query string
            **kwargs: Configuration parameters
            
        Returns:
            Dict containing chunks and metadata
        """
        # Check if enhanced retrieval is enabled
        if not getattr(self, 'use_enhanced_retrieval', True):
            return self.query_legacy(query, **kwargs)
            
        logger.info("Using enhanced retrieval system")
        
        # Step 1: Entity extraction and query analysis
        entities = self.nlp.naive_extract_graph(query.split("\n")[0])
        entities = entities["nouns"]
        
        if len(entities) == 0:
            # Fallback to dense retrieval only
            logger.info("No entities found, using dense retrieval fallback")
            chunk_ids = self.dense_retrieval(query, kwargs.get("max_chunk_setting", self.budget_N))
            result = {"chunks": self.format_res(chunk_ids)}
            if kwargs.get("debug", True):
                supplement_info = self._build_supplement_info(chunk_ids, entities, chunk_ids, list(chunk_ids.keys()), len(chunk_ids), [])
                result.update(supplement_info)
                result["retrieval_type"] = "Enhanced Dense Fallback"
            return result
        
        # Step 2: Parallel recall execution
        start_time = time.time()
        
        tree_chunks = self.tree_recall(entities, self.tree_top_k)
        graph_chunks = self.graph_recall(entities, self.graph_top_k, self.graph_max_hops)
        bm25_chunks = self.bm25_recall(query, self.bm25_top_k)
        dense_chunks = self.dense_recall_enhanced(query, self.dense_top_k)
        
        recall_time = time.time() - start_time
        logger.debug(f"Parallel recall completed in {recall_time:.3f}s")
        
        # Step 3: Hard overlap detection
        hard_overlap = self.detect_hard_overlap(tree_chunks, graph_chunks)
        
        # Step 4: Collect all candidate chunks with sources
        all_candidates = self._collect_candidates_with_sources(
            tree_chunks, graph_chunks, bm25_chunks, dense_chunks
        )
        
        # Step 5: Apply enhanced scoring and ranking
        ranked_chunks = self._apply_enhanced_scoring(query, entities, all_candidates, hard_overlap)
        
        # Step 6: Budget allocation and final selection
        final_chunks = self._allocate_budget_and_select(ranked_chunks, hard_overlap, entities, kwargs)
        
        # Step 7: Format results
        result = {"chunks": self.format_res(final_chunks)}
        
        if kwargs.get("debug", True):
            debug_info = self._build_enhanced_supplement_info(
                final_chunks, entities, all_candidates, hard_overlap, 
                len(final_chunks), recall_time
            )
            result.update(debug_info)
            result["retrieval_type"] = "Enhanced Parallel Retrieval"
        
        return result

    def query_legacy(self, query, **kwargs):
        # step 1: extract the Entities from the query.
        # reuse the naive_extract_graph function, which is used in the graph building process.
        entities = self.nlp.naive_extract_graph(query.split("\n")[0])
        entities = entities["nouns"]

        # step 2.0: set up the parameters.
        shortest_path_k = kwargs.get("shortest_path_k", 4)

        # step 2.1: short circuit, if there is no entity, then return the naive dense retrieval.
        if len(entities) == 0:
            chunk_ids = self.dense_retrieval(query, kwargs.get("max_chunk_setting", 25))
            result = {"chunks":self.format_res(chunk_ids)}
            if kwargs.get("debug", True):
                supplement_info = self._build_supplement_info(chunk_ids, entities, chunk_ids, list(chunk_ids.keys()), len(chunk_ids), [])
                result.update(supplement_info)
                result["retrieval_type"] = "Global Search"
            return result

        # step 2.2: initialize the chunks by wasd method.
        local_res = self.local_retrieval(entities, shortest_path_k)

        # step 2.2: check the result.
            # if the chunk count is larger than the max chunk setting, then change the setting, decrease the shortest path k, until the chunk count is less than the max chunk setting.
            # if the chunk count is 0, take it as dense retrieval.
    
        chunk_count = self._count_chunks(local_res)
        # record the chunk count history, for debug.
        chunk_counts_history = []
        chunk_counts_history.append((shortest_path_k, chunk_count))

        # if the chunk count is 0, occurrence ranking.
        if chunk_count == 0:          
            query_embed = self.embedder.encode(query).reshape(1, -1)
            # retrieve the top 2 k chunks.
            _, condidate_chunks_indexs = self.faiss_index.search(query_embed, k = kwargs.get("max_chunk_setting", 25)*2)
            condidate_chunks_indexs = condidate_chunks_indexs[0]
            condidate_chunk_ids = [self.collapse_tree_ids[i] for i in condidate_chunks_indexs]
            filtered_chunk_ids = self.occurrence_ranking(condidate_chunk_ids, entities) 
            # return the entity_entityB: [chunk_id1, chunk_id2, ...]
            res_str = self.format_res(filtered_chunk_ids)

            result = {"chunks":res_str}
            if kwargs.get("debug", True):
                supplement_info = self._build_supplement_info(filtered_chunk_ids, entities, filtered_chunk_ids, list(filtered_chunk_ids.keys()), 25, chunk_counts_history)
                result.update(supplement_info)
                result["retrieval_type"] = "Occurrence Rerank"
            return result
        
        while chunk_count > kwargs.get("max_chunk_setting", 25):
            prev_local_res = copy.deepcopy(local_res)
            # if the chunk count is larger than the max chunk setting
            # then change the setting, increase the min count and decrease the shortest path k.
            shortest_path_k -= 1
            # update the result with new restrictions.
            local_res = self.local_retrieval(entities, shortest_path_k)
            chunk_count = self._count_chunks(local_res)
            chunk_counts_history.append((shortest_path_k, chunk_count))

        # if the chunk count is 0, dense retrieval + entity filter.
        # if the chunk count is not 0, return the result.
        if chunk_count != 0:
            # format the result.
            logger.debug(f"BOTTOM2TOP: final chunk_count {chunk_count}")
            res_str = self.format_res(local_res)

            result = {"chunks":res_str}
            if kwargs.get("debug", True):
                supplement_info = self._build_supplement_info(local_res, entities, local_res, list(local_res.keys()), chunk_count, chunk_counts_history)
                result.update(supplement_info)
                result["retrieval_type"] = f"Local, Loop for {len(chunk_counts_history)-1} times"
            return result
        else:
            # the previous local result is not empty, so we can use it as candidate chunks.
            candidate_chunks = prev_local_res
            res_ids = self.entityaware_filter(candidate_chunks, entities)
            chunk_count = self._count_chunks(res_ids)
            res_str = self.format_res(res_ids)
            result = {"chunks":res_str}
            result["chunk_counts_history"] = chunk_counts_history
            if kwargs.get("debug", True):
                supplement_info = self._build_supplement_info(res_ids, entities, res_ids, list(res_ids.keys()), 25, chunk_counts_history)
                result.update(supplement_info)
                result["retrieval_type"] = f"EntityAware Filter, Loop for {len(chunk_counts_history)-1} times"
            return result        

    # ==================== Enhanced Retrieval Methods ====================
    
    def tree_recall(self, query_entities: List[str], top_k: int) -> Dict[str, List[str]]:
        """
        Tree recall: Top-down traversal of summary tree to get top-Kt leaf chunks.
        
        Args:
            query_entities: List of entities extracted from query
            top_k: Number of top chunks to retrieve
            
        Returns:
            Dict mapping entities to chunk IDs
        """
        candidates = []
        
        # Find root nodes (nodes with no parents)
        root_nodes = [node_id for node_id, node_data in self.cache_tree.items() 
                     if not node_data.get('parent') and node_id.startswith('summary_')]
        
        if not root_nodes:
            # If no explicit root, find highest level summaries
            max_level = max([int(node_id.split('_')[1]) for node_id in self.cache_tree.keys() 
                           if node_id.startswith('summary_')], default=-1)
            root_nodes = [node_id for node_id in self.cache_tree.keys() 
                         if node_id.startswith(f'summary_{max_level}_')]
        
        # Traverse tree top-down to collect leaf candidates
        for root in root_nodes:
            leaf_candidates = self._traverse_tree_for_entities(root, query_entities, set())
            candidates.extend(leaf_candidates)
        
        # Add direct leaf matches
        for entity in query_entities:
            if entity in self.index:
                for chunk_id in self.index[entity]:
                    if chunk_id.startswith('leaf_'):
                        score = self.appearance_count.get(chunk_id, {}).get(entity, 0)
                        candidates.append((chunk_id, entity, score))
        
        # Remove duplicates and sort by score
        unique_candidates = {}
        for chunk_id, entity, score in candidates:
            key = (chunk_id, entity)
            if key not in unique_candidates or unique_candidates[key][2] < score:
                unique_candidates[key] = (chunk_id, entity, score)
        
        # Sort by score and take top-k
        sorted_candidates = sorted(unique_candidates.values(), key=lambda x: x[2], reverse=True)[:top_k]
        
        # Group by entity
        result = defaultdict(list)
        for chunk_id, entity, score in sorted_candidates:
            result[entity].append(chunk_id)
        
        logger.debug(f"Tree recall found {len(sorted_candidates)} candidates for {len(query_entities)} entities")
        return dict(result)
    
    def _traverse_tree_for_entities(self, node_id: str, query_entities: List[str], visited: Set[str]) -> List[Tuple[str, str, float]]:
        """Helper method to traverse tree and collect leaf nodes containing entities."""
        if node_id in visited:
            return []
        visited.add(node_id)
        
        candidates = []
        node_data = self.cache_tree.get(node_id, {})
        children = node_data.get('children', [])
        
        if not children or all(child.startswith('leaf_') for child in children):
            # This is a leaf or parent of leaves
            for child in children:
                if child.startswith('leaf_'):
                    chunk_stats = self.appearance_count.get(child, {})
                    for entity in query_entities:
                        if entity in chunk_stats:
                            score = chunk_stats[entity]
                            candidates.append((child, entity, score))
        else:
            # Continue traversing
            for child in children:
                candidates.extend(self._traverse_tree_for_entities(child, query_entities, visited))
        
        return candidates
    
    def graph_recall(self, query_entities: List[str], top_k: int, max_hops: int) -> Dict[str, List[str]]:
        """
        Graph recall: Entity-seeded expansion with 1-2 hop limits to get evidence chunks.
        
        Args:
            query_entities: List of entities extracted from query
            top_k: Number of top chunks to retrieve
            max_hops: Maximum number of hops for expansion (1-2)
            
        Returns:
            Dict mapping entity combinations to chunk IDs
        """
        expanded_entities = set(query_entities)
        
        # Expand entities through graph traversal
        for entity in query_entities:
            if entity in self.G.nodes():
                # 1-hop expansion
                neighbors_1hop = list(self.G.neighbors(entity))
                expanded_entities.update(neighbors_1hop)
                
                # 2-hop expansion if max_hops >= 2
                if max_hops >= 2:
                    for neighbor in neighbors_1hop:
                        neighbors_2hop = list(self.G.neighbors(neighbor))
                        expanded_entities.update(neighbors_2hop)
        
        # Find chunks containing expanded entities
        candidates = []
        for entity in expanded_entities:
            if entity in self.index:
                for chunk_id in self.index[entity]:
                    # Calculate score based on entity appearance and graph distance
                    appearance_score = self.appearance_count.get(chunk_id, {}).get(entity, 0)
                    
                    # Calculate minimum graph distance from query entities
                    min_distance = float('inf')
                    for query_entity in query_entities:
                        if query_entity in self.G.nodes() and entity in self.G.nodes():
                            try:
                                distance = nx.shortest_path_length(self.G, query_entity, entity)
                                min_distance = min(min_distance, distance)
                            except nx.NetworkXNoPath:
                                continue
                    
                    if min_distance == float('inf'):
                        min_distance = 0 if entity in query_entities else max_hops + 1
                    
                    # Score inversely proportional to distance
                    distance_score = 1.0 / (1 + min_distance) if min_distance <= max_hops else 0.0
                    final_score = appearance_score * distance_score
                    
                    candidates.append((chunk_id, entity, final_score))
        
        # Remove duplicates and sort
        unique_candidates = {}
        for chunk_id, entity, score in candidates:
            key = (chunk_id, entity)
            if key not in unique_candidates or unique_candidates[key][2] < score:
                unique_candidates[key] = (chunk_id, entity, score)
        
        # Sort by score and take top-k
        sorted_candidates = sorted(unique_candidates.values(), key=lambda x: x[2], reverse=True)[:top_k]
        
        # Group by entity combinations (for pairs found together)
        result = self._group_entities_in_chunks(sorted_candidates, query_entities)
        
        logger.debug(f"Graph recall found {len(sorted_candidates)} candidates with {len(expanded_entities)} expanded entities")
        return result
    
    def _group_entities_in_chunks(self, candidates: List[Tuple[str, str, float]], query_entities: List[str]) -> Dict[str, List[str]]:
        """Group candidates by entity combinations found in the same chunks."""
        chunk_entities = defaultdict(set)
        chunk_scores = defaultdict(float)
        
        # Collect entities per chunk
        for chunk_id, entity, score in candidates:
            chunk_entities[chunk_id].add(entity)
            chunk_scores[chunk_id] += score
        
        # Create entity combination keys
        result = defaultdict(list)
        for chunk_id, entities in chunk_entities.items():
            # Find which query entities are in this chunk
            query_entities_in_chunk = [e for e in query_entities if e in entities]
            
            if query_entities_in_chunk:
                key = "_".join(sorted(query_entities_in_chunk))
                result[key].append(chunk_id)
        
        return dict(result)
    
    def bm25_recall(self, query: str, top_k: int) -> Dict[str, List[str]]:
        """
        BM25 recall: Lexical matching using TF-IDF approximation.
        
        Args:
            query: Query string
            top_k: Number of top chunks to retrieve
            
        Returns:
            Dict with empty key mapping to chunk IDs
        """
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            logger.warning("BM25 index not available, returning empty results")
            return {"": []}
        
        try:
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarity scores
            similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top-k indices
            top_indices = np.argsort(similarity_scores)[::-1][:top_k]
            
            # Map back to chunk IDs
            top_chunks = []
            for idx in top_indices:
                if similarity_scores[idx] > 0:  # Only include non-zero similarities
                    chunk_id = self.collapse_tree_ids[idx]
                    top_chunks.append(chunk_id)
            
            logger.debug(f"BM25 recall found {len(top_chunks)} relevant chunks")
            return {"": top_chunks}
            
        except Exception as e:
            logger.warning(f"BM25 recall failed: {e}")
            return {"": []}
    
    def dense_recall_enhanced(self, query: str, top_k: int) -> Dict[str, List[str]]:
        """
        Enhanced dense recall using semantic similarity.
        
        Args:
            query: Query string
            top_k: Number of top chunks to retrieve
            
        Returns:
            Dict with empty key mapping to chunk IDs
        """
        if self.embedder is None or self.faiss_index is None:
            logger.warning("Dense retrieval not available, returning empty results")
            return {"": []}
        
        try:
            # Encode query
            query_embed = self.embedder.encode(query).reshape(1, -1)
            
            # Search in FAISS index
            similarities, indices = self.faiss_index.search(query_embed, k=top_k)
            
            # Map to chunk IDs
            top_chunks = []
            for i, idx in enumerate(indices[0]):
                if similarities[0][i] > 0:  # Only include positive similarities
                    chunk_id = self.collapse_tree_ids[idx]
                    top_chunks.append(chunk_id)
            
            logger.debug(f"Dense recall found {len(top_chunks)} semantically similar chunks")
            return {"": top_chunks}
            
        except Exception as e:
            logger.warning(f"Dense recall failed: {e}")
            return {"": []}
    
    def detect_hard_overlap(self, tree_chunks: Dict[str, List[str]], graph_chunks: Dict[str, List[str]]) -> Set[str]:
        """
        Detect hard overlap: I = Tree ∩ Graph
        
        Args:
            tree_chunks: Results from tree recall
            graph_chunks: Results from graph recall
            
        Returns:
            Set of chunk IDs that appear in both tree and graph results
        """
        tree_chunk_ids = set()
        for chunk_list in tree_chunks.values():
            tree_chunk_ids.update(chunk_list)
        
        graph_chunk_ids = set()
        for chunk_list in graph_chunks.values():
            graph_chunk_ids.update(chunk_list)
        
        overlap = tree_chunk_ids & graph_chunk_ids
        
        logger.debug(f"Hard overlap detected: {len(overlap)} chunks from {len(tree_chunk_ids)} tree + {len(graph_chunk_ids)} graph")
        return overlap
    
    def _collect_candidates_with_sources(self, tree_chunks: Dict, graph_chunks: Dict, 
                                       bm25_chunks: Dict, dense_chunks: Dict) -> Dict[str, Dict]:
        """
        Collect all candidate chunks with their sources and rankings.
        
        Returns:
            Dict mapping chunk_id to source information
        """
        candidates = {}
        
        # Process tree chunks
        rank = 0
        for entity, chunk_list in tree_chunks.items():
            for chunk_id in chunk_list:
                if chunk_id not in candidates:
                    candidates[chunk_id] = {'sources': set(), 'ranks': {}}
                candidates[chunk_id]['sources'].add('tree')
                candidates[chunk_id]['ranks']['tree'] = rank
                rank += 1
        
        # Process graph chunks
        rank = 0
        for entity, chunk_list in graph_chunks.items():
            for chunk_id in chunk_list:
                if chunk_id not in candidates:
                    candidates[chunk_id] = {'sources': set(), 'ranks': {}}
                candidates[chunk_id]['sources'].add('graph')
                candidates[chunk_id]['ranks']['graph'] = rank
                rank += 1
        
        # Process BM25 chunks
        rank = 0
        for chunk_list in bm25_chunks.values():
            for chunk_id in chunk_list:
                if chunk_id not in candidates:
                    candidates[chunk_id] = {'sources': set(), 'ranks': {}}
                candidates[chunk_id]['sources'].add('bm25')
                candidates[chunk_id]['ranks']['bm25'] = rank
                rank += 1
        
        # Process dense chunks
        rank = 0
        for chunk_list in dense_chunks.values():
            for chunk_id in chunk_list:
                if chunk_id not in candidates:
                    candidates[chunk_id] = {'sources': set(), 'ranks': {}}
                candidates[chunk_id]['sources'].add('dense')
                candidates[chunk_id]['ranks']['dense'] = rank
                rank += 1
        
        return candidates
    
    def _apply_enhanced_scoring(self, query: str, entities: List[str], 
                              candidates: Dict[str, Dict], hard_overlap: Set[str]) -> List[Tuple[str, float]]:
        """
        Apply enhanced linear scoring to all candidates.
        
        Returns:
            List of (chunk_id, final_score) tuples sorted by score
        """
        scored_chunks = []
        
        # Detect question type for adaptive weights
        question_type = self.detect_question_type(query)
        weights = self.question_type_weights.get(question_type, self.scoring_weights)
        
        for chunk_id, source_info in candidates.items():
            # Calculate all scoring components
            scores = {
                'sim_emb': self.calculate_embedding_similarity(query, chunk_id),
                'sim_lex': self.calculate_lexical_similarity(query, chunk_id),
                'ent_overlap': self.calculate_entity_overlap(entities, chunk_id),
                'path_score': self.calculate_path_score(entities, chunk_id, self.graph_max_hops),
                'level_boost': self.calculate_level_boost(chunk_id),
                'recency': self.calculate_recency_boost(chunk_id),
                'authority': self.calculate_authority_score(chunk_id),
                'overlap_bonus': self.calculate_soft_overlap_score(chunk_id, source_info)
            }
            
            # Calculate weighted final score
            final_score = sum(weights.get(component, 0) * score for component, score in scores.items())
            
            # Boost score for hard overlap chunks
            if chunk_id in hard_overlap:
                final_score *= 1.2  # 20% boost for hard overlap
            
            scored_chunks.append((chunk_id, final_score))
        
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Scored {len(scored_chunks)} candidates using {question_type} weights")
        return scored_chunks
    
    def _allocate_budget_and_select(self, ranked_chunks: List[Tuple[str, float]], 
                                  hard_overlap: Set[str], entities: List[str], kwargs: Dict) -> Dict[str, List[str]]:
        """
        Allocate budget and apply final selection with diversity constraints.
        
        Returns:
            Final selected chunks grouped by entities
        """
        budget = kwargs.get("max_chunk_setting", self.budget_N)
        cap_I = int(budget * self.cap_I_ratio)
        
        # Separate hard overlap and remainder
        hard_overlap_candidates = [(chunk_id, score) for chunk_id, score in ranked_chunks if chunk_id in hard_overlap]
        remainder_candidates = [(chunk_id, score) for chunk_id, score in ranked_chunks if chunk_id not in hard_overlap]
        
        # Allocate hard overlap chunks (capped)
        selected_hard = hard_overlap_candidates[:cap_I]
        
        # Calculate remaining budget
        remaining_budget = budget - len(selected_hard)
        
        # Select from remainder with MMR and diversity
        selected_remainder = self._mmr_selection(remainder_candidates, selected_hard, remaining_budget)
        
        # Combine selections
        final_selected = [chunk_id for chunk_id, _ in selected_hard + selected_remainder]
        
        # Group by entities for output format
        result = self._group_selected_by_entities(final_selected, entities)
        
        logger.info(f"Final selection: {len(selected_hard)} hard overlap + {len(selected_remainder)} remainder = {len(final_selected)} total")
        return result
    
    def _mmr_selection(self, candidates: List[Tuple[str, float]], 
                      already_selected: List[Tuple[str, float]], remaining_budget: int) -> List[Tuple[str, float]]:
        """
        MMR-based selection for diversity.
        
        Returns:
            List of selected (chunk_id, score) tuples
        """
        if remaining_budget <= 0 or not candidates:
            return []
        
        selected = []
        selected_chunk_ids = [chunk_id for chunk_id, _ in already_selected]
        
        candidate_pool = candidates.copy()
        
        for _ in range(min(remaining_budget, len(candidate_pool))):
            if not candidate_pool:
                break
            
            best_mmr_score = -float('inf')
            best_candidate = None
            best_idx = -1
            
            for i, (chunk_id, relevance_score) in enumerate(candidate_pool):
                # Calculate max similarity to already selected
                max_sim = 0.0
                if selected_chunk_ids:
                    max_sim = max(self._calculate_chunk_similarity(chunk_id, selected_id) 
                                for selected_id in selected_chunk_ids)
                
                # MMR score
                mmr_score = self.mmr_lambda * relevance_score - (1 - self.mmr_lambda) * max_sim
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate = (chunk_id, relevance_score)
                    best_idx = i
            
            if best_candidate:
                selected.append(best_candidate)
                selected_chunk_ids.append(best_candidate[0])
                candidate_pool.pop(best_idx)
        
        return selected
    
    def _calculate_chunk_similarity(self, chunk_id1: str, chunk_id2: str) -> float:
        """Calculate similarity between two chunks for MMR."""
        if chunk_id1 == chunk_id2:
            return 1.0
        
        # Simple text similarity using embeddings if available
        if self.embedder is not None:
            try:
                text1 = self.cache_tree.get(chunk_id1, {}).get('text', '')
                text2 = self.cache_tree.get(chunk_id2, {}).get('text', '')
                
                if text1 and text2:
                    embed1 = self.embedder.encode(text1)
                    embed2 = self.embedder.encode(text2)
                    similarity = np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
                    return max(0.0, similarity)  # Ensure non-negative
            except:
                pass
        
        # Fallback: simple overlap of entities
        entities1 = set(self.inverse_index.get(chunk_id1, []))
        entities2 = set(self.inverse_index.get(chunk_id2, []))
        
        if not entities1 and not entities2:
            return 0.0
        
        intersection = len(entities1 & entities2)
        union = len(entities1 | entities2)
        
        return intersection / union if union > 0 else 0.0
    
    def _group_selected_by_entities(self, selected_chunks: List[str], entities: List[str]) -> Dict[str, List[str]]:
        """Group selected chunks by entities for output format."""
        result = defaultdict(list)
        
        for chunk_id in selected_chunks:
            chunk_entities = self.inverse_index.get(chunk_id, [])
            
            # Find which query entities are in this chunk
            matching_entities = [e for e in entities if e in chunk_entities]
            
            if matching_entities:
                key = "_".join(sorted(matching_entities))
                result[key].append(chunk_id)
            else:
                # If no entity match, use generic key
                result[""].append(chunk_id)
        
        return dict(result)

    # ==================== Scoring Component Methods ====================
    
    def calculate_embedding_similarity(self, query: str, chunk_id: str) -> float:
        """Calculate query-chunk cosine similarity using embeddings."""
        if self.embedder is None:
            return 0.0
        
        try:
            chunk_text = self.cache_tree.get(chunk_id, {}).get('text', '')
            if not chunk_text:
                return 0.0
            
            query_embed = self.embedder.encode(query)
            chunk_embed = self.embedder.encode(chunk_text)
            
            # Ensure both embeddings are 1D arrays
            if query_embed.ndim > 1:
                query_embed = query_embed.flatten()
            if chunk_embed.ndim > 1:
                chunk_embed = chunk_embed.flatten()
            
            similarity = np.dot(query_embed, chunk_embed) / (np.linalg.norm(query_embed) * np.linalg.norm(chunk_embed))
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"Embedding similarity calculation failed: {e}")
            return 0.0
    
    def calculate_lexical_similarity(self, query: str, chunk_id: str) -> float:
        """Calculate lexical similarity: BM25×0.7 + token_coverage×0.2 + proximity×0.1"""
        chunk_text = self.cache_tree.get(chunk_id, {}).get('text', '')
        if not chunk_text:
            return 0.0
        
        # BM25 component using TF-IDF approximation
        bm25_score = 0.0
        if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
            try:
                query_vector = self.tfidf_vectorizer.transform([query])
                chunk_idx = self.chunk_id_to_doc_idx.get(chunk_id)
                if chunk_idx is not None:
                    chunk_vector = self.tfidf_matrix[chunk_idx:chunk_idx+1]
                    bm25_score = cosine_similarity(query_vector, chunk_vector)[0][0]
            except:
                pass
        
        # Token coverage component
        query_tokens = set(query.lower().split())
        chunk_tokens = set(chunk_text.lower().split())
        coverage = len(query_tokens & chunk_tokens) / len(query_tokens) if query_tokens else 0.0
        
        # Proximity component (simplified: inverse of position difference)
        proximity = 0.5  # Default neutral value
        
        # Combine components
        lexical_score = 0.7 * bm25_score + 0.2 * coverage + 0.1 * proximity
        return max(0.0, min(1.0, lexical_score))
    
    def calculate_entity_overlap(self, query_entities: List[str], chunk_id: str) -> float:
        """Calculate weighted Jaccard entity overlap."""
        chunk_entities = set(self.inverse_index.get(chunk_id, []))
        query_entity_set = set(query_entities)
        
        if not query_entity_set:
            return 0.0
        
        intersection = query_entity_set & chunk_entities
        union = query_entity_set | chunk_entities
        
        # Simple Jaccard for now (can be enhanced with entity type weighting)
        jaccard = len(intersection) / len(union) if union else 0.0
        
        return max(0.0, min(1.0, jaccard))
    
    def calculate_path_score(self, query_entities: List[str], chunk_id: str, max_hops: int) -> float:
        """Calculate graph path score with relation matching, edge weights, and time decay."""
        chunk_entities = set(self.inverse_index.get(chunk_id, []))
        
        if not query_entities or not chunk_entities:
            return 0.0
        
        total_score = 0.0
        path_count = 0
        
        for query_entity in query_entities:
            for chunk_entity in chunk_entities:
                if query_entity in self.G.nodes() and chunk_entity in self.G.nodes():
                    try:
                        path = nx.shortest_path(self.G, query_entity, chunk_entity)
                        path_length = len(path) - 1
                        
                        if path_length <= max_hops:
                            # Calculate path score based on edge weights
                            path_weight = 1.0
                            for i in range(len(path) - 1):
                                edge_data = self.G.get_edge_data(path[i], path[i+1], {})
                                edge_weight = edge_data.get('weight', 1.0)
                                path_weight *= edge_weight
                            
                            # Apply hop penalty
                            hop_penalty = 1.0 / (1 + path_length)
                            
                            # Time decay (simplified - using constant decay for now)
                            time_decay = 1.0
                            
                            score = path_weight * hop_penalty * time_decay
                            total_score += score
                            path_count += 1
                            
                    except nx.NetworkXNoPath:
                        continue
        
        # Average score across paths
        avg_score = total_score / path_count if path_count > 0 else 0.0
        return max(0.0, min(1.0, avg_score))
    
    def calculate_level_boost(self, chunk_id: str) -> float:
        """Calculate tree level boost: 0.6×tree_sim + 0.3×depth_norm + 0.1×is_leaf"""
        # Tree similarity component (simplified)
        tree_sim = 0.5  # Default neutral value
        
        # Depth normalization
        if chunk_id.startswith('leaf_'):
            depth_norm = 1.0  # Leaves get full depth score
            is_leaf = 1.0
        elif chunk_id.startswith('summary_'):
            try:
                level = int(chunk_id.split('_')[1])
                depth_norm = 1.0 / (1 + level)  # Higher levels get lower scores
                is_leaf = 0.0
            except:
                depth_norm = 0.5
                is_leaf = 0.0
        else:
            depth_norm = 0.5
            is_leaf = 0.0
        
        level_boost = 0.6 * tree_sim + 0.3 * depth_norm + 0.1 * is_leaf
        return max(0.0, min(1.0, level_boost))
    
    def calculate_recency_boost(self, chunk_id: str) -> float:
        """Calculate recency boost: exp(-age/H) clamped to [0.2, 1]"""
        # Simplified: assume all chunks are recent for now
        # In a real system, this would use document timestamps
        recency = 1.0  # Default to recent
        
        return max(self.recency_min, min(1.0, recency))
    
    def calculate_authority_score(self, chunk_id: str) -> float:
        """Calculate authority score: log1p(citations) normalized within group"""
        # Simplified: use entity count as proxy for authority
        chunk_entities = len(self.inverse_index.get(chunk_id, []))
        authority = math.log1p(chunk_entities) / math.log1p(10)  # Normalize by log(11)
        
        return max(0.0, min(1.0, authority))
    
    def calculate_soft_overlap_score(self, chunk_id: str, source_info: Dict) -> float:
        """Calculate soft overlap score using RRF-style scoring."""
        sources = source_info.get('sources', set())
        ranks = source_info.get('ranks', {})
        
        # RRF-style scoring
        rrf_score = 0.0
        for source in ['tree', 'graph', 'bm25', 'dense']:
            if source in sources:
                rank = ranks.get(source, 0)
                rrf_score += 1.0 / (self.rrf_k + rank)
        
        # Normalize by number of possible sources
        normalized_score = rrf_score / (4 * (1.0 / self.rrf_k))  # 4 sources max
        
        return max(0.0, min(1.0, normalized_score))
    
    def detect_question_type(self, query: str) -> str:
        """Detect question type for adaptive weights."""
        query_lower = query.lower()
        
        # Definition/comprehension patterns
        if any(word in query_lower for word in ['what is', 'what are', 'define', 'definition', 'explain', 'describe', 'who is', 'who are']):
            return 'definition'
        
        # Relation/comparison patterns  
        if any(word in query_lower for word in ['compare', 'versus', 'vs', 'relationship', 'connection', 'related', 'how does', 'how is', 'why does']):
            return 'relation'
        
        # Recent/temporal patterns
        if any(word in query_lower for word in ['recent', 'latest', 'new', 'current', 'today', 'now', 'recently', 'breakthrough']):
            return 'recent'
        
        # Default to baseline weights
        return 'baseline'
    
    def _build_enhanced_supplement_info(self, final_chunks: Dict, entities: List[str], 
                                      all_candidates: Dict, hard_overlap: Set[str], 
                                      len_chunks: int, recall_time: float) -> Dict:
        """Build enhanced debug information."""
        return {
            "chunk_ids": final_chunks,
            "entities": entities,
            "len_chunks": len_chunks,
            "hard_overlap_count": len(hard_overlap),
            "total_candidates": len(all_candidates),
            "recall_time": recall_time,
            "enhanced_retrieval": True
        }

    # ==================== End Enhanced Methods ====================

    def _build_supplement_info(self, chunk_ids, entities, neighbor_nodes, keys, len_chunks, chunk_counts_history):
        return {
            "chunk_ids": chunk_ids,
            "entities": entities,
            "neighbor_nodes": neighbor_nodes,
            "keys": keys,
            "len_chunks": len_chunks,
            "chunk_counts_history": chunk_counts_history
        }
