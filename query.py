from extract_graph import Extractor
from build_tree import sequential_merge
from typing import List, Tuple, Dict, Set
from itertools import combinations
import networkx as nx
import faiss
import spacy
from collections import defaultdict
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import random
import logging
random.seed(1)
import copy
import numpy as np

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

    def _entity_scoring(self, query_entities: List[str], candidate_chunks: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate entity-based scores for chunks based on entity frequency, co-occurrence, and importance"""
        entity_scores = {}
        
        for key, chunk_ids in candidate_chunks.items():
            for chunk_id in chunk_ids:
                # Calculate entity score based on frequency and co-occurrence
                entity_score = 0.0
                key_entities = key.split("_") if key else []
                
                # Get appearance count for this chunk
                chunk_appearance = self.appearance_count.get(chunk_id, {})
                
                # Calculate frequency-based score
                for entity in query_entities:
                    entity_freq = chunk_appearance.get(entity, 0)
                    entity_score += entity_freq
                    
                # Add bonus for co-occurrence (entities appearing together)
                if len(query_entities) > 1:
                    co_occurrence_bonus = len([e for e in query_entities if e in key_entities]) / len(query_entities)
                    entity_score *= (1 + co_occurrence_bonus)
                    
                entity_scores[chunk_id] = entity_score
                
        return entity_scores

    def _summary_scoring(self, query: str, k: int = 50) -> Dict[str, float]:
        """Calculate semantic similarity scores for chunks using dense retrieval"""
        if self.embedder is None or self.faiss_index is None:
            return {}
            
        # Get dense retrieval results with scores
        query_embed = self.embedder.encode(query).reshape(1, -1)
        scores, chunk_indices = self.faiss_index.search(query_embed, k=k)
        
        # Convert to chunk_id -> score mapping
        summary_scores = {}
        for i, chunk_idx in enumerate(chunk_indices[0]):
            if chunk_idx >= 0:  # Valid index
                chunk_id = self.collapse_tree_ids[chunk_idx]
                summary_scores[chunk_id] = float(scores[0][i])
                
        return summary_scores

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to [0,1] range using min-max normalization"""
        if not scores:
            return scores
            
        score_values = list(scores.values())
        min_score = min(score_values)
        max_score = max(score_values)
        
        # Avoid division by zero
        if max_score == min_score:
            return {chunk_id: 0.5 for chunk_id in scores.keys()}
            
        normalized_scores = {}
        for chunk_id, score in scores.items():
            normalized_scores[chunk_id] = (score - min_score) / (max_score - min_score)
            
        return normalized_scores

    def _calculate_dynamic_weights(self, query: str, entities: List[str]) -> Tuple[float, float]:
        """Calculate dynamic fusion weights based on query characteristics"""
        if not query.strip():
            return 0.5, 0.5
            
        # Calculate entity ratio
        query_tokens = query.split()
        entity_ratio = len(entities) / len(query_tokens) if query_tokens else 0
        
        # Dynamic weight adjustment
        if entity_ratio > 0.5:
            # High entity density - emphasize entity matching
            alpha = 0.7  # entity weight
            beta = 0.3   # summary weight
        elif entity_ratio < 0.2:
            # Low entity density - emphasize semantic matching  
            alpha = 0.3  # entity weight
            beta = 0.7   # summary weight
        else:
            # Balanced approach
            alpha = 0.5
            beta = 0.5
            
        return alpha, beta

    def _fuse_scores(self, entity_scores: Dict[str, float], summary_scores: Dict[str, float], 
                     alpha: float, beta: float, threshold: float = 0.2) -> Dict[str, float]:
        """Fuse normalized entity and summary scores with given weights"""
        # Get all unique chunk IDs
        all_chunk_ids = set(entity_scores.keys()) | set(summary_scores.keys())
        
        fused_scores = {}
        for chunk_id in all_chunk_ids:
            entity_score = entity_scores.get(chunk_id, 0.0)
            summary_score = summary_scores.get(chunk_id, 0.0)
            
            # Calculate fused score
            fused_score = alpha * entity_score + beta * summary_score
            
            # Apply threshold filter
            if fused_score >= threshold:
                fused_scores[chunk_id] = fused_score
                
        return fused_scores

    def _cross_encoder_rerank(self, query: str, chunk_ids: List[str], top_n: int = 20) -> List[str]:
        """
        Re-rank chunks using a cross-encoder for more precise relevance scoring.
        This is optional but strongly recommended for better results.
        """
        try:
            # For now, use a simple heuristic re-ranking based on chunk content similarity
            # In a production system, you would use a proper cross-encoder model
            if not chunk_ids or len(chunk_ids) <= top_n:
                return chunk_ids
                
            # Get chunk texts
            chunk_texts = {}
            for chunk_id in chunk_ids:
                if chunk_id in self.cache_tree:
                    chunk_texts[chunk_id] = self.cache_tree[chunk_id]["text"]
            
            # Simple text-based re-ranking (placeholder for cross-encoder)
            query_words = set(query.lower().split())
            
            def calculate_overlap_score(text: str) -> float:
                text_words = set(text.lower().split())
                overlap = len(query_words.intersection(text_words))
                return overlap / max(len(query_words), 1)
            
            # Re-rank based on text overlap
            chunk_scores = []
            for chunk_id in chunk_ids:
                if chunk_id in chunk_texts:
                    score = calculate_overlap_score(chunk_texts[chunk_id])
                    chunk_scores.append((chunk_id, score))
            
            # Sort by score and return top N
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            return [chunk_id for chunk_id, score in chunk_scores[:top_n]]
            
        except Exception as e:
            logger.warning(f"Cross-encoder re-ranking failed: {e}, returning original order")
            return chunk_ids[:top_n]

    def fusion_retrieval(self, query: str, entities: List[str], **kwargs) -> Dict[str, List[str]]:
        """
        Fusion retrieval that combines entity graph and summary tree retrieval
        following the pipeline described in the problem statement
        """
        max_chunks = kwargs.get("max_chunk_setting", 25)
        k_candidates = max_chunks * 2  # Retrieve more candidates for fusion
        
        # Step 1: Independent retrieval and scoring
        logger.debug("Step 1: Independent retrieval and scoring")
        
        # Entity graph retrieval
        entity_candidates = {}
        if entities:
            # Use local retrieval to get entity-based candidates
            shortest_path_k = kwargs.get("shortest_path_k", 4)
            entity_candidates = self.local_retrieval(entities, shortest_path_k)
            
            # If no results from local retrieval, try occurrence ranking
            if self._count_chunks(entity_candidates) == 0:
                # Get candidates from dense retrieval for occurrence ranking
                dense_candidates = self.dense_retrieval(query, k_candidates)
                candidate_chunk_ids = dense_candidates.get("", [])
                entity_candidates = self.occurrence_ranking(candidate_chunk_ids, entities)
        
        # Calculate entity scores
        entity_scores = self._entity_scoring(entities, entity_candidates)
        
        # Summary tree retrieval and scoring
        summary_scores = self._summary_scoring(query, k_candidates)
        
        # Step 2: Score normalization
        logger.debug("Step 2: Score normalization")
        entity_scores_norm = self._normalize_scores(entity_scores)
        summary_scores_norm = self._normalize_scores(summary_scores)
        
        # Step 3: Dynamic weight calculation and fusion
        logger.debug("Step 3: Dynamic weight calculation and fusion")
        alpha, beta = self._calculate_dynamic_weights(query, entities)
        logger.debug(f"Dynamic weights: alpha={alpha}, beta={beta}")
        
        fused_scores = self._fuse_scores(entity_scores_norm, summary_scores_norm, alpha, beta)
        
        # Step 4: Re-ranking with cross-encoder (optional but recommended)
        use_rerank = kwargs.get("use_cross_encoder_rerank", True)
        if use_rerank:
            logger.debug("Step 4: Cross-encoder re-ranking")
            sorted_chunks = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Get more candidates for re-ranking
            rerank_candidates = min(max_chunks * 2, len(sorted_chunks))
            candidate_chunk_ids = [chunk_id for chunk_id, score in sorted_chunks[:rerank_candidates]]
            
            # Apply cross-encoder re-ranking
            reranked_chunk_ids = self._cross_encoder_rerank(query, candidate_chunk_ids, max_chunks)
        else:
            logger.debug("Step 4: Skipping cross-encoder re-ranking")
            sorted_chunks = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            reranked_chunk_ids = [chunk_id for chunk_id, score in sorted_chunks[:max_chunks]]
        
        # Step 5: Result merging and deduplication
        logger.debug("Step 5: Result merging and deduplication")
        result_chunks = reranked_chunk_ids
        
        # Group chunks by entities they contain for consistent format
        result_dict = {}
        for chunk_id in result_chunks:
            # Find which entities this chunk contains
            chunk_entities = []
            for entity in entities:
                if chunk_id in self.index.get(entity, []):
                    chunk_entities.append(entity)
            
            if chunk_entities:
                key = "_".join(sorted(chunk_entities))
                result_dict.setdefault(key, []).append(chunk_id)
            else:
                # For chunks without entities (from summary retrieval)
                result_dict.setdefault("", []).append(chunk_id)
                
        return self.merge_keys(result_dict)

    def query_fusion(self, query: str, **kwargs) -> dict:
        """
        Main fusion query method that implements the complete fusion pipeline
        """
        # Step 1: Extract entities from query
        entities = self.nlp.naive_extract_graph(query.split("\n")[0])
        entities = entities["nouns"]
        logger.debug(f"Extracted entities: {entities}")

        # If no embedder available, fallback to original query method
        if self.embedder is None or self.faiss_index is None:
            logger.warning("Embedder or FAISS index not available, falling back to original retrieval")
            return self.query(query, **kwargs)

        # Use fusion retrieval
        fusion_result = self.fusion_retrieval(query, entities, **kwargs)
        res_str = self.format_res(fusion_result)
        
        result = {"chunks": res_str}
        
        if kwargs.get("debug", True):
            supplement_info = self._build_supplement_info(
                fusion_result, entities, fusion_result, 
                list(fusion_result.keys()), 
                self._count_chunks(fusion_result), 
                []  # No chunk count history for fusion method
            )
            result.update(supplement_info)
            result["retrieval_type"] = "Fusion Retrieval"
            
        return result

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
        # Check if fusion mode is requested
        if kwargs.get("use_fusion", False):
            return self.query_fusion(query, **kwargs)
            
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

    def _build_supplement_info(self, chunk_ids, entities, neighbor_nodes, keys, len_chunks, chunk_counts_history):
        return {
            "chunk_ids": chunk_ids,
            "entities": entities,
            "neighbor_nodes": neighbor_nodes,
            "keys": keys,
            "len_chunks": len_chunks,
            "chunk_counts_history": chunk_counts_history
        }
