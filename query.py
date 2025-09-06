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

    def tree_semantic_retrieval(self, query, k=25):
        """
        Tree-based semantic retrieval: search in the summary tree to find relevant 
        high-level nodes and leaf nodes using semantic similarity.
        Returns a dict with separate lists for high-level nodes and leaf nodes.
        """
        if self.embedder is None:
            logger.warning("Embedder is None, cannot perform tree semantic retrieval")
            return {"high_level_nodes": [], "leaf_nodes": []}
        
        query_embed = self.embedder.encode(query).reshape(1, -1)
        
        # Separate high-level summary nodes and leaf nodes
        high_level_nodes = []
        high_level_texts = []
        leaf_nodes = []
        leaf_texts = []
        
        for node_id, node_data in self.cache_tree.items():
            if node_id.startswith("leaf_"):
                leaf_nodes.append(node_id)
                leaf_texts.append(node_data["text"])
            elif node_id.startswith("summary_"):
                high_level_nodes.append(node_id)
                high_level_texts.append(node_data["text"])
        
        # Encode all texts
        if high_level_texts:
            high_level_embeds = self.embedder.encode(high_level_texts, batch_size=16, device=self.device)
        if leaf_texts:
            leaf_embeds = self.embedder.encode(leaf_texts, batch_size=16, device=self.device)
        
        result_high_level = []
        result_leaf = []
        
        # Search in high-level nodes - get top relevant summary nodes
        if high_level_texts:
            high_level_index = faiss.IndexFlatIP(high_level_embeds.shape[1])
            high_level_index.add(high_level_embeds)
            
            # Get more high-level nodes as they provide essential background framework
            high_level_k = min(k // 2, len(high_level_nodes))
            _, high_level_indices = high_level_index.search(query_embed, k=high_level_k)
            high_level_indices = high_level_indices[0]
            result_high_level = [high_level_nodes[i] for i in high_level_indices]
        
        # Search in leaf nodes 
        if leaf_texts:
            leaf_index = faiss.IndexFlatIP(leaf_embeds.shape[1])
            leaf_index.add(leaf_embeds)
            
            leaf_k = min(k - len(result_high_level), len(leaf_nodes))
            _, leaf_indices = leaf_index.search(query_embed, k=leaf_k)
            leaf_indices = leaf_indices[0]
            result_leaf = [leaf_nodes[i] for i in leaf_indices]
        
        return {
            "high_level_nodes": result_high_level,
            "leaf_nodes": result_leaf
        }

    def _count_chunks(self, res:Dict[str, List[str]]) -> int:
        # count the chunks.
        count = 0
        for chunk_ids in res.values():
            count += len(chunk_ids)
        return count

    def hybrid_result_fusion(self, tree_result, graph_result, entities, max_chunks=25):
        """
        Fuse results from tree semantic retrieval and graph entity retrieval.
        
        Fusion rules:
        1. Keep all high-level summary nodes (essential background framework)
        2. Keep overlapping parts between leaf nodes (from tree) and chunks (from graph)
        3. When not reaching max_chunks limit, supplement with other chunks based on priority rules:
           a. Chunks containing more different entities have higher priority
           b. Chunks with more neighbor nodes have higher priority  
           c. Chunks with higher entity appearance frequency have higher priority
        """
        # Step 1: Keep all high-level nodes - they provide essential background
        final_result = {}
        high_level_nodes = tree_result.get("high_level_nodes", [])
        if high_level_nodes:
            final_result["high_level_summary"] = high_level_nodes
        
        # Step 2: Get leaf nodes from tree and chunks from graph
        leaf_nodes_from_tree = tree_result.get("leaf_nodes", [])
        chunks_from_graph = []
        for chunk_list in graph_result.values():
            chunks_from_graph.extend(chunk_list)
        
        # Step 3: Find overlapping parts (intersection) between tree leaf nodes and graph chunks
        overlapping_chunks = list(set(leaf_nodes_from_tree) & set(chunks_from_graph))
        if overlapping_chunks:
            final_result["overlapping_chunks"] = overlapping_chunks
        
        # Step 4: Count current chunks
        current_count = len(high_level_nodes) + len(overlapping_chunks)
        
        # Step 5: If we haven't reached max_chunks, add supplementary chunks with priority ranking
        if current_count < max_chunks:
            remaining_slots = max_chunks - current_count
            
            # Get all candidate supplementary chunks (union - intersection)
            all_tree_leaf = set(leaf_nodes_from_tree)
            all_graph_chunks = set(chunks_from_graph)
            supplementary_candidates = (all_tree_leaf | all_graph_chunks) - set(overlapping_chunks)
            supplementary_candidates = [chunk for chunk in supplementary_candidates if chunk.startswith("leaf_")]
            
            if supplementary_candidates and len(supplementary_candidates) > 0:
                # Apply enhanced ranking logic
                ranked_supplements = self.enhanced_chunk_ranking(list(supplementary_candidates), entities)
                selected_supplements = ranked_supplements[:remaining_slots]
                
                if selected_supplements:
                    final_result["supplementary_chunks"] = selected_supplements
        
        return final_result

    def enhanced_chunk_ranking(self, candidate_chunks, entities):
        """
        Enhanced ranking logic for supplementary chunks based on:
        1. Number of different entities in the chunk (higher priority)
        2. Number of neighbor nodes (higher priority) 
        3. Entity appearance frequency (higher priority)
        """
        if not candidate_chunks:
            return []
        
        chunks_info = []
        
        for chunk_id in candidate_chunks:
            if not chunk_id.startswith("leaf_"):
                continue
                
            # Count different entities in this chunk
            chunk_entities = set()
            entity_appearance_count = 0
            
            chunk_appearance_stat = self.appearance_count.get(chunk_id, {})
            for entity in entities:
                if entity in chunk_appearance_stat and chunk_appearance_stat[entity] > 0:
                    chunk_entities.add(entity)
                    entity_appearance_count += chunk_appearance_stat[entity]
            
            # Count neighbor nodes
            neighbor_nodes_count = len(self._detect_neighbor_nodes(chunk_entities, chunk_id)) if chunk_entities else 0
            
            chunk_info = {
                "chunk_id": chunk_id,
                "unique_entities_count": len(chunk_entities),  # Rule 1: more different entities
                "neighbor_nodes_count": neighbor_nodes_count,   # Rule 2: more neighbor nodes
                "entity_frequency_sum": entity_appearance_count  # Rule 3: higher appearance frequency
            }
            chunks_info.append(chunk_info)
        
        # Sort by priority rules (descending order for all criteria)
        sorted_chunks = sorted(chunks_info, 
                             key=lambda x: (x["unique_entities_count"], 
                                          x["neighbor_nodes_count"], 
                                          x["entity_frequency_sum"]), 
                             reverse=True)
        
        return [chunk["chunk_id"] for chunk in sorted_chunks]

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
        # step 1: extract the Entities from the query.
        # reuse the naive_extract_graph function, which is used in the graph building process.
        entities = self.nlp.naive_extract_graph(query.split("\n")[0])
        entities = entities["nouns"]

        # step 2.0: set up the parameters.
        shortest_path_k = kwargs.get("shortest_path_k", 4)
        max_chunks = kwargs.get("max_chunk_setting", 25)

        # step 2.1: short circuit, if there is no entity, then return the naive dense retrieval.
        if len(entities) == 0:
            chunk_ids = self.dense_retrieval(query, max_chunks)
            result = {"chunks":self.format_res(chunk_ids)}
            if kwargs.get("debug", True):
                supplement_info = self._build_supplement_info(chunk_ids, entities, chunk_ids, list(chunk_ids.keys()), len(chunk_ids), [])
                result.update(supplement_info)
                result["retrieval_type"] = "Global Search"
            return result

        # HYBRID RETRIEVAL STRATEGY IMPLEMENTATION
        # Step 1: Primary retrieval from summary tree (semantic search)
        tree_result = self.tree_semantic_retrieval(query, k=max_chunks)
        
        # Step 2: Auxiliary retrieval from entity graph (existing method with shortest_path_k=4)
        graph_result = self.local_retrieval(entities, shortest_path_k)
        
        # Track chunk counts for debugging
        chunk_counts_history = []
        initial_graph_count = self._count_chunks(graph_result)
        chunk_counts_history.append((shortest_path_k, initial_graph_count))
        
        # Step 3: Result fusion and deduplication
        fused_result = self.hybrid_result_fusion(tree_result, graph_result, entities, max_chunks)
        
        # Check if we need to adjust graph retrieval if too many chunks
        current_total = sum(len(chunks) for chunks in fused_result.values())
        
        # If still too many chunks, reduce graph retrieval scope
        while current_total > max_chunks and shortest_path_k > 1:
            shortest_path_k -= 1
            graph_result = self.local_retrieval(entities, shortest_path_k)
            graph_count = self._count_chunks(graph_result)
            chunk_counts_history.append((shortest_path_k, graph_count))
            
            # Re-fuse with reduced graph results
            fused_result = self.hybrid_result_fusion(tree_result, graph_result, entities, max_chunks)
            current_total = sum(len(chunks) for chunks in fused_result.values())
        
        # Format the result for output
        res_str = self.format_res(fused_result)
        
        result = {"chunks": res_str}
        
        if kwargs.get("debug", True):
            # Build debug information
            supplement_info = {
                "chunk_ids": fused_result,
                "entities": entities,
                "neighbor_nodes": graph_result,
                "keys": list(graph_result.keys()),
                "len_chunks": current_total,
                "chunk_counts_history": chunk_counts_history,
                "tree_high_level_count": len(tree_result.get("high_level_nodes", [])),
                "tree_leaf_count": len(tree_result.get("leaf_nodes", [])),
                "graph_chunk_count": initial_graph_count
            }
            result.update(supplement_info)
            result["retrieval_type"] = f"Hybrid Tree-Graph Retrieval, {len(chunk_counts_history)} iterations"
        
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
