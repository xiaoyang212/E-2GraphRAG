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

    def hybrid_retrieval(self, query, entities, **kwargs):
        """
        Implement the primary-auxiliary collaborative hybrid retrieval strategy.
        
        1. Primary retrieval with summary tree: semantic search for high-level and leaf nodes
        2. Auxiliary retrieval with entity graph: find entities and their relationship networks  
        3. Result fusion and deduplication: keep all high-level summaries, deduplicate overlaps
        4. Supplementary completion: use entityaware_filter for remaining chunks
        """
        max_chunk_setting = kwargs.get("max_chunk_setting", 25)
        
        # Step 1: Primary retrieval - semantic search in summary tree
        tree_results = self.tree_semantic_search(query, max_chunk_setting * 2, **kwargs)
        
        # Step 2: Auxiliary retrieval - entity graph search  
        graph_results = self.local_retrieval(entities, kwargs.get("shortest_path_k", 4))
        
        # Step 3: Result fusion and deduplication
        fused_results = self.fuse_and_deduplicate(tree_results, graph_results, **kwargs)
        
        # Step 4: Supplementary completion with remaining slots
        current_count = self._count_chunks(fused_results)
        if current_count < max_chunk_setting:
            remaining_slots = max_chunk_setting - current_count
            fused_results = self.supplement_with_entityaware_filter(
                fused_results, entities, remaining_slots, **kwargs
            )
        
        return fused_results
        
    def tree_semantic_search(self, query, k, **kwargs):
        """
        Perform semantic search in the summary tree to find relevant high-level nodes and leaf nodes.
        Returns a dict with high_level_summaries and leaf_nodes separated.
        """
        if self.embedder is None or self.faiss_index is None:
            logger.warning("Embedder or FAISS index not available, falling back to empty tree results")
            return {"high_level_summaries": {}, "leaf_nodes": {}}
            
        # Encode query for similarity search
        try:
            query_embed = self.embedder.encode(query).reshape(1, -1)
        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            return {"high_level_summaries": {}, "leaf_nodes": {}}
        
        # Search in the collapsed tree (which contains all nodes)
        try:
            search_k = min(k, len(self.collapse_tree_ids))
            if search_k == 0:
                return {"high_level_summaries": {}, "leaf_nodes": {}}
                
            _, candidate_indices = self.faiss_index.search(query_embed, k=search_k)
            candidate_indices = candidate_indices[0]
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return {"high_level_summaries": {}, "leaf_nodes": {}}
        
        # Separate results by node type 
        high_level_summaries = {}
        leaf_nodes = {}
        
        for idx in candidate_indices:
            if idx >= len(self.collapse_tree_ids):
                continue  # Skip invalid indices
                
            chunk_id = self.collapse_tree_ids[idx]
            
            if chunk_id.startswith("leaf_"):
                # This is a leaf node - add to leaf_nodes
                leaf_nodes.setdefault("tree_leaf", []).append(chunk_id)
            elif chunk_id.startswith("summary_"):
                # This is a summary node - determine its level
                try:
                    level = int(chunk_id.split("_")[1])
                    if level >= 1:  # Level 1 and above are considered "high-level"
                        high_level_summaries.setdefault(f"high_level_{level}", []).append(chunk_id)
                    else:
                        # Level 0 summaries are closer to leaves
                        leaf_nodes.setdefault("tree_summary", []).append(chunk_id)
                except (IndexError, ValueError):
                    # Invalid chunk_id format, skip
                    logger.warning(f"Invalid chunk_id format: {chunk_id}")
                    continue
        
        return {"high_level_summaries": high_level_summaries, "leaf_nodes": leaf_nodes}
    
    def fuse_and_deduplicate(self, tree_results, graph_results, **kwargs):
        """
        Fuse results from tree and graph retrieval, removing duplicates while preserving all high-level summaries.
        """
        fused_results = {}
        
        # Step 1: Keep ALL high-level summary nodes (these provide indispensable background)
        high_level_summaries = tree_results.get("high_level_summaries", {})
        if high_level_summaries:
            fused_results.update(high_level_summaries)
        
        # Step 2: Collect all leaf-level chunks from both sources
        all_leaf_chunks = set()
        
        # From tree search (leaf nodes)
        tree_leaves = tree_results.get("leaf_nodes", {})
        for key, chunks in tree_leaves.items():
            if isinstance(chunks, list):
                all_leaf_chunks.update(chunks)
        
        # From graph search  
        for key, chunks in graph_results.items():
            if isinstance(chunks, list):
                all_leaf_chunks.update(chunks)
        
        # Step 3: Deduplicate leaf chunks - only keep one copy of each
        if all_leaf_chunks:
            fused_results["deduplicated_leaves"] = list(all_leaf_chunks)
            
        return fused_results
    
    def supplement_with_entityaware_filter(self, current_results, entities, remaining_slots, **kwargs):
        """
        Fill remaining slots using entityaware_filter strategy on candidates not yet selected.
        """
        if remaining_slots <= 0 or not entities:
            return current_results
            
        # Get all currently selected chunks to avoid duplicates
        selected_chunks = set()
        for chunks in current_results.values():
            if isinstance(chunks, list):
                selected_chunks.update(chunks)
        
        # Only proceed if we have embedder and FAISS index
        if self.embedder is None or self.faiss_index is None:
            logger.warning("Cannot supplement chunks: embedder or FAISS index not available")
            return current_results
        
        # Get additional candidates through dense retrieval
        query = kwargs.get("query", "")
        if not query:
            logger.warning("No query provided for supplementary retrieval")
            return current_results
            
        try:
            query_embed = self.embedder.encode(query).reshape(1, -1)
            search_k = min(remaining_slots * 3, len(self.collapse_tree_ids))
            _, candidate_indices = self.faiss_index.search(query_embed, k=search_k)
            candidate_indices = candidate_indices[0]
        except Exception as e:
            logger.error(f"Failed to get supplementary candidates: {e}")
            return current_results
        
        # Filter out already selected chunks and create candidate dict
        candidate_chunks = {}
        for idx in candidate_indices:
            if idx >= len(self.collapse_tree_ids):
                continue
                
            chunk_id = self.collapse_tree_ids[idx] 
            if chunk_id not in selected_chunks:
                # Find which entities this chunk contains
                chunk_entities = []
                for entity in entities:
                    if chunk_id in self.index.get(entity, []):
                        chunk_entities.append(entity)
                
                if chunk_entities:
                    key = "_".join(sorted(chunk_entities))
                    candidate_chunks.setdefault(key, []).append(chunk_id)
        
        # Apply entityaware_filter to get the best remaining chunks
        if candidate_chunks:
            try:
                filtered_additional = self.entityaware_filter(candidate_chunks, entities)
                
                # Limit to remaining slots
                additional_count = 0
                for key, chunks in filtered_additional.items():
                    if additional_count >= remaining_slots:
                        break
                        
                    # Take only what we need
                    chunks_to_take = min(len(chunks), remaining_slots - additional_count)
                    if chunks_to_take > 0:
                        current_results.setdefault(f"supplementary_{key}", []).extend(chunks[:chunks_to_take])
                        additional_count += chunks_to_take
            except Exception as e:
                logger.error(f"Failed to apply entityaware_filter: {e}")
        
        return current_results

    def query(self, query, **kwargs):
        # step 1: extract the Entities from the query.
        # reuse the naive_extract_graph function, which is used in the graph building process.
        entities = self.nlp.naive_extract_graph(query.split("\n")[0])
        entities = entities["nouns"]

        # step 2.0: set up the parameters.
        shortest_path_k = kwargs.get("shortest_path_k", 4)
        
        # Check if hybrid retrieval is enabled (new parameter)
        use_hybrid = kwargs.get("use_hybrid", True)

        # step 2.1: short circuit, if there is no entity, then return the naive dense retrieval.
        if len(entities) == 0:
            chunk_ids = self.dense_retrieval(query, kwargs.get("max_chunk_setting", 25))
            result = {"chunks":self.format_res(chunk_ids)}
            if kwargs.get("debug", True):
                supplement_info = self._build_supplement_info(chunk_ids, entities, chunk_ids, list(chunk_ids.keys()), len(chunk_ids), [])
                result.update(supplement_info)
                result["retrieval_type"] = "Global Search"
            return result

        # NEW: Use hybrid retrieval strategy if enabled
        if use_hybrid:
            hybrid_results = self.hybrid_retrieval(query, entities, query=query, **kwargs)
            result = {"chunks": self.format_res(hybrid_results)}
            if kwargs.get("debug", True):
                supplement_info = self._build_supplement_info(
                    hybrid_results, entities, hybrid_results, 
                    list(hybrid_results.keys()), self._count_chunks(hybrid_results), []
                )
                result.update(supplement_info)
                result["retrieval_type"] = "Hybrid Primary-Auxiliary Collaborative Retrieval"
            return result

        # EXISTING CODE: Original retrieval strategy (fallback)
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
