"""
Main script for Dual-Index GraphRAG
Integrates dual-index builder and dual-path retriever into the pipeline
"""

import multiprocessing as mp
from extract_graph import load_nlp
from utils import Timer, sequential_split, logger
import yaml
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from prompt_dict import Prompts
import os
import json
import numpy as np
import traceback
import sys
import argparse
import time
from process_utils import build_tree_task, clean_cuda_memory
import gc
from datetime import datetime
from utils import load_dataset
from dual_index_builder import build_dual_index
from dual_retriever import create_dual_retriever


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    return config


def parallel_build_dual_index(text, configs, cache_folder, length, overlap, merge_num):
    """
    Build dual-index in parallel: summary tree + concept graph with sentence-level indexing
    """
    timer = Timer()
    device_id = int(configs["llm"]["llm_device"].split(':')[1]) if ':' in configs["llm"]["llm_device"] else 0
    
    try:
        with timer.timer("total"):
            logger.info("Starting dual-index construction...")
            
            # Build summary tree (existing functionality)
            build_args = (
                configs["llm"]["llm_path"],
                configs["llm"]["llm_device"],
                text,
                cache_folder,
                configs["llm"]["llm_path"],
                length,
                overlap,
                merge_num,
                torch.float16,
                configs.get("language", "en")
            )
            
            logger.info("Building summary tree...")
            tree, build_time_cost = build_tree_task(build_args)
            logger.info(f"Summary tree built in {build_time_cost}s")
            
            # Build dual-index (concept graph + sentence indexing)
            logger.info("Building dual-index (concept graph + sentence indexing)...")
            dual_index_start = time.time()
            
            G, concept_to_sentences, sentence_to_chunk, concept_vectors, sentences = build_dual_index(
                chunks=text,
                cache_folder=cache_folder,
                embedder_model=configs["retriever"]["kwargs"].get("embedder", "BAAI/bge-m3"),
                device=configs["retriever"]["kwargs"]["device"],
                language=configs.get("language", "en"),
                use_cache=not configs["cluster"].get("force_Reextract", False)
            )
            
            dual_index_time = time.time() - dual_index_start
            logger.info(f"Dual-index built in {dual_index_time}s")
            
    except Exception as e:
        logger.error(f"Error occurred in parallel_build_dual_index: {e}")
        logger.error("traceback:")
        logger.error(traceback.format_exc())
        raise e
    
    finally:
        clean_cuda_memory(device_id)
        gc.collect()
    
    logger.info("-" * 15)
    logger.info(f"total time: {timer['total']} seconds")
    logger.info(f"tree build time: {build_time_cost} seconds")
    logger.info(f"dual-index build time: {dual_index_time} seconds")
    logger.info("-" * 15)
    
    if dual_index_time != -1 and build_time_cost != -1:
        with open(os.path.join(cache_folder, "time_cost.txt"), "w") as f:
            f.write(f"total time: ||{timer['total']}|| seconds\n")
            f.write(f"tree build time: ||{build_time_cost}|| seconds\n")
            f.write(f"dual-index build time: ||{dual_index_time}|| seconds\n")
    
    return tree, (G, concept_to_sentences, sentence_to_chunk, concept_vectors, sentences)


def main():
    try:
        date = datetime.now().strftime("%Y%m%d")
        configs = parse_args()
        device_id = int(configs["llm"]["llm_device"].split(':')[1]) if ':' in configs["llm"]["llm_device"] else 0
        
        # Load dataset
        dataset = load_dataset(configs["dataset"]["dataset_name"], configs["dataset"]["dataset_path"])
        
        # Load tokenizer for text splitting
        tokenizer = AutoTokenizer.from_pretrained(configs["llm"]["llm_path"])
        
        try:
            for i, data_piece in enumerate(dataset):
                if i < configs["resume"]["resumeIndex"]:
                    continue
                
                text = data_piece["book"]
                if configs.get("split_method", "sequential") == "sequential":
                    text = sequential_split(text, tokenizer, configs["cluster"]["length"], configs["cluster"]["overlap"])
                elif configs.get("split_method", "sequential") == "nn":
                    logger.info("split_method: nn")
                    text = text.split("\n\n")
                qa = data_piece["qa"]
                
                piece_name = dataset.available_ids[i]
                cache_folder = os.path.join(configs["paths"]["cache_path"], configs["dataset"]["dataset_name"], str(piece_name))
                if not os.path.exists(cache_folder):
                    os.makedirs(cache_folder)
                
                # Build dual-index
                tree, dual_index = parallel_build_dual_index(
                    text, configs, cache_folder,
                    configs["cluster"]["length"], configs["cluster"]["overlap"],
                    configs["cluster"]["merge_num"]
                )
                
                G, concept_to_sentences, sentence_to_chunk, concept_vectors, sentences = dual_index
                
                # Load model for QA
                if configs["dataset"]["dataset_name"] == "NovelQA" or configs["dataset"]["dataset_name"] == "InfiniteChoice":
                    if "Qwen2" in configs["llm"]["llm_path"]:
                        from transformers import Qwen2ForCausalLM
                        llm = Qwen2ForCausalLM.from_pretrained(
                            configs["llm"]["llm_path"],
                            torch_dtype=torch.bfloat16,
                            low_cpu_mem_usage=True
                        )
                    else:
                        llm = AutoModelForCausalLM.from_pretrained(
                            configs["llm"]["llm_path"],
                            torch_dtype=torch.bfloat16
                        )
                    llm.eval()
                    llm.to(configs["llm"]["llm_device"])
                elif configs["dataset"]["dataset_name"] == "InfiniteQALoader":
                    llm = pipeline("text-generation", model=configs["llm"]["llm_path"], 
                                 tokenizer=tokenizer, device=configs["llm"]["llm_device"])
                else:
                    raise ValueError("Invalid dataset")
                
                try:
                    # Create dual-path retriever
                    if "retriever" not in locals():
                        retriever = create_dual_retriever(
                            cache_tree=tree,
                            concept_graph=G,
                            concept_to_sentences=concept_to_sentences,
                            sentence_to_chunk=sentence_to_chunk,
                            concept_vectors=concept_vectors,
                            sentences=sentences,
                            embedder_model=configs["retriever"]["kwargs"].get("embedder", "BAAI/bge-m3"),
                            device=configs["retriever"]["kwargs"]["device"],
                            concept_top_k=configs["retriever"]["kwargs"].get("concept_top_k", 20),
                            sentence_top_k=configs["retriever"]["kwargs"].get("sentence_top_k", 30),
                            tree_top_k=configs["retriever"]["kwargs"].get("tree_top_k", 25),
                            concept_threshold=configs["retriever"]["kwargs"].get("concept_threshold", 0.6)
                        )
                    else:
                        retriever.update(tree, G, concept_to_sentences, sentence_to_chunk, 
                                       concept_vectors, sentences)
                    
                    res = []
                    os.makedirs(configs["paths"]["answer_path"], exist_ok=True)
                    
                    # Answer questions
                    for j, qa_piece in enumerate(qa):
                        question = qa_piece["question"]
                        answer = qa_piece["answer"]
                        
                        try:
                            query_start_time = time.time()
                            model_supplement = retriever.query(question, 
                                                              debug=configs["retriever"]["kwargs"].get("debug", True))
                            query_end_time = time.time()
                            
                            with open(os.path.join(configs["paths"]["answer_path"], f"{date}_query_time.txt"), "a") as f:
                                f.write(f"question {i}-{j}: query time: {query_end_time - query_start_time}\n")
                            
                            evidences = model_supplement["chunks"]
                            logger.info(f"Retrieved {model_supplement.get('total_count', 0)} total chunks")
                            logger.info(f"Path A: {model_supplement.get('path_a_count', 0)} chunks")
                            logger.info(f"Path B: {model_supplement.get('path_b_count', 0)} nodes")
                            
                        except Exception as e:
                            logger.error(f"Error occurred: {e}")
                            logger.error("traceback:")
                            logger.error(traceback.format_exc())
                            raise e
                        
                        if configs["dataset"]["dataset_name"] == "NovelQA" or configs["dataset"]["dataset_name"] == "InfiniteChoice":
                            input_text = Prompts["QA_prompt_options"].format(question=question, evidence=evidences)
                            try:
                                inputs = tokenizer(input_text, return_tensors="pt").to(configs["llm"]["llm_device"])
                                with torch.no_grad():
                                    logger.info(f"inputs token length: {inputs.input_ids.shape[-1]}")
                                    output_logits = llm(**inputs).logits[0, -1]
                            except Exception as e:
                                logger.error(f"Error occurred: {e}")
                                logger.error("traceback:")
                                logger.error(traceback.format_exc())
                                raise e
                            finally:
                                clean_cuda_memory(device_id)
                            
                            probs = torch.nn.functional.softmax(
                                torch.tensor([
                                    output_logits[tokenizer("A").input_ids[-1]],
                                    output_logits[tokenizer("B").input_ids[-1]],
                                    output_logits[tokenizer("C").input_ids[-1]],
                                    output_logits[tokenizer("D").input_ids[-1]],
                                ]).float(),
                                dim=0,
                            ).detach().cpu().numpy()
                            output_text = ["A", "B", "C", "D"][np.argmax(probs)]
                        
                        elif configs["dataset"]["dataset_name"] == "InfiniteQALoader":
                            if configs.get("language", "en") == "zh":
                                input_text = Prompts["QA_prompt_answer_zh"].format(question=question,
                                                                                  evidence=evidences)
                            else:
                                input_text = Prompts["QA_prompt_answer"].format(question=question,
                                                                               evidence=evidences)
                            logger.info(f"input_text length: {len(input_text)}")
                            output = llm(input_text, max_new_tokens=300)
                            output_text = output[0]["generated_text"]
                            output_text = output_text[len(input_text):]
                            logger.info(f"output_text: {output_text}")
                        else:
                            raise ValueError("Invalid dataset")
                        
                        res.append({
                            "question": question,
                            "answer": answer,
                            "output_text": output_text,
                            "evidences": qa_piece.get("evidence", None)
                        })
                    
                    os.makedirs(configs["paths"]["answer_path"], exist_ok=True)
                    os.makedirs(os.path.join(configs["paths"]["answer_path"], configs["dataset"]["dataset_name"]), exist_ok=True)
                    
                    # Save results
                    res_path = os.path.join(configs["paths"]["answer_path"], configs["dataset"]["dataset_name"], f"book_{i}.json")
                    with open(res_path, "w") as f:
                        json.dump(res, f, indent=4)
                
                except Exception as e:
                    logger.error(f"Error occurred during QA processing: {e}")
                    logger.error("traceback:")
                    logger.error(traceback.format_exc())
                    logger.error(f"TODO: Error occurred during book {i} processing. Set resumeIndex to {i}.")
                    raise e
                finally:
                    if 'llm' in locals():
                        del llm
                        logger.info("llm deleted")
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
        
        except Exception as e:
            logger.error(f"Error occurred during dataset processing: {e}")
            logger.error("traceback:")
            logger.error(traceback.format_exc())
            raise e
        finally:
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    except Exception as e:
        logger.error(f"Error occurred in main: {e}")
        logger.error("traceback:")
        logger.error(traceback.format_exc())
        clean_cuda_memory(device_id)
        raise e


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    try:
        main()
    except Exception as e:
        logger.error(f"Program terminated with error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
