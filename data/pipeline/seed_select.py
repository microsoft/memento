#!/usr/bin/env python3
"""
Stage 1: Seed Selection
Select initial prompts + responses from OpenThoughts dataset.

Currently uses a fixed set of 3 examples, but can be extended to select
a random subset or apply other selection criteria.

Output:
- {root_dir}/datasets/openthoughts-{args.num_questions} -> subset dataset can be loaded with datasets.load_from_disk
- {root_dir}/openthoughts/{args.output_dir}/data.jsonl -> JSONL file of selected seeds
- {root_dir}/openthoughts/{args.output_dir}/examples/ -> directory for human-readable txt files of sampled reasonings
"""

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import json, random, argparse, os
import numpy as np
from collections import defaultdict

# Use relative path for root_dir
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Only load dataset if not in test mode (will be checked in main)
ds = None
tokenizer = None

def get_reasoning(index):
    return ds[index]["conversations"][1]["value"]

def get_question(index):
    return ds[index]["conversations"][0]["value"]

def check_chinese_in_text(text):
    """
    Returns True if the input string contains any Chinese character.
    """
    for ch in text:
        # Common CJK Unified Ideographs: U+4E00...U+9FFF
        # This covers most Chinese, Japanese, and Korean, but is mostly Chinese in practice
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def compute_num_tokens(text):
    return len(tokenizer.encode(text))

def is_think_end(text):
    return "</think>" in text

def main():
    global ds, tokenizer
    
    parser = argparse.ArgumentParser(description='Stage 1: Select seed examples from OpenThoughts')
    parser.add_argument('--output-dir', type=str, default='stage1_seed_select',
                        help='Output directory for selected seeds')
    parser.add_argument('--num-questions', type=int, default=625,
                        help='Number of questions to draw')
    parser.add_argument("--readable-num-samples", type=int, default=10,
                        help="Number of samples for human to read")
    parser.add_argument('--dataset-dir', type=str, default=None,
                        help='Path to pre-downloaded OpenThoughts dataset directory (avoids re-downloading)')
    parser.add_argument('--test', action='store_true',
                        help='Use test files from test_file/ directory instead of downloading dataset')
    args = parser.parse_args()

    output_dir = f"{root_dir}/{args.output_dir}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/examples", exist_ok=True)
    
    # If test mode, use local test files
    if args.test:
        print("Test mode: Using local test files from test_file/")
        
        test_files = [
            f'{root_dir}/test_file/test_cot_1.txt',
            f'{root_dir}/test_file/test_cot_2.txt',
            f'{root_dir}/test_file/test_cot_3.txt'
        ]
        
        # Create data.jsonl from test files
        with open(f"{output_dir}/data.jsonl", "w", encoding="utf-8") as f:
            for test_file in test_files:
                with open(test_file, 'r') as tf:
                    content = tf.read()
                
                # Extract task ID
                task_id_line = [line for line in content.split('\n') if line.startswith('TASK ID:')][0]
                task_id = task_id_line.split('TASK ID:')[1].strip()
                
                # Extract source
                source_line = [line for line in content.split('\n') if line.startswith('SOURCE:')][0]
                source = source_line.split('SOURCE:')[1].strip()
                
                # Extract domain
                domain_line = [line for line in content.split('\n') if line.startswith('DOMAIN:')][0]
                domain = domain_line.split('DOMAIN:')[1].strip()
                
                # Extract problem
                problem_start = content.find('PROBLEM:\n')
                problem_end = content.find('────────────────────────────────────────────────────────────────────────────────\nCHAIN OF THOUGHT:')
                problem = content[problem_start+9:problem_end].strip()
                
                # Extract COT (everything after "CHAIN OF THOUGHT:" divider)
                cot_start = content.find('────────────────────────────────────────────────────────────────────────────────\n\n')
                if cot_start != -1:
                    cot = content[cot_start:].split('\n\n', 1)[1].strip()
                else:
                    cot_start = content.find('CHAIN OF THOUGHT:')
                    cot = content[cot_start:].split('\n', 2)[2].strip()
                
                entry = {
                    "task_id": task_id,
                    "cot": cot,
                    "prompt": problem,
                    "source": source,
                    "domain": domain,
                    "difficulty": ""
                }
                f.write(json.dumps(entry) + "\n")
                
                # Copy test file to examples directory with task_id as filename
                with open(f"{output_dir}/examples/{task_id}.txt", "w", encoding="utf-8") as out:
                    out.write(content)
        
        print(f"\nStage 1 complete! Output in {output_dir}/")
        print(f"  - JSONL file saved: {output_dir}/data.jsonl (3 test samples)")
        print(f"  - Example files copied to examples/ directory")
        return
    
    # Load dataset and tokenizer
    # For small samples, only process a small random subset to speed up processing
    if args.num_questions < 100:
        # Sample size: aim for ~50x the requested questions to ensure diversity after filtering
        sample_size = min(args.num_questions * 50, 5000)
        print(f"Small sample requested ({args.num_questions} questions), loading random {sample_size} examples...")
        
        if args.dataset_dir and os.path.exists(args.dataset_dir):
            print(f"Loading from {args.dataset_dir}...")
            ds_full = load_dataset(args.dataset_dir, split="train")
            
            # Randomly sample indices
            random.seed(42)  # For reproducibility
            total_examples = len(ds_full)
            selected_indices = sorted(random.sample(range(total_examples), sample_size))
            
            print(f"Selecting {len(selected_indices)} random examples from {total_examples} total...")
            ds = ds_full.select(selected_indices)
        else:
            print("Downloading OpenThoughts dataset...")
            ds = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train")
            # Sample after loading
            random.seed(42)
            total_examples = len(ds)
            selected_indices = sorted(random.sample(range(total_examples), sample_size))
            ds = ds.select(selected_indices)
    else:
        if args.dataset_dir and os.path.exists(args.dataset_dir):
            print(f"Loading full dataset from {args.dataset_dir}...")
            ds = load_dataset(args.dataset_dir, split="train")
        else:
            print("Downloading OpenThoughts dataset...")
            ds = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train")
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/QwQ-32B")

    if os.path.exists(os.path.join(output_dir, "dataset_dict.json")):
        with open(os.path.join(output_dir, "dataset_dict.json"), "r", encoding="utf-8") as f_in:
            dataset_dict = json.load(f_in)
    else:

        dataset_dict = {}

        for i in tqdm(range(len(ds))):    
            question = get_question(i)
            reasoning = get_reasoning(i)
            if not ds[i]["difficulty"] in dataset_dict: 
                dataset_dict[ds[i]["difficulty"]] = {}
            if not ds[i]["source"] in dataset_dict[ds[i]["difficulty"]]: 
                dataset_dict[ds[i]["difficulty"]][ds[i]["source"]] = {}
            if not ds[i]["domain"] in dataset_dict[ds[i]["difficulty"]][ds[i]["source"]]: 
                dataset_dict[ds[i]["difficulty"]][ds[i]["source"]][ds[i]["domain"]] = {}
            if not question in dataset_dict[ds[i]["difficulty"]][ds[i]["source"]][ds[i]["domain"]]: 
                # 1 means keep, 0 means abandon
                dataset_dict[ds[i]["difficulty"]][ds[i]["source"]][ds[i]["domain"]][question] = {"ids": [], "valid": 1}

            if dataset_dict[ds[i]["difficulty"]][ds[i]["source"]][ds[i]["domain"]][question]["valid"]:
                if check_chinese_in_text(reasoning) or (not is_think_end(reasoning)):
                    dataset_dict[ds[i]["difficulty"]][ds[i]["source"]][ds[i]["domain"]][question]["valid"] = 0
                
                dataset_dict[ds[i]["difficulty"]][ds[i]["source"]][ds[i]["domain"]][question]["ids"].append(i)

        # Save dataset_dict to a JSON file for inspection
        with open(os.path.join(output_dir, "dataset_dict.json"), "w", encoding="utf-8") as f_out:
            json.dump(dataset_dict, f_out, ensure_ascii=False, indent=2)
            
    # Target domain ratios (for 10K examples = 625 questions × 16 answers)
    # Math: 66.7%, Code: 21.3%, Science: 12.0%
    target_domain_ratios = {
        "math": 0.667,
        "code": 0.213,
        "science": 0.120
    }
    
    total_samples = args.num_questions  # 625 questions
    answers_per_question = 16
    
    # Filter questions where ALL 16 answers pass quality tests
    all_questions = []
    
    for difficulty in dataset_dict:
        for source in dataset_dict[difficulty]:
            for domain in dataset_dict[difficulty][source]:
                for question in dataset_dict[difficulty][source][domain]:
                    q_obj = dataset_dict[difficulty][source][domain][question]
                    if not q_obj["valid"]:
                        continue
                    
                    # Check if we have at least 16 answers
                    if len(q_obj["ids"]) < answers_per_question:
                        continue
                    
                    # Check if ALL first 16 answers pass quality filters
                    all_valid = True
                    for idx in q_obj["ids"][:answers_per_question]:
                        reasoning = get_reasoning(idx)
                        if check_chinese_in_text(reasoning) or not is_think_end(reasoning):
                            all_valid = False
                            break
                    
                    if all_valid:
                        all_questions.append({
                            "difficulty": difficulty,
                            "source": source,
                            "domain": domain,
                            "question": question,
                            "ids": q_obj["ids"][:answers_per_question]
                        })
    
    # Group questions by domain
    domain_questions = defaultdict(list)
    for q in all_questions:
        domain_questions[q["domain"]].append(q)
    
    # Calculate target questions per domain
    target_questions_per_domain = {}
    for domain, ratio in target_domain_ratios.items():
        target_questions_per_domain[domain] = int(total_samples * ratio)
    
    # Adjust if we don't have enough questions in a domain
    actual_questions_per_domain = {}
    domains_to_adjust = []
    total_deficit = 0
    
    for domain in target_domain_ratios.keys():
        available = len(domain_questions[domain])
        target = target_questions_per_domain[domain]
        
        if available < target:
            print(f"Warning: Domain '{domain}' has only {available} valid questions (target: {target})")
            actual_questions_per_domain[domain] = available
            total_deficit += (target - available)
        else:
            actual_questions_per_domain[domain] = target
            domains_to_adjust.append(domain)
    
    # Distribute deficit among domains that have excess
    if total_deficit > 0 and domains_to_adjust:
        print(f"Distributing deficit of {total_deficit} questions among domains with excess...")
        # Distribute proportionally among domains with excess
        excess_total = sum(len(domain_questions[d]) - actual_questions_per_domain[d] for d in domains_to_adjust)
        
        for domain in domains_to_adjust:
            available = len(domain_questions[domain])
            current = actual_questions_per_domain[domain]
            excess = available - current
            
            if excess > 0:
                additional = int((excess / excess_total) * total_deficit)
                actual_questions_per_domain[domain] = min(current + additional, available)
    
    # Final adjustment to reach exactly total_samples
    total_assigned = sum(actual_questions_per_domain.values())
    if total_assigned < total_samples:
        # Add remaining to domain with most excess
        for domain in sorted(domains_to_adjust, key=lambda d: len(domain_questions[d]) - actual_questions_per_domain[d], reverse=True):
            available = len(domain_questions[domain])
            current = actual_questions_per_domain[domain]
            can_add = min(available - current, total_samples - total_assigned)
            if can_add > 0:
                actual_questions_per_domain[domain] += can_add
                total_assigned += can_add
                if total_assigned >= total_samples:
                    break
    
    print(f"\nFinal domain distribution:")
    for domain in sorted(target_domain_ratios.keys()):
        target = target_questions_per_domain[domain]
        actual = actual_questions_per_domain[domain]
        available = len(domain_questions[domain])
        print(f"  {domain}: {actual} questions (target: {target}, available: {available})")
    
    # Stratified sampling within each domain by (difficulty, source)
    random.seed(42)
    final_samples = []
    
    for domain in target_domain_ratios.keys():
        domain_q_list = domain_questions[domain]
        n_to_sample = actual_questions_per_domain[domain]
        
        if n_to_sample == 0:
            continue
        
        # Group by (difficulty, source) within this domain
        domain_groups = defaultdict(list)
        for q in domain_q_list:
            key = (q["difficulty"], q["source"])
            domain_groups[key].append(q)
        
        # Calculate proportional samples per group
        total_in_domain = len(domain_q_list)
        group_keys = list(domain_groups.keys())
        samples_per_group = {}
        cumulative = 0
        
        for idx, key in enumerate(group_keys):
            group_count = len(domain_groups[key])
            if idx == len(group_keys) - 1:
                # Last group gets remainder
                samples_per_group[key] = n_to_sample - cumulative
            else:
                n = round(group_count / total_in_domain * n_to_sample)
                samples_per_group[key] = min(n, group_count)
                cumulative += samples_per_group[key]
        
        # Sample from each group
        for key in group_keys:
            group_list = domain_groups[key]
            sample_n = samples_per_group[key]
            if sample_n > len(group_list):
                sample_n = len(group_list)
            
            if sample_n > 0:
                group_sample = random.sample(group_list, sample_n)
                final_samples.extend(group_sample)
    
    # Collect all IDs (16 answers per question)
    ids = []
    for sample in final_samples:
        ids.extend(sample["ids"])


    # Check any output is too long
    for i, index in enumerate(ids):
        reasoning = get_reasoning(index)
        num_tokens = len(tokenizer.encode(reasoning))
        if num_tokens > 20000:
            print(1)

    # Select subset and add original index to each record
    subset_ds = ds.select(ids)
    
    # Add original dataset index to each record for traceability
    def add_index(example, idx):
        example['original_index'] = ids[idx]
        return example
    
    subset_ds = subset_ds.map(add_index, with_indices=True)
    subset_ds.save_to_disk(f"{root_dir}/datasets/openthoughts-{args.num_questions}")
    
    # Count actual examples per domain in final dataset
    domain_counts = defaultdict(int)
    for index in ids:
        domain_counts[ds[index].get("domain", "")] += 1
    
    print(f"\nActual examples distribution (total {len(ids)} examples):")
    total_examples = len(ids)
    for domain in sorted(target_domain_ratios.keys()):
        count = domain_counts[domain]
        percentage = (count / total_examples * 100) if total_examples > 0 else 0
        target_pct = target_domain_ratios[domain] * 100
        print(f"  {domain}: {count} examples ({percentage:.1f}%, target: {target_pct:.1f}%)")
    
    with open(f"{output_dir}/data.jsonl", "w", encoding="utf-8") as f:
        for i, index in enumerate(ids):
            # Extract task_id from the dataset (OpenThoughts doesn't have 'id', use index)
            task_id = ds[index].get("id", f"ot3-train-{index:07d}")
            reasoning = get_reasoning(index)
            question = get_question(index)
            
            # Create entry compatible with sentence_split.py, preserving all original fields
            entry = {
                "task_id": task_id,
                "original_index": index,  # Store original dataset index
                "cot": reasoning,
                "prompt": question,
                "source": ds[index].get("source", ""),
                "domain": ds[index].get("domain", ""),
                "difficulty": ds[index].get("difficulty", ""),
            }
            # Preserve any other fields from original dataset
            for key in ds[index].keys():
                if key not in entry and key != 'conversations':
                    entry[key] = ds[index][key]
            
            f.write(json.dumps(entry) + "\n")
    
    print(f"\nStage 1 complete!")
    print(f"  - HuggingFace dataset saved in {root_dir}/datasets/openthoughts-{args.num_questions}")
    print(f"  - JSONL file saved: {output_dir}/data.jsonl ({len(ids)} samples)")

    # Save args.readable_num_samples sampled reasonings into a human-readable txt file
    sample_size = min(args.readable_num_samples, len(ids))
    sampled_ids = random.sample(ids, sample_size)

    for i, index in enumerate(sampled_ids):
        content = f"""
================================================================================
TASK ID: {ds[index].get("id", f"ot3-train-{index:07d}")}
SOURCE: {ds[index].get("source", "")}
DOMAIN: {ds[index].get("domain", "")}
DIFFICULTY: {ds[index].get("difficulty", "")}
================================================================================

PROBLEM:
{get_question(index)}

────────────────────────────────────────────────────────────────────────────────
CHAIN OF THOUGHT:
────────────────────────────────────────────────────────────────────────────────
{get_reasoning(index)}
"""

        with open(f"{output_dir}/examples/ot3-train-{index:07d}.txt", "w", encoding="utf-8") as f:
            f.write(content)

    print(f"{sample_size} sampled reasonings saved to {output_dir}/examples/")

if __name__ == '__main__':
    main()
