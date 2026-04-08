#!/usr/bin/env python3
"""
Stage 3: Boundary Scoring
Score boundaries between sentences using an LLM.

For each boundary between sentences, the LLM scores how good a place it is
to break the text for summarization (0=bad, 1=weak, 2=moderate, 3=good).

Input: stage2_sentence_split/data.jsonl
Output:
- stage3_score/data.jsonl: Tasks with boundary_scores field
- stage3_score/examples/*.txt: Human-readable boundary scores
"""

import json
import argparse
import re
from pathlib import Path
from typing import List
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# Prompt paths
PROMPT_DIR = Path(__file__).parent / "prompts"
BOUNDARY_SYSTEM_PROMPT_FILE = PROMPT_DIR / "boundary_score_system_prompt.txt"
BOUNDARY_SCORE_PROMPT_FILE = PROMPT_DIR / "boundary_score_prompt.txt"

from client import get_llm_client


# Load prompts from files
with open(BOUNDARY_SYSTEM_PROMPT_FILE, 'r') as f:
    BOUNDARY_SYSTEM_PROMPT = f.read().strip()

with open(BOUNDARY_SCORE_PROMPT_FILE, 'r') as f:
    SCORING_PROMPT = f.read().strip()


def score_boundaries_batch(sentences: List[str], start_idx: int, end_idx: int, client) -> tuple[List[float], dict]:
    """Score a batch of boundaries using the LLM.
    
    Returns:
        Tuple of (scores, metadata) where metadata contains endpoint info
    """
    
    # Build the text with boundary markers
    text_parts = []
    boundaries_list = []
    
    for i in range(start_idx, end_idx + 1):
        text_parts.append(sentences[i])
        if i < end_idx:
            text_parts.append(f"\n<<<BOUNDARY_{i}>>>\n")
            boundaries_list.append(f"BOUNDARY_{i}: between sentence {i} and {i+1}")
    
    text = ''.join(text_parts)
    boundaries_str = '\n'.join(boundaries_list)
    expected_count = end_idx - start_idx
    
    prompt = SCORING_PROMPT.format(text=text, boundaries=boundaries_str, count=expected_count)
    
    # Retry logic with exponential backoff
    max_retries = 5
    retry_delay = 1.0  # seconds
    
    for attempt in range(max_retries):
        try:
            # Get current model from client (set by get_llm_client)
            model = client.model
            response = client.create_chat_completion(
                messages=[
                    {"role": "system", "content": BOUNDARY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
            )
            
            content = response.choices[0].message.content.strip()
            
            # Check for empty response
            if not content:
                if attempt < max_retries - 1:
                    print(f"  [WARN] Empty response from LLM (attempt {attempt + 1}/{max_retries}), retrying...")
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
                else:
                    print(f"  [ERROR] Empty response after {max_retries} attempts")
                    return [0.0] * expected_count, {'endpoint': 'error', 'model': model, 'boundaries_scored': expected_count}
            
            # Strip markdown code fences (e.g., ```json ... ```) that some models add
            cleaned = content
            if '```' in cleaned:
                match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1).strip()
            
            data = json.loads(cleaned)
            batch_scores = data["scores"]
            
            # Validate score count matches expected
            if len(batch_scores) != expected_count:
                if attempt < max_retries - 1:
                    print(f"  [WARN] Expected {expected_count} scores, got {len(batch_scores)} (attempt {attempt + 1}/{max_retries}), retrying...")
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
                else:
                    print(f"  [ERROR] Score count mismatch after {max_retries} attempts, padding/truncating")
                    batch_scores = list(batch_scores)[:expected_count]
                    while len(batch_scores) < expected_count:
                        batch_scores.append(0.0)
            
            # Success! Return scores and endpoint metadata
            endpoint_url = str(client.client.base_url) if hasattr(client.client, 'base_url') else 'unknown'
            metadata = {
                'endpoint': endpoint_url,
                'model': model,
                'boundaries_scored': expected_count
            }
            return [float(s) for s in batch_scores], metadata
        
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"  [WARN] JSON parse error (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                time.sleep(retry_delay * (2 ** attempt))
                continue
            else:
                print(f"  [ERROR] JSON parse failed after {max_retries} attempts: {e}")
                return [0.0] * expected_count, {'endpoint': 'error', 'model': model, 'boundaries_scored': expected_count}
        
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [WARN] Batch scoring error (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                time.sleep(retry_delay * (2 ** attempt))
                continue
            else:
                print(f"  [ERROR] Batch scoring failed after {max_retries} attempts: {e}")
                return [0.0] * expected_count, {'endpoint': 'error', 'model': client.model, 'boundaries_scored': expected_count}
    
    # Should never reach here, but just in case
    return [0.0] * expected_count, {'endpoint': 'error', 'model': client.model, 'boundaries_scored': expected_count}


def score_task(task: dict, client, use_two_pass: bool = True) -> dict:
    """Score all boundaries in a task.
    
    If use_two_pass=True, scores boundaries using two passes with coprime window sizes
    and averages the results for more robust scoring.
    """
    
    task_id = task['task_id']
    sentences = task['sentences']
    num_boundaries = len(sentences) - 1
    
    print(f"  {task_id}: scoring {num_boundaries} boundaries...")
    
    if num_boundaries == 0:
        return {
            'task_id': task_id,
            'num_sentences': len(sentences),
            'boundary_scores': []
        }
    
    if use_two_pass:
        # Two-pass scoring with coprime window sizes
        print(f"    Using two-pass scoring with coprime windows")
        
        # Pass 1: window size 16
        batch_size_1 = 16
        scores_pass1, metadata_pass1 = score_boundaries_with_window(sentences, num_boundaries, batch_size_1, client, pass_num=1)
        
        # Pass 2: window size 11 (coprime to 16)
        batch_size_2 = 11
        scores_pass2, metadata_pass2 = score_boundaries_with_window(sentences, num_boundaries, batch_size_2, client, pass_num=2)
        
        # Average the two passes
        all_scores = [(s1 + s2) / 2.0 for s1, s2 in zip(scores_pass1, scores_pass2)]
        print(f"    Averaged scores from both passes")
        
        result = {
            'task_id': task_id,
            'num_sentences': len(sentences),
            'boundary_scores': all_scores,
            'boundary_scores_pass1': scores_pass1,
            'boundary_scores_pass2': scores_pass2,
            'scoring_metadata': {
                'pass1_batches': metadata_pass1,
                'pass2_batches': metadata_pass2
            }
        }
        
    else:
        # Single-pass scoring (original behavior)
        batch_size = 16
        all_scores, batch_metadata = score_boundaries_with_window(sentences, num_boundaries, batch_size, client, pass_num=1)
        
        result = {
            'task_id': task_id,
            'num_sentences': len(sentences),
            'boundary_scores': all_scores,
            'scoring_metadata': {
                'batches': batch_metadata
            }
        }
    
    # Preserve metadata if present
    if 'metadata' in task:
        result['metadata'] = task['metadata']
    
    return result


def score_boundaries_with_window(sentences: List[str], num_boundaries: int, batch_size: int, client, pass_num: int) -> tuple[List[float], List[dict]]:
    """Score boundaries using a specific window size.
    
    Returns:
        Tuple of (scores, metadata_list) where:
        - scores: List of averaged scores for all boundaries
        - metadata_list: List of endpoint metadata for each batch
    """
    
    # Track all scores for each boundary (will average multiple scores per boundary)
    boundary_scores = [[] for _ in range(num_boundaries)]
    batch_metadata = []
    
    for batch_start in range(0, num_boundaries, batch_size):
        batch_end = min(batch_start + batch_size, num_boundaries)
        
        # Indices for sentences (need batch_end + 1 sentences to get batch_end boundaries)
        sent_start = batch_start
        sent_end = batch_end  # This gives us boundaries from batch_start to batch_end-1
        
        print(f"    [pass {pass_num}, batch {batch_start//batch_size + 1}/{(num_boundaries-1)//batch_size + 1}] scoring {batch_end - batch_start} boundaries...")
        
        # Get fresh client from pool for each batch to maximize endpoint usage
        batch_client = get_llm_client()
        scores, metadata = score_boundaries_batch(sentences, sent_start, sent_end, batch_client)
        batch_metadata.append(metadata)
        
        # Record scores for each boundary in this batch
        for i, score in enumerate(scores):
            boundary_idx = batch_start + i
            boundary_scores[boundary_idx].append(score)
        
        time.sleep(0.5)  # Rate limiting
    
    # Average scores for each boundary (most will have 1 score, some at window overlaps will have 2+)
    averaged_scores = []
    for boundary_idx, scores in enumerate(boundary_scores):
        if scores:
            avg_score = sum(scores) / len(scores)
            averaged_scores.append(avg_score)
        else:
            # Should not happen, but handle gracefully
            print(f"    [WARN] Boundary {boundary_idx} has no scores, defaulting to 0.0")
            averaged_scores.append(0.0)
    
    return averaged_scores, batch_metadata


def main():
    parser = argparse.ArgumentParser(description='Stage 3: Score boundaries between sentences')
    parser.add_argument('--input-dir', type=str, default='stage2_sentence_split',
                        help='Input directory from Stage 2')
    parser.add_argument('--output-dir', type=str, default='stage3_score',
                        help='Output directory for boundary scores')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--two-pass', action='store_true',
                        help='Use two-pass scoring with coprime windows (16 and 11) and average results')
    parser.add_argument('--test', action='store_true',
                        help='Generate example .txt files (for testing/debugging)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = output_dir / 'examples'
    examples_dir.mkdir(exist_ok=True)
    
    # Load input data
    input_jsonl = input_dir / 'data.jsonl'
    tasks = []
    with open(input_jsonl, 'r') as f:
        for line in f:
            tasks.append(json.loads(line))
    
    print(f"Loaded {len(tasks)} tasks from {input_jsonl}")
    if args.two_pass:
        print(f"Using two-pass scoring with coprime windows (16, 11)")
    print(f"Starting scoring with {args.workers} workers...")
    
    # Create output file and lock for thread-safe incremental writes
    output_jsonl = output_dir / 'data.jsonl'
    file_lock = threading.Lock()
    results = []  # Keep for stats and example generation
    
    # Initialize empty output file
    with open(output_jsonl, 'w') as f:
        pass
    
    def process_and_save_task(task, client, use_two_pass):
        """Process a task and immediately append to output file."""
        result = score_task(task, client, use_two_pass=use_two_pass)
        
        # Thread-safe append to JSONL
        with file_lock:
            with open(output_jsonl, 'a') as f:
                f.write(json.dumps(result) + '\n')
        
        return result
    
    # Process tasks
    if args.workers == 1:
        # Sequential processing
        client = get_llm_client()
        for task in tasks:
            result = process_and_save_task(task, client, args.two_pass)
            results.append(result)
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_and_save_task, task, get_llm_client(), args.two_pass) for task in tasks]
            for future in futures:
                results.append(future.result())
    
    print(f"\nSaved scores to {output_jsonl}")
    
    # Create human-readable examples (only in test mode)
    if args.test:
        print(f"\nGenerating example .txt files (--test mode)...")
        # Load sentences for context
        sentences_map = {t['task_id']: t['sentences'] for t in tasks}
        
        for result in results:
            task_id = result['task_id']
            scores = result['boundary_scores']
            sentences = sentences_map[task_id]
            
            example_file = examples_dir / f'{task_id}.txt'
            with open(example_file, 'w') as f:
                f.write(f"Task ID: {task_id}\n")
                f.write(f"Sentences: {len(sentences)}\n")
                f.write(f"Boundaries: {len(scores)}\n")
                f.write("=" * 80 + "\n\n")
                
                for i, sent in enumerate(sentences):
                    f.write(f"[{i}] {sent[:100]}{'...' if len(sent) > 100 else ''}\n")
                    
                    if i < len(scores):
                        score = scores[i]
                        f.write(f"    └─ BOUNDARY SCORE: {score}\n")
                    
                    f.write("\n")
            
            print(f"  Saved example: {example_file.name}")
    
    # Print statistics
    all_scores = [s for r in results for s in r['boundary_scores']]
    if all_scores:
        import numpy as np
        print(f"\nScore statistics:")
        print(f"  Mean: {np.mean(all_scores):.3f}")
        print(f"  Std: {np.std(all_scores):.3f}")
        print(f"  Distribution: 0={sum(s==0 for s in all_scores)}, 1={sum(s==1 for s in all_scores)}, 2={sum(s==2 for s in all_scores)}, 3={sum(s==3 for s in all_scores)}")
    
    print(f"\nStage 3 complete! Output in {output_dir}/")


if __name__ == '__main__':
    main()
