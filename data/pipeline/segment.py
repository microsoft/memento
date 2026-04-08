#!/usr/bin/env python3
"""
Stage 4: Segmentation
Find optimal breaking points using DP or greedy algorithms.

Uses boundary scores from Stage 3 to segment the text into blocks.
Supports multiple algorithms:
- variance: Maximize avg_score - variance_weight × CV(block_sizes)
- average: Maximize average boundary score
- greedy: Greedily select highest-scoring boundaries

Input: stage3_score/data.jsonl
Output:
- stage4_segment/data.jsonl: Tasks with blocks field [(start, end), ...]
- stage4_segment/examples/*.txt: Human-readable segmentations
"""

import json
import argparse
import threading
from pathlib import Path
from typing import List, Tuple, Optional
import math
import tiktoken


def segment_variance_dp(
    num_sentences: int,
    scores: List[float],
    sentences: List[str],
    min_blocks: int = 3,
    max_blocks: Optional[int] = None,
    max_block_size: Optional[int] = None,
    min_block_tokens: int = 200,
    variance_weight: float = 1.0,
) -> List[Tuple[int, int]]:
    """
    Segment using DP to maximize: avg_score - variance_weight * CV(block_token_sizes)
    where CV = std_dev(block_token_sizes) / mean(block_token_sizes)
    Block sizes are measured in tokens, not sentence counts.
    
    Args:
        min_block_tokens: Minimum number of tokens per block (default: 200)
    """
    
    if max_blocks is None:
        max_blocks = num_sentences
    
    if max_block_size is None:
        max_block_size = num_sentences
    
    # Compute token counts for each sentence using tiktoken (cl100k_base for GPT-4)
    tokenizer = tiktoken.get_encoding('cl100k_base')
    token_counts = [len(tokenizer.encode(sent)) for sent in sentences]
    
    n = num_sentences
    INF = float('-inf')
    
    # dp[i][k] = (best_objective, total_score, sum_sizes, sum_sizes_sq, prev_j)
    dp = [[None for _ in range(max_blocks + 1)] for _ in range(n + 1)]
    
    # Base case: 0 sentences, 0 blocks
    dp[0][0] = (0.0, 0.0, 0, 0, -1)
    
    # Fill DP table
    for i in range(1, n + 1):
        for k in range(1, min(i, max_blocks) + 1):
            best = None
            
            # Try all possible previous block endings
            for j in range(max(0, i - max_block_size), i):
                if dp[j][k-1] is None:
                    continue
                
                prev_obj, prev_total_score, prev_sum_sizes, prev_sum_sizes_sq, _ = dp[j][k-1]
                
                # New block from j to i-1 (inclusive)
                # block_size is the total number of tokens in this block
                block_size = sum(token_counts[j:i])
                
                # Enforce minimum block size constraint
                if block_size < min_block_tokens:
                    continue
                
                # Boundary score is the boundary crossed to enter this block
                # That's the boundary after sentence j-1 (between j-1 and j)
                # For the first block (j=0), there is no boundary, so score is 0
                if j == 0:
                    boundary_score = 0.0
                elif j-1 < len(scores):
                    boundary_score = scores[j-1]
                else:
                    boundary_score = 0.0
                
                # Update cumulative statistics
                new_total_score = prev_total_score + boundary_score
                new_sum_sizes = prev_sum_sizes + block_size
                new_sum_sizes_sq = prev_sum_sizes_sq + block_size * block_size
                
                # Calculate objective
                # avg_score = total_score / num_blocks
                # This is the average score per block (first block contributes 0)
                # As k increases, this approaches the true average boundary score
                avg_score = new_total_score / k
                mean_size = new_sum_sizes / k
                variance = (new_sum_sizes_sq / k) - (mean_size * mean_size)
                std_dev = math.sqrt(max(0, variance))
                cv = std_dev / mean_size if mean_size > 0 else 0
                
                objective = avg_score - variance_weight * cv
                
                if best is None or objective > best[0]:
                    best = (objective, new_total_score, new_sum_sizes, new_sum_sizes_sq, j)
            
            if best is not None:
                dp[i][k] = best
    
    # Find best solution with k in [min_blocks, max_blocks]
    best_k = None
    best_obj = INF
    
    for k in range(min_blocks, max_blocks + 1):
        if dp[n][k] is not None:
            obj = dp[n][k][0]
            if obj > best_obj:
                best_obj = obj
                best_k = k
    
    if best_k is None:
        # Fallback: use all sentences as one block
        return [(0, n - 1)]
    
    # Reconstruct solution
    blocks = []
    i = n
    k = best_k
    
    while k > 0:
        _, _, _, _, j = dp[i][k]
        blocks.append((j, i - 1))
        i = j
        k -= 1
    
    blocks.reverse()
    return blocks


def segment_task(task: dict, sentences: List[str], strategy: str, **kwargs) -> dict:
    """Segment a single task."""
    
    task_id = task['task_id']
    num_sentences = task['num_sentences']
    scores = task['boundary_scores']
    
    print(f"  {task_id}: {num_sentences} sentences, {len(scores)} boundaries")
    
    if strategy == 'variance':
        blocks = segment_variance_dp(
            num_sentences, scores, sentences,
            min_blocks=kwargs.get('min_blocks', 10),
            max_blocks=kwargs.get('max_blocks', 20),
            variance_weight=kwargs.get('variance_weight', 1.0)
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Calculate statistics
    # Compute token counts for reporting
    tokenizer = tiktoken.get_encoding('cl100k_base')
    token_counts = [len(tokenizer.encode(sent)) for sent in sentences]
    
    # Block sizes in sentences (for backward compatibility)
    block_sizes = [end - start + 1 for start, end in blocks]
    
    # Block sizes in tokens (what we actually optimize for)
    block_token_sizes = [sum(token_counts[start:end+1]) for start, end in blocks]
    
    # Average score is the average of boundaries crossed (excluding first block which has no boundary before it)
    # Each block after the first crossed a boundary at position start-1
    if len(blocks) > 1:
        boundary_scores_crossed = [scores[b[0]-1] for b in blocks[1:] if b[0]-1 < len(scores)]
        avg_score = sum(boundary_scores_crossed) / len(boundary_scores_crossed) if boundary_scores_crossed else 0.0
    else:
        avg_score = 0.0
    
    result = {
        'task_id': task_id,
        'num_sentences': num_sentences,
        'num_blocks': len(blocks),
        'blocks': blocks,
        'block_sizes': block_sizes,
        'block_token_sizes': block_token_sizes,
        'avg_score': avg_score
    }
    
    # Preserve metadata if present
    if 'metadata' in task:
        result['metadata'] = task['metadata']
    
    print(f"    -> {len(blocks)} blocks, avg_score={avg_score:.3f}")
    print(f"       Sentence sizes: min={min(block_sizes)}, max={max(block_sizes)}, avg={sum(block_sizes)/len(block_sizes):.1f}")
    print(f"       Token sizes: min={min(block_token_sizes)}, max={max(block_token_sizes)}, avg={sum(block_token_sizes)/len(block_token_sizes):.1f}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Stage 4: Segment text into blocks')
    parser.add_argument('--input-dir', type=str, default='stage3_score',
                        help='Input directory from Stage 3')
    parser.add_argument('--output-dir', type=str, default='stage4_segment',
                        help='Output directory for segmentations')
    parser.add_argument('--strategy', type=str, default='variance',
                        choices=['variance'],
                        help='Segmentation strategy')
    parser.add_argument('--min-blocks', type=int, default=10,
                        help='Minimum number of blocks')
    parser.add_argument('--max-blocks', type=int, default=20,
                        help='Maximum number of blocks')
    parser.add_argument('--variance-weight', type=float, default=1.0,
                        help='Weight for variance penalty (variance strategy)')
    parser.add_argument('--test', action='store_true',
                        help='Save example .txt files for inspection')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.test:
        examples_dir = output_dir / 'examples'
        examples_dir.mkdir(exist_ok=True)
    
    # Load input data
    input_jsonl = input_dir / 'data.jsonl'
    tasks = []
    with open(input_jsonl, 'r') as f:
        for line in f:
            tasks.append(json.loads(line))
    
    print(f"Loaded {len(tasks)} tasks from {input_jsonl}")
    print(f"Segmentation strategy: {args.strategy}")
    
    # Load sentences for examples
    sentence_input = Path(args.input_dir).parent / 'stage2_sentence_split' / 'data.jsonl'
    sentences_map = {}
    with open(sentence_input, 'r') as f:
        for line in f:
            data = json.loads(line)
            sentences_map[data['task_id']] = data['sentences']
    
    # Initialize output file and lock for incremental writes
    output_jsonl = output_dir / 'data.jsonl'
    file_lock = threading.Lock()
    
    # Create empty output file
    with open(output_jsonl, 'w') as f:
        pass
    
    # Process tasks with incremental writes
    results = []
    for task in tasks:
        task_id = task['task_id']
        sentences = sentences_map.get(task_id, [])
        result = segment_task(
            task,
            sentences,
            strategy=args.strategy,
            min_blocks=args.min_blocks,
            max_blocks=args.max_blocks,
            variance_weight=args.variance_weight
        )
        
        # Write result immediately
        with file_lock:
            with open(output_jsonl, 'a') as f:
                f.write(json.dumps(result) + '\n')
        
        results.append(result)
    
    print(f"\nSaved segmentations to {output_jsonl}")
    
    # Create human-readable examples (only in test mode)
    if args.test:
        for result in results:
            task_id = result['task_id']
            blocks = result['blocks']
            sentences = sentences_map[task_id]
            scores = next(t['boundary_scores'] for t in tasks if t['task_id'] == task_id)
            
            example_file = examples_dir / f'{task_id}.txt'
            with open(example_file, 'w') as f:
                f.write(f"Task ID: {task_id}\n")
                f.write(f"Total sentences: {result['num_sentences']}\n")
                f.write(f"Number of blocks: {result['num_blocks']}\n")
                f.write(f"Average boundary score: {result['avg_score']:.3f}\n")
                f.write(f"Sentence sizes: min={min(result['block_sizes'])}, max={max(result['block_sizes'])}, avg={sum(result['block_sizes'])/len(result['block_sizes']):.1f}\n")
                f.write(f"Token sizes: min={min(result['block_token_sizes'])}, max={max(result['block_token_sizes'])}, avg={sum(result['block_token_sizes'])/len(result['block_token_sizes']):.1f}\n")
                f.write("=" * 80 + "\n\n")
                
                for block_idx, (start, end) in enumerate(blocks, 1):
                    size_sentences = end - start + 1
                    size_tokens = result['block_token_sizes'][block_idx - 1]
                    # Boundary score is the score of the boundary crossed to enter this block
                    # That's the boundary after sentence start-1 (between start-1 and start)
                    boundary_score = scores[start-1] if start > 0 and start-1 < len(scores) else None
                    
                    f.write(f"BLOCK {block_idx}/{len(blocks)}\n")
                    f.write(f"Sentences: {start}-{end} | {size_sentences} sents | {size_tokens} tokens")
                    if boundary_score is not None:
                        f.write(f" | Boundary score: {boundary_score:.1f}")
                    f.write("\n")
                    f.write("-" * 80 + "\n")
                    
                    for i in range(start, end + 1):
                        sent = sentences[i]
                        f.write(f"[{i}] {sent}\n")
                    
                    f.write("\n")
            
            print(f"  Saved example: {example_file.name}")
    
    print(f"\nStage 4 complete! Output in {output_dir}/")


if __name__ == '__main__':
    main()
