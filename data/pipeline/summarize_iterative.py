#!/usr/bin/env python3
"""
Stage 5: Summarization with Iterative Refinement
Generate summaries for each block with LLM-judge feedback loop.

Uses an LLM to create concise summaries, then judges quality and refines up to MAX_ITERATIONS times.

Input: stage4_segment/data.jsonl + stage2_sentence_split/data.jsonl
Output:
- stage5_summarize/data.jsonl: Tasks with summaries field
- stage5_summarize/examples/*.txt: Human-readable final results
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import os
import re
import time
import tiktoken
import threading


def retry_with_backoff(func, max_retries=5, initial_delay=1.0):
    """
    Retry a function with exponential backoff on 429 errors.
    
    Args:
        func: Callable that may raise exceptions
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds (doubles each retry)
    
    Returns:
        Result of func() if successful
    
    Raises:
        Last exception if all retries exhausted
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_str = str(e)
            
            # Check if it's a 429 rate limit error
            if '429' in error_str or 'rate limit' in error_str.lower() or 'token limit' in error_str.lower():
                if attempt < max_retries:
                    # Extract wait time from error message if available
                    wait_time = delay
                    if 'Try again in' in error_str:
                        try:
                            # Extract seconds from message like "Try again in 46 seconds"
                            match = re.search(r'Try again in (\d+) seconds?', error_str)
                            if match:
                                suggested_wait = int(match.group(1))
                                wait_time = max(wait_time, suggested_wait)
                        except:
                            pass
                    
                    print(f"  [RETRY] Rate limit hit, waiting {wait_time:.1f}s before retry {attempt+1}/{max_retries}")
                    time.sleep(wait_time)
                    delay *= 2  # Exponential backoff
                    continue
            
            # For non-429 errors, raise immediately
            raise
    
    # All retries exhausted
    raise last_exception


# Import client pool for multi-endpoint load balancing
from client import get_llm_client


# Load prompts
PROMPT_DIR = Path(__file__).parent / "prompts"
SUMMARY_PROMPT_FILE = PROMPT_DIR / "summary_prompt.txt"
JUDGE_PROMPT_FILE = PROMPT_DIR / "judge_prompt.txt"


def load_prompt_pair(prompt_file: Path) -> tuple[str, str]:
    """Load system and user prompts from a prompt file."""
    text = prompt_file.read_text(encoding="utf-8")
    parts = re.split(r"=+\s*USER Prompt\s*=*", text, maxsplit=1, flags=re.IGNORECASE)
    system_prompt = parts[0].strip()
    user_template = parts[1].strip() if len(parts) > 1 else ""
    return system_prompt, user_template


SUMMARY_SYSTEM, SUMMARY_USER_TEMPLATE = load_prompt_pair(SUMMARY_PROMPT_FILE)
JUDGE_SYSTEM, JUDGE_USER_TEMPLATE = load_prompt_pair(JUDGE_PROMPT_FILE)


class DefaultFormatDict(dict):
    """Dict that returns missing keys as {key} for format_map."""
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def render_user_prompt(template: str, **kwargs) -> str:
    """Render user prompt with safe placeholder handling."""
    fill = DefaultFormatDict(kwargs)
    if not template:
        return kwargs.get('full_trace_with_blocks', '')
    try:
        return template.format_map(fill)
    except ValueError:
        # Fallback to simple replacement if braces confuse format_map
        out = template
        for k, v in fill.items():
            out = out.replace("{" + k + "}", v)
        return out


# Regex to parse "Summary N: ..." format
SUMMARY_BLOCK_RE = re.compile(
    r"Summary\s+(\d+):\s*(.*?)\s*(?=\n\nSummary\s+\d+:|\[\[\s*##\s*completed\s*##\s*\]\]|$)",
    re.IGNORECASE | re.DOTALL,
)


def parse_summary_blocks(text: str) -> List[str]:
    """Parse summaries from LLM response."""
    cleaned = re.sub(r"\[\[\s*##\s*summaries\s*##\s*\]\]", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[\[\s*##\s*completed\s*##\s*\]\]", "", cleaned, flags=re.IGNORECASE)
    blocks: List[str] = []
    for m in SUMMARY_BLOCK_RE.finditer(cleaned):
        txt = m.group(2).strip()
        blocks.append(txt)
    return blocks


def parse_judge_score(text: str) -> Tuple[float, str]:
    """Parse score and feedback from judge response."""
    score = 0.0
    feedback = ""
    
    # Extract score
    score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)\s*/\s*10", text, re.IGNORECASE)
    if score_match:
        score = float(score_match.group(1))
    
    # Extract feedback section
    feedback_match = re.search(
        r"FEEDBACK:\s*(.*?)\s*(?:\[\[\s*##\s*completed\s*##\s*\]\]|$)",
        text,
        re.IGNORECASE | re.DOTALL
    )
    if feedback_match:
        feedback = feedback_match.group(1).strip()
    
    return score, feedback


def judge_summary(block_text: str, summary_text: str, problem_text: str, client) -> Tuple[float, str, str, dict]:
    """Judge a summary and return (score, feedback, raw_response, metadata)."""
    
    user_prompt = render_user_prompt(
        JUDGE_USER_TEMPLATE,
        problem_text=problem_text,
        block_text=block_text,
        summary_text=summary_text,
    )
    
    def _judge_call():
        model = client.model
        response = client.create_chat_completion(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=16000,
        )
        
        content = response.choices[0].message.content or ""
        score, feedback = parse_judge_score(content)
        
        endpoint_url = str(client.client.base_url) if hasattr(client.client, 'base_url') else 'unknown'
        metadata = {
            'endpoint': endpoint_url,
            'model': model,
            'operation': 'judge'
        }
        
        return score, feedback, content, metadata
    
    try:
        return retry_with_backoff(_judge_call, max_retries=5, initial_delay=2.0)
    
    except Exception as e:
        print(f"  [ERROR] Judge call failed after retries: {e}")
        return 0.0, str(e), "", {'endpoint': 'error', 'model': 'unknown', 'operation': 'judge'}


def refine_summary(block_text: str, old_summary: str, feedback: str, problem_text: str, client) -> Tuple[str, dict]:
    """Refine a summary based on judge feedback.
    
    Returns:
        Tuple of (refined_summary, metadata)
    """
    
    # Build refinement prompt
    refinement_context = f"""The following summary was evaluated and received feedback for improvement.

ORIGINAL BLOCK:
<<<BLOCK_BEGIN>>>
{block_text}
<<<BLOCK_END>>>

PREVIOUS SUMMARY:
<<<SUMMARY_BEGIN>>>
{old_summary}
<<<SUMMARY_END>>>

JUDGE FEEDBACK:
{feedback}

Please generate an IMPROVED summary that addresses all the feedback points above.
Follow the same output format as before (just the improved summary text, no extra commentary).
"""

    user_prompt = render_user_prompt(
        SUMMARY_USER_TEMPLATE,
        full_trace_with_blocks=f"BEGIN BLOCK 1\n{block_text}\nEND BLOCK 1",
        problem_text=problem_text,
    )
    
    # Prepend refinement context
    user_prompt = refinement_context + "\n\n" + user_prompt
    
    def _refine_call():
        model = client.model
        response = client.create_chat_completion(
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=16000,
        )
        
        content = response.choices[0].message.content or ""
        
        endpoint_url = str(client.client.base_url) if hasattr(client.client, 'base_url') else 'unknown'
        metadata = {
            'endpoint': endpoint_url,
            'model': model,
            'operation': 'refine'
        }
        
        # Parse improved summary
        summaries = parse_summary_blocks(content)
        if summaries:
            return summaries[0], metadata
        else:
            # Fallback: return content as-is
            return (content.strip() or old_summary), metadata
    
    try:
        return retry_with_backoff(_refine_call, max_retries=5, initial_delay=2.0)
    
    except Exception as e:
        print(f"  [ERROR] Refinement failed after retries: {e}")
        return old_summary, {'endpoint': 'error', 'model': 'unknown', 'operation': 'refine'}


def summarize_blocks_iterative(
    blocks: List[tuple],
    sentences: List[str],
    client,
    problem_text: str = "",
    max_iterations: int = 2,
    score_threshold: float = 8.0,
) -> Tuple[List[str], List[Dict]]:
    """
    Summarize all blocks with iterative refinement using judge feedback.
    
    Returns:
        summaries: List of final summaries (one per block)
        metadata: List of dicts with iteration history for each block
    """
    
    # Step 1: Generate initial summaries for all blocks (one LLM call)
    full_trace_lines = []
    for block_idx, (start, end) in enumerate(blocks, 1):
        full_trace_lines.append(f"BEGIN BLOCK {block_idx}")
        block_sentences = sentences[start:end+1]
        full_trace_lines.extend(block_sentences)
        full_trace_lines.append(f"END BLOCK {block_idx}")
    
    full_trace_with_blocks = '\n'.join(full_trace_lines)
    
    user_prompt = render_user_prompt(
        SUMMARY_USER_TEMPLATE,
        full_trace_with_blocks=full_trace_with_blocks,
        problem_text=problem_text,
    )
    
    def _initial_summarize():
        model = client.model
        response = client.create_chat_completion(
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=16000,
        )
        
        content = response.choices[0].message.content or ""
        summaries = parse_summary_blocks(content)
        
        endpoint_url = str(client.client.base_url) if hasattr(client.client, 'base_url') else 'unknown'
        metadata = {
            'endpoint': endpoint_url,
            'model': model,
            'operation': 'initial_summarize'
        }
        
        if not summaries:
            summaries = [content.strip()] * len(blocks)
        elif len(summaries) < len(blocks):
            summaries.extend(["[Missing summary]"] * (len(blocks) - len(summaries)))
        elif len(summaries) > len(blocks):
            summaries = summaries[:len(blocks)]
        
        return summaries, metadata
    
    try:
        summaries, initial_metadata = retry_with_backoff(_initial_summarize, max_retries=5, initial_delay=2.0)
    
    except Exception as e:
        print(f"  [ERROR] Initial summarization failed after retries: {e}")
        summaries = ["Summarization failed"] * len(blocks)
        initial_metadata = {'endpoint': 'error', 'model': 'unknown', 'operation': 'initial_summarize'}
    
    # Step 2: Iteratively refine each block's summary (in parallel)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    metadata_list = [None] * len(blocks)  # Pre-allocate to preserve order
    
    # Initialize tokenizer for token counting
    tokenizer = tiktoken.get_encoding('cl100k_base')
    
    def refine_block(block_idx, start, end, initial_summary):
        """Refine a single block's summary with iterative judge feedback."""
        block_sentences = sentences[start:end+1]
        block_text = '\n'.join(block_sentences)
        current_summary = summaries[block_idx]
        
        # Count tokens in the block
        block_tokens = len(tokenizer.encode(block_text))
        
        iteration_history = {
            'block_idx': block_idx,
            'block_tokens': block_tokens,
            'initial_summarize_metadata': initial_metadata if block_idx == 0 else None,
            'iterations': []
        }
        
        for iteration in range(max_iterations):
            # Judge current summary (get fresh client from pool)
            judge_client = get_llm_client()
            score, feedback, raw_judge, judge_metadata = judge_summary(
                block_text=block_text,
                summary_text=current_summary,
                problem_text=problem_text,
                client=judge_client,
            )
            
            # Count tokens in summary
            summary_tokens = len(tokenizer.encode(current_summary))
            compression_ratio = summary_tokens / block_tokens if block_tokens > 0 else 0.0
            
            iteration_history['iterations'].append({
                'iteration': iteration,
                'score': score,
                'feedback': feedback,
                'summary': current_summary,
                'summary_tokens': summary_tokens,
                'compression_ratio': compression_ratio,
                'judge_metadata': judge_metadata,
            })
            
            print(f"    Block {block_idx+1}: Iteration {iteration+1}, Score: {score:.1f}/10")
            
            # Check if we should stop
            if score >= score_threshold:
                print(f"    Block {block_idx+1}: Stopping early (score >= {score_threshold})")
                break
            
            # Refine if not last iteration (get fresh client from pool)
            if iteration < max_iterations - 1:
                refine_client = get_llm_client()
                improved_summary, refine_metadata = refine_summary(
                    block_text=block_text,
                    old_summary=current_summary,
                    feedback=feedback,
                    problem_text=problem_text,
                    client=refine_client,
                )
                current_summary = improved_summary
                summaries[block_idx] = improved_summary
                # Store refine metadata in next iteration's data
                iteration_history['iterations'][-1]['refine_metadata'] = refine_metadata
        
        # Record final score, tokens, and compression ratio
        final_iter = iteration_history['iterations'][-1]
        iteration_history['final_score'] = final_iter['score']
        iteration_history['final_summary_tokens'] = final_iter['summary_tokens']
        iteration_history['final_compression_ratio'] = final_iter['compression_ratio']
        
        return block_idx, current_summary, iteration_history
    
    # Process all blocks in parallel
    with ThreadPoolExecutor(max_workers=len(blocks)) as executor:
        futures = [
            executor.submit(refine_block, block_idx, start, end, summaries[block_idx])
            for block_idx, (start, end) in enumerate(blocks)
        ]
        
        for future in as_completed(futures):
            try:
                block_idx, final_summary, history = future.result()
                summaries[block_idx] = final_summary
                metadata_list[block_idx] = history
            except Exception as e:
                print(f"  [ERROR] Block refinement failed: {e}")
    
    return summaries, metadata_list


def summarize_task(
    task_id: str,
    blocks: List[tuple],
    sentences: List[str],
    client,
    problem_text: str = "",
    max_iterations: int = 2,
    score_threshold: float = 8.0,
) -> dict:
    """Summarize all blocks in a task with iterative refinement."""
    
    print(f"  {task_id}: summarizing {len(blocks)} blocks with iterative refinement...")
    
    summaries, metadata = summarize_blocks_iterative(
        blocks=blocks,
        sentences=sentences,
        client=client,
        problem_text=problem_text,
        max_iterations=max_iterations,
        score_threshold=score_threshold,
    )
    
    return {
        'task_id': task_id,
        'num_blocks': len(blocks),
        'blocks': blocks,
        'summaries': summaries,
        'refinement_metadata': metadata,
    }


def main():
    parser = argparse.ArgumentParser(description='Stage 5: Summarize blocks with iterative refinement')
    parser.add_argument('--input-dir', type=str, default='stage4_segment',
                        help='Input directory from Stage 4')
    parser.add_argument('--output-dir', type=str, default='stage5_summarize',
                        help='Output directory for summaries')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--max-iterations', type=int, default=2,
                        help='Maximum refinement iterations per block')
    parser.add_argument('--score-threshold', type=float, default=8.0,
                        help='Score threshold for early stopping (0-10)')
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
    print(f"Config: max_iterations={args.max_iterations}, score_threshold={args.score_threshold}")
    
    # Load sentences
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
    
    # Process tasks
    results = []
    
    if args.workers == 1:
        # Sequential processing
        client = get_llm_client()
        for task in tasks:
            task_id = task['task_id']
            blocks = [tuple(b) for b in task['blocks']]
            sentences = sentences_map[task_id]
            problem_text = task.get('metadata', {}).get('problem', '')
            
            result = summarize_task(
                task_id=task_id,
                blocks=blocks,
                sentences=sentences,
                client=client,
                problem_text=problem_text,
                max_iterations=args.max_iterations,
                score_threshold=args.score_threshold,
            )
            
            # Write result immediately
            with file_lock:
                with open(output_jsonl, 'a') as f:
                    f.write(json.dumps(result) + '\n')
            
            results.append(result)
    else:
        # Parallel processing
        from concurrent.futures import ThreadPoolExecutor
        
        def process_task_wrapper(task):
            task_id = task['task_id']
            blocks = [tuple(b) for b in task['blocks']]
            sentences = sentences_map[task_id]
            problem_text = task.get('metadata', {}).get('problem', '')
            client = get_llm_client()
            result = summarize_task(
                task_id=task_id,
                blocks=blocks,
                sentences=sentences,
                client=client,
                problem_text=problem_text,
                max_iterations=args.max_iterations,
                score_threshold=args.score_threshold,
            )
            
            # Write result immediately after completion
            with file_lock:
                with open(output_jsonl, 'a') as f:
                    f.write(json.dumps(result) + '\n')
            
            return result
        
        print(f"Using {args.workers} workers for parallel summarization...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(process_task_wrapper, tasks))
    
    print(f"\nSaved summaries to {output_jsonl}")
    
    # Compute and display statistics
    all_final_scores = []
    all_iterations_used = []
    all_block_tokens = []
    all_summary_tokens = []
    all_compression_ratios = []
    
    for result in results:
        for block_meta in result.get('refinement_metadata', []):
            all_final_scores.append(block_meta['final_score'])
            all_iterations_used.append(len(block_meta['iterations']))
            all_block_tokens.append(block_meta.get('block_tokens', 0))
            all_summary_tokens.append(block_meta.get('final_summary_tokens', 0))
            all_compression_ratios.append(block_meta.get('final_compression_ratio', 0.0))
    
    if all_final_scores:
        avg_score = sum(all_final_scores) / len(all_final_scores)
        avg_iterations = sum(all_iterations_used) / len(all_iterations_used)
        avg_block_tokens = sum(all_block_tokens) / len(all_block_tokens)
        avg_summary_tokens = sum(all_summary_tokens) / len(all_summary_tokens)
        avg_compression = sum(all_compression_ratios) / len(all_compression_ratios)
        
        print(f"\nRefinement Statistics:")
        print(f"  Average final score: {avg_score:.2f}/10")
        print(f"  Average iterations: {avg_iterations:.2f}")
        print(f"  Total blocks processed: {len(all_final_scores)}")
        print(f"\nToken Statistics:")
        print(f"  Average block tokens: {avg_block_tokens:.1f}")
        print(f"  Average summary tokens: {avg_summary_tokens:.1f}")
        print(f"  Average compression ratio: {avg_compression:.3f} ({avg_compression*100:.1f}%)")
    
    # Create human-readable examples (only in test mode)
    if args.test:
        for result in results:
            task_id = result['task_id']
            blocks = result['blocks']
            summaries = result['summaries']
            sentences = sentences_map[task_id]
            refinement_meta = result.get('refinement_metadata', [])
            
            example_file = examples_dir / f'{task_id}.txt'
            with open(example_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write(f"TASK: {task_id}\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Total sentences: {len(sentences)}\n")
                f.write(f"Number of blocks: {len(blocks)}\n")
                f.write("\n")
                
                for block_idx, ((start, end), summary) in enumerate(zip(blocks, summaries)):
                    size = end - start + 1
                    
                    f.write("=" * 80 + "\n")
                    f.write(f"BLOCK {block_idx+1}/{len(blocks)}\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"Sentences: {start}-{end} | Size: {size}\n")
                    
                    # Add refinement info with token counts and compression ratio
                    if block_idx < len(refinement_meta):
                        meta = refinement_meta[block_idx]
                        final_score = meta['final_score']
                        num_iterations = len(meta['iterations'])
                        block_tokens = meta.get('block_tokens', 0)
                        summary_tokens = meta.get('final_summary_tokens', 0)
                        compression = meta.get('final_compression_ratio', 0.0)
                        f.write(f"Refinement: {num_iterations} iteration(s), final score: {final_score:.1f}/10\n")
                        f.write(f"Tokens: block={block_tokens}, summary={summary_tokens}, compression={compression:.3f} ({compression*100:.1f}%)\n")
                    
                    f.write("-" * 80 + "\n\n")
                    
                    # Summary
                    f.write("SUMMARY:\n")
                    f.write(summary + "\n\n")
                    
                    # Full text (wrapped)
                    f.write("FULL TEXT:\n")
                    f.write("-" * 80 + "\n")
                    block_text = ' '.join(sentences[start:end+1])
                    
                    # Word wrap at 100 chars
                    import textwrap
                    wrapped = textwrap.fill(block_text, width=100)
                    f.write(wrapped + "\n\n")
                    
            print(f"  Saved example: {example_file.name}")
    
    print(f"\nStage 5 (iterative) complete! Output in {output_dir}/")
    print("\nFinal segmented and summarized chains-of-thought are ready!")


if __name__ == '__main__':
    main()
