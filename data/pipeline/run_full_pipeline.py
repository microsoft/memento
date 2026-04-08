#!/usr/bin/env python3
"""
Unified pipeline runner for processing CoTs through all 5 stages.

This script processes traces incrementally, one at a time or in batches,
through all pipeline stages:
  Stage 1: Seed Selection (reads from input JSONL)
  Stage 2: Sentence Splitting
  Stage 3: Boundary Scoring
  Stage 4: Segmentation into blocks
  Stage 5: Iterative Summarization

Advantages:
- Streams through large datasets without loading everything into memory
- Processes each trace through all stages before moving to the next
- Incremental checkpointing: can resume if interrupted
- Modular: uses functions from individual stage files
- Scales to 10K+ traces efficiently

Usage:
  # With OpenAI
  export OPENAI_API_KEY=sk-...
  python run_full_pipeline.py \\
    --input traces.jsonl \\
    --output-dir runs/run1 \\
    --model gpt-4o \\
    --workers 4

  # With local vLLM server
  python run_full_pipeline.py \\
    --input traces.jsonl \\
    --output-dir runs/run1 \\
    --model Qwen/Qwen3-32B \\
    --base-url http://localhost:8000/v1 \\
    --api-key no-key
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import stage processing functions (same directory)
from sentence_split import split_into_sentences
from score import score_task as score_task_internal
from segment import segment_variance_dp
from summarize_iterative import summarize_task
from client import get_llm_client


class FailureTracker:
    """
    Tracks consecutive failures and triggers early stop when threshold exceeded.
    
    Detects patterns like:
    - 'Summarization failed' in block_summaries
    - avg_final_score == 0 (indicating LLM call failures)
    - HTTP 403 authorization errors
    """
    
    def __init__(self, max_consecutive_failures: int = 10):
        self.max_consecutive_failures = max_consecutive_failures
        self.consecutive_failures = 0
        self.total_failures = 0
        self.last_failure_reason = None
        self.lock = threading.Lock()
        self._stop_requested = False
    
    def record_success(self):
        """Reset consecutive failure count on success."""
        with self.lock:
            self.consecutive_failures = 0
    
    def record_failure(self, reason: str = None):
        """Record a failure and check if we should stop."""
        with self.lock:
            self.consecutive_failures += 1
            self.total_failures += 1
            self.last_failure_reason = reason
            
            if self.consecutive_failures >= self.max_consecutive_failures:
                self._stop_requested = True
                print(f"\n{'='*60}")
                print(f"EARLY STOP: {self.consecutive_failures} consecutive failures detected!")
                print(f"Last failure reason: {reason}")
                print(f"Total failures: {self.total_failures}")
                print(f"Stopping pipeline to prevent further wasted API calls.")
                print(f"{'='*60}\n")
    
    def should_stop(self) -> bool:
        """Check if pipeline should stop."""
        with self.lock:
            return self._stop_requested
    
    def check_result_for_failure(self, result: dict) -> bool:
        """
        Check if a result indicates a failure pattern.
        Returns True if the result looks like a failure.
        """
        if not result:
            return False  # None results are handled separately (skipped tasks)
        
        # Check for summarization failures
        summaries = result.get('block_summaries', [])
        for summary in summaries:
            if 'failed' in str(summary).lower():
                return True
        
        # Check for zero score (indicates LLM call failures)
        if result.get('avg_final_score', 1) == 0:
            # Check refinement metadata for specific error patterns
            refinement_meta = result.get('refinement_metadata', [])
            for block_meta in refinement_meta:
                for iteration in block_meta.get('iterations', []):
                    feedback = str(iteration.get('feedback', ''))
                    # Check for auth errors
                    if '403' in feedback or 'authorization' in feedback.lower():
                        return True
                    if 'error' in feedback.lower():
                        return True
            return True  # Zero score is always suspicious
        
        return False


class PipelineState:
    """Manages pipeline state and checkpointing."""
    
    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self.processed_ids = set()
        self.lock = threading.Lock()
        self.load_checkpoint()
    
    def load_checkpoint(self):
        """Load processed task IDs from checkpoint file."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                self.processed_ids = set(data.get('processed_ids', []))
            print(f"Loaded checkpoint: {len(self.processed_ids)} tasks already processed")
    
    def save_checkpoint(self):
        """Save processed task IDs to checkpoint file."""
        with self.lock:
            with open(self.checkpoint_file, 'w') as f:
                json.dump({'processed_ids': list(self.processed_ids)}, f)
    
    def is_processed(self, task_id: str) -> bool:
        """Check if task has been processed."""
        with self.lock:
            return task_id in self.processed_ids
    
    def mark_processed(self, task_id: str):
        """Mark task as processed."""
        with self.lock:
            self.processed_ids.add(task_id)


class IncrementalWriter:
    """Thread-safe incremental JSONL writer."""
    
    def __init__(self, output_file: Path):
        self.output_file = output_file
        self.lock = threading.Lock()
        # Create file only if it doesn't exist (append mode for resume support)
        if not self.output_file.exists():
            with open(self.output_file, 'w') as f:
                pass
    
    def write(self, data: Dict[str, Any]):
        """Write a single record to the JSONL file."""
        with self.lock:
            with open(self.output_file, 'a') as f:
                f.write(json.dumps(data) + '\n')


def process_single_trace(
    task: Dict[str, Any],
    client: Any,
    args: argparse.Namespace,
    state: PipelineState,
    failure_tracker: FailureTracker = None
) -> Optional[Dict[str, Any]]:
    """
    Process a single trace through all 5 pipeline stages.
    
    Returns the final result dict or None if already processed.
    """
    task_id = task['task_id']
    
    # Skip if already processed
    if state.is_processed(task_id):
        print(f"  {task_id}: already processed (from checkpoint)")
        return None
    
    try:
        print(f"  {task_id}: starting pipeline...")
        
        # Stage 1: Seed selection (input is already selected, just validate)
        # Handle both JSONL format and HuggingFace dataset format
        if 'conversations' in task:
            # HuggingFace format: extract from conversations
            conversations = task['conversations']
            problem_text = ''
            cot = ''
            for msg in conversations:
                if msg.get('from') == 'human':
                    problem_text = msg.get('value', '')
                elif msg.get('from') == 'gpt':
                    cot = msg.get('value', '')
        else:
            # JSONL format
            cot = task.get('cot', task.get('full_cot', ''))
            problem_text = task.get('problem', '')
        
        # Stage 2: Sentence splitting
        sentences, prefix, suffix = split_into_sentences(cot, extract_think=True)
        print(f"    Stage 2: {len(sentences)} sentences")
        
        if len(sentences) < 2:
            print(f"    Skipping {task_id}: too few sentences")
            state.mark_processed(task_id)
            return None
        
        # Stage 3: Boundary scoring (get fresh client from pool)
        score_task_dict = {
            'task_id': task_id,
            'sentences': sentences,
            'num_sentences': len(sentences)
        }
        if args.include_problem:
            score_task_dict['problem'] = problem_text
        
        scoring_client = get_llm_client()
        score_result = score_task_internal(score_task_dict, scoring_client, use_two_pass=args.two_pass_scoring)
        scores = score_result['boundary_scores']
        print(f"    Stage 3: scored {len(scores)} boundaries")
        
        # Stage 4: Segmentation
        blocks = segment_variance_dp(
            num_sentences=len(sentences),
            scores=scores,
            sentences=sentences,
            min_blocks=1,
            max_blocks=None,
            max_block_size=args.max_block_size,
            min_block_tokens=args.min_block_tokens,
            variance_weight=args.variance_penalty
        )
        print(f"    Stage 4: {len(blocks)} blocks")
        
        if len(blocks) < 1:
            print(f"    Skipping {task_id}: no blocks created")
            state.mark_processed(task_id)
            return None
        
        # Stage 5: Iterative summarization (get fresh client from pool)
        summary_client = get_llm_client()
        summary_result = summarize_task(
            task_id=task_id,
            blocks=blocks,
            sentences=sentences,
            client=summary_client,
            problem_text=problem_text,
            max_iterations=args.max_iterations,
            score_threshold=args.score_threshold
        )
        
        # Extract refinement metadata for display
        refinement_meta = summary_result.get('refinement_metadata', [])
        avg_final_score = sum(m.get('final_score', 0) for m in refinement_meta) / len(refinement_meta) if refinement_meta else 0
        total_iterations = sum(len(m.get('iterations', [])) for m in refinement_meta) if refinement_meta else 0
        print(f"    Stage 5: avg_score={avg_final_score:.2f}, total_iterations={total_iterations}")
        
        # Build final result with all intermediate outputs
        result = {
            'task_id': task_id,
            
            # Stage 2 outputs
            'sentences': sentences,
            'num_sentences': len(sentences),
            'think_prefix': prefix,
            'think_suffix': suffix,
            
            # Stage 3 outputs
            'boundary_scores': scores,
            'scoring_metadata': score_result.get('scoring_metadata', {}),
            
            # Stage 4 outputs
            'blocks': blocks,
            'num_blocks': len(blocks),
            
            # Stage 5 outputs
            'block_summaries': summary_result.get('summaries', []),
            'refinement_metadata': refinement_meta,
            'avg_final_score': avg_final_score,
            'total_iterations': total_iterations
        }
        
        # Preserve OpenThoughts metadata fields
        for field in ['original_index', 'source', 'domain', 'difficulty', 'dataset_index']:
            if field in task:
                result[field] = task[field]
        
        # Preserve original fields if needed
        if args.include_problem:
            result['problem'] = problem_text
        if args.include_original_cot:
            result['original_cot'] = cot
        
        # Preserve all other original fields from input task (excluding large ones)
        excluded_keys = {'cot', 'full_cot', 'conversations', 'sentences', 'boundary_scores', 
                        'blocks', 'block_summaries', 'refinement_metadata'}
        for key in task:
            if key not in result and key not in excluded_keys:
                result[f'original_{key}'] = task[key]
        
        state.mark_processed(task_id)
        
        # Check for failure patterns in the result
        if failure_tracker:
            if failure_tracker.check_result_for_failure(result):
                failure_tracker.record_failure(f"Task {task_id}: Summarization/scoring failed")
                print(f"  {task_id}: completed but detected failure pattern ⚠")
            else:
                failure_tracker.record_success()
                print(f"  {task_id}: completed all stages ✓")
        else:
            print(f"  {task_id}: completed all stages ✓")
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"  {task_id}: ERROR in pipeline - {e}")
        import traceback
        traceback.print_exc()
        
        # Record exception as failure
        if failure_tracker:
            # Check for auth errors specifically
            if '403' in error_msg or 'Forbidden' in error_msg:
                failure_tracker.record_failure(f"Authorization error (403): {error_msg[:100]}")
            else:
                failure_tracker.record_failure(f"Exception: {error_msg[:100]}")
        
        return None


def process_batch(
    tasks: List[Dict[str, Any]],
    args: argparse.Namespace,
    state: PipelineState,
    writer: IncrementalWriter,
    failure_tracker: FailureTracker = None
) -> bool:
    """
    Process a batch of tasks through the pipeline.
    
    Returns False if early stop was triggered, True otherwise.
    """
    
    if args.workers == 1:
        # Sequential processing - get fresh client on each call
        for task in tasks:
            if failure_tracker and failure_tracker.should_stop():
                return False
            result = process_single_trace(task, None, args, state, failure_tracker)
            if result:
                writer.write(result)
        return not (failure_tracker and failure_tracker.should_stop())
    else:
        # Parallel processing - get fresh client on each call within the task
        def process_wrapper(task):
            if failure_tracker and failure_tracker.should_stop():
                return None  # Skip if stop requested
            # Pass None for client, functions will call get_llm_client() as needed
            result = process_single_trace(task, None, args, state, failure_tracker)
            if result:
                writer.write(result)
            return task['task_id']
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_wrapper, task) for task in tasks]
            for future in as_completed(futures):
                if failure_tracker and failure_tracker.should_stop():
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    return False
                try:
                    task_id = future.result()
                except Exception as e:
                    print(f"    ERROR processing task: {e}")
        
        return not (failure_tracker and failure_tracker.should_stop())


def main():
    parser = argparse.ArgumentParser(description='Run full pipeline on CoT dataset')
    
    # Input/output
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSONL file or HuggingFace dataset directory with CoTs')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    
    # Processing options
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1 for sequential)')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Process this many traces before checkpointing')
    parser.add_argument('--checkpoint-every', type=int, default=10,
                        help='Save checkpoint every N tasks')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of tasks to process (for testing)')
    
    # Stage 3: Scoring options
    parser.add_argument('--include-problem', action='store_true',
                        help='Include problem text in output JSON')
    parser.add_argument('--two-pass-scoring', action='store_true',
                        help='Use two-pass method for boundary scoring (more accurate but slower)')
    
    # Stage 4: Segmentation options
    parser.add_argument('--variance-penalty', type=float, default=0.5,
                        help='Variance penalty for segmentation (default: 0.5)')
    parser.add_argument('--max-block-size', type=int, default=None,
                        help='Maximum block size in sentences')
    parser.add_argument('--min-block-tokens', type=int, default=200,
                        help='Minimum block size in tokens (default: 200)')
    
    # Stage 5: Summarization options
    parser.add_argument('--max-iterations', type=int, default=3,
                        help='Maximum refinement iterations for summaries')
    parser.add_argument('--score-threshold', type=float, default=8.0,
                        help='Stop refinement when judge score >= this threshold (out of 10)')
    
    # Output options
    parser.add_argument('--include-original-cot', action='store_true',
                        help='Include original CoT in output (increases file size)')
    
    # Failure detection / early stopping
    parser.add_argument('--max-consecutive-failures', type=int, default=20,
                        help='Stop pipeline after this many consecutive failures (default: 20)')
    parser.add_argument('--no-early-stop', action='store_true',
                        help='Disable early stopping on consecutive failures')
    
    # LLM API configuration
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='Model name (default: gpt-4o)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='API key (default: uses OPENAI_API_KEY env var)')
    parser.add_argument('--base-url', type=str, default=None,
                        help='Base URL for OpenAI-compatible API (default: uses OPENAI_BASE_URL env var or OpenAI)')
    
    args = parser.parse_args()
    
    # Initialize the client pool with user-specified API settings
    from client import init_client_pool
    init_client_pool(model=args.model, api_key=args.api_key, base_url=args.base_url)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize checkpoint and writer
    checkpoint_file = output_dir / 'checkpoint.json'
    state = PipelineState(checkpoint_file)
    
    output_file = output_dir / 'pipeline_results.jsonl'
    writer = IncrementalWriter(output_file)
    
    # Initialize failure tracker for early stopping
    if args.no_early_stop:
        failure_tracker = None
        print(f"  Early stopping: DISABLED")
    else:
        failure_tracker = FailureTracker(max_consecutive_failures=args.max_consecutive_failures)
        print(f"  Early stopping: enabled (after {args.max_consecutive_failures} consecutive failures)")
    
    # Load input tasks
    input_path = Path(args.input)
    tasks = []
    print(f"Loading tasks from {input_path}...")
    
    # Check if input is a directory (HuggingFace dataset) or JSONL file
    if input_path.is_dir():
        # Load HuggingFace dataset
        print(f"Detected HuggingFace dataset directory")
        from datasets import load_from_disk
        ds = load_from_disk(str(input_path))
        
        # Convert to task format
        limit = args.limit if args.limit else len(ds)
        for i in range(min(limit, len(ds))):
            task = dict(ds[i])
            # Generate task_id if not present, and preserve dataset index
            if 'task_id' not in task:
                task['task_id'] = f"{input_path.name}-{i:05d}"
            task['dataset_index'] = i  # Store the original index in the dataset
            tasks.append(task)
        
        print(f"Loaded {len(tasks)} tasks from HuggingFace dataset")
    else:
        # Load JSONL file
        with open(input_path, 'r') as f:
            for i, line in enumerate(f):
                task = json.loads(line)
                # Generate task_id if not present
                if 'task_id' not in task:
                    task['task_id'] = f"task-{i:05d}"
                tasks.append(task)
                if args.limit and len(tasks) >= args.limit:
                    break
        
        print(f"Loaded {len(tasks)} tasks from JSONL file")
    print(f"Configuration:")
    print(f"  Workers: {args.workers}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Checkpoint every: {args.checkpoint_every} tasks")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Score threshold: {args.score_threshold}")
    print()
    
    # Process in batches
    total_processed = 0
    checkpoint_counter = 0
    early_stopped = False
    
    for i in range(0, len(tasks), args.batch_size):
        batch = tasks[i:i + args.batch_size]
        print(f"\nProcessing batch {i // args.batch_size + 1} ({len(batch)} tasks)...")
        
        continue_processing = process_batch(batch, args, state, writer, failure_tracker)
        
        total_processed += len(batch)
        checkpoint_counter += len(batch)
        
        # Periodic checkpointing
        if checkpoint_counter >= args.checkpoint_every:
            state.save_checkpoint()
            checkpoint_counter = 0
            print(f"  Checkpoint saved ({len(state.processed_ids)} tasks completed)")
        
        # Check for early stop
        if not continue_processing:
            early_stopped = True
            print(f"\n*** Pipeline stopped early due to consecutive failures ***")
            break
    
    # Final checkpoint
    state.save_checkpoint()
    
    print(f"\n{'='*60}")
    if early_stopped:
        print(f"Pipeline STOPPED EARLY!")
        print(f"  Reason: {failure_tracker.last_failure_reason if failure_tracker else 'Unknown'}")
        print(f"  Consecutive failures: {failure_tracker.consecutive_failures if failure_tracker else 'N/A'}")
        print(f"  Total failures: {failure_tracker.total_failures if failure_tracker else 'N/A'}")
        sys.exit(1)  # Exit with error code
    else:
        print(f"Pipeline complete!")
    print(f"  Total processed: {len(state.processed_ids)} tasks")
    print(f"  Output: {output_file}")
    print(f"  Checkpoint: {checkpoint_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
