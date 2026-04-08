# Memento Data Pipeline

This pipeline converts raw chain-of-thought (CoT) reasoning traces into the **Memento format** — with block boundaries and compressed summaries — used for SFT training.

## Pipeline Overview

The pipeline processes each CoT trace through 5 stages:

| Stage | Script | Description | Requires LLM? |
|-------|--------|-------------|----------------|
| 1 | `seed_select.py` | Select traces from OpenThoughts dataset | No |
| 2 | `sentence_split.py` | Split CoT into sentences (preserves code/math blocks) | No |
| 3 | `score.py` | Score boundary quality between sentences (0-3) | **Yes** |
| 4 | `segment.py` | Optimal segmentation into blocks via DP | No |
| 5 | `summarize_iterative.py` | Generate & refine block summaries with judge feedback | **Yes** |

The unified runner `run_full_pipeline.py` chains all stages together.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API access

The pipeline works with any **OpenAI-compatible API**:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Or any compatible provider (Together AI, Fireworks, Groq, etc.)
export OPENAI_API_KEY=your-key
export OPENAI_BASE_URL=https://api.together.xyz/v1

# Or local vLLM server (no key needed)
export OPENAI_BASE_URL=http://localhost:8000/v1
```

### 3. Run the pipeline

```bash
cd data/pipeline

# Process a single trace (smoke test)
python run_full_pipeline.py \
    --input ../examples/example_trace.jsonl \
    --output-dir output/ \
    --model gpt-4o \
    --limit 1 \
    --include-problem

# Process a full dataset with parallelism
python run_full_pipeline.py \
    --input /path/to/openthoughts_subset.jsonl \
    --output-dir runs/my_run \
    --model gpt-4o \
    --workers 8 \
    --batch-size 10 \
    --checkpoint-every 100

# Using a local vLLM server
python run_full_pipeline.py \
    --input traces.jsonl \
    --output-dir runs/local_run \
    --model Qwen/Qwen3-32B \
    --base-url http://localhost:8000/v1 \
    --api-key no-key \
    --workers 4
```

### Supported Providers

| Provider | `--base-url` | Example `--model` |
|----------|-------------|-------------------|
| OpenAI (default) | *(not needed)* | `gpt-4o`, `gpt-4o-mini` |
| Together AI | `https://api.together.xyz/v1` | `Qwen/Qwen3-32B` |
| Fireworks | `https://api.fireworks.ai/inference/v1` | `accounts/fireworks/models/qwen3-32b` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| OpenRouter | `https://openrouter.ai/api/v1` | `openai/gpt-4o` |
| Local vLLM | `http://localhost:8000/v1` | *(your served model)* |
| Ollama | `http://localhost:11434/v1` | `llama3.1` |

## Pipeline Stages in Detail

### Stage 1: Seed Selection (`seed_select.py`)

Selects and filters traces from the [OpenThoughts](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) dataset. Filters out:
- Traces with Chinese/CJK characters
- Traces without complete `<think>...</think>` tags
- Traces that are too short

### Stage 2: Sentence Splitting (`sentence_split.py`)

Splits CoT text into semantically coherent sentences while preserving:
- Code blocks (fenced and indented)
- Math expressions (LaTeX `$$...$$`, `$...$`, `\[...\]`)
- Multi-line derivations
- List structures

### Stage 3: Boundary Scoring (`score.py`)

Uses an LLM to score each boundary between sentences (0-3):
- **0**: Poor break (mid-thought, mid-calculation)
- **1**: Weak break (minor transition)
- **2**: Good break (clear transition)
- **3**: Strong break (major topic shift)

Supports two-pass scoring with coprime window sizes (16, 11) for more robust results.

### Stage 4: Segmentation (`segment.py`)

Optimal block segmentation using dynamic programming:
- Maximizes: `avg_boundary_score - variance_weight × CV(block_token_sizes)`
- Enforces minimum block size in tokens
- Produces balanced blocks that align with natural reasoning boundaries

### Stage 5: Iterative Summarization (`summarize_iterative.py`)

For each block, generates a compressed summary with iterative refinement:
1. Initial summarization of all blocks
2. LLM judge scores each summary (0-10 rubric)
3. If score < threshold, refine with judge feedback
4. Repeat up to `--max-iterations` times

Target compression: ~10-20% of block tokens while preserving all logically relevant information.

## CLI Reference

```
python run_full_pipeline.py [OPTIONS]

Required:
  --input PATH              Input JSONL file or HuggingFace dataset directory
  --output-dir PATH         Output directory for results

API Configuration:
  --model MODEL             Model name (default: gpt-4o)
  --api-key KEY             API key (default: OPENAI_API_KEY env var)
  --base-url URL            Base URL (default: OPENAI_BASE_URL env var or OpenAI)

Processing:
  --workers N               Parallel workers (default: 1)
  --batch-size N            Tasks per checkpoint batch (default: 10)
  --checkpoint-every N      Checkpoint interval (default: 10)
  --limit N                 Max tasks to process

Scoring (Stage 3):
  --two-pass-scoring        Use two-pass coprime window scoring
  --include-problem         Include problem text in output

Segmentation (Stage 4):
  --variance-penalty F      Block size variance penalty (default: 0.5)
  --max-block-size N        Max sentences per block
  --min-block-tokens N      Min tokens per block (default: 200)

Summarization (Stage 5):
  --max-iterations N        Max refinement iterations (default: 3)
  --score-threshold F       Early stop score threshold (default: 8.0)

Output:
  --include-original-cot    Include original CoT in output
  --no-early-stop           Disable failure-based early stopping
  --max-consecutive-failures N  Failure threshold (default: 20)
```

## Output Format

The pipeline produces `pipeline_results.jsonl` where each line is a JSON object:

```json
{
  "task_id": "ot3-train-00001",
  "sentences": ["First sentence...", "Second sentence..."],
  "boundary_scores": [0.0, 2.5, 1.0, 3.0],
  "blocks": [[0, 5], [6, 12], [13, 20]],
  "block_summaries": ["Summary of block 1...", "Summary of block 2...", "Summary of block 3..."],
  "avg_final_score": 8.5,
  "num_blocks": 3,
  "num_sentences": 21
}
```

