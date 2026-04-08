# Memento

[**Paper (PDF)**](docs/memento.pdf) | [**OpenMementos Dataset**](https://huggingface.co/datasets/microsoft/OpenMementos)

**Memento** extends the effective output length of large language models by splitting chain-of-thought reasoning into **blocks** and **summaries** (memento). After each reasoning block, the model generates a short summary, then the block content is evicted from the KV cache. The model continues from the summary with a shorter context, enabling more reasoning within a fixed context window.

## Special tokens

| Token | Purpose |
|-------|---------|
| `<think>` / `</think>` | Reasoning wrapper |
| `<\|block_start\|>` / `<\|block_end\|>` | Reasoning block boundaries |
| `<\|summary_start\|>` / `<\|summary_end\|>` | Summary (memento block) boundaries |

**Block structure:** `<|block_start|> reasoning <|block_end|> <|summary_start|> summary <|summary_end|>`

## Repository layout

| Directory | Description |
|-----------|-------------|
| [`data/`](data/) | Data pipeline — converts raw CoT traces into the Memento format (block boundaries + summaries) for SFT training |
| [`vllm/`](vllm/) | vLLM overlay — adds KV cache block masking to stock vLLM for efficient Memento inference |

## Quick start

### Data pipeline

Convert chain-of-thought traces into Memento training data:

```bash
pip install -r data/requirements.txt
export OPENAI_API_KEY=sk-...   # or any OpenAI-compatible provider

cd data/pipeline
python run_full_pipeline.py \
    --input ../examples/example_trace.jsonl \
    --output-dir output/ \
    --model gpt-4o \
    --limit 1
```

See [data/README.md](data/README.md) for full documentation.

### vLLM inference with block masking

* Step 1: Set Up the Environment
    Build a customized vllm with block masking support:
    ```bash
    pip install vllm==0.13.0
    cd vllm
    bash install_overlay.sh
    ```

* Step 2: Serve a Memento Model with KV Cache Compaction
    To expose the model through an API-compatible server, run:
    ```bash
    python -m vllm.entrypoints.openai.api_server \
        --model /path/to/memento-checkpoint \
        --served-model-name memento \
        --port 8010 \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code \
        --chat-template chat_templates/memento_nosys.jinja \
        --block-masking-config '{
            "enable": true,
            "keep_last_n_blocks": 0,
            "mask_delimiters": false,
            "compact_on_summary_end": true,
            "require_assistant_section": true,
            "debug": true
        }'
    ```

    See [vllm/README.md](vllm/README.md) for full documentation, including API usage and alternative setup options.

## License

This project is licensed under the [MIT License](LICENSE).
