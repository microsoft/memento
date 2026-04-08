# vLLM Block Masking Overlay

Custom modifications to [vLLM](https://github.com/vllm-project/vllm) that add **block masking** support for Memento-style inference. During generation, completed reasoning blocks are compacted (evicted) from the KV cache, keeping only their summaries. This allows the model to produce more reasoning tokens within a fixed context window.

## How it works

A Memento model generates structured chain-of-thought with special tokens:

```
<think>
<|block_start|> ... reasoning ... <|block_end|>
<|summary_start|> ... compressed summary ... <|summary_end|>

<|block_start|> ... more reasoning ... <|block_end|>
<|summary_start|> ... summary ... <|summary_end|>
...
</think>
Final answer
```

When block masking is enabled, the vLLM scheduler watches for `<|summary_end|>` during token-by-token generation. Each time a block's summary completes, the block's reasoning content is evicted from the KV cache. The model continues from the summary with a shorter effective context.

With `mask_delimiters=False` (Qwen3 style), delimiter tokens (`<|block_start|>`, `<|block_end|>`) are **preserved** in the cache; only the content between them is evicted. With `mask_delimiters=True` (Phi3/Phi4 style), the delimiters are evicted too.

## What's here

This directory contains **only the files modified or added** relative to stock vLLM. It is not a full vLLM installation — it's an overlay that patches an existing vLLM install.

**New files:**
- `vllm/config/block_masking.py` — `BlockMaskingConfig` dataclass
- `vllm/v1/core/block_masking/` — Per-request state tracker and token processor
- `vllm/v1/sample/logits_processor/block_length_cap.py` — Optional block length cap
- `chat_templates/` — Chat templates for Memento models
- `install_overlay.sh` — Automated installer

**Modified vLLM files** — scheduler, engine, KV cache manager, worker, config, and request types (see `vllm/` tree).

## Base version

Compatible with **vLLM 0.13.x**. Overlay files were extracted from stock vLLM `0.13.1.dev0` with block masking patches applied.

## Installation

### Quick install (overlay on existing vLLM)

```bash
# 1. Install stock vLLM 0.13.x
pip install vllm==0.13.0

# 2. Apply the Memento overlay
cd vllm/
bash install_overlay.sh
```

### How the overlay installer works

The block masking changes are pure Python — no C++/CUDA recompilation needed. The installer patches `.py` files on top of a stock vLLM wheel while preserving the pre-compiled extensions:

1. **Find the installed vLLM** — locates the `vllm/` package in your Python environment's site-packages (carefully skipping the local `vllm/` directory in the repo).

2. **Download the stock wheel** — fetches the matching release wheel from PyPI (e.g. `vllm==0.13.0`) and extracts its `.py` files to `/tmp/vllm_stock/` for reference.

3. **rsync all `.py` files** — copies every Python file from `vllm/vllm/` in this repo on top of the installed package. This applies all block masking modifications (config, scheduler, engine, worker, KV cache manager, etc.) in one shot.

4. **Restore `.so`-interface files** — the overlay's upstream base may have newer function signatures than the compiled `.so` extensions expect. The installer restores two critical files from the stock wheel:
   - `vllm_flash_attn/flash_attn_interface.py` — FlashAttention wrapper (arg count must match the compiled `_C` extension)
   - `_custom_ops.py` — custom op bindings

5. **Verify** — imports `BlockMaskingConfig`, `BlockMaskingProcessor`, `BlockMaskingState`, `LLM`, and `_custom_ops` to confirm everything loads.

### From source

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm && git checkout 85f55c943

# Copy overlay files on top
cp -r /path/to/memento/vllm/vllm/* vllm/

# Build
pip install -e .
```

## Usage

### OpenAI-compatible server

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

Then query it with any OpenAI-compatible client:

```bash
curl http://localhost:8010/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "memento",
        "messages": [{"role": "user", "content": "What is 2+3? Put your answer in \\boxed{}."}],
        "max_tokens": 16384,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "skip_special_tokens": false
    }'
```

> Set `skip_special_tokens: false` to see the `<|block_start|>` / `<|summary_end|>` markers in the response text.

### Python API

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_path = "path/to/memento-checkpoint"
tokenizer = AutoTokenizer.from_pretrained(model_path)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 2+3? Put your answer in \\boxed{}."}],
    tokenize=False,
    add_generation_prompt=True,
)

llm = LLM(
    model=model_path,
    block_masking_config={
        "enable": True,
        "keep_last_n_blocks": 0,
        "mask_delimiters": False,
        "compact_on_summary_end": True,
    },
    max_model_len=32768,
    enable_prefix_caching=False,
)

params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=16384)
outputs = llm.generate([prompt], params)
print(outputs[0].outputs[0].text)
```

## Block masking parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `false` | Enable block masking |
| `keep_last_n_blocks` | int | `0` | Blocks to keep in KV cache. `0` = compact all, `-1` = disabled (no compaction) |
| `mask_delimiters` | bool | `false` | Include delimiter tokens in compaction range. `false` for Qwen3/OLMo3, `true` for Phi3/Phi4 |
| `compact_on_summary_end` | bool | `true` | Trigger compaction when `<|summary_end|>` is generated |
| `require_assistant_section` | bool | `true` | Only activate block masking inside assistant turns |
| `restart_mode` | bool | `false` | Evict block + recompute summary KV (alternative to compact mode) |
| `keep_last_block_for_answer` | bool | `false` | Defer last block compaction to preserve context for the final answer |
| `max_block_tokens` | int | `0` | Cap block length (0 = unlimited) |
| `debug` | bool | `false` | Print `[BlockMasking]` events to stdout |

## Chat templates

- `chat_templates/memento_nosys.jinja` — For Qwen3 models trained **without** a system prompt. System messages are silently ignored.

Use with `--chat-template` when launching the server.

## Important notes

- **Prefix caching must be disabled** when using block masking (`enable_prefix_caching=False`). The overlay does not set this automatically.
- **`skip_special_tokens=False`** is needed in the sampling request to preserve block/summary markers in the output text.
- Debug mode (`"debug": true`) logs every block lifecycle event (`started`, `ended`, `summary started/ended`, `compaction triggered`) with token positions to stdout.

## License

vLLM is licensed under [Apache License 2.0](https://github.com/vllm-project/vllm/blob/main/LICENSE). Our modifications are released under the same license.
