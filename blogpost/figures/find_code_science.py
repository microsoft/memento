#!/usr/bin/env python3
"""Find good Code (LCB) and Science (GPQA) examples for the blog animation."""
import json
import sys
import re

def count_blocks(text):
    return text.count("<|block_start|>")

def get_segments(text):
    """Parse response into segments."""
    # Remove <think> wrapper
    text = re.sub(r'^<think>\s*', '', text)
    text = re.sub(r'</think>\s*', '', text)
    
    segments = []
    remaining = text
    block_idx = 0
    
    while remaining:
        bs = remaining.find("<|block_start|>")
        if bs == -1:
            # No more blocks - rest is answer
            answer_text = remaining.strip()
            if answer_text:
                segments.append({"type": "answer", "text": answer_text, "chars": len(answer_text)})
            break
        
        # Skip any text before first block_start
        if bs > 0:
            pre = remaining[:bs].strip()
            if pre:
                segments.append({"type": "pre", "text": pre, "chars": len(pre)})
        
        remaining = remaining[bs + len("<|block_start|>"):]
        block_idx += 1
        
        be = remaining.find("<|block_end|>")
        if be == -1:
            break
        
        block_text = remaining[:be].strip()
        segments.append({"type": "block", "idx": block_idx, "text": block_text, "chars": len(block_text)})
        remaining = remaining[be + len("<|block_end|>"):]
        
        ss = remaining.find("<|summary_start|>")
        if ss == -1:
            break
        remaining = remaining[ss + len("<|summary_start|>"):]
        
        se = remaining.find("<|summary_end|>")
        if se == -1:
            break
        
        summary_text = remaining[:se].strip()
        segments.append({"type": "summary", "idx": block_idx, "text": summary_text, "chars": len(summary_text)})
        remaining = remaining[se + len("<|summary_end|>"):]
    
    return segments

def score_example(d):
    """Score an example for quality."""
    if not d.get("correct"):
        return -1
    
    nblocks = d.get("block_start_count", 0)
    if nblocks < 3 or nblocks > 8:
        return -1
    
    ntokens = d.get("num_tokens", 0)
    if ntokens < 800 or ntokens > 6000:
        return -1
    
    fr = d.get("full_response", "")
    segs = get_segments(fr)
    blocks = [s for s in segs if s["type"] == "block"]
    summaries = [s for s in segs if s["type"] == "summary"]
    
    if len(blocks) != len(summaries):
        return -1
    
    # Prefer 3-5 blocks, moderate length
    score = 100
    score -= abs(nblocks - 4) * 10  # prefer 4 blocks
    score -= abs(ntokens - 2500) / 50  # prefer ~2500 tokens
    
    # Check block length variance (prefer balanced blocks)
    if blocks:
        avg_len = sum(b["chars"] for b in blocks) / len(blocks)
        variance = sum((b["chars"] - avg_len)**2 for b in blocks) / len(blocks)
        score -= (variance ** 0.5) / 100  # penalize high variance
    
    return score

# Read from stdin
data_type = sys.argv[1]  # "gpqa" or "lcb"
results = []

for line in sys.stdin:
    d = json.loads(line.strip())
    s = score_example(d)
    if s > 0:
        results.append((s, d))

results.sort(key=lambda x: -x[0])

print(f"\n=== Top 5 {data_type} examples ===")
for i, (score, d) in enumerate(results[:5]):
    pidx = d["problem_idx"]
    ridx = d["rep_idx"]
    nblocks = d["block_start_count"]
    ntokens = d["num_tokens"]
    fr = d.get("full_response", "")
    # Get problem text
    problem = d.get("problem", "")
    if not problem:
        # Try to extract from prompt
        problem = fr[:100]
    
    print(f"\n--- #{i+1}: p={pidx} r={ridx} score={score:.1f} blocks={nblocks} tokens={ntokens} ---")
    print(f"Problem: {problem[:200]}...")
    
    segs = get_segments(fr)
    for j, seg in enumerate(segs):
        print(f"  {seg['type']}{seg.get('idx','')}({seg['chars']}c): {seg['text'][:80]}...")
