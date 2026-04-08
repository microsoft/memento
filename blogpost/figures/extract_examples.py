#!/usr/bin/env python3
"""Extract specific examples from eval results and save as JSON for animation."""
import json
import re
import sys

def parse_response(full_response):
    """Parse a Memento response into segments."""
    text = full_response
    # Remove <think> wrapper
    text = re.sub(r'^<think>\s*', '', text)
    text = re.sub(r'</think>\s*', '', text)
    
    segments = []
    remaining = text
    block_idx = 0
    
    while remaining:
        bs = remaining.find("<|block_start|>")
        if bs == -1:
            answer_text = remaining.strip()
            if answer_text:
                segments.append({
                    "type": "answer",
                    "block_idx": 0,
                    "text": answer_text,
                    "char_len": len(answer_text),
                    "approx_tokens": len(answer_text) // 4
                })
            break
        
        if bs > 0:
            pre = remaining[:bs].strip()
            if pre:
                segments.append({
                    "type": "pre",
                    "block_idx": 0,
                    "text": pre,
                    "char_len": len(pre),
                    "approx_tokens": len(pre) // 4
                })
        
        remaining = remaining[bs + len("<|block_start|>"):]
        block_idx += 1
        
        be = remaining.find("<|block_end|>")
        if be == -1:
            break
        
        block_text = remaining[:be].strip()
        segments.append({
            "type": "block",
            "block_idx": block_idx,
            "text": block_text,
            "char_len": len(block_text),
            "approx_tokens": len(block_text) // 4
        })
        remaining = remaining[be + len("<|block_end|>"):]
        
        ss = remaining.find("<|summary_start|>")
        if ss == -1:
            break
        remaining = remaining[ss + len("<|summary_start|>"):]
        
        se = remaining.find("<|summary_end|>")
        if se == -1:
            break
        
        summary_text = remaining[:se].strip()
        segments.append({
            "type": "summary",
            "block_idx": block_idx,
            "text": summary_text,
            "char_len": len(summary_text),
            "approx_tokens": len(summary_text) // 4
        })
        remaining = remaining[se + len("<|summary_end|>"):]
    
    return segments

# Config: which examples to extract
targets = [
    {"source": "gpqa_diamond", "problem_idx": 39, "rep_idx": 0, 
     "label": "Science", "source_name": "GPQA Diamond"},
    {"source": "lcb_v6", "problem_idx": 380, "rep_idx": 0,
     "label": "Code", "source_name": "LiveCodeBench v6"},
]

source = sys.argv[1]  # gpqa_diamond or lcb_v6
target = [t for t in targets if t["source"] == source][0]

for line in sys.stdin:
    d = json.loads(line.strip())
    if d["problem_idx"] == target["problem_idx"] and d["rep_idx"] == target["rep_idx"]:
        segments = parse_response(d["full_response"])
        
        result = {
            "problem": d.get("problem", ""),
            "problem_source": target["source_name"],
            "label": target["label"],
            "answer": d.get("prediction", d.get("ground_truth", "")),
            "segments": segments
        }
        
        outfile = f"example_{target['label'].lower()}.json"
        with open(outfile, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"Wrote {outfile}")
        print(f"  Segments: {len(segments)}")
        for i, seg in enumerate(segments):
            print(f"  {i}: {seg['type']} idx={seg['block_idx']} chars={seg['char_len']} tokens≈{seg['approx_tokens']}")
        break
