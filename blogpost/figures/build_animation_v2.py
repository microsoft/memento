#!/usr/bin/env python3
"""Generate animation.html with multi-example support (Math/Code/Science)."""
import json

def escape_js_backtick(text):
    """Escape text for JS template literal (backtick)."""
    text = text.replace("\\", "\\\\")   # \ -> \\
    text = text.replace("`", "\\`")     # ` -> \`
    # Escape ${ to prevent template literal interpolation
    text = text.replace("${", "\\${")
    return text

def build_example_js(data):
    """Build JS object literal for one example."""
    problem = escape_js_backtick(data["problem"])
    source = data.get("problem_source", "")
    answer = escape_js_backtick(str(data.get("answer", "")))
    
    seg_lines = []
    for seg in data["segments"]:
        t = seg["type"]
        idx = seg["block_idx"]
        tokens = seg["approx_tokens"]
        text = escape_js_backtick(seg["text"])
        seg_lines.append(f'      {{type:"{t}", idx:{idx}, text:`{text}`, tokens:{tokens}}}')
    
    segs_str = ",\n".join(seg_lines)
    
    return f"""{{
    source: `{source}`,
    problem: `{problem}`,
    answer: `{answer}`,
    segments: [\n{segs_str}\n    ]
  }}"""

# Load all 3 examples
examples = {}
for fname, key in [("example_response.json", "math"), 
                    ("example_code.json", "code"), 
                    ("example_science.json", "science")]:
    with open(fname) as f:
        examples[key] = json.load(f)

# Build the EXAMPLES JS object
example_entries = []
for key, data in examples.items():
    example_entries.append(f"  {key}: {build_example_js(data)}")
examples_js = "const EXAMPLES = {\n" + ",\n".join(example_entries) + "\n};"

# Read existing animation.html
with open("animation.html") as f:
    html = f.read()

# 1. Replace the SEGMENTS declaration with EXAMPLES + active tracking
old_seg_start = html.find("const SEGMENTS = [")
old_seg_end = html.find("];\n", old_seg_start)
# Find the end of the SEGMENTS array (the ];)
# We need to find the closing ]; that ends the array
import re
# Find from old_seg_start to the next line that is just "];""
pos = old_seg_start
depth = 0
i = pos
while i < len(html):
    if html[i] == '[':
        depth += 1
    elif html[i] == ']':
        depth -= 1
        if depth == 0:
            old_seg_end = i + 2  # include ];\n
            break
    i += 1

# Also remove the comment lines before SEGMENTS
comment_start = html.rfind("// ===", 0, old_seg_start)
if comment_start > 0:
    # Go back to find all comment lines
    line_start = html.rfind("\n", 0, comment_start) + 1
    old_seg_start = line_start

old_segment_block = html[old_seg_start:old_seg_end]

new_segment_block = f"""{examples_js}

let currentExample = 'math';
let SEGMENTS = EXAMPLES.math.segments;"""

html = html[:old_seg_start] + new_segment_block + html[old_seg_end:]

# 2. Add tab buttons in the header (after the h3)
old_header_h3 = '<h3>Memento Generation &mdash; Qwen3-32B on AIME 2025 Problem 5</h3>'
new_header = """<h3 id="demoTitle">Memento Generation &mdash; Qwen3-32B</h3>
    <div class="example-tabs">
      <button class="tab active" data-key="math" onclick="switchExample('math')">Math</button>
      <button class="tab" data-key="code" onclick="switchExample('code')">Code</button>
      <button class="tab" data-key="science" onclick="switchExample('science')">Science</button>
    </div>"""
html = html.replace(old_header_h3, new_header)

# 3. Update problem text to use an ID
old_problem = """<div class="problem-text">
        <strong>Problem (AIME 2025 Problem 5)</strong><br>
        An isosceles trapezoid has an inscribed circle tangent to each of its four sides. The radius of the circle is 3, and the area of the trapezoid is 72. Let the parallel sides of the trapezoid have lengths r and s, with r \u2260 s. Find r\u00b2 + s\u00b2.
      </div>"""
new_problem = """<div class="problem-text" id="problemText">
        <strong id="problemSource">Problem (AIME 2025 Problem 5)</strong><br>
        <span id="problemBody">An isosceles trapezoid has an inscribed circle tangent to each of its four sides. The radius of the circle is 3, and the area of the trapezoid is 72. Let the parallel sides of the trapezoid have lengths r and s, with r \u2260 s. Find r\u00b2 + s\u00b2.</span>
      </div>"""
html = html.replace(old_problem, new_problem)

# 4. Add tab CSS before the closing </style>
tab_css = """
/* ---- Example tabs ---- */
.example-tabs {
  display: flex;
  gap: 4px;
}

.example-tabs .tab {
  background: #fff;
  color: #268bd2;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 4px 12px;
  font-size: 0.78rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s;
}

.example-tabs .tab:hover {
  background: #d6eaf8;
}

.example-tabs .tab.active {
  background: #268bd2;
  color: #fff;
  border-color: #268bd2;
}
"""
html = html.replace("</style>", tab_css + "</style>")

# 5. Add switchExample function before the BOOT section
switch_fn = """
// ================================================================
// EXAMPLE SWITCHING
// ================================================================
function switchExample(key) {
  if (key === currentExample && !state.finished && state.segIdx === 0) return;
  
  // Update active tab
  document.querySelectorAll('.example-tabs .tab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.key === key);
  });
  
  currentExample = key;
  SEGMENTS = EXAMPLES[key].segments;
  
  // Update problem text
  document.getElementById('problemSource').textContent = 'Problem (' + EXAMPLES[key].source + ')';
  document.getElementById('problemBody').textContent = EXAMPLES[key].problem;
  
  // Reset animation
  resetDemo();
}
"""

boot_marker = "// ================================================================\n// BOOT"
html = html.replace(boot_marker, switch_fn + boot_marker)

with open("animation.html", "w") as f:
    f.write(html)

print(f"Wrote animation.html ({len(html)} chars)")
print(f"  Math segments: {len(examples['math']['segments'])}")
print(f"  Code segments: {len(examples['code']['segments'])}")
print(f"  Science segments: {len(examples['science']['segments'])}")
