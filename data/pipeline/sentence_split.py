#!/usr/bin/env python3
"""
Stage 2: Split chains-of-thought into sentences

Takes CoT from Stage 1 and splits into semantically coherent sentences,
preserving code blocks and math formulas intact.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional

# Try to import tiktoken for token counting
try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("o200k_base")
except Exception:
    _ENCODER = None


def rough_token_count(text: str) -> int:
    """Approximate token count: tiktoken if available, else word count."""
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    return len(text.split())


def extract_think_content(text: str) -> Tuple[str, str, str]:
    """
    Extract content before, inside, and after <think>...</think> tags.
    
    Returns:
        (prefix, thinking, suffix) where:
        - prefix: content before <think> tag (empty if no tag)
        - thinking: content inside <think>...</think> tags (or full text if no tags)
        - suffix: content after </think> tag (empty if no tag)
    """
    # Case-insensitive search for think tags
    think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
    match = think_pattern.search(text)
    
    if match:
        prefix = text[:match.start()].strip()
        thinking = match.group(1).strip()
        suffix = text[match.end():].strip()
        return prefix, thinking, suffix
    else:
        # No think tags - process entire text
        return "", text.strip(), ""


def reconstruct_with_think_tags(prefix: str, thinking: str, suffix: str) -> str:
    """
    Reconstruct the full text with think tags if they were present.
    
    Args:
        prefix: content before <think> tag
        thinking: processed thinking content
        suffix: content after </think> tag
    
    Returns:
        Reconstructed text with tags if prefix or suffix exist, otherwise just thinking
    """
    if prefix or suffix:
        result = ""
        if prefix:
            result += prefix + "\n\n"
        result += "<think>\n" + thinking + "\n</think>"
        if suffix:
            result += "\n\n" + suffix
        return result
    else:
        # No tags in original, return just the thinking
        return thinking


def _looks_like_code_line(line: str) -> bool:
    """
    Heuristic to detect if a line looks like code.
    Returns True if the line has code-like characteristics.
    """
    stripped = line.strip()
    if not stripped:
        return False
    
    # Skip if already in a comment or string explanation
    if stripped.startswith('//') or stripped.startswith('#'):
        # Could be code comment, check context
        pass
    
    # Strong code indicators
    # 1. Variable assignment
    if re.match(r'^[a-zA-Z_]\w*\s*[+\-*/]?=', stripped):
        return True
    
    # 2. Function/class definition
    if re.match(r'^(def|class|function|fn|func|int|void|double|float|bool|char|string|auto)\s+', stripped):
        return True
    
    # 3. Control flow keywords at start
    if re.match(r'^(if|else|elif|for|while|return|break|continue|switch|case|try|catch|throw)\s*[\(\{]', stripped):
        return True
    
    # 4. Array/struct/pointer syntax
    if re.search(r'\w+\[\w*\]|\w+->\w+|\w+::\w+', stripped):
        return True
    
    # 5. Function calls with parentheses (but not prose)
    if re.search(r'\w+\([^\)]*\)', stripped) and not stripped.endswith('.'):
        # Exclude common prose patterns like "function(x) returns"
        if not re.search(r'\)\s+(returns?|gives?|means?|is|are|will|should)', stripped, re.IGNORECASE):
            return True
    
    # 6. Import statements
    if re.match(r'^(import|from|#include|using)\s+', stripped):
        return True
    
    # 7. Multiple semicolons or braces (C/C++/Java style)
    if ';' in stripped or stripped in ['{', '}']:
        return True
    
    # 8. Indented line with code-like operators
    if line.startswith('    ') and re.search(r'[=+\-*/()[\]{}]', stripped):
        return True
    
    return False


def _detect_indented_code_blocks(text: str) -> str:
    """
    Detect and fence code blocks that appear as consistently indented sections
    (4+ spaces or tabs) with code-like content, even without explicit markers.
    """
    lines = text.split('\n')
    result = []
    i = 0
    in_existing_fence = False
    
    while i < len(lines):
        line = lines[i]
        
        # Track if we're inside existing fence
        if '```' in line:
            in_existing_fence = not in_existing_fence
            result.append(line)
            i += 1
            continue
        
        # Skip math blocks
        if '$$' in line:
            result.append(line)
            i += 1
            continue
        
        # Skip if we're already inside a fence
        if in_existing_fence:
            result.append(line)
            i += 1
            continue
        
        # Check if this line starts a code block (indented + code-like)
        # Must have consistent 4-space indentation (not arbitrary whitespace)
        if _looks_like_code_line(line) and line.startswith('    ') and not line.startswith('     '):
            # Collect consecutive code-like lines with similar indentation
            code_block = [line]
            j = i + 1
            
            while j < len(lines):
                next_line = lines[j]
                next_stripped = next_line.strip()
                
                # Stop at math markers
                if '$$' in next_line or '```' in next_line:
                    break
                
                # Empty lines are ok within code (single newline only)
                if not next_stripped:
                    # Check if next non-empty line is still code
                    k = j + 1
                    while k < len(lines) and not lines[k].strip():
                        k += 1
                    if k < len(lines) and lines[k].startswith('    ') and _looks_like_code_line(lines[k]):
                        code_block.append(next_line)
                        j += 1
                    else:
                        # Empty line followed by non-code, stop
                        break
                    continue
                
                # Check if still in code block (must maintain 4-space indent)
                if next_line.startswith('    ') and _looks_like_code_line(next_line):
                    code_block.append(next_line)
                    j += 1
                elif next_stripped in ['}', '})', '];', '}:', '},']:
                    # Closing braces can have different indent
                    code_block.append(next_line)
                    j += 1
                else:
                    # Code block ended
                    break
            
            # Only fence if we have at least 4 lines of actual code
            code_lines = [l for l in code_block if l.strip()]
            if len(code_lines) >= 4:
                # Determine language from content
                code_text = '\n'.join(code_block)
                if any(kw in code_text for kw in ['#include', 'int ', 'void ', 'scanf', 'printf', 'std::', 'scanf', 'static_cast']):
                    lang = 'cpp'
                else:
                    lang = 'python'
                
                # Remove common indentation (4 spaces)
                dedented = [l[4:] if l.startswith('    ') else l for l in code_block]
                
                result.append(f'```{lang}')
                result.extend(dedented)
                result.append('```')
                i = j
            else:
                # Too short, keep as is
                result.append(line)
                i += 1
        else:
            result.append(line)
            i += 1
    
    return '\n'.join(result)


def _protect_raw_code_blocks(text: str) -> str:
    """
    Protect raw code blocks (C++/Python/etc.) that appear after markers like "Sample code:"
    without triple backticks. Wrap them in ``` so they're preserved.
    """
    # Match from marker through all code until we hit prose paragraph
    # But skip if code is already fenced
    def wrap_in_fence_cpp(match):
        content = match.group(0)
        # Don't double-fence
        if '```' in content:
            return content
        return f"```cpp\n{content}\n```"
    
    def wrap_in_fence_python(match):
        content = match.group(0)
        # Don't double-fence
        if '```' in content:
            return content
        return f"```python\n{content}\n```"
    
    # C/C++ code (starts with #include)
    text = re.sub(
        r'(Sample code:|Solution Code:|Code:)\s*\n+#include\b.*?(?=\n{2,}[A-Z][a-z]+(?:[^a-zA-Z]|(?:\s+[a-z]+))[^\n]*[.?!]|\n{2,}#{2,}|\n{2,}\*\*[A-Z]|\n{2,}Step\s+\d+:|\Z)',
        wrap_in_fence_cpp,
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # Python code (starts with import, def, or class)
    text = re.sub(
        r'(Sample code:|Solution Code:|Code:)\s*\n+(?:import\s+|from\s+\w+\s+import\s+|def\s+\w+|class\s+\w+).*?(?=\n{2,}[A-Z][a-z]+(?:[^a-zA-Z]|(?:\s+[a-z]+))[^\n]*[.?!]|\n{2,}#{2,}|\n{2,}\*\*[A-Z]|\n{2,}Step\s+\d+:|\Z)',
        wrap_in_fence_python,
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # Python code blocks that start with variable assignments (informal pseudocode)
    # Match patterns like "The code for that would be:" or "code would be something like:" followed by variable assignments
    # This handles cases where each line of code is separated by double newlines
    def wrap_pseudocode(match):
        marker = match.group(1)  # "The code for that would be:" or similar
        code = match.group(2)     # The actual code
        return f"{marker}\n\n```python\n{code}\n```"
    
    text = re.sub(
        r'((?:The\s+)?code\s+(?:in\s+\w+\s+)?(?:for\s+that\s+)?would\s+be(?:\s+something\s+like)?:|code\s+is:)\s*\n{1,2}((?:def\s+\w+|[a-zA-Z_]\w*\s*=).*?)(?=\n{2,}(?:Then|This|That|So|But|Now|Wait|Hmm|The\s+[A-Z]|Testing)[^\n]*[.?!]|\Z)',
        wrap_pseudocode,
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # Code blocks after "as follows:" or "The formula:" with variable assignments
    # These often have mixed indentation and double newlines between statements
    # IMPORTANT: This must run BEFORE _detect_indented_code_blocks to catch the full block
    def wrap_formula_code(match):
        intro = match.group(1)     # "as follows:" or similar intro
        marker = match.group(2)    # Optional "The formula:" etc
        code = match.group(3)      # The actual code
        if marker:
            return f"{intro}\n\n{marker}\n\n```python\n{code}\n```"
        else:
            return f"{intro}\n\n```python\n{code}\n```"
    
    text = re.sub(
        r'((?:coding\s+in\s+\w+\s+)?as\s+follows:)(?:\s*\n{1,2}(The\s+formula:))?\s*\n{1,2}([a-zA-Z_]\w*\s*=.*?)(?=\n{2,}(?:So|Wait|Thus|Then|This|That)[^\n]*[.?!]|\Z)',
        wrap_formula_code,
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # Finally, detect and fence remaining indented code blocks
    # This runs last so it doesn't interfere with marker-based patterns above
    text = _detect_indented_code_blocks(text)
    
    return text


def _looks_like_math_line(line: str) -> bool:
    """
    Heuristic to detect if a line looks like a math expression without delimiters.
    Returns True if the line has mathematical characteristics.
    """
    line = line.strip()
    if not line or len(line) < 3:
        return False
    
    # Skip bullet points and list items - these are prose, not math equations
    # Patterns: "- text", "* text", "1. text", "   - text"
    if re.match(r'^[\s]*[-*+]\s+', line) or re.match(r'^[\s]*\d+\.\s+', line):
        return False
    
    # Skip lines that look like prose (have many words)
    # If line has 5+ words and ends with punctuation, it's likely prose
    words = re.findall(r'\b\w+\b', line)
    if len(words) >= 5 and line.rstrip().endswith(('.', '!', '?', ':')):
        return False
    
    # Skip lines with code compound assignment operators
    if re.search(r'[+\-*/]=', line):
        return False
    
    # Skip simple variable assignments (common in code)
    # These are too simple to be interesting math
    if re.match(r'^[a-zA-Z_]\w*\s*=\s*[0-9]+\s*$', line):
        return False
    if re.match(r'^[a-zA-Z_]\w*\s*=\s*[a-zA-Z_]\w*\s*$', line):
        return False
    
    # Count mathematical operators and symbols
    math_chars = sum(1 for c in line if c in '=≤≥≠<>±×÷∑∏∫∂∇√')
    
    # Has multiple math operators (strong signal)
    if math_chars >= 2:
        return True
    
    # Single equation with mathematical expression (not just simple assignment)
    if '=' in line and math_chars >= 1:
        # Check for mathematical operations around the =
        if re.search(r'[+\-*/\^]', line):
            return True
        # Or LaTeX-style patterns
        if re.search(r'\\[a-zA-Z]+|\{|\}|_\{|\^\{', line):
            return True
    
    # Has fraction-like patterns: a/b where a and b could be expressions
    if re.search(r'\w+\s*/\s*\w+', line) and not _looks_like_domain_or_file(line):
        return True
    
    # Has superscript/subscript notation patterns
    if re.search(r'[a-zA-Z]\^[0-9{]|\w+_[0-9{]', line):
        return True
    
    return False


def _protect_raw_math_expressions(text: str) -> str:
    """
    Detect and wrap raw math expressions (without $ delimiters) in display math $$ $$.
    Also converts LaTeX \\[...\\] display math to $$ $$ for consistent handling.
    This helps preserve multi-line mathematical derivations.
    """
    # First, convert \[...\] to $$...$$ for consistent handling
    # In the text, \[ appears as literal backslash + [
    # In regex, we need r'\\\[' which matches \ followed by [
    text = re.sub(r'\\\[', '$$', text)
    text = re.sub(r'\\\]', '$$', text)
    
    lines = text.split('\n')
    protected_lines = []
    in_math_block = False
    in_code_block = False
    math_block = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Track code blocks to avoid detecting code as math
        # Check if ``` appears anywhere in the line (not just at start)
        if '```' in line:
            in_code_block = not in_code_block
            if in_math_block and math_block:
                # Flush accumulated math before code block
                protected_lines.append('$$\n' + '\n'.join(math_block) + '\n$$')
                math_block = []
                in_math_block = False
            protected_lines.append(line)
            continue
        
        # Skip processing inside code blocks
        if in_code_block:
            protected_lines.append(line)
            continue
        
        # Skip if already protected with $
        if '$' in line:
            if in_math_block and math_block:
                # Flush accumulated math
                protected_lines.append('$$\n' + '\n'.join(math_block) + '\n$$')
                math_block = []
                in_math_block = False
            protected_lines.append(line)
            continue
        
        # Check if this line looks like math
        is_math = _looks_like_math_line(line)
        
        if is_math:
            if not in_math_block:
                in_math_block = True
            math_block.append(line)
        else:
            # Not math - flush any accumulated math block
            if in_math_block and math_block:
                # Only wrap if we have 2+ consecutive math lines
                if len(math_block) >= 2:
                    protected_lines.append('$$\n' + '\n'.join(math_block) + '\n$$')
                else:
                    # Single line - just keep as is
                    protected_lines.extend(math_block)
                math_block = []
                in_math_block = False
            protected_lines.append(line)
    
    # Flush any remaining math block
    if math_block:
        if len(math_block) >= 2:
            protected_lines.append('$$\n' + '\n'.join(math_block) + '\n$$')
        else:
            protected_lines.extend(math_block)
    
    return '\n'.join(protected_lines)


def _split_text_into_math_code_and_plain(text: str) -> List[Tuple[str, str]]:
    """
    Split into ('math', chunk), ('code', chunk), or ('text', chunk).
    First separates $$ blocks, then code blocks, then plain text.
    """
    parts: List[Tuple[str, str]] = []
    
    # First pass: separate display math blocks $$...$$
    segments = []
    current = []
    in_display_math = False
    i = 0
    while i < len(text):
        if i < len(text) - 1 and text[i:i+2] == '$$':
            if not in_display_math:
                # Start of math block
                if current:
                    segments.append(('text', ''.join(current)))
                    current = []
                in_display_math = True
                current.append('$$')
                i += 2
            else:
                # End of math block
                current.append('$$')
                segments.append(('math', ''.join(current)))
                current = []
                in_display_math = False
                i += 2
        else:
            current.append(text[i])
            i += 1
    
    if current:
        typ = 'math' if in_display_math else 'text'
        segments.append((typ, ''.join(current)))
    
    # Second pass: for text segments, separate code blocks
    final_parts = []
    for typ, content in segments:
        if typ == 'math':
            final_parts.append(('math', content))
        else:
            # Split by code fences
            code_parts = _split_text_into_code_and_plain(content)
            final_parts.extend(code_parts)
    
    return final_parts


def _split_text_into_code_and_plain(text: str) -> List[Tuple[str, str]]:
    """
    Split into ('code', chunk) or ('text', chunk) using ``` fences.
    Handles cases where ``` may appear mid-line with surrounding text.
    Unmatched fences are treated best-effort.
    """
    parts: List[Tuple[str, str]] = []
    in_code = False
    plain_buf: List[str] = []
    code_buf: List[str] = []

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        
        # Check if line contains ``` (possibly with text before/after)
        if "```" in line:
            fence_pos = line.find("```")
            
            if not in_code:
                # Opening fence
                # Text before fence goes to plain buffer
                before_fence = line[:fence_pos]
                if before_fence.strip():
                    plain_buf.append(before_fence.rstrip() + '\n')
                
                # Flush plain text
                if plain_buf:
                    parts.append(("text", "".join(plain_buf)))
                    plain_buf = []
                
                # Start code block with fence line (from fence onward)
                in_code = True
                code_buf = [line[fence_pos:]]
            else:
                # Closing fence
                # Add everything up to and including fence to code buffer
                end_fence = fence_pos + 3
                code_buf.append(line[:end_fence])
                
                # Flush code block
                parts.append(("code", "".join(code_buf)))
                code_buf = []
                in_code = False
                
                # Text after fence starts new plain buffer
                after_fence = line[end_fence:]
                if after_fence.strip():
                    plain_buf.append(after_fence.lstrip())
        else:
            # No fence in this line
            if in_code:
                code_buf.append(line)
            else:
                plain_buf.append(line)

    # flush any leftovers
    if in_code and code_buf:
        parts.append(("code", "".join(code_buf)))
    elif plain_buf:
        parts.append(("text", "".join(plain_buf)))

    return parts


_ABBREVIATIONS = {
    "e.g.", "i.e.", "etc.", "vs.", "dr.", "mr.", "mrs.", "ms.",
    "prof.", "sr.", "jr.", "fig.", "eq.", "approx."
}


def _looks_like_domain_or_file(token: str) -> bool:
    """
    Heuristic: token looks like a URL, domain, or filename.
    e.g., "example.com", "file.txt", "script.py"
    """
    if "http://" in token or "https://" in token:
        return True
    # something.xxx[/...]
    if re.search(r"\.[a-zA-Z0-9]{2,5}(/|$)", token):
        return True
    return False


def _is_list_marker(token: str) -> bool:
    """
    Numbered / lettered list markers like "1.", "2)", "(a)", "(b)".
    """
    token = token.strip()
    if re.fullmatch(r"\d+[\.\)]", token):
        return True
    if re.fullmatch(r"\([a-zA-Z]\)", token):
        return True
    return False


def _split_plain_text_into_sentences(text: str) -> List[str]:
    r"""
    Heuristic sentence splitter for math / code-ish reasoning traces.
    Improvements:
      - Only splits on '.', '?', '!' when NOT inside:
          * parentheses / brackets / braces: (...), [...], {...}
          * LaTeX math with $...$
          * LaTeX environments \begin{...} ... \end{...}
          * Markdown bold (**text**) or italic (*text*, _text_)
      - Avoids splitting on:
          * != operator (exclamation followed by equals)
          * decimals, abbreviations, domains, list markers
          * ellipsis (...)
          * method calls (word.word())
      - Treats newlines as spaces inside sentences.
    """
    text = text.strip()
    if not text:
        return []

    sentences: List[str] = []
    buf: List[str] = []
    n = len(text)
    i = 0

    # Context tracking
    paren_depth = 0    # ()
    bracket_depth = 0  # []
    brace_depth = 0    # {}
    env_depth = 0      # \begin{...}...\end{...}
    in_math_dollar = False  # $...$
    in_markdown_bold = False  # **text**

    while i < n:
        ch = text[i]
        buf.append(ch)

        # Track LaTeX environments \begin{...} ... \end{...}
        if text.startswith(r'\begin{', i):
            env_depth += 1
        elif text.startswith(r'\end{', i):
            if env_depth > 0:
                env_depth -= 1

        # Track markdown bold **text**
        if ch == '*' and i + 1 < n and text[i + 1] == '*':
            in_markdown_bold = not in_markdown_bold

        # Track bracket depths
        if ch == '(':
            paren_depth += 1
        elif ch == ')':
            if paren_depth > 0:
                paren_depth -= 1
        elif ch == '[':
            bracket_depth += 1
        elif ch == ']':
            if bracket_depth > 0:
                bracket_depth -= 1
        elif ch == '{':
            brace_depth += 1
        elif ch == '}':
            if brace_depth > 0:
                brace_depth -= 1
        elif ch == '$':
            # Toggle simple $...$ math mode
            in_math_dollar = not in_math_dollar

        # Sentence-ending punctuation
        if ch in ".?!":
            prev_char = text[i - 1] if i > 0 else ""
            next_char = text[i + 1] if i + 1 < n else ""
            next_next_char = text[i + 2] if i + 2 < n else ""

            # != operator: don't split on the !
            if ch == "!" and next_char == "=":
                i += 1
                continue

            # Ellipsis: ... -> don't split, treat as continuation
            if ch == "." and next_char == "." and next_next_char == ".":
                i += 3  # Skip all three dots
                buf.append(next_char)
                buf.append(next_next_char)
                continue

            # Do NOT end sentence while inside any structured container
            if (
                paren_depth > 0
                or bracket_depth > 0
                or brace_depth > 0
                or env_depth > 0
                or in_math_dollar
                or in_markdown_bold
            ):
                i += 1
                continue

            # Decimal number: digit '.' digit -> don't split
            if ch == "." and prev_char.isdigit() and next_char.isdigit():
                i += 1
                continue

            # Method call or attribute access: word.word( -> don't split
            # Look back to see if we have identifier before '.'
            # Look forward to see if we have identifier or '(' after '.'
            if ch == ".":
                # Check if it's a method call like result.append() or obj.attr
                lookback = i - 1
                while lookback >= 0 and (text[lookback].isalnum() or text[lookback] == '_'):
                    lookback -= 1
                # Now lookback is at the character before the identifier
                has_identifier_before = lookback < i - 1
                
                lookforward = i + 1
                while lookforward < n and text[lookforward].isspace():
                    lookforward += 1
                has_identifier_or_paren_after = (lookforward < n and 
                    (text[lookforward].isalnum() or text[lookforward] == '_' or text[lookforward] == '('))
                
                if has_identifier_before and has_identifier_or_paren_after:
                    i += 1
                    continue

            # Figure out the token containing this punctuation
            start_word = i
            while start_word > 0 and not text[start_word - 1].isspace():
                start_word -= 1
            end_word = i
            while end_word + 1 < n and not text[end_word + 1].isspace():
                end_word += 1
            token = text[start_word : end_word + 1]

            # Normalize token for checks
            token_clean = token.lower().rstrip(",;:")

            # Abbreviations like "etc.,", "e.g.," -> don't split
            if token_clean in _ABBREVIATIONS:
                i += 1
                continue

            # URLs / domains / filenames -> don't split
            if _looks_like_domain_or_file(token_clean):
                i += 1
                continue

            # Numbered list marker like "1.", "2)" -> don't split
            if _is_list_marker(token_clean):
                i += 1
                continue

            # Real sentence boundary
            sentence = "".join(buf).strip()
            if sentence:
                sentences.append(sentence)
                buf = []
            i += 1
            continue

        # Handle newlines: convert to space, but double newlines may indicate paragraph breaks
        if ch in "\r\n":
            # Check if this is part of a double newline (paragraph break)
            if i + 1 < n and text[i + 1] in "\r\n":
                # Double newline - treat as sentence boundary if we have content
                sentence = "".join(buf).strip()
                if sentence:
                    sentences.append(sentence)
                    buf = []
                # Skip the second newline
                i += 1
                if i + 1 < n and text[i + 1] in "\r\n":
                    i += 1  # Skip \r\n\r\n case
            else:
                # Single newline - treat as space
                buf[-1] = " "

        i += 1

    leftover = "".join(buf).strip()
    if leftover:
        sentences.append(leftover)

    return sentences


def _merge_leading_punct(sentences: List[str]) -> List[str]:
    """
    If a segment starts with only closing punctuation or leading punctuation
    (quotes, brackets, semicolon, colon), merge it into the previous segment.
    """
    if not sentences:
        return []

    leading_chars = {":", ";", ")", "]", "}", '"', """, """, "'", "'"}

    new: List[str] = []
    for s in sentences:
        stripped = s.lstrip()
        if new and stripped and stripped[0] in leading_chars:
            new[-1] = new[-1].rstrip() + " " + stripped
        else:
            new.append(s)
    return new


def _merge_short_segments(sentences: List[str], min_tokens: int = 5) -> List[str]:
    """
    Merge very short segments into their previous neighbor, based on token length.
    This helps avoid standalone fragments like "Hmm." or "Yes." becoming separate segments.
    """
    if not sentences:
        return []

    new: List[str] = [sentences[0]]
    for s in sentences[1:]:
        tokens = rough_token_count(s)
        if tokens < min_tokens:
            new[-1] = new[-1].rstrip() + " " + s.lstrip()
        else:
            new.append(s)
    return new


def _merge_short_math_code_sentences(sentences: List[str], max_words: int = 30) -> List[str]:
    """
    Second pass: Merge consecutive short sentences that both contain math/code symbols.
    
    This catches terse computational steps like:
    - "i=0 (company A):"
    - "temp is (1-p1) * (1-p2)."
    - "p1 is 1 →1-p1 =0. So temp is (0) * (0)."
    
    Also merges when current sentence ends with a short math line and next is short math.
    
    Args:
        sentences: List of sentences after first-pass merging
        max_words: Maximum word count to consider a sentence "short"
    
    Returns:
        List with short math/code sentences merged
    """
    import re
    
    if len(sentences) <= 1:
        return sentences
    
    # Symbols indicating math or code content
    math_code_symbols = [r'=', r'\+', r'->', r'→', r'⇒', r'\*', r'/', r'\(', r'\)', 
                         r'\[', r'\]', r'\{', r'\}', r'<', r'>', r'\$', r'_', r'\^', r'\|']
    
    def has_math_code(text: str) -> bool:
        """Check if text contains math or code symbols."""
        return any(re.search(pattern, text) for pattern in math_code_symbols)
    
    def is_short(text: str, max_words: int) -> bool:
        """Check if text is short (word count <= max_words)."""
        word_count = len(text.split())
        return word_count <= max_words
    
    def last_line_is_short_math(text: str, max_words: int) -> bool:
        """Check if the last line of a multi-line sentence is short and has math."""
        lines = text.strip().split('\n')
        if not lines:
            return False
        last_line = lines[-1]
        return is_short(last_line, max_words) and has_math_code(last_line)
    
    merged = []
    i = 0
    
    while i < len(sentences):
        current = sentences[i]
        
        # Look ahead to merge short math/code sentences
        while i + 1 < len(sentences):
            next_sent = sentences[i + 1]
            
            # Merge if:
            # 1. Both are short and both have math/code, OR
            # 2. Current ends with short math line and next is short math
            should_merge = (
                (is_short(current, max_words) and is_short(next_sent, max_words) and
                 has_math_code(current) and has_math_code(next_sent)) or
                (last_line_is_short_math(current, max_words) and 
                 is_short(next_sent, max_words) and has_math_code(next_sent))
            )
            
            if should_merge:
                # Merge next sentence into current
                current = current.rstrip() + "\n" + next_sent.lstrip()
                i += 1
            else:
                break
        
        merged.append(current)
        i += 1
    
    return merged


def _merge_math_derivation_chains(sentences: List[str]) -> List[str]:
    """
    Merge consecutive sentences that form a continuous mathematical derivation or list structure.
    
    Keeps together:
    - Math derivation chains (equations with continuations)
    - Bullet/numbered list items (-, *, 1., 2., etc.)
    - Edge case enumerations
    
    This prevents the scorer from splitting logically related content.
    """
    import re
    
    if len(sentences) <= 1:
        return sentences
    
    continuation_words = {'therefore', 'thus', 'so', 'hence', 'then', 'substituting', 'right', 'left'}
    math_patterns = [r'\$\$', r'⇒', r'→', r'≥', r'≤', r'=', r'\^', r'\_', r'\\']
    # Patterns for calculation labels that are part of derivations
    calculation_labels = [r'right\s+side', r'left\s+side', r'simplify', r'expand']
    # List item patterns: "- ", "* ", "1. ", "2. ", etc.
    list_item_pattern = r'^\s*[-*•]|\d+\.'
    # Algorithmic pseudocode patterns
    algo_keywords = [r'^\s*for each\b', r'^\s*compute\b', r'^\s*calculate\b', r'^\s*initialize\b', 
                     r'^\s*return\b', r'^\s*if\b', r'^\s*else\b', r'^\s*while\b', r'^\s*set\b']
    
    def has_math(text: str) -> bool:
        """Check if text contains mathematical notation or is a calculation label."""
        text_lower = text.lower()
        # Check for calculation labels (like "Right side: 0")
        if any(re.search(pattern, text_lower) for pattern in calculation_labels):
            return True
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in math_patterns)
    
    def is_list_item(sent: str) -> bool:
        """Check if sentence is a list item (bullet or numbered)."""
        return bool(re.match(list_item_pattern, sent.strip()))
    
    def is_algo_step(sent: str) -> bool:
        """Check if sentence is an algorithmic pseudocode step."""
        sent_lower = sent.lower()
        return any(re.search(pattern, sent_lower, re.IGNORECASE) for pattern in algo_keywords)
    
    def contains_algo_keywords(sent: str) -> bool:
        """Check if sentence contains algorithmic keywords anywhere (not just at start)."""
        sent_lower = sent.lower()
        return any(re.search(r'\b' + keyword.replace(r'^\s*', '').replace(r'\b', ''), sent_lower) 
                   for keyword in algo_keywords)
    
    def ends_with_colon(sent: str) -> bool:
        """Check if sentence ends with colon (like 'for each company i:')."""
        return sent.strip().endswith(':')
    
    def has_assignment_or_operation(sent: str) -> bool:
        """Check if sentence has variable assignment or operations (=, +=, -=, *=, product, sum)."""
        return bool(re.search(r'(total|prob|sum|product|count)\s*[+\-*/]?=|product_|sum_', sent.lower()))
    
    def starts_with_continuation(sent: str) -> bool:
        """Check if sentence starts with continuation word."""
        first_word = sent.lower().strip().split()[0] if sent.strip() else ""
        return any(first_word.startswith(word) for word in continuation_words)
    
    def starts_with_contradiction(sent: str) -> bool:
        """Check if sentence starts with contradiction/negation words (But, However, This contradicts, etc)."""
        sent_lower = sent.lower().strip()
        contradiction_starts = ['but ', 'however ', 'this contradicts', 'this is a contradiction', 
                                'which contradicts', 'which is impossible', 'this is impossible']
        return any(sent_lower.startswith(start) for start in contradiction_starts)
    
    def starts_with_assumption(sent: str) -> bool:
        """Check if sentence starts with assumption words (Suppose, Assume, Let's say, etc)."""
        sent_lower = sent.lower().strip()
        assumption_starts = ['suppose ', 'assume ', "let's say ", 'say ', 'if we assume', 'consider the case']
        return any(sent_lower.startswith(start) for start in assumption_starts)
    
    def ends_with_incomplete_math(sent: str) -> bool:
        """Check if sentence ends with incomplete math (equation mid-derivation)."""
        sent = sent.strip()
        if not has_math(sent):
            return False
        # Ends with =, ⇒, or equation but no period/conclusion
        if sent.endswith(('=', '⇒', '→', '...')):
            return True
        # Has math and doesn't end with sentence-ending punctuation
        if has_math(sent) and not sent.endswith(('.', '!', '?', ':', ';')):
            return True
        return False
    
    merged = []
    i = 0
    
    while i < len(sentences):
        current = sentences[i]
        
        # Look ahead to merge continuous math derivations or list items
        while i + 1 < len(sentences):
            next_sent = sentences[i + 1]
            
            # Check if we should merge
            should_merge = (
                # ALWAYS merge if current ends with colon (introduces what follows)
                ends_with_colon(current) or
                # Math derivation continuation
                (ends_with_incomplete_math(current) and has_math(next_sent)) or
                (has_math(current) and starts_with_continuation(next_sent) and has_math(next_sent)) or
                # List items (both current and next are list items)
                (is_list_item(current) and is_list_item(next_sent)) or
                # Algorithmic pseudocode (both are algo steps)
                (is_algo_step(current) and is_algo_step(next_sent)) or
                # Algorithmic pseudocode (current has algo keywords, next starts with algo or has assignment)
                (contains_algo_keywords(current) and (is_algo_step(next_sent) or has_assignment_or_operation(next_sent))) or
                # Algorithmic pseudocode (current has assignment, next has algo keywords or assignment)
                (has_assignment_or_operation(current) and (contains_algo_keywords(next_sent) or has_assignment_or_operation(next_sent)))
            )
            
            if should_merge:
                # Merge next sentence into current
                current = current.rstrip() + "\n" + next_sent.lstrip()
                i += 1
            else:
                break
        
        merged.append(current)
        i += 1
    
    return merged


def split_into_sentences(text: str, min_tokens: int = 5, extract_think: bool = True) -> Tuple[List[str], Optional[str], Optional[str]]:
    """
    Top-level splitter:
      - Optionally extracts and handles <think>...</think> tags
      - First wraps raw code blocks (like "Sample code:" sections) in triple backticks
      - Detects and protects raw math expressions (without $ delimiters) by wrapping in $$
      - Preserves fenced code blocks (```...```) as single segments
      - Preserves math blocks ($$...$$) as single segments
      - Heuristically splits plain text into sentences with context awareness
      - Avoids splitting inside parentheses, brackets, LaTeX math, etc.
      - Merges leading punctuation segments
      - Merges ultra-short fragments by token count
    
    Args:
        text: Input text (may contain <think> tags)
        min_tokens: Minimum token count for standalone segments
        extract_think: If True, extract and return think tag boundaries
    
    Returns:
        (sentences, prefix, suffix) where:
        - sentences: list of sentence strings (only from thinking content)
        - prefix: content before <think> tag (None if no tags or extract_think=False)
        - suffix: content after </think> tag (None if no tags or extract_think=False)
    """
    text = text.strip()
    if not text:
        return [], None, None

    # Step 0: Extract think content if requested
    prefix, thinking, suffix = None, None, None
    if extract_think:
        prefix, thinking, suffix = extract_think_content(text)
        # If there were no think tags, prefix and suffix will be empty strings
        # Convert empty strings to None for clarity
        prefix = prefix if prefix else None
        suffix = suffix if suffix else None
        text = thinking
    
    # Step 1: Protect raw code blocks by wrapping in fences
    text = _protect_raw_code_blocks(text)
    
    # Step 2: Protect raw math expressions by wrapping in $$
    text = _protect_raw_math_expressions(text)

    # Step 3: Split into math, code, and plain text chunks
    parts = _split_text_into_math_code_and_plain(text)
    segments: List[str] = []

    for kind, chunk in parts:
        if kind in ("code", "math"):
            s = chunk.strip()
            if s:
                segments.append(s)
        else:  # plain text
            sents = _split_plain_text_into_sentences(chunk)
            segments.extend(sents)

    # Step 4: Post-process across the whole sequence
    segments = _merge_leading_punct(segments)
    segments = _merge_short_segments(segments, min_tokens=min_tokens)
    
    # Step 5: Merge mathematical derivation chains (optional, enabled by default)
    # This keeps continuous math derivations together for better segmentation later
    segments = _merge_math_derivation_chains(segments)
    
    # Step 6: Second pass - merge short consecutive math/code sentences
    # This keeps terse computational steps together
    segments = _merge_short_math_code_sentences(segments)

    # Final cleanup
    segments = [s.strip() for s in segments if s.strip()]
    return segments, prefix, suffix


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Split chains-of-thought into sentences')
    parser.add_argument('--input-dir', type=str, default='stage1_seed_select',
                        help='Input directory from Stage 1')
    parser.add_argument('--output-dir', type=str, default='stage2_sentence_split',
                        help='Output directory for sentence splits')
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
    
    # Process each task
    results = []
    for task in tasks:
        task_id = task['task_id']
        cot = task.get('cot', task.get('full_cot', ''))
        
        # Split into sentences (extracts think tags if present)
        sentences, prefix, suffix = split_into_sentences(cot, extract_think=True)
        
        result = {
            'task_id': task_id,
            'num_sentences': len(sentences),
            'sentences': sentences
        }
        
        # Store think tag boundaries if they existed
        if prefix is not None:
            result['think_prefix'] = prefix
        if suffix is not None:
            result['think_suffix'] = suffix
        
        # Preserve original fields if needed
        if 'prompt' in task:
            result['prompt'] = task['prompt']
        if 'response' in task:
            result['response'] = task['response']
        
        results.append(result)
        
        tag_info = ""
        if prefix is not None or suffix is not None:
            tag_info = " (with <think> tags)"
        print(f"  {task_id}: {len(sentences)} sentences{tag_info}")
        
        # Save example (only in test mode)
        if args.test:
            example_path = examples_dir / f"{task_id}.txt"
            with open(example_path, 'w') as f:
                if prefix is not None:
                    f.write("=== PREFIX (before <think>) ===\n")
                    f.write(prefix + "\n\n")
                    f.write("=== THINKING (inside <think>...</think>) ===\n\n")
                
                for i, sent in enumerate(sentences):
                    f.write(f"[{i}] {sent}\n\n")
                
                if suffix is not None:
                    f.write("\n=== SUFFIX (after </think>) ===\n")
                    f.write(suffix + "\n")
            print(f"    Saved example: {task_id}.txt")
    
    # Write results
    output_jsonl = output_dir / 'data.jsonl'
    with open(output_jsonl, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\nStage 2 complete! Output in {output_dir}/")
    print(f"  - data.jsonl: {len(results)} tasks with sentences")
    if args.test:
        print(f"  - examples/: {len(results)} .txt files")


if __name__ == '__main__':
    main()
