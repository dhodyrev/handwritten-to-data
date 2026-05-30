"""All Qwen-VL prompts in one place. Lifted from the official baseline notebook.

Phase 1 extension: ``with_few_shot`` injects in-context examples per writing_type.
"""
from __future__ import annotations


QWEN_CLASSIFY_PAGE_PROMPT = """Look at this school/student page. What type of content dominates?
Return JSON: {"page_type": "text"} or {"page_type": "math"}
- "text"  — mostly handwritten text (language, literature, history, biology,
            geography, social studies, essays)
- "math"  — formulas, equations, numbers, calculations, tables (math, physics,
            chemistry problems)
ONLY valid JSON."""


QWEN_BLOCK_DETECT_PROMPT = """Detect content BLOCKS in this document image.

block_type:
- "text_block"  (paragraph of text lines)
- "table"       (rows + columns)
- "formula"     (standalone math/chemistry equation on its OWN line, NOT inline)
- "image"       (stamp / seal / drawing)
- "graph"       (chart / diagram with axes)
- "annotation"  (grade / date / page number / exercise number / teacher mark / signature)

writing_type (per block):
- "handwritten" — written by hand
- "printed"     — typed / typeset / machine-printed / stamped

Rules:
- Merge consecutive text lines into ONE text_block
- Full-page text = ONE text_block
- Tables, annotations = ALWAYS separate blocks
- Formula = separate block ONLY if it stands alone on its own line(s)
- Text above/below a table = separate text_blocks
- A page can MIX handwritten and printed blocks (typed body + handwritten
  signature/note). Classify writing_type per block, not per page.
- Skip vertical text (rotated > 45°)
- Ignore ruled/grid/squared background lines — they are NOT text

Return: {"blocks": [{"bbox_2d": [x1,y1,x2,y2], "block_type": "text_block", "writing_type": "handwritten"}]}
Coordinates 0-1000 scale. ONLY valid JSON."""


QWEN_LINE_DETECT_PROMPT = """Detect each TEXT LINE in this image (one visual row = one line).
Return: {"lines": [{"bbox_2d": [x1,y1,x2,y2]}]}
Coordinates 0-1000 scale, top-to-bottom order. ONLY valid JSON."""


QWEN_SCHOOL_PERLINE_PROMPT = """Detect every LINE in this school notebook page (grades 5-11, various subjects).

Each line = one visual row. Do NOT merge multiple lines.
Ignore ruled/grid lines in the background — they are NOT text.

For each line, classify:
- "formula"     — line contains math/chemistry: equations (=), variables,
                  fractions, chemical formulas
- "text_block"  — plain text (Ukrainian language, literature, geography,
                  history, etc.)
- "annotation"  — date, page number, exercise number (e.g. "Вправа 430"),
                  grade, teacher mark
- "table"       — ENTIRE vertical calculation block with a dividing line.
                  ONE bbox covering ALL rows.

Return: {"blocks": [{"bbox_2d": [x1,y1,x2,y2], "block_type": "text_block"}]}
Coordinates 0-1000 scale. ONLY valid JSON."""


QWEN_UNIVERSITY_PERLINE_PROMPT = """Detect every LINE in this university student work (math, formulas, text).

Each line = one visual row. Do NOT merge multiple lines.

For each line, classify:
- "formula"     — line contains math symbols, equations, set notation,
                  matrices, variables, inequalities
- "text_block"  — plain text without math
- "annotation"  — grade, date, task number
- "image"       — drawing, diagram without axes
- "graph"       — chart with coordinate axes
- "table"       — table row

Return: {"blocks": [{"bbox_2d": [x1,y1,x2,y2], "block_type": "formula"}]}
Coordinates 0-1000 scale. ONLY valid JSON."""


QWEN_TRANSCRIBE_PROMPT = """Transcribe ONLY the text physically written in this crop. Read left to right.

RULES:
- This crop contains ONE main line of text. Transcribe ONLY that central line.
  ANY text from the line above or below — even if it looks fully written — is OUT OF SCOPE. Do NOT include it. Output the central line only.
- If you cannot read a word, write the LETTERS you see — do NOT substitute a similar real word from context.
- Numbered list lines start with a DIGIT + period ("1.", "2.", "3.", "10."). Never a Cyrillic letter (а., в.) when it is clearly a digit.
- Ukrainian Cyrillic ONLY, never Latin. These glyphs are Cyrillic, not Latin: В Н С Р Т К Е О А М Х.
- Edge cut-off: visible part with hyphen ("вирі-"). Crossed-out: ~~закреслено~~ or ~~old~~{new}. No memory completion.
- Formulas: plain text with spacing ("2x + 3 = 7", "H₂SO₄"). Tables: cells separated by |.

Return JSON: {"text": "..."}"""


QWEN_TRANSCRIBE_LINES_PROMPT = """Transcribe EVERY handwritten text line in this image, top to bottom.

RULES:
- One JSON array entry per visual line (one written row = one entry). Do NOT merge
  two lines into one entry; do NOT split one line across entries.
- Transcribe ONLY text physically written here. No completion from memory.
- If you cannot read a word, write the LETTERS you see — do NOT substitute a
  similar real word from context.
- Numbered list lines start with a DIGIT + period ("1.", "2.", "10."). Never a
  Cyrillic letter (а., в.) when it is clearly a digit.
- Ukrainian Cyrillic ONLY, never Latin. These glyphs are Cyrillic, not Latin:
  В Н С Р Т К Е О А М Х.
- Edge cut-off: keep the visible part with a hyphen ("вирі-"). Crossed-out:
  ~~text~~ or ~~old~~{new}.
- Inline formulas: plain text with spacing ("2x + 3 = 7", "H₂SO₄").

Return JSON: {"lines": ["first line", "second line", "..."]}"""


QWEN_FORMULA_PROMPT = """Transcribe this handwritten math/chemistry formula exactly as written.

Examples of correct transcription:
- "P(A) = 1/60 · 36 ? P(B) = 1/10"
- "x - 41 = 0"
- "|-57| · |x| = 0"
- "R₁ ∩ R₂ = {(a, b)}"
- "TC(Q) = 4Q^2 + 40Q + 400"
- "CH_3-COOH + NaOH → CH_3COONa + H_2O"
- "НСК (56, 35) = 2 * 2 * 2 * 7 * 5 = 280"
- "\\frac{d}{dx} f(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}"
- "\\sqrt{16} = 4"
- "S = 4 · 0,000 03 = 0,00012 м^2"

Rules:
- Use LaTeX for fractions (\\frac{}{}), roots (\\sqrt{}), limits (\\lim), arrows (\\to)
- Use ^{} for superscripts, _{} for subscripts: x^{2}, H_{2}O
- Simple cases: x^2, x_3 (no braces for single char)
- Ukrainian Cyrillic for text (В ≠ B, С ≠ C, Н ≠ H)
- Preserve exact notation as written

Return JSON: {"text": "..."}"""


QWEN_TABLE_PROMPT = ('Transcribe this table. Each row on a new line, cells separated by |. '
                    'Return JSON: {"text": "..."}')

QWEN_ANNOTATION_PROMPT = ('Transcribe this annotation (grade / date / page number / exercise number). '
                         'Return JSON: {"text": "..."}')


# ── Phase 1: few-shot extension ──────────────────────────────────────

def with_few_shot(prompt: str, examples: list[tuple[str, str]]) -> str:
    """Append a few-shot example block to a transcribe prompt.

    examples: list of (crop_description, gt_text). The crop image itself is
    NOT inlined here (Qwen3-VL only accepts a single image per turn in our
    vLLM config); examples are descriptive text + gold transcription.

    For richer few-shot with multiple images, switch to multi-turn messages
    in client.qwen_call — see docs/few-shot.md (TODO).
    """
    if not examples:
        return prompt
    block = "\n\nReference transcriptions from this dataset (do not copy verbatim — match the style):\n"
    for desc, gold in examples:
        block += f"- {desc!r} → {gold!r}\n"
    return prompt + block
