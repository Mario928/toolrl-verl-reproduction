"""
reparse_result_fix.py

Reads the existing result.json for the 3B raw baseline and re-extracts
tool_calls from raw_output using robust logic instead of split()[-1].

Original broken logic (generate.py line 91):
    tool_calls = assistant_output.split("<tool_call>")[-1].split("</tool_call>")[0].strip()

Problem: [-1] takes content after the LAST <tool_call> tag. When the model
outputs multiple <tool_call> opening tags, the correct JSON (in the first
block) is discarded and garbage after the last tag is taken instead.

Real output patterns observed across 597 examples:
  127 — <tool_call>{json}</tool_call>       (closed, 1 open)
  294 — <tool_call>{json}<tool_call>...     (unclosed — no </tool_call>)
   72 — <tool_call>{j1}<tool_call>{j2}</tool_call>  (multiple opens)
   48 — <obs>{json}</obs>                   (no <tool_call> at all)
   56 — neither tag                         (pure text responses)

Fix strategy per pattern:
  - If any <tool_call> exists: take content after the FIRST <tool_call>,
    then stop at the first </tool_call> OR the next <tool_call> OR end-of-string.
    This recovers the first call regardless of whether the block is closed.
  - If no <tool_call> but <obs> exists: take content of first <obs> block.
    (The model hallucinated the observation format but the JSON is correct.)
  - Otherwise: empty → no tool calls.

Does NOT touch generate.py, generate_batch.py, or evaluate.py.
Writes result_fix.json alongside the original result.json.
"""
import json
import re

RESULT_PATH = "PATH_TO_YOUR_SCORE_ROOT/_app_models_Qwen2.5-3B-Instruct/result.json"
OUTPUT_PATH = "PATH_TO_YOUR_SCORE_ROOT/_app_models_Qwen2.5-3B-Instruct/result_fix.json"


def extract_tool_calls(raw):
    """
    Return list of parsed tool-call dicts from raw model output.
    Handles closed blocks, unclosed blocks, multi-open blocks, and <obs> fallback.
    """
    if "<tool_call>" in raw:
        # Take everything after the FIRST <tool_call>
        after_first = raw.split("<tool_call>", 1)[1]
        # Stop at first </tool_call> or next <tool_call>, whichever comes first
        end = len(after_first)
        for stopper in ["</tool_call>", "<tool_call>"]:
            idx = after_first.find(stopper)
            if idx != -1 and idx < end:
                end = idx
        block = after_first[:end].strip()
    elif "<obs>" in raw:
        # Fallback: model hallucinated <obs> tag instead of <tool_call>
        m = re.search(r"<obs>(.*?)</obs>", raw, re.DOTALL)
        block = m.group(1).strip() if m else ""
    else:
        block = ""

    tool_calls = []
    for line in block.split("\n"):
        line = line.strip()
        if line:
            try:
                tool_calls.append(json.loads(line))
            except Exception:
                pass
    return tool_calls


results = json.load(open(RESULT_PATH, "r", encoding="utf-8"))

changed = 0
for key, result in results.items():
    raw = result.get("raw_output", "")
    new_tool_calls = extract_tool_calls(raw)
    old_tool_calls = result.get("tool_calls", [])
    if new_tool_calls != old_tool_calls:
        changed += 1
    result["tool_calls"] = new_tool_calls

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Total entries      : {len(results)}")
print(f"tool_calls changed : {changed}")
print(f"Written to         : {OUTPUT_PATH}")
