"""
evaluate_result_fix.py

Scores result_fix.json using the identical soft-match logic from evaluate.py.
Prints Overall, Level 1, Level 2, Level 3 accuracy.
Also prints a comparison against the original result.json scores,
and shows 3 example cases where the old parser failed but the fix recovered.

Does NOT touch evaluate.py, generate.py, or generate_batch.py.
"""
import json

RESULT_ORIG = "PATH_TO_YOUR_SCORE_ROOT/_app_models_Qwen2.5-3B-Instruct/result.json"
RESULT_FIX  = "PATH_TO_YOUR_SCORE_ROOT/_app_models_Qwen2.5-3B-Instruct/result_fix.json"

def score_result(results):
    record = {"c1": 0, "c2": 0, "c3": 0, "t1": 0, "t2": 0, "t3": 0}
    per_key = {}
    for key, result in results.items():
        tool_calls = result["tool_calls"]
        answer = result["data"]["answer"]
        if isinstance(answer, list):
            answer = answer[0]
        answer_name = answer["name"]
        answer_parameters = answer["parameters"]

        score = 0
        try:
            for tc in tool_calls:
                if isinstance(tc, str):
                    tc = json.loads(tc)
                if "name" not in tc or "parameters" not in tc:
                    name, parameters = answer_name, tc
                else:
                    name, parameters = tc["name"], tc["parameters"]
                if name == answer_name and parameters == answer_parameters:
                    score = 1
                    break
        except Exception:
            pass

        level = key[5]  # 'Level1_0' -> '1'
        if level == "1":
            record["t1"] += 1
            record["c1"] += score
        elif level == "2":
            record["t2"] += 1
            record["c2"] += score
        elif level == "3":
            record["t3"] += 1
            record["c3"] += score

        per_key[key] = score
    return record, per_key


orig_results = json.load(open(RESULT_ORIG, "r", encoding="utf-8"))
fix_results  = json.load(open(RESULT_FIX,  "r", encoding="utf-8"))

orig_record, orig_scores = score_result(orig_results)
fix_record,  fix_scores  = score_result(fix_results)

def acc(c, t):
    return round(c / t * 100, 2) if t > 0 else 0.0

print("=" * 55)
print(f"{'':30} {'ORIGINAL':>10} {'FIXED':>10}")
print("=" * 55)
for label, c_o, t_o, c_f, t_f in [
    ("Level 1", orig_record["c1"], orig_record["t1"], fix_record["c1"], fix_record["t1"]),
    ("Level 2", orig_record["c2"], orig_record["t2"], fix_record["c2"], fix_record["t2"]),
    ("Level 3", orig_record["c3"], orig_record["t3"], fix_record["c3"], fix_record["t3"]),
]:
    print(f"  {label:28} {acc(c_o,t_o):>9.2f}%  {acc(c_f,t_f):>9.2f}%")

c_orig = orig_record["c1"]+orig_record["c2"]+orig_record["c3"]
t_orig = orig_record["t1"]+orig_record["t2"]+orig_record["t3"]
c_fix  = fix_record["c1"]+fix_record["c2"]+fix_record["c3"]
t_fix  = fix_record["t1"]+fix_record["t2"]+fix_record["t3"]
print("-" * 55)
print(f"  {'Overall':28} {acc(c_orig,t_orig):>9.2f}%  {acc(c_fix,t_fix):>9.2f}%")
print("=" * 55)

# How many changed
flipped_wrong_to_right = [k for k in orig_scores if orig_scores[k]==0 and fix_scores[k]==1]
flipped_right_to_wrong = [k for k in orig_scores if orig_scores[k]==1 and fix_scores[k]==0]
print(f"\nExamples fixed (0→1): {len(flipped_wrong_to_right)}")
print(f"Examples broken (1→0): {len(flipped_right_to_wrong)}  (should be 0)")

# Show 3 recovered examples in detail
print("\n" + "=" * 55)
print("3 EXAMPLES: old parser FAILED, fix RECOVERED")
print("=" * 55)
for key in flipped_wrong_to_right[:3]:
    orig_r = orig_results[key]
    fix_r  = fix_results[key]
    answer = orig_r["data"]["answer"]
    if isinstance(answer, list):
        answer = answer[0]
    print(f"\n--- {key} ---")
    print(f"Expected name      : {answer['name']}")
    print(f"Expected params    : {answer['parameters']}")
    print(f"Old tool_calls     : {orig_r['tool_calls']}")
    print(f"Fixed tool_calls   : {fix_r['tool_calls']}")
    raw = orig_r["raw_output"]
    # show the relevant part of raw output
    snip_start = max(0, raw.find("<tool_call>") - 30)
    print(f"raw_output excerpt : ...{raw[snip_start:snip_start+300]}...")
