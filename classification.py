import os
import json
import re
from dotenv import load_dotenv
import openai
import pandas as pd

load_dotenv()
api_key = os.getenv("API_KEY")
client = openai.OpenAI(api_key=api_key)

CLASSES = ["Functional", "Non-Functional", "Others"]

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT CHAIN
# ─────────────────────────────────────────────────────────────────────────────

software_requirements = pd.read_csv("Trimmed Data Input.csv")

# ── Chain 1: Split requirements into atomic statements ────────────────────────
response = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {
            "role": "user",
            "content": f"""You are a requirements engineer.

Your task is to split requirements into atomic statements. Split any requirement if:
- It contains more than one distinct behavior, condition, or constraint, OR
- It is excessively long based on your judgment of length and complexity.

Each resulting statement must express exactly one intent and be concise.

When splitting a requirement with ID REQ-N, label the children REQ-Na, REQ-Nb, etc.
If a requirement does not need splitting, keep its original ID.

Do not classify or alter the text beyond splitting.

{software_requirements.to_string(index=False)}"""
        }
    ]
)
chain1 = response.choices[0].message.content
print("=== Chain 1 Output (Split) ===")
print(chain1)

# ── Chain 2: Refine grammar and clarity ───────────────────────────────────────
response2 = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {
            "role": "user",
            "content": f"""You are a requirements engineer.

Your task is to refine the split requirements below by:
1. Cleaning up grammar and clarity.
2. Ensuring each requirement is a complete, well-formed sentence.

Preserve all RequirementIDs exactly as given. Do not split, merge, or reclassify.

{chain1}"""
        }
    ]
)
chain2 = response2.choices[0].message.content
print("\n=== Chain 2 Output (Refined) ===")
print(chain2)

# ── Chain 3: Classify requirements → structured JSON output ───────────────────
response3 = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {
            "role": "user",
            "content": f"""You are a requirements engineer.

Your task is to classify each requirement below.

Classification rules:
- Functional: Describes what the system shall do (behavior, feature, process).
- Non-Functional: Describes how the system performs (quality attribute).
- Others: On-screen / UI display requirements only.

Return ONLY a valid JSON array — no markdown, no code fences, no explanation.
Each element must have exactly these three keys:
  "RequirementID", "Requirement", "Classification"

Example format:
[
  {{"RequirementID": "REQ-1", "Requirement": "The system shall ...", "Classification": "Functional"}},
  {{"RequirementID": "REQ-2a", "Requirement": "The system shall ...", "Classification": "Non-Functional"}}
]

Requirements to classify:
{chain2}"""
        }
    ]
)
chain3_raw = response3.choices[0].message.content

# ─────────────────────────────────────────────────────────────────────────────
# PARSE CHAIN 3 OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def parse_chain3(raw_text: str) -> pd.DataFrame:
    """
    Parse chain3 JSON output into a DataFrame.
    Strips markdown fences if the model includes them despite instructions.
    """
    # Strip ```json ... ``` or ``` ... ``` fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text).strip().rstrip("```").strip()

    try:
        records = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"chain3 did not return valid JSON. JSONDecodeError: {e}\n"
            f"Raw output:\n{raw_text}"
        )

    df = pd.DataFrame(records, columns=["RequirementID", "Requirement", "Classification"])
    return df


classified_df = parse_chain3(chain3_raw)

print("\n=== Chain 3 Output (Classified Requirements) ===")
print(classified_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# METRICS EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
 
def normalize_req_id(req_id: str) -> str:
    """
    Strip trailing letter suffix from split child IDs so they map back to
    their parent. E.g. 'REQ-1a' → 'REQ-1', 'REQ-F7a' → 'REQ-F7'.
    IDs without a suffix are returned unchanged.
    """
    return re.sub(r"^([A-Z]+-?[A-Z0-9]*\d)[a-z]$", r"\1", str(req_id).strip())
 
 
def evaluate_metrics(classified_df: pd.DataFrame, ground_truth_path: str) -> None:
    """
    Compare classified requirements against a ground truth CSV.
 
    Matching strategy:
    - The ground truth CSV and the original input CSV are row-aligned by
      position (row 0 in GT = row 0 in input). Since requirement IDs repeat
      across different projects, we match using a (RowIndex, ParentID) key
      rather than ID alone.
    - Split children (e.g. REQ-1a, REQ-1b) inherit their parent row's ground
      truth label. Extra splits not in ground truth are still evaluated if
      their parent ID appears at that row position.
 
    Prints:
        1. Per-class TP, FP, FN → Precision, Recall, F1
        2. Macro-averaged summary with overall Accuracy
    """
    # ── Load ground truth and input (needed for row-position mapping) ─────────
    gt_df  = pd.read_csv(ground_truth_path)
    inp_df = pd.read_csv("Trimmed Data Input.csv")  # columns: RequirementID, Requirement
 
    required_cols = {"RequirementID", "Classification"}
    if not required_cols.issubset(gt_df.columns):
        raise ValueError(
            f"Ground truth CSV must contain columns: {required_cols}. "
            f"Found: {set(gt_df.columns)}"
        )
 
    # Normalize labels — handle "Other" vs "Others" inconsistency
    gt_df["Classification"] = (gt_df["Classification"]
                                .astype(str).str.strip()
                                .replace("Other", "Others"))
 
    # ── Build position-aware lookup: (row_index, parent_id) → label ──────────
    # inp_df row i  →  gt_df row i  (they are aligned by position)
    # We store: parent_id → list of (row_index, label) so we can match the
    # right project's label when the same ID appears in multiple projects.
    from collections import defaultdict
    parent_to_positions = defaultdict(list)  # parent_id → [(row_idx, label)]
 
    # Input columns: RequirementID, Requirement
    # Ground truth columns: RequirementID, Classification
    for row_idx in range(len(inp_df)):
        inp_id   = str(inp_df.iloc[row_idx]["RequirementID"]).strip()
        gt_label = str(gt_df.iloc[row_idx]["Classification"]).strip()
        parent_to_positions[inp_id].append((row_idx, gt_label))
 
    # ── Match classified rows to ground truth ─────────────────────────────────
    # We walk the classified_df in order. For each parent_id we pop the next
    # available position from the queue — this handles repeated IDs across
    # projects correctly because the model processes projects in order.
    position_cursor = defaultdict(int)  # parent_id → next index into its list
 
    rows      = []
    unmatched = []
 
    for _, row in classified_df.iterrows():
        parent_id = normalize_req_id(row["RequirementID"])
        predicted = str(row["Classification"]).strip()
 
        positions = parent_to_positions.get(parent_id, [])
        cursor    = position_cursor[parent_id]
 
        if cursor < len(positions):
            # Use the next available ground truth entry for this parent_id.
            # Split children all share the same cursor entry (don't advance
            # until we see the next *different* original requirement).
            _, actual = positions[cursor]
            rows.append({
                "RequirementID": row["RequirementID"],
                "ParentID":      parent_id,
                "Predicted":     predicted,
                "Actual":        actual,
            })
        else:
            unmatched.append(row["RequirementID"])
 
    # Advance the cursor: after collecting all rows, deduplicate cursor
    # advancement so that REQ-1a and REQ-1b both map to the same REQ-1 entry
    # but the cursor only advances once we move past all children of REQ-1.
    # Re-do the walk with proper cursor advancement.
    position_cursor = defaultdict(int)
    rows = []
    unmatched = []
    prev_parent = None
 
    for _, row in classified_df.iterrows():
        parent_id = normalize_req_id(row["RequirementID"])
        predicted = str(row["Classification"]).strip()
 
        # Advance cursor only when the parent_id changes from the previous row
        if prev_parent is not None and parent_id != prev_parent:
            position_cursor[prev_parent] += 1
 
        prev_parent = parent_id
        positions   = parent_to_positions.get(parent_id, [])
        cursor      = position_cursor[parent_id]
 
        if cursor < len(positions):
            _, actual = positions[cursor]
            rows.append({
                "RequirementID": row["RequirementID"],
                "ParentID":      parent_id,
                "Predicted":     predicted,
                "Actual":        actual,
            })
        else:
            unmatched.append(row["RequirementID"])
 
    # Advance for the very last parent
    if prev_parent is not None:
        position_cursor[prev_parent] += 1
 
    if unmatched:
        print(f"\n⚠  {len(unmatched)} predictions had no ground truth match and were skipped.")
        print(f"   Skipped IDs: {unmatched[:20]}{'...' if len(unmatched) > 20 else ''}")
 
    if not rows:
        print("No matched requirements to evaluate.")
        return
 
    cmp_df = pd.DataFrame(rows)
 
    # ── Validate class labels ─────────────────────────────────────────────────
    valid = set(CLASSES)
    bad_pred   = cmp_df[~cmp_df["Predicted"].isin(valid)]["Predicted"].unique()
    bad_actual = cmp_df[~cmp_df["Actual"].isin(valid)]["Actual"].unique()
 
    if len(bad_pred):
        print(f"\n⚠  Unexpected predicted labels (will count as wrong): {bad_pred}")
    if len(bad_actual):
        print(f"\n⚠  Unexpected ground truth labels: {bad_actual}")
 
    # ── Per-class TP / FP / FN ────────────────────────────────────────────────
    # For multi-class one-vs-rest:
    #   TP(c) = predicted c AND actual c
    #   FP(c) = predicted c BUT actual ≠ c
    #   FN(c) = actual c BUT predicted ≠ c
 
    per_class = {}
    for cls in CLASSES:
        tp = int(((cmp_df["Predicted"] == cls) & (cmp_df["Actual"] == cls)).sum())
        fp = int(((cmp_df["Predicted"] == cls) & (cmp_df["Actual"] != cls)).sum())
        fn = int(((cmp_df["Predicted"] != cls) & (cmp_df["Actual"] == cls)).sum())
 
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
 
        per_class[cls] = {
            "TP": tp, "FP": fp, "FN": fn,
            "Precision": precision,
            "Recall":    recall,
            "F1":        f1,
        }
 
    # ── Overall accuracy ──────────────────────────────────────────────────────
    total    = len(cmp_df)
    correct  = int((cmp_df["Predicted"] == cmp_df["Actual"]).sum())
    accuracy = correct / total if total > 0 else 0.0
 
    # ── Macro averages ────────────────────────────────────────────────────────
    macro_precision = sum(v["Precision"] for v in per_class.values()) / len(CLASSES)
    macro_recall    = sum(v["Recall"]    for v in per_class.values()) / len(CLASSES)
    macro_f1        = sum(v["F1"]        for v in per_class.values()) / len(CLASSES)
 
    # ─────────────────────────────────────────────────────────────────────────
    # PRINT REPORT
    # ─────────────────────────────────────────────────────────────────────────
 
    sep  = "=" * 65
    sep2 = "-" * 65
 
    print(f"\n{sep}")
    print("  CLASSIFICATION METRICS REPORT")
    print(f"  Total evaluated: {total}  |  Correct: {correct}  |  Overall Accuracy: {accuracy:.2%}")
    print(sep)
 
    # Per-class table
    print(f"\n{'Class':<18} {'TP':>4} {'FP':>4} {'FN':>4}  {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(sep2)
    for cls in CLASSES:
        m = per_class[cls]
        print(
            f"{cls:<18} {m['TP']:>4} {m['FP']:>4} {m['FN']:>4}"
            f"  {m['Precision']:>10.2%} {m['Recall']:>8.2%} {m['F1']:>8.2%}"
        )
 
    # Macro summary
    print(sep2)
    print(
        f"{'Macro Average':<18} {'':>4} {'':>4} {'':>4}"
        f"  {macro_precision:>10.2%} {macro_recall:>8.2%} {macro_f1:>8.2%}"
    )
    print(sep)
 
    # Per-class breakdown with raw counts for clarity
    print("\n── Per-Class Detail ──────────────────────────────────────────")
    for cls in CLASSES:
        m = per_class[cls]
        print(f"\n  {cls}")
        print(f"    TP={m['TP']}  FP={m['FP']}  FN={m['FN']}")
        print(f"    Precision : {m['Precision']:.4f}  ({m['TP']} correctly labelled {cls} / {m['TP']+m['FP']} predicted {cls})")
        print(f"    Recall    : {m['Recall']:.4f}  ({m['TP']} correctly labelled {cls} / {m['TP']+m['FN']} actual {cls})")
        print(f"    F1        : {m['F1']:.4f}")
 
    print(f"\n{sep}\n")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# RUN EVALUATION
# Replace "ground_truth.csv" with the path to your ground truth file.
# Expected columns: RequirementID, Classification
# ─────────────────────────────────────────────────────────────────────────────
 
evaluate_metrics(classified_df, "Ground Truth Input.csv")