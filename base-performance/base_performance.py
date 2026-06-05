#!/usr/bin/env python3
"""
base_performance.py

Raw, unintervened, pronoun resolution accuracy in the 0- and 1-distractor
settings:
- 0 distractors: eo_task.tsv     (occupation context + target)
- 1 distractor:  eo_ep_task.tsv  (occupation context + distractor + target)

Same ___ blank + forced multiple-choice prompt as DAS training. Accuracy is argmax over the log-probs of the three pronoun candidates at the response position. This reproduces the base-performance table (Appendix).
"""

import torch
import json
import random
import argparse
import csv
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from model_registry import _MODEL_CONFIGS


ALL_PRONOUNS = ["he", "she", "they"]

# Each base pronoun in its three grammatical cases.
PRONOUN_FORMS = {
    "he":   {"NOM": "he",   "ACC": "him",  "POSS": "his"},
    "she":  {"NOM": "she",  "ACC": "her",  "POSS": "her"},
    "they": {"NOM": "they", "ACC": "them", "POSS": "their"},
}

# The three answer options shown for each case.
OPTIONS_BY_CASE = {
    "NOM":  ["he",  "she", "they"],
    "ACC":  ["him", "her", "them"],
    "POSS": ["his", "her", "their"],
}

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "Please select the correct pronoun from the options below.\n\n"
    "### Input:\n"
    "{input_text}\n\n"
    "### Question:\n"
    "{question}\n\n"
    "### Response:\n"
)


def normalize_pronoun(pronoun: str) -> str | None:
    # Collapse any case form to its base; None if it isn't he/she/they.
    p = pronoun.lower().strip()
    for base, forms in PRONOUN_FORMS.items():
        if p in forms.values() or p == base:
            return base
    return None


def detect_case(pronoun_type: str) -> str:
    # Read NOM/ACC/POSS off the pronoun_type column, defaulting to NOM.
    pt = pronoun_type.upper()
    if "NOM" in pt:
        return "NOM"
    if "ACC" in pt:
        return "ACC"
    if "POSS" in pt:
        return "POSS"
    return "NOM"


def make_question(case: str, rng: random.Random) -> str:
    # Randomize the options so position can't leak the answer.
    options = OPTIONS_BY_CASE.get(case, OPTIONS_BY_CASE["NOM"])[:]
    rng.shuffle(options)
    return f"What pronoun should be used to fill the blank? Options: {options[0]}, {options[1]}, {options[2]}"


def replace_placeholders(sentence: str) -> str:
    # The TSV marks the target slot with a $..._PRONOUN placeholder. we blank it.
    return (sentence
            .replace("$NOM_PRONOUN", "___")
            .replace("$ACC_PRONOUN", "___")
            .replace("$POSS_PRONOUN", "___"))


def get_pronoun_token_ids(tokenizer) -> dict:
    # Token ID for each pronoun in each case, so scoring matches the question's case.
    return {
        case: {
            base: tokenizer.encode(form, add_special_tokens=False)[0]
            for base, form in {
                "he":   OPTIONS_BY_CASE[case][0],
                "she":  OPTIONS_BY_CASE[case][1],
                "they": OPTIONS_BY_CASE[case][2],
            }.items()
        }
        for case in ("NOM", "ACC", "POSS")
    }


def load_tsv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def build_examples(rows: list[dict], num_distractors: int, rng: random.Random) -> list[dict]:
    """
    Turn raw TSV rows into prompt-ready examples.

    0 distractors -> eo_task.tsv    (confuse_pronoun empty)
    1 distractor  -> eo_ep_task.tsv (confuse_pronoun set)
    """
    examples = []
    for row in rows:
        pronoun_base = normalize_pronoun(row["pronoun"])
        if pronoun_base is None:
            continue  # skip xe / other non-standard pronouns or words like 'the'

        if num_distractors == 1:
            confuse_base = normalize_pronoun(row.get("confuse_pronoun", "") or "")
            if confuse_base is None:
                continue  # skip rows whose distractor isn't he/she/they

        case = detect_case(row["pronoun_type"])
        correct_form = PRONOUN_FORMS[pronoun_base][case]

        input_text = replace_placeholders(row["sentence"])
        question = make_question(case, rng)
        prompt = PROMPT_TEMPLATE.format(input_text=input_text, question=question)

        examples.append({
            "uid": row["uid"],
            "occupation": row["occupation"].lower().strip(),
            "pronoun": pronoun_base,
            "confuse_pronoun": normalize_pronoun(row.get("confuse_pronoun", "") or "") if num_distractors == 1 else None,
            "case": case,
            "correct_form": correct_form, 
            "prompt": prompt,
        })

    return examples


@torch.no_grad()
def score_example(model, tokenizer, prompt: str, case: str,
                  pronoun_token_ids: dict, device: str) -> str:
    # Predicted base pronoun via forced choice over he/she/they.
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # Score at the last token, i.e. the blank right after ### Response:.
    last_pos = inputs["input_ids"].shape[1] - 1
    logits = outputs.logits[0, last_pos]
    log_probs = F.log_softmax(logits, dim=0)

    scores = {
        base: log_probs[pronoun_token_ids[case][base]].item()
        for base in ALL_PRONOUNS
    }
    return max(scores, key=scores.get)


def evaluate(examples: list[dict], model, tokenizer, pronoun_token_ids: dict,
             device: str, desc: str) -> dict:
    # Overall accuracy plus breakdowns by pronoun and by case.
    total = 0
    correct = 0
    by_pronoun = defaultdict(lambda: {"total": 0, "correct": 0})
    by_case = defaultdict(lambda: {"total": 0, "correct": 0})

    for ex in tqdm(examples, desc=desc):
        predicted_base = score_example(
            model, tokenizer, ex["prompt"], ex["case"],
            pronoun_token_ids, device
        )
        is_correct = (predicted_base == ex["pronoun"])

        total += 1
        if is_correct:
            correct += 1

        bp = ex["pronoun"]
        by_pronoun[bp]["total"] += 1
        if is_correct:
            by_pronoun[bp]["correct"] += 1

        c = ex["case"]
        by_case[c]["total"] += 1
        if is_correct:
            by_case[c]["correct"] += 1

    def pct(c, t):
        return round(100 * c / t, 2) if t > 0 else 0.0

    return {
        "overall_accuracy": pct(correct, total),
        "correct": correct,
        "total": total,
        "by_pronoun": {
            p: {"accuracy": pct(v["correct"], v["total"]), **v}
            for p, v in sorted(by_pronoun.items())
        },
        "by_case": {
            c: {"accuracy": pct(v["correct"], v["total"]), **v}
            for c, v in sorted(by_case.items())
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Measure base pronoun resolution accuracy")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model name or local path")
    parser.add_argument("--data_dir", default="RUFF_data",
                        help="Directory containing eo_task.tsv and eo_ep_task.tsv")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Cap examples per setting (useful for quick tests)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for option shuffling")
    parser.add_argument("--output", default="base_performance_results.json",
                        help="Path to write JSON results")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    data_dir = Path(args.data_dir)
    eo_task_path = data_dir / "eo_task.tsv"
    eo_ep_task_path = data_dir / "eo_ep_task.tsv"

    for p in (eo_task_path, eo_ep_task_path):
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p}")

    print(f"Loading data...")
    rows_0 = load_tsv(eo_task_path)
    rows_1 = load_tsv(eo_ep_task_path)
    print(f"  eo_task.tsv:    {len(rows_0):,} rows")
    print(f"  eo_ep_task.tsv: {len(rows_1):,} rows")

    examples_0 = build_examples(rows_0, num_distractors=0, rng=rng)
    examples_1 = build_examples(rows_1, num_distractors=1, rng=rng)
    print(f"  0-distractor examples: {len(examples_0):,}")
    print(f"  1-distractor examples: {len(examples_1):,}")

    if args.max_examples:
        examples_0 = examples_0[: args.max_examples]
        examples_1 = examples_1[: args.max_examples]
        print(f"  Capped to {args.max_examples} examples per setting")

    print(f"\nLoading model: {args.model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = _MODEL_CONFIGS[args.model]
    tokenizer = cfg["tok_cls"].from_pretrained(args.model)
    model = cfg["model_cls"].from_pretrained(args.model, **cfg["model_kwargs"])
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pronoun_token_ids = get_pronoun_token_ids(tokenizer)
    print(f"Pronoun token IDs (NOM): {pronoun_token_ids['NOM']}")

    print(f"\nEvaluating -- 0 distractors (eo_task.tsv)")
    results_0 = evaluate(examples_0, model, tokenizer, pronoun_token_ids, device,
                         desc="0-distractor")

    print(f"\nEvaluating -- 1 distractor (eo_ep_task.tsv)")
    results_1 = evaluate(examples_1, model, tokenizer, pronoun_token_ids, device,
                         desc="1-distractor")

    results = {
        "model": args.model,
        "seed": args.seed,
        "settings": {
            "0_distractors": results_0,
            "1_distractor":  results_1,
        },
    }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults summary")
    for setting, res in results["settings"].items():
        print(f"\n{setting}:")
        print(f"  Overall accuracy: {res['overall_accuracy']:.2f}%  "
              f"({res['correct']}/{res['total']})")
        print(f"  By pronoun:  " +
              "  ".join(f"{p}={v['accuracy']:.1f}%" for p, v in res["by_pronoun"].items()))
        print(f"  By case:     " +
              "  ".join(f"{c}={v['accuracy']:.1f}%" for c, v in res["by_case"].items()))

    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()