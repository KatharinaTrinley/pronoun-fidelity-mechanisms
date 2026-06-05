#!/usr/bin/env python3
"""
context_free_bias.py
Measures each model's context-free pronoun bias per occupation: blank out the pronoun in the first sentence and see which of he/she/they and variants the model predicts.
Runs each model 3 times and averages, since predictions can vary run to run. This is what fills the stereotype table (mechanism S targets) used elsewhere.
"""

import os
import gc
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

login(os.environ.get("HF_TOKEN"))  # set HF_TOKEN

CSV_FILE = "add/path/data/base_he_source_pairs.csv"  # add path later
OUTPUT_DIR = "bias_results"

N_RUNS = 3

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "allenai/OLMo-2-0425-1B-Instruct",
    "allenai/OLMo-2-1124-7B-Instruct",
    "allenai/OLMo-2-1124-13B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it",
]

MODEL_KWARGS = {
    "meta-llama/Llama-3.1-8B-Instruct":
        dict(torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"),

    "allenai/OLMo-2-0425-1B-Instruct":
        dict(torch_dtype=torch.float16, device_map="auto",
             attn_implementation="eager", trust_remote_code=True),

    "allenai/OLMo-2-1124-7B-Instruct":
        dict(torch_dtype=torch.float16, device_map="auto",
             attn_implementation="eager", trust_remote_code=True),

    "allenai/OLMo-2-1124-13B-Instruct":
        dict(torch_dtype=torch.float16, device_map="auto",
             attn_implementation="eager", trust_remote_code=True),

    "Qwen/Qwen2.5-7B-Instruct":
        dict(torch_dtype="auto", device_map="auto",
             attn_implementation="eager"),

    "google/gemma-2-9b-it":
        dict(torch_dtype=torch.bfloat16, device_map="auto",
             attn_implementation="eager"),
}


def model_slug(name):
    return name.split("/")[-1].lower()


def get_pronoun_token_ids(tokenizer):
    # We only score the nominative forms here (context-free probe).
    return {
        p: tokenizer.encode(p, add_special_tokens=False)[0]
        for p in ("he", "she", "they")
    }


def extract_first_sentence(text):
    return text.split('.')[0].strip() + '.'


def replace_first_pronoun_with_blank(sentence):
    # Blank out the first pronoun (any case) so the model has to predict it.
    pronouns = {
        'he', 'she', 'they',
        'him', 'her', 'them',
        'his', 'hers', 'their'
    }

    words = sentence.split()

    for i, word in enumerate(words):
        if word.lower().strip('.,!?;:') in pronouns:
            words[i] = '___'
            break

    return ' '.join(words)


def get_prediction(model, tokenizer, prompt, pronoun_tokens, device):
    # Forced choice over he/she/they at the last position.
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
        log_probs = F.log_softmax(logits, dim=0)

    scores = {
        p: log_probs[tid].item()
        for p, tid in pronoun_tokens.items()
    }

    return max(scores, key=scores.get), scores


def analyze(csv_file, model, tokenizer, device):
    df = pd.read_csv(csv_file)

    pronoun_tokens = get_pronoun_token_ids(tokenizer)

    results = []

    # Per-occupation tally of which pronoun the model picked.
    occ_preds = defaultdict(
        lambda: {'he': 0, 'she': 0, 'they': 0, 'count': 0}
    )

    for _, row in tqdm(df.iterrows(), total=len(df)):
        occupation = row['occupation'].lower().strip()

        sentence = extract_first_sentence(row['base_sentence'])
        blanked = replace_first_pronoun_with_blank(sentence)

        # Skip rows where no pronoun was found to blank.
        if '___' not in blanked:
            continue

        pred, scores = get_prediction(
            model,
            tokenizer,
            blanked,
            pronoun_tokens,
            device
        )

        results.append({
            'occupation': occupation,
            'blanked_sentence': blanked,
            'predicted_pronoun': pred,
            **{f'score_{p}': s for p, s in scores.items()}
        })

        occ_preds[occupation][pred] += 1
        occ_preds[occupation]['count'] += 1

    return pd.DataFrame(results), occ_preds


def aggregate_average(run_results):
    # Average the per-occupation counts across the N_RUNS runs.
    combined = defaultdict(
        lambda: {'he': [], 'she': [], 'they': [], 'count': []}
    )

    for occ_preds in run_results:
        for occ, counts in occ_preds.items():
            combined[occ]['he'].append(counts['he'])
            combined[occ]['she'].append(counts['she'])
            combined[occ]['they'].append(counts['they'])
            combined[occ]['count'].append(counts['count'])

    rows = []

    for occ, vals in combined.items():

        he_avg = sum(vals['he']) / len(vals['he'])
        she_avg = sum(vals['she']) / len(vals['she'])
        they_avg = sum(vals['they']) / len(vals['they'])
        total_avg = sum(vals['count']) / len(vals['count'])

        rows.append({
            'occupation': occ,

            # The stereotype pronoun = whichever case wins on average.
            'most_likely_pronoun': max(
                ['he', 'she', 'they'],
                key=lambda p: {
                    'he': he_avg,
                    'she': she_avg,
                    'they': they_avg
                }[p]
            ),

            'he_count_avg': round(he_avg, 2),
            'she_count_avg': round(she_avg, 2),
            'they_count_avg': round(they_avg, 2),

            'he_pct_avg': round((he_avg / total_avg) * 100, 1),
            'she_pct_avg': round((she_avg / total_avg) * 100, 1),
            'they_pct_avg': round((they_avg / total_avg) * 100, 1),

            'total_samples_avg': round(total_avg, 2),
        })

    return pd.DataFrame(rows).sort_values('occupation')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name in MODELS:

        slug = model_slug(model_name)

        print(f"\n{'='*70}")
        print(model_name)
        print(f"{'='*70}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **MODEL_KWARGS.get(model_name, {"device_map": "auto"})
            )

            model.eval()

        except Exception as e:
            print(f"[SKIP] {e}")
            continue

        all_run_preds = []
        all_detailed = []

        for run_idx in range(N_RUNS):

            print(f"\n--- RUN {run_idx + 1}/{N_RUNS} ---")

            detailed_df, occ_preds = analyze(
                CSV_FILE,
                model,
                tokenizer,
                device
            )

            detailed_df["run"] = run_idx + 1

            all_detailed.append(detailed_df)
            all_run_preds.append(occ_preds)

        avg_df = aggregate_average(all_run_preds)

        detailed_combined = pd.concat(all_detailed, ignore_index=True)

        avg_df.to_csv(
            f"{OUTPUT_DIR}/{slug}_aggregated_avg.csv",
            index=False
        )

        detailed_combined.to_csv(
            f"{OUTPUT_DIR}/{slug}_detailed_all_runs.csv",
            index=False
        )

        print(f"\nSaved averaged results for {slug}")

        # Free the model before loading the next one.
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()