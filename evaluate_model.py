"""Evaluate the base model, or the base model + a LoRA adapter, on the held-out test split."""
import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import DATASET_NAME, LABELS, MODEL_NAME, TEST_SPLIT, build_prompt, compute_dtype, parse_label


def compute_metrics(records):
    n = len(records)
    gold_counts = Counter(r["gold"] for r in records)
    metrics = {
        "n": n,
        "accuracy": sum(r["pred"] == r["gold"] for r in records) / n,
        "strict_accuracy": sum(r["strict"] and r["pred"] == r["gold"] for r in records) / n,
        "format_rate": sum(r["strict"] for r in records) / n,
        "unparseable_rate": sum(r["pred"] is None for r in records) / n,
        "majority_baseline": max(gold_counts.values()) / n,
    }
    per_class = {}
    for label in LABELS:
        tp = sum(r["pred"] == label and r["gold"] == label for r in records)
        pred_n = sum(r["pred"] == label for r in records)
        gold_n = gold_counts[label]
        precision = tp / pred_n if pred_n else 0.0
        recall = tp / gold_n if gold_n else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": gold_n}
    metrics["macro_f1"] = sum(c["f1"] for c in per_class.values()) / len(LABELS)
    metrics["per_class"] = per_class
    metrics["confusion"] = {  # gold -> predicted -> count
        g: dict(Counter(r["pred"] or "UNPARSEABLE" for r in records if r["gold"] == g)) for g in LABELS
    }
    return metrics


def load_predictions(name, out_dir="outputs/eval"):
    """Read the per-tweet predictions written by evaluate_model()."""
    with open(Path(out_dir) / f"{name}_predictions.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate_model(name, adapter=None, limit=None, batch_size=64, out_dir="outputs/eval"):
    """Generate an answer for every test tweet, score it, save metrics + predictions, return metrics."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"  # batched generation with a decoder-only model needs left padding
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=compute_dtype()).to("cuda")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    test = load_dataset(DATASET_NAME, split=TEST_SPLIT)
    if limit:  # the split is sorted by label, so shuffle before taking a subset
        test = test.shuffle(seed=42).select(range(limit))

    records = []
    start = time.time()
    for i in range(0, len(test), batch_size):
        batch = test[i : i + batch_size]
        prompts = [build_prompt(tokenizer, text) for text in batch["text"]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=10, do_sample=False)
        outputs = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for text, gold, output in zip(batch["text"], batch["label"], outputs):
            pred, strict = parse_label(output)
            records.append({"text": text, "gold": LABELS[gold], "output": output, "pred": pred, "strict": strict})

    metrics = compute_metrics(records)
    metrics.update({"name": name, "adapter": adapter, "eval_seconds": round(time.time() - start, 1)})

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with open(out_path / f"{name}_predictions.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    del model  # free GPU memory so the next run in the same process starts clean
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="run name used for the output files")
    parser.add_argument("--adapter", default=None, help="path to a saved LoRA adapter; omit for the base model")
    parser.add_argument("--limit", type=int, default=None, help="only evaluate a random subset of N examples")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", default="outputs/eval")
    args = parser.parse_args()

    metrics = evaluate_model(args.name, args.adapter, args.limit, args.batch_size, args.out_dir)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("per_class", "confusion")}, indent=2))
    print("confusion (gold -> predicted):", json.dumps(metrics["confusion"]))


if __name__ == "__main__":
    main()
