"""LoRA fine-tuning of Qwen2.5-0.5B-Instruct on the training split."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from common import DATASET_NAME, LABELS, MODEL_NAME, build_prompt, compute_dtype

MAX_LENGTH = 256  # tweets are at most ~90 tokens, so nothing gets truncated


def train_lora(
    output_dir,
    r=8,
    alpha=None,
    dropout=0.05,
    lr=2e-4,
    epochs=1,
    batch_size=4,
    grad_accum=4,
    limit=None,
    seed=42,
):
    """Train one LoRA adapter, save it to <output_dir>/adapter and return a summary dict.

    batch_size 4 x grad_accum 4 = effective batch 16. A per-device batch of 16 peaked at ~7 GB:
    the logits over Qwen's 152k-token vocabulary dominate memory, and on an 8 GB card that
    spills into slow shared memory.
    """
    alpha = alpha if alpha is not None else 2 * r
    dtype = compute_dtype()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train = load_dataset(DATASET_NAME, split="train")
    if limit:  # the split is sorted by label, so shuffle before taking a subset
        train = train.shuffle(seed=seed).select(range(limit))

    def process(example):
        # Prompt tokens get label -100 (no loss); only the answer word and the end marker are learned.
        prompt_ids = tokenizer(build_prompt(tokenizer, example["text"]), add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(LABELS[example["label"]] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + answer_ids)[:MAX_LENGTH]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:MAX_LENGTH]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

    tokenized = train.map(process, remove_columns=train.column_names)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type="linear",
        logging_steps=20,
        save_strategy="no",
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        report_to="none",
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer.train()
    runtime = time.time() - start

    out_dir = Path(output_dir)
    model.save_pretrained(out_dir / "adapter")  # only the LoRA weights (a few MB)
    summary = {
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "train_examples": len(tokenized),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_seconds": round(runtime, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "log_history": trainer.state.log_history,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    del trainer, model  # free GPU memory so the next run in the same process starts clean
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=None, help="lora_alpha; defaults to 2 * r")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="only train on a random subset of N examples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = train_lora(
        args.output_dir, r=args.r, alpha=args.alpha, dropout=args.dropout, lr=args.lr, epochs=args.epochs,
        batch_size=args.batch_size, grad_accum=args.grad_accum, limit=args.limit, seed=args.seed,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "log_history"}, indent=2))


if __name__ == "__main__":
    main()
