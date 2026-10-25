#!/usr/bin/env bash
# Full experiment: base model eval -> LoRA r=8 train + eval -> LoRA r=16 train + eval.
# lora_alpha = 2 * r in both runs, so the LoRA scaling (alpha / r) stays at 2 and only r changes.
set -euo pipefail
PY=${PYTHON:-python}
mkdir -p outputs/logs

$PY evaluate_model.py --name base 2>&1 | tee outputs/logs/eval_base.log

$PY train.py --output-dir outputs/lora_r8 --r 8 --alpha 16 2>&1 | tee outputs/logs/train_r8.log
$PY evaluate_model.py --name lora_r8 --adapter outputs/lora_r8/adapter 2>&1 | tee outputs/logs/eval_r8.log

$PY train.py --output-dir outputs/lora_r16 --r 16 --alpha 32 2>&1 | tee outputs/logs/train_r16.log
$PY evaluate_model.py --name lora_r16 --adapter outputs/lora_r16/adapter 2>&1 | tee outputs/logs/eval_r16.log
