# LoRA Fine-tuning Qwen2.5-0.5B-Instruct for Financial Tweet Sentiment

A small, end-to-end LoRA experiment: fine-tune a 0.5B-parameter instruction model to classify financial news tweets as **Bearish / Bullish / Neutral**, and compare it with the untouched base model on a held-out test set. A second run doubles the LoRA rank (r=8 vs r=16) to see whether the extra capacity helps.

The experiment runs on a single 8 GB laptop GPU. All numbers below come from the runs saved in [`lora_finetune.ipynb`](lora_finetune.ipynb).

## Setup

| | |
|---|---|
| Base model | [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (494M parameters, Apache-2.0) |
| Dataset | [`zeroshot/twitter-financial-news-sentiment`](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) (MIT) |
| Train / test | 9,543 tweets in `train` / 2,388 tweets in `validation`, used as the **test set** (the dataset has no test split, and nothing in `validation` is used for training or tuning). No tweet appears in both splits. |
| Class balance (test) | Neutral 65.6%, Bullish 19.9%, Bearish 14.5% |
| Metrics | Accuracy, macro-F1 (the classes are imbalanced), per-class F1, and format rate (share of outputs that are exactly one label word) |

### Prompt and answer format

Every tweet is wrapped in Qwen's own chat template (with its default system prompt). The model must answer with a single word:

```
<|im_start|>user
Classify the sentiment of the following financial tweet as Bearish, Bullish, or Neutral. Answer with exactly one word.

Tweet: $BYND - JPMorgan reels in expectations on Beyond Meat https://t.co/bd0xbFGjkT<|im_end|>
<|im_start|>assistant
Bearish<|im_end|>
```

During training, all prompt tokens get the label `-100`, so the loss only covers the answer tokens (e.g. `Bear`, `ish`, `<|im_end|>`). Evaluation uses greedy decoding with at most 10 new tokens.

### LoRA and training configuration

| Setting | Value | Why |
|---|---|---|
| `target_modules` | `q_proj`, `k_proj`, `v_proj` in all 24 layers | Attention projections; already enough for this task |
| `r` | 8 and 16 | The variable compared in this experiment |
| `lora_alpha` | 2 x r (16 and 32) | Keeps the LoRA scaling `alpha / r` at 2 in both runs, so only the rank changes |
| `lora_dropout` | 0.05 | Light regularisation |
| Trainable parameters | 737,280 (r=8, 0.15%) / 1,474,560 (r=16, 0.30%) | |
| Learning rate | 2e-4, linear decay, no warmup | The `TrainingArguments` default of 5e-5 barely moved the loss in an early LoRA test |
| Effective batch size | 16 (batch 4 x gradient accumulation 4) | A per-device batch of 16 peaked at ~7 GB and spilled into shared memory on the 8 GB card |
| Epochs | 1 | Chosen up front; not tuned on the test set |
| Precision | bf16 (fp16 on GPUs without bf16 support) | |
| Seed | 42 | |

## Results

Test set: 2,388 tweets.

| Model | Accuracy | Macro-F1 | F1 Bearish | F1 Bullish | F1 Neutral | Format rate |
|---|---|---|---|---|---|---|
| Majority class (always "Neutral") | 0.656 | - | - | - | - | - |
| Base model (no fine-tuning) | 0.396 | 0.284 | 0.000 | 0.346 | 0.505 | 1.000 |
| **LoRA r=8** | **0.882** | **0.848** | 0.805 | 0.824 | 0.915 | 1.000 |
| **LoRA r=16** | **0.891** | **0.859** | 0.814 | 0.841 | 0.922 | 1.000 |

Training on an RTX 4070 Laptop GPU (8 GB): 597 optimizer steps, peak GPU memory 2.65 GB (r=8) / 2.66 GB (r=16). Wall-clock time per run was about 6 minutes from the command line and 8-23 minutes in the notebook run. Other load on the laptop was not controlled, so the timings are approximate.

### What the numbers show

- **The base model is worse than always guessing "Neutral".** It follows the one-word format perfectly, but it never answers "Bearish" (0 of 347 Bearish tweets) and labels 994 of 1,566 Neutral tweets as "Bullish".
- **LoRA fixes the judgment, not the format.** The format rate was already 100% before fine-tuning, so the jump from 0.396 to 0.882 accuracy comes from the model making better calls, not from learning to answer in one word.
- **r=16 is slightly better than r=8** (+0.9 points accuracy, +0.011 macro-F1). On the tweets where the two adapters disagree, r=16 is right 44 times and r=8 is right 22 times (McNemar exact test, p = 0.009), so the gap is unlikely to be test-set noise **for these two trained adapters**.
- **Caveat: each setting was trained only once.** An earlier run of the same scripts with the same seed gave 0.880 (r=8) and 0.897 (r=16), because GPU training is not bit-for-bit deterministic. The size of the r=8 vs r=16 gap therefore moves between runs; multiple seeds would be needed to pin it down.
- **Remaining errors lean towards "Neutral".** For r=16, the most common mistakes are Bullish -> Neutral (77) and Bearish -> Neutral (65), which matches the 65% Neutral share of the training data.

## How to run

Tested on Windows 11 with Python 3.11 and an NVIDIA RTX 4070 Laptop GPU (CUDA 12.6 build of PyTorch).

```bash
conda create -n qwen-lora python=3.11 -y
conda activate qwen-lora
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Run the full experiment (base eval, then train + eval for r=8 and r=16) from the command line. On Windows, use Git Bash:

```bash
bash run_experiments.sh
```

Or run the steps one at a time:

```bash
python evaluate_model.py --name base
python train.py --output-dir outputs/lora_r8 --r 8 --alpha 16
python evaluate_model.py --name lora_r8 --adapter outputs/lora_r8/adapter
```

Or open `lora_finetune.ipynb` and run all cells (pick the `qwen-lora` environment as the kernel).

Metrics and per-tweet predictions are written to `outputs/eval/`, adapters to `outputs/<run>/adapter/`. Model weights and adapters are not committed to this repository.

**Google Colab:** the notebook has a setup cell that installs the pinned libraries and clones this repo, and the code falls back to fp16 on GPUs without bf16 (such as the free T4). This path has **not been tested** yet.

## Project structure

```
common.py            model / dataset names, prompt template, answer parsing
train.py             LoRA training (train_lora function + CLI)
evaluate_model.py    batched generation, scoring and metrics (evaluate_model function + CLI)
run_experiments.sh   full experiment in one command
lora_finetune.ipynb  the same experiment as a notebook, with saved outputs
requirements.txt     pinned library versions
```

## Limitations

- One training run per setting (seed 42); no variance across seeds.
- The base model is evaluated zero-shot with a single prompt. Better prompting (e.g. few-shot examples) might narrow the gap; that was not tried.
- Hyperparameters (learning rate, epochs, target modules) were not tuned.
- The test set is the dataset's official `validation` split, since there is no separate test split.

## Acknowledgements

The overall workflow follows the LoRA lesson of the Chinese-language Hugging Face Transformers tutorial series [zyds/transformers-code](https://github.com/zyds/transformers-code). That repository has no license, so no code is copied from it; the code here is written for this project, Qwen2.5 and transformers 5.x.
