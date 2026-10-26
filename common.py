"""Shared settings: model, dataset, prompt format and answer parsing."""
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "zeroshot/twitter-financial-news-sentiment"
TEST_SPLIT = "validation"  # the dataset has no official test split; this split is held out from training
LABELS = ["Bearish", "Bullish", "Neutral"]  # list index == dataset label id

INSTRUCTION = (
    "Classify the sentiment of the following financial tweet as Bearish, Bullish, or Neutral. "
    "Answer with exactly one word.\n\nTweet: {text}"
)


def compute_dtype():
    """bf16 on GPUs that support it (e.g. RTX 30/40 series), fp16 otherwise (e.g. Colab's T4)."""
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def build_prompt(tokenizer, text):
    """Wrap a tweet in Qwen's chat template, ending right where the assistant starts answering."""
    messages = [{"role": "user", "content": INSTRUCTION.format(text=text)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_label(output):
    """Map a model output to a label.

    Returns (label or None, strict) where strict means the output was exactly one label word
    (case-insensitive, optional trailing period). A non-strict output still counts if it
    mentions exactly one of the three labels.
    """
    cleaned = output.strip().rstrip(".").strip().lower()
    for label in LABELS:
        if cleaned == label.lower():
            return label, True
    found = [label for label in LABELS if label.lower() in output.lower()]
    if len(found) == 1:
        return found[0], False
    return None, False
