"""
Dataset preparation script for SentinelLM semantic NER model.

Processes training data from multiple sources:
1. Synthetic obfuscated PII examples (JSONL)
2. Hard negative examples (JSONL)

Converts all data to BIO format, tokenizes with DistilBERT tokenizer,
aligns labels with subword tokens, and saves as a HuggingFace Dataset.
"""

import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# --- Config ---
# Uses local base model so Docker doesn't need internet access.
# Run model/download_base_model.py once first to populate this directory.
MODEL_NAME = "./model/base_model"
OUTPUT_DIR = Path(__file__).parent / "processed"
SYNTHETIC_PATH = Path(__file__).parent / "synthetic_obfuscated.jsonl"
HARD_NEGATIVES_PATH = Path(__file__).parent / "hard_negatives.jsonl"

LABEL_LIST = ["O", "B-PII", "I-PII", "B-SECRET", "I-SECRET"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


def load_jsonl(filepath: Path) -> list[dict]:
    """Load a JSONL file and return a list of examples."""
    examples = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def text_to_bio(text: str, entities: list[dict]) -> tuple[list[str], list[str]]:
    """
    Convert a text + entity spans to token-level BIO labels.

    Args:
        text: The input text.
        entities: List of {"start": int, "end": int, "label": str} dicts.

    Returns:
        Tuple of (tokens, labels) where tokens are whitespace-split words
        and labels are BIO tags.
    """
    # Create character-level labels
    char_labels = ["O"] * len(text)

    # Sort entities by start position
    sorted_entities = sorted(entities, key=lambda e: e["start"])

    for entity in sorted_entities:
        start = entity["start"]
        end = entity["end"]
        label = entity["label"]  # "PII" or "SECRET"

        # Mark the first character as B-
        if start < len(char_labels):
            char_labels[start] = f"B-{label}"

        # Mark remaining characters as I-
        for i in range(start + 1, min(end, len(char_labels))):
            char_labels[i] = f"I-{label}"

    # Split into words and assign word-level labels
    tokens = text.split()
    labels = []
    char_idx = 0

    for token in tokens:
        # Find the token in the text
        while char_idx < len(text) and text[char_idx] == " ":
            char_idx += 1

        # Get the label for the first character of this token
        if char_idx < len(char_labels):
            label = char_labels[char_idx]
        else:
            label = "O"

        labels.append(label)
        char_idx += len(token)

    return tokens, labels


def align_labels_with_tokens(
    labels: list[int], word_ids: list[int | None]
) -> list[int]:
    """
    Align word-level BIO labels with subword tokens.

    When DistilBERT tokenizes "gmail" into ["gm", "##ail"], both subword
    tokens get the same label. B- labels on continuation tokens become I-.

    Args:
        labels: Word-level label IDs.
        word_ids: Word IDs from the tokenizer (None for special tokens).

    Returns:
        Token-level label IDs, with -100 for special tokens (ignored in loss).
    """
    new_labels = []
    current_word = None

    for word_id in word_ids:
        if word_id is None:
            # Special token ([CLS], [SEP], [PAD])
            new_labels.append(-100)
        elif word_id != current_word:
            # First token of a new word
            current_word = word_id
            if word_id < len(labels):
                new_labels.append(labels[word_id])
            else:
                new_labels.append(LABEL2ID["O"])
        else:
            # Continuation of a word — B- becomes I-
            if word_id < len(labels):
                label = labels[word_id]
                # B-PII (1) -> I-PII (2), B-SECRET (3) -> I-SECRET (4)
                if label in (LABEL2ID["B-PII"], LABEL2ID["B-SECRET"]):
                    new_labels.append(label + 1)
                else:
                    new_labels.append(label)
            else:
                new_labels.append(LABEL2ID["O"])

    return new_labels


def prepare_dataset():
    """Main dataset preparation pipeline."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    all_tokens = []
    all_labels = []

    # --- Load synthetic obfuscated data ---
    if SYNTHETIC_PATH.exists():
        print(f"Loading synthetic data from {SYNTHETIC_PATH}...")
        synthetic = load_jsonl(SYNTHETIC_PATH)
        for example in synthetic:
            tokens, labels = text_to_bio(example["text"], example["entities"])
            all_tokens.append(tokens)
            all_labels.append([LABEL2ID.get(l, 0) for l in labels])
        print(f"  Loaded {len(synthetic)} synthetic examples")

        # Oversampling removed — dataset is now large enough (500+ examples)
        # that duplicating causes the model to memorize rather than generalize,
        # resulting in artificially inflated F1=1.0 on the test split.

    # --- Load hard negatives (capped to avoid drowning out PII signal) ---
    # Loading all 154 negatives caused the model to predict all-O (F1=0).
    # Cap at 40 — enough to prevent false positives, not so many that the
    # model forgets PII/SECRET labels exist.
    MAX_NEGATIVES = 80  # raised from 40 — more positives now so we can afford more negatives
    if HARD_NEGATIVES_PATH.exists():
        print(f"Loading hard negatives from {HARD_NEGATIVES_PATH}...")
        negatives = load_jsonl(HARD_NEGATIVES_PATH)[:MAX_NEGATIVES]
        for example in negatives:
            tokens, labels = text_to_bio(example["text"], example["entities"])
            all_tokens.append(tokens)
            all_labels.append([LABEL2ID.get(l, 0) for l in labels])
        print(f"  Loaded {len(negatives)} hard negative examples (capped at {MAX_NEGATIVES})")

    print(f"Total examples: {len(all_tokens)} ({len(all_tokens) - len(negatives)} positive, {len(negatives)} negative)")

    # --- Tokenize and align labels ---
    print("Tokenizing and aligning labels...")

    tokenized_inputs = []
    aligned_labels = []

    for tokens, labels in zip(all_tokens, all_labels):
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=128,   # reduced from 512 — PII examples are short, saves RAM
            padding="max_length",
        )
        word_ids = encoding.word_ids()
        token_labels = align_labels_with_tokens(labels, word_ids)

        # Pad labels to max_length
        while len(token_labels) < 128:
            token_labels.append(-100)

        tokenized_inputs.append(encoding)
        aligned_labels.append(token_labels)

    # --- Create HuggingFace Dataset ---
    print("Creating dataset...")

    dataset_dict = {
        "input_ids": [t["input_ids"] for t in tokenized_inputs],
        "attention_mask": [t["attention_mask"] for t in tokenized_inputs],
        "labels": aligned_labels,
        "tokens": all_tokens,
    }

    dataset = Dataset.from_dict(dataset_dict)

    # --- Split: 80% train, 10% validation, 10% test ---
    print("Splitting dataset...")
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    test_val = train_test["test"].train_test_split(test_size=0.5, seed=42)

    final_dataset = DatasetDict({
        "train": train_test["train"],
        "validation": test_val["train"],
        "test": test_val["test"],
    })

    print(f"Train: {len(final_dataset['train'])} examples")
    print(f"Validation: {len(final_dataset['validation'])} examples")
    print(f"Test: {len(final_dataset['test'])} examples")

    # --- Save ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_dataset.save_to_disk(str(OUTPUT_DIR))
    print(f"Dataset saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    prepare_dataset()
