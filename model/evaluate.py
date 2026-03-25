"""
Evaluation script for the SentinelLM semantic NER model.

Runs the trained model against the held-out test set and produces
a detailed evaluation report per PRD Section 9.6.
"""

import json
import os

import torch
from datasets import load_from_disk
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForTokenClassification, AutoTokenizer

# --- Config ---
MODEL_PATH = "./model/trained"
DATA_DIR = "./model/data/processed"
OUTPUT_PATH = "./docs/model_eval.json"

LABEL_LIST = ["O", "B-PII", "I-PII", "B-SECRET", "I-SECRET"]


def main():
    """Run evaluation on the test set."""
    print("=" * 60)
    print("SentinelLM — Model Evaluation")
    print("=" * 60)

    # --- Load model ---
    print(f"\nLoading model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    model.eval()

    # --- Load test data ---
    print(f"Loading test data from: {DATA_DIR}")
    dataset = load_from_disk(DATA_DIR)
    test_data = dataset["test"]
    print(f"  Test examples: {len(test_data)}")

    # --- Evaluate ---
    print("\nRunning evaluation...")
    all_true = []
    all_pred = []

    for example in test_data:
        tokens = example["tokens"]
        true_labels_ids = example["labels"]

        inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
        word_ids = inputs.word_ids()

        true_labels = []
        pred_labels = []
        prev_word_id = None

        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != prev_word_id:
                if word_id < len(true_labels_ids):
                    true_label_id = true_labels_ids[word_id]
                    # Skip -100 labels
                    if true_label_id != -100:
                        true_labels.append(LABEL_LIST[true_label_id])
                        pred_labels.append(LABEL_LIST[predictions[idx]])
            prev_word_id = word_id

        if true_labels:
            all_true.append(true_labels)
            all_pred.append(pred_labels)

    # --- Report ---
    print("\n" + "=" * 60)
    print("Classification Report:")
    print("=" * 60)
    report = classification_report(all_true, all_pred)
    print(report)

    metrics = {
        "precision": precision_score(all_true, all_pred),
        "recall": recall_score(all_true, all_pred),
        "f1": f1_score(all_true, all_pred),
        "num_test_examples": len(all_true),
    }

    print(f"Overall Precision: {metrics['precision']:.4f}")
    print(f"Overall Recall:    {metrics['recall']:.4f}")
    print(f"Overall F1:        {metrics['f1']:.4f}")

    # --- Save metrics ---
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
