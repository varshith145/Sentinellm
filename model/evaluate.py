"""
Evaluation script for the SentinelLM semantic NER model.

Runs the trained model against the held-out test set and produces
a detailed evaluation report per PRD Section 9.6.

Uses the pre-computed input_ids and labels from the dataset directly,
avoiding re-tokenization (which caused word_id misalignment in v1).
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
from transformers import AutoModelForTokenClassification

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
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    model.eval()

    # --- Load test data ---
    print(f"Loading test data from: {DATA_DIR}")
    dataset = load_from_disk(DATA_DIR)
    test_data = dataset["test"]
    print(f"  Test examples: {len(test_data)}")

    # --- Evaluate ---
    # Use the stored input_ids and labels directly — no re-tokenization.
    # The labels array has -100 at positions for [CLS], [SEP], [PAD] and
    # continuation subword tokens; we skip those when building sequences.
    print("\nRunning evaluation...")
    all_true = []
    all_pred = []
    pred_counts = {label: 0 for label in LABEL_LIST}

    for example in test_data:
        input_ids = torch.tensor([example["input_ids"]])
        attention_mask = torch.tensor([example["attention_mask"]])
        stored_labels = example["labels"]   # list of ints, -100 where ignored

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        pred_ids = torch.argmax(outputs.logits, dim=2)[0].tolist()

        true_seq = []
        pred_seq = []

        for pred_id, true_id in zip(pred_ids, stored_labels):
            if true_id == -100:
                # [CLS], [SEP], [PAD], or subword continuation — skip
                continue
            true_seq.append(LABEL_LIST[true_id])
            pred_seq.append(LABEL_LIST[pred_id])
            pred_counts[LABEL_LIST[pred_id]] += 1

        if true_seq:
            all_true.append(true_seq)
            all_pred.append(pred_seq)

    # --- Prediction distribution (sanity check) ---
    print("\nPrediction distribution (should NOT be all-O):")
    total_preds = sum(pred_counts.values())
    for label, count in pred_counts.items():
        pct = count / total_preds * 100 if total_preds > 0 else 0
        marker = " ← good" if label != "O" and count > 0 else ""
        print(f"  {label:12s}: {count:4d} ({pct:.1f}%){marker}")

    if all(v == 0 for k, v in pred_counts.items() if k != "O"):
        print("\n  ⚠️  Model is predicting all-O — class weighting may need tuning.")
    else:
        print("\n  ✅ Model is predicting non-O labels — training worked!")

    # --- Report ---
    print("\n" + "=" * 60)
    print("Classification Report:")
    print("=" * 60)

    if not all_true:
        print("No valid examples to evaluate.")
        metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "num_test_examples": 0}
    else:
        report = classification_report(all_true, all_pred, zero_division=0)
        print(report)

        metrics = {
            "precision": precision_score(all_true, all_pred, zero_division=0),
            "recall": recall_score(all_true, all_pred, zero_division=0),
            "f1": f1_score(all_true, all_pred, zero_division=0),
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
