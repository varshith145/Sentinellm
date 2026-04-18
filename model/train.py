"""
Training script for the SentinelLM semantic NER model.

Fine-tunes distilbert-base-uncased for token classification with BIO tags:
  O, B-PII, I-PII, B-SECRET, I-SECRET

Per PRD Section 9.4.
"""

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from seqeval.metrics import classification_report, f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# --- Config ---
MODEL_NAME = "./model/base_model"
OUTPUT_DIR = "./model/trained"
DATA_DIR = "./model/data/processed"
NUM_LABELS = 5  # O, B-PII, I-PII, B-SECRET, I-SECRET

LABEL_LIST = ["O", "B-PII", "I-PII", "B-SECRET", "I-SECRET"]
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}

# Class weights: O is common so we weight it low.
# PII/SECRET labels are rare so we weight them high — the model pays a
# much bigger penalty for missing them than for a false alarm on O.
# [O,  B-PII, I-PII, B-SECRET, I-SECRET]
CLASS_WEIGHTS = [0.3, 10.0, 10.0, 10.0, 10.0]


class WeightedLossTrainer(Trainer):
    """
    Custom Trainer that uses class-weighted cross-entropy loss.

    Without this, the model learns to predict all-O because O tokens
    outnumber PII/SECRET tokens and the default loss accepts that trade-off.
    With class weights, a missed PII prediction is 33x more costly than
    a missed O prediction, forcing the model to actually learn PII patterns.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        weights = torch.tensor(
            CLASS_WEIGHTS,
            dtype=torch.float,
            device=logits.device,
        )

        loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        loss = loss_fct(
            logits.view(-1, NUM_LABELS),
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss


def compute_metrics(p):
    """Compute sequence-level NER metrics using seqeval."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [LABEL_LIST[l] for (p_i, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_predictions = [
        [LABEL_LIST[p_i] for (p_i, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    f1 = f1_score(true_labels, true_predictions)
    print(f"\n  Epoch F1: {f1:.4f}")

    return {
        "f1": f1,
        "report": classification_report(true_labels, true_predictions),
    }


def print_label_distribution(dataset):
    """Print label counts so we can confirm PII labels exist in training data."""
    counts = {label: 0 for label in LABEL_LIST}
    for example in dataset:
        for label_id in example["labels"]:
            if label_id != -100 and 0 <= label_id < len(LABEL_LIST):
                counts[LABEL_LIST[label_id]] += 1
    print("\n  Label distribution in training set:")
    total = sum(counts.values())
    for label, count in counts.items():
        pct = count / total * 100 if total > 0 else 0
        print(f"    {label:12s}: {count:5d} ({pct:.1f}%)")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("SentinelLM — Training Semantic NER Model")
    print("=" * 60)

    # --- Load tokenizer and model ---
    print(f"\nLoading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # --- Load dataset ---
    print(f"Loading dataset from: {DATA_DIR}")
    dataset = load_from_disk(DATA_DIR)
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Validation: {len(dataset['validation'])} examples")
    print(f"  Test: {len(dataset['test'])} examples")

    # Confirm PII labels actually exist — if all zeros, data prep failed
    print_label_distribution(dataset["train"])

    # --- Data Collator ---
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=12,  # reduced from 20 — large dataset needs fewer passes to avoid memorization
        weight_decay=0.01,
        warmup_ratio=0.06,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        fp16=False,
        seed=42,
        report_to="none",
        dataloader_num_workers=0,
    )

    # --- Weighted Trainer ---
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- Train ---
    print("\nStarting training...")
    print(f"Class weights: {dict(zip(LABEL_LIST, CLASS_WEIGHTS))}")
    trainer.train()

    # --- Save best model ---
    print(f"\nSaving best model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # --- Final evaluation on validation set ---
    print("\nFinal evaluation on validation set:")
    results = trainer.evaluate()
    print(f"  F1: {results.get('eval_f1', 'N/A'):.4f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
