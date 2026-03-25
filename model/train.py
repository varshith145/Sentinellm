"""
Training script for the SentinelLM semantic NER model.

Fine-tunes distilbert-base-uncased for token classification with BIO tags:
  O, B-PII, I-PII, B-SECRET, I-SECRET

Per PRD Section 9.4.
"""

import numpy as np
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
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./model/trained"
DATA_DIR = "./model/data/processed"
NUM_LABELS = 5  # O, B-PII, I-PII, B-SECRET, I-SECRET

LABEL_LIST = ["O", "B-PII", "I-PII", "B-SECRET", "I-SECRET"]
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}


def compute_metrics(p):
    """Compute sequence-level NER metrics using seqeval."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (-100) and convert to label strings
    true_labels = [
        [LABEL_LIST[l] for (p_i, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_predictions = [
        [LABEL_LIST[p_i] for (p_i, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    return {
        "f1": f1_score(true_labels, true_predictions),
        "report": classification_report(true_labels, true_predictions),
    }


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
    )

    # --- Load dataset ---
    print(f"Loading dataset from: {DATA_DIR}")
    dataset = load_from_disk(DATA_DIR)
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Validation: {len(dataset['validation'])} examples")
    print(f"  Test: {len(dataset['test'])} examples")

    # --- Data Collator ---
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        fp16=False,  # Set True if GPU available
        seed=42,
        report_to="none",  # Disable wandb/tensorboard
    )

    # --- Trainer ---
    trainer = Trainer(
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
    trainer.train()

    # --- Save ---
    print(f"\nSaving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # --- Final evaluation ---
    print("\nFinal evaluation on validation set:")
    results = trainer.evaluate()
    print(f"  F1: {results.get('eval_f1', 'N/A')}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
