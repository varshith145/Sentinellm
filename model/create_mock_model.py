"""
SentinelLM — Mock model generator

Creates a minimal valid DistilBERT token-classification model with random
weights so the semantic detector loads and runs immediately — even before
you've run the full training pipeline.

The model will have low accuracy (random weights) but the detector
WILL be active rather than silently disabled.

Run this once to unblock development:
    python model/create_mock_model.py

Then run ./train.sh when you're ready to replace it with a real model.
"""

import json
import os
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "trained"

LABEL_LIST = ["O", "B-PII", "I-PII", "B-SECRET", "I-SECRET"]
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}


def create_mock_model():
    print("=" * 60)
    print("SentinelLM — Creating mock semantic NER model")
    print("=" * 60)

    try:
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            DistilBertConfig,
        )
    except ImportError:
        print("\n❌ transformers not installed.")
        print("   Run: pip install transformers torch")
        print("   Or use ./train.sh to run inside Docker.\n")
        sys.exit(1)

    print(f"\nOutput directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create a tiny DistilBERT config (much smaller than default)
    # so it loads fast in development
    print("Creating minimal DistilBERT config (random weights)...")
    config = DistilBertConfig(
        vocab_size=30522,
        max_position_embeddings=512,
        sinusoidal_pos_embds=False,
        n_layers=2,          # 2 layers instead of 6 — tiny, fast
        n_heads=4,           # 4 heads instead of 12
        dim=256,             # 256 dims instead of 768
        hidden_dim=512,
        dropout=0.1,
        attention_dropout=0.1,
        activation="gelu",
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        finetuning_task="token-classification",
    )

    model = AutoModelForTokenClassification.from_config(config)

    print("Saving model...")
    model.save_pretrained(str(OUTPUT_DIR))

    print("Saving tokenizer (downloads from HuggingFace)...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Save label map as a separate JSON for easy inspection
    label_map_path = OUTPUT_DIR / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID}, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ Mock model created!")
    print(f"   Location: {OUTPUT_DIR}")
    print("")
    print("⚠️  This model has RANDOM weights — low accuracy.")
    print("   Run ./train.sh to replace it with a properly trained model.")
    print("=" * 60)


if __name__ == "__main__":
    create_mock_model()
