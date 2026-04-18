"""
SentinelLM — Download base model for offline training

Run this ONCE on your Mac before running ./train.sh
It downloads distilbert-base-uncased to model/base_model/ so Docker
doesn't need internet access during training.

Uses huggingface_hub to download raw files — no PyTorch needed.

Usage:
    pip3 install transformers --quiet
    python3 model/download_base_model.py
"""

from pathlib import Path

SAVE_DIR = Path(__file__).parent / "base_model"
REPO_ID = "distilbert-base-uncased"

def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("\n❌ huggingface_hub not installed. Run:")
        print("   pip3 install transformers")
        raise SystemExit(1)

    print("=" * 60)
    print("Downloading distilbert-base-uncased to model/base_model/")
    print("(This only needs to be done once — ~270MB)")
    print("=" * 60)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n📥 Downloading all model files for {REPO_ID}...")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(SAVE_DIR),
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
    )

    print("\n" + "=" * 60)
    print(f"✅ Base model saved to: {SAVE_DIR}")
    print("   Files downloaded:")
    for f in sorted(SAVE_DIR.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"   {f.name:40s} {size_mb:.1f} MB")
    print("\n   Now run: ./train.sh")
    print("=" * 60)

if __name__ == "__main__":
    main()
