"""
download.py
-----------
Downloads the Xenova/all-MiniLM-L6-v2 ONNX model from HuggingFace Hub.

Run once before anything else:
    python download.py

What it saves:
    models/Xenova/all-MiniLM-L6-v2/tokenizer.json
    models/Xenova/all-MiniLM-L6-v2/model.onnx
    models/Xenova/all-MiniLM-L6-v2/model.onnx_data  (if the model uses a data sidecar)

Why ONNX?
    The sentence-transformers library pulls in PyTorch (~4.8 GB).
    The ONNX Runtime is ~147 MB.  Same vectors, no PyTorch, runs anywhere.
"""

import os
import shutil
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

# Silence HuggingFace telemetry and noisy log output
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Different repos name their ONNX file differently — check these in order
ONNX_CANDIDATES = [
    "onnx/model.onnx",
    "onnx/encoder_model.onnx",
    "model.onnx",
]


def download(repo, dest="models"):
    """
    Download tokenizer + ONNX model from a HuggingFace repo.

    Args:
        repo: HuggingFace repo ID, e.g. "Xenova/all-MiniLM-L6-v2"
        dest: local folder to save into (default: "models")
    """
    dest = Path(dest) / repo
    dest.mkdir(parents=True, exist_ok=True)

    # List all files in the repo to find which ONNX variant exists
    files = list_repo_files(repo_id=repo)
    onnx_file = next((c for c in ONNX_CANDIDATES if c in files), None)
    if not onnx_file:
        raise FileNotFoundError(f"No ONNX model found in {repo}")

    # Download tokenizer and model, rename to standard local names
    for remote, local in [
        ("tokenizer.json", "tokenizer.json"),
        (onnx_file, "model.onnx"),
    ]:
        src = hf_hub_download(repo_id=repo, filename=remote)
        dst = dest / local
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"  saved  {dst}")
        else:
            print(f"  exists {dst}")

    # Some large models split weights into a separate _data sidecar file
    onnx_ext = onnx_file + "_data"
    if onnx_ext in files:
        src = hf_hub_download(repo_id=repo, filename=onnx_ext)
        dst = dest / "model.onnx_data"
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"  saved  {dst}")
        else:
            print(f"  exists {dst}")


if __name__ == "__main__":
    print("Downloading Xenova/all-MiniLM-L6-v2 ...")
    download("Xenova/all-MiniLM-L6-v2")
    print("Done.")
