"""
embedder.py
-----------
A lightweight text embedder using ONNX Runtime + HuggingFace tokenizers.

No PyTorch.  No sentence-transformers.  Same vectors — 30x smaller footprint.

Usage:
    from embedder import Embedder

    embed = Embedder()                        # loads model from ./models/
    v = embed.encode("some text")             # returns np.ndarray, shape (384,)
    X = embed.encode_batch(["a", "b", "c"])  # returns np.ndarray, shape (3, 384)
"""

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from pathlib import Path


class Embedder:
    """
    Wraps an ONNX sentence-transformer model.

    The pipeline for each call:
      1. Tokenize  — convert text to token IDs + attention mask
      2. Inference — run the ONNX model, get hidden states per token
      3. Pool      — mean-pool over non-padding tokens (masked mean)
      4. Normalize — L2-normalize so dot product == cosine similarity
    """

    def __init__(self, path="models/Xenova/all-MiniLM-L6-v2"):
        path = Path(path)

        # HuggingFace fast tokenizer (no Python tokenizer dependency)
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))

        # ONNX Runtime session — CPU only, no CUDA needed
        self.session = ort.InferenceSession(
            str(path / "model.onnx"),
            providers=["CPUExecutionProvider"]
        )

        # Collect which input names this model actually expects
        # (some models don't use token_type_ids)
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Embed a single string.

        Returns:
            np.ndarray of shape (384,)
        """
        return self.encode_batch([text], normalize=normalize)[0]

    def encode_batch(self, texts: list, normalize: bool = True) -> np.ndarray:
        """
        Embed a list of strings in one ONNX call.

        Padding is enabled so all sequences in the batch have the same length.

        Returns:
            np.ndarray of shape (len(texts), 384)
        """
        # Enable padding so sequences batch together cleanly
        self.tokenizer.enable_padding()

        encoded = self.tokenizer.encode_batch(texts)

        # Build the feed dict with only the inputs this model expects
        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = np.array(
                [e.ids for e in encoded], dtype=np.int64
            )
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )

        # Run ONNX inference — first output is token-level hidden states
        hidden = self.session.run(None, feed)[0]   # shape: (batch, seq_len, 384)

        # Masked mean pooling:
        #   multiply each token's hidden state by its attention mask (0 for padding),
        #   sum across the sequence, then divide by the number of real tokens.
        mask = feed["attention_mask"][..., None]   # shape: (batch, seq_len, 1)
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)   # shape: (batch, 384)

        # L2 normalize so that dot product == cosine similarity
        if normalize:
            pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)

        return pooled
