from __future__ import annotations

import threading

import numpy as np
import torch


def _pick_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        batch_size: int = 32,
        normalize: bool = True,
        max_length: int = 512,
    ):
        self.model_id = model_id
        self.device = _pick_device(device)
        self.batch_size = batch_size
        self.normalize = normalize
        self.max_length = max_length
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id)
            self._model.to(self.device)
            self._model.eval()

    def _mean_pool(self, outputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(outputs.dtype)
        summed = (outputs * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @torch.no_grad()
    def embed(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        all_vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self._model(**encoded)
            hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            vectors = self._mean_pool(hidden, encoded["attention_mask"])
            if self.normalize:
                vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            all_vectors.append(vectors.cpu().float().numpy())
        if not all_vectors:
            return np.zeros((0, self._model.config.hidden_size), dtype=np.float32)
        return np.vstack(all_vectors)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
