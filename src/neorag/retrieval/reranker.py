from __future__ import annotations

import threading

import torch

from neorag.logging_utils import logger
from neorag.retrieval.passage import Passage


class CrossEncoderReranker:
    def __init__(self, model_id: str, device: str = "auto", batch_size: int = 16):
        self.model_id = model_id
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        with self._lock:
            if self._model is not None:
                return True
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
                self._model.to(self.device)
                self._model.eval()
                return True
            except Exception as exc:
                logger.warning(f"Reranker unavailable ({exc}); skipping reranking.")
                return False

    @torch.no_grad()
    def rerank(self, query: str, passages: list[Passage]) -> list[Passage]:
        if not self._ensure_loaded():
            return passages
        pairs = [(query, p.text) for p in passages]
        scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            logits = self._model(**encoded).logits
            flat = logits.reshape(-1)
            if flat.numel() != len(batch):
                flat = flat[:, 0] if logits.dim() > 1 else flat
            scores.extend(flat.float().cpu().tolist())
        ranked = sorted(zip(passages, scores), key=lambda t: t[1], reverse=True)
        return [p for p, _ in ranked]
