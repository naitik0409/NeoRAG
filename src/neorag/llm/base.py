from __future__ import annotations

from typing import Protocol

Message = dict[str, str]


class LLM(Protocol):
    def generate(
        self,
        messages: list[Message],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str: ...


def build_llm(cfg):
    if cfg.backend == "api":
        from neorag.llm.openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(
            api_base=cfg.api_base,
            api_key_env=cfg.api_key_env,
            model=cfg.api_model or cfg.model_id,
            reasoning_effort=cfg.reasoning_effort,
        )
    from neorag.llm.local_hf import LocalHFLLM

    return LocalHFLLM(
        model_id=cfg.model_id,
        dtype=cfg.dtype,
        quantize_4bit=cfg.quantize_4bit,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        repetition_penalty=cfg.repetition_penalty,
    )
