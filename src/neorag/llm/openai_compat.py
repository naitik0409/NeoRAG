from __future__ import annotations

import os

from neorag.logging_utils import logger

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class OpenAICompatLLM:
    def __init__(
        self,
        api_base: str,
        api_key_env: str,
        model: str,
        timeout: int = 120,
        reasoning_effort: str = "low",
    ):
        if not api_base:
            raise ValueError("llm.api_base must be set when backend='api' (use a free-tier endpoint)")
        self.api_base = api_base.rstrip("/")
        self.api_key_env = api_key_env
        self.model = model
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort

    def _missing_key_error(self) -> RuntimeError:
        return RuntimeError(
            f"Missing API key: set {self.api_key_env}=<your key> in the .env file at the project root"
        )

    def _generate_sdk(self, messages, max_tokens: int, temperature: float, top_p: float, key: str) -> str:
        from openai import OpenAI

        client = OpenAI(base_url=self.api_base, api_key=key, timeout=self.timeout)
        kwargs: dict = {}
        if self.reasoning_effort and "gpt-oss" in self.model.lower():
            kwargs["reasoning_effort"] = self.reasoning_effort
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        choices = getattr(resp, "choices", []) or []
        if not choices:
            raise RuntimeError("LLM API returned no choices")
        message = choices[0].message
        content = (message.content or "").strip()
        if not content:
            reasoning = (getattr(message, "reasoning", None) or "").strip()
            if reasoning:
                logger.warning("LLM content empty; using reasoning text as fallback")
                content = reasoning
        if not content:
            logger.warning("LLM API returned empty content")
        return content

    def _generate_requests(self, messages, payload: dict, key: str) -> str:
        import requests

        headers = {
            "User-Agent": BROWSER_UA,
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        resp = requests.post(
            f"{self.api_base}/chat/completions", json=payload, headers=headers, timeout=self.timeout
        )
        if resp.status_code != 200:
            raise RuntimeError(f"LLM API error {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM API returned no choices: {data}")
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            logger.warning("LLM API returned empty content")
        return content

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise self._missing_key_error()
        max_tokens = max_new_tokens or 512
        temperature = temperature if temperature is not None else 0.7
        top_p = top_p if top_p is not None else 0.9
        try:
            return self._generate_sdk(messages, max_tokens, temperature, top_p, key)
        except ImportError:
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            return self._generate_requests(messages, payload, key)
        except Exception as exc:
            name = type(exc).__name__
            if name in {"HTTPError", "APIConnectionError", "Timeout", "ConnectionError"} or name.startswith("API"):
                raise RuntimeError(f"LLM API {name}: {exc}") from exc
            raise
