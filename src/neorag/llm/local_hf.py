from __future__ import annotations

import torch

from neorag.logging_utils import logger


class LocalHFLLM:
    def __init__(
        self,
        model_id: str,
        dtype: str = "auto",
        quantize_4bit: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
    ):
        self.model_id = model_id
        self.dtype = dtype
        self.quantize_4bit = bool(quantize_4bit)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self._tokenizer = None
        self._model = None

    def _resolve_dtype(self) -> torch.dtype:
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cuda = torch.cuda.is_available()
        kwargs: dict = {"local_files_only": False}
        dtype = self._resolve_dtype()
        try:
            model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype, **kwargs)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=dtype, **kwargs)

        if self.quantize_4bit and cuda:
            try:
                from transformers import BitsAndBytesConfig

                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, quantization_config=bnb, **kwargs
                )
            except Exception as exc:
                logger.warning(f"4-bit quantization unavailable ({exc}); loading unquantized.")

        model.to("cuda" if cuda else "cpu")
        model.eval()
        self._model = model
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def build_prompt(self, messages: list[dict[str, str]]) -> str:
        self._ensure_loaded()
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        self._ensure_loaded()
        temperature = self.temperature if temperature is None else temperature
        top_p = self.top_p if top_p is None else top_p
        max_new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens

        input_ids = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self._model.device)
        attention_mask = torch.ones_like(input_ids)

        do_sample = (temperature or 0.0) > 0.0
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)

        output = self._model.generate(**gen_kwargs)
        new_tokens = output[0][input_ids.shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)
