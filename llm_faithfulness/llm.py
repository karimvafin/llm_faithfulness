import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

Message = dict[str, str]


class LLM:
    def __init__(
        self,
        model_name: str,
        *,
        use_api: bool = False,
        api_base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ):
        self.model_name = model_name
        self.use_api = use_api
        if use_api:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=api_base_url,
                api_key=os.environ.get(api_key_env, "dummy"),
            )
            self.tokenizer = None
            self.model = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map,
            )
            self.model.eval()

    def apply_chat_template(
        self,
        messages: list[Message],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        if self.tokenizer is None:
            parts = []
            for m in messages:
                parts.append(f"{m['role'].upper()}: {m['content']}")
            return "\n".join(parts)
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # tokenizer's chat template doesn't accept enable_thinking
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    @torch.inference_mode()
    def generate(self, prompts: list[str], max_new_tokens: int = 512) -> list[str]:
        if self.use_api:
            return [self._api_generate(p, max_new_tokens) for p in prompts]
        return self._hf_generate(prompts, max_new_tokens)

    def _hf_generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        ).to(self.model.device)

        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        input_len = enc["input_ids"].shape[1]
        gen = out[:, input_len:]
        texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        return texts

    def _api_generate(self, prompt: str, max_new_tokens: int) -> str:
        resp = self._client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        return resp.choices[0].text
