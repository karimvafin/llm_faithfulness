import os
from typing import Any

import torch
import torch.nn.functional as F
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

    @torch.inference_mode()
    def generate_with_entropy(
        self,
        prompts: list[str],
        max_new_tokens: int = 512,
        markers: list[str] | None = None,
    ) -> list[tuple[str, list[float], dict[str, int] | None]]:
        """HF only. Returns [(text, entropies, marker_positions)] per prompt.
        entropies[t] is the Shannon entropy (nats) of the next-token distribution at step t.
        marker_positions maps each marker substring to the token index right after which
        the marker first fully appears in the decoded text (None if `markers` is None)."""
        if self.use_api:
            raise NotImplementedError(
                "generate_with_entropy requires the local HF backend; "
                "full next-token distribution is not available via OpenAI-style API."
            )

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
            output_scores=True,
            return_dict_in_generate=True,
        )
        input_len = enc["input_ids"].shape[1]
        gen_ids = out.sequences[:, input_len:]
        scores = out.scores  # tuple of length S, each [batch, vocab]
        eos_id = self.tokenizer.eos_token_id

        results: list[tuple[str, list[float], dict[str, int] | None]] = []
        for i in range(gen_ids.shape[0]):
            seq = gen_ids[i]
            if eos_id is not None:
                eos_positions = (seq == eos_id).nonzero(as_tuple=False).flatten()
                length = int(eos_positions[0].item()) if eos_positions.numel() > 0 else seq.shape[0]
            else:
                length = seq.shape[0]
            length = min(length, len(scores))

            entropies: list[float] = []
            for t in range(length):
                logits = scores[t][i].to(torch.float32)
                logp = F.log_softmax(logits, dim=-1)
                p = logp.exp()
                h = -(p * logp).sum().item()
                entropies.append(h)

            text = self.tokenizer.decode(seq[:length], skip_special_tokens=True)
            marker_positions: dict[str, int] | None = None
            if markers is not None:
                marker_positions = self._find_marker_positions(seq[:length], text, markers)
            results.append((text, entropies, marker_positions))
        return results

    def _find_marker_positions(
        self,
        token_ids: torch.Tensor,
        full_text: str,
        markers: list[str],
    ) -> dict[str, int]:
        """For each marker found in full_text, return the token index after which
        the marker's start character offset has been crossed in the decoded prefix."""
        targets: list[tuple[str, int]] = []
        for m in markers:
            idx = full_text.find(m)
            if idx >= 0:
                targets.append((m, idx))
        if not targets:
            return {}
        targets.sort(key=lambda kv: kv[1])

        result: dict[str, int] = {}
        cursor = 0
        n = int(token_ids.shape[0])
        for t in range(1, n + 1):
            if cursor >= len(targets):
                break
            cum_len = len(self.tokenizer.decode(token_ids[:t], skip_special_tokens=True))
            while cursor < len(targets) and cum_len >= targets[cursor][1]:
                result[targets[cursor][0]] = t
                cursor += 1
        return result

    def _api_generate(self, prompt: str, max_new_tokens: int) -> str:
        resp = self._client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        return resp.choices[0].text
