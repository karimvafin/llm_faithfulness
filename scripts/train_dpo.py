import argparse
import gc
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO fine-tuning with optional LoRA.")
    p.add_argument("--model-name", required=True)
    p.add_argument("--train-file", required=True)
    p.add_argument("--eval-file", default=None)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-prompt-length", type=int, default=1792)

    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--no-lora", action="store_true")

    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")

    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = args.max_length

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.enable_input_require_grads()

    if not args.no_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def _apply_chat_template(tokenizer, messages, *, add_generation_prompt):
    """Mirror llm.LLM.apply_chat_template: keep thinking DISABLED so the training prompt
    matches the evaluation prompt exactly. For Qwen3, enable_thinking=False makes the
    template append an empty <think></think> block after the assistant header; omitting
    it (template default True) yields a different prefix, and the DPO preference then
    fails to transfer to the distribution scored at eval."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        # tokenizer's chat template doesn't accept enable_thinking (e.g. Falcon, Llama)
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def make_format_func(tokenizer):
    eos = tokenizer.eos_token or ""

    def _format(example):
        assistant_prefix = example.get("assistant_prefix")
        if assistant_prefix:
            prompt_text = _apply_chat_template(
                tokenizer,
                [{"role": "user", "content": example["prompt"]}],
                add_generation_prompt=False,
            )
            prompt_text += assistant_prefix
        else:
            prompt_text = _apply_chat_template(
                tokenizer,
                [{"role": "user", "content": example["prompt"]}],
                add_generation_prompt=True,
            )
        return {
            "prompt": prompt_text,
            "chosen": example["chosen"] + eos,
            "rejected": example["rejected"] + eos,
        }

    return _format


def main() -> None:
    args = parse_args()
    fix_seed(args.seed)

    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file
    raw_datasets = load_dataset("json", data_files=data_files)

    model, tokenizer = build_model_and_tokenizer(args)

    raw_datasets = raw_datasets.map(make_format_func(tokenizer))
    use_eval = "validation" in raw_datasets

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        optim="adamw_torch_fused",
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if use_eval else "no",
        eval_steps=args.save_steps if use_eval else None,
        load_best_model_at_end=use_eval,
        metric_for_best_model="eval_loss" if use_eval else None,
        greater_is_better=False if use_eval else None,
        seed=args.seed,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["validation"] if use_eval else None,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved {args.output_dir}")

    if not args.no_lora:
        del trainer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        merged_dir = os.path.join(args.output_dir, "merged")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        peft_model = PeftModel.from_pretrained(base_model, args.output_dir)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        AutoTokenizer.from_pretrained(args.model_name).save_pretrained(merged_dir)
        print(f"Merged model saved to {merged_dir}")


if __name__ == "__main__":
    main()
