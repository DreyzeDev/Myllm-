from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from src.tokenizer import ByteBPETokenizer
from src.utils import load_model_checkpoint, select_device


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    top_p: float,
    generated: list[int],
    repetition_penalty: float,
) -> int:
    scores = logits.clone()
    if repetition_penalty != 1.0 and generated:
        for token_id in set(generated):
            scores[token_id] = scores[token_id] * repetition_penalty if scores[token_id] < 0 else scores[token_id] / repetition_penalty
    if temperature <= 0:
        return int(torch.argmax(scores).item())
    scores = scores / temperature
    if top_k is not None and top_k > 0:
        threshold = torch.topk(scores, min(top_k, scores.numel())).values[-1]
        scores[scores < threshold] = -torch.inf
    if 0.0 < top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[sorted_indices[remove]] = -torch.inf
    probabilities = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1).item())


@torch.inference_mode()
def generate_tokens(
    model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.05,
    seed: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> list[int]:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not prompt_ids:
        raise ValueError("prompt_ids cannot be empty")
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    device = next(model.parameters()).device
    output_ids = list(prompt_ids[-model.config.context_length :])
    past_key_values = None
    for _ in range(max_new_tokens):
        if past_key_values is not None and past_key_values[0][0].shape[-2] < model.config.context_length:
            current = torch.tensor([[output_ids[-1]]], dtype=torch.long, device=device)
        else:
            past_key_values = None
            current = torch.tensor([output_ids[-model.config.context_length :]], dtype=torch.long, device=device)
        output = model(current, past_key_values=past_key_values, use_cache=True)
        past_key_values = output.past_key_values
        next_id = _sample_next_token(
            output.logits[0, -1].float(), temperature, top_k, top_p, output_ids, repetition_penalty
        )
        output_ids.append(next_id)
        if eos_token_id is not None and next_id == eos_token_id:
            break
    return output_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a model checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory, e.g. checkpoints/step_00500")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = select_device()
    model = load_model_checkpoint(args.checkpoint, device)
    tokenizer = ByteBPETokenizer.load(args.tokenizer)
    bos_id = tokenizer.token_id("<bos>")
    eos_id = tokenizer.token_id("<eos>")

    def answer(prompt: str) -> str:
        prompt_ids = tokenizer.encode(prompt, add_bos=True)
        result = generate_tokens(
            model,
            prompt_ids or [bos_id],
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            args.repetition_penalty,
            args.seed,
            eos_id,
        )
        return tokenizer.decode(result)

    if args.prompt is not None:
        print(answer(args.prompt))
        return
    print("From-scratch V1 chat. До обучения ответы будут случайными и бессмысленными. Выход: /exit")
    while True:
        try:
            prompt = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"/exit", "/quit"}:
            break
        if prompt:
            print(answer(prompt))


if __name__ == "__main__":
    main()

