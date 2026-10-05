from pathlib import Path

import torch

from src.generate import generate_tokens
from src.sft_data import encode_chat_prompt, encode_messages, to_training_pair
from src.tokenizer import ByteBPETokenizer


def _tokenizer() -> ByteBPETokenizer:
    root = Path(__file__).resolve().parents[1]
    return ByteBPETokenizer.load(root / "tokenizer" / "tokenizer.json")


def test_sft_loss_masks_everything_except_assistant_text_and_matches_chat_prefix() -> None:
    tokenizer = _tokenizer()
    conversations = [
        [
            {"role": "system", "content": "Отвечай коротко."},
            {"role": "user", "content": "Привет"},
            {"role": "assistant", "content": "Здравствуйте!"},
        ],
        [
            {"role": "user", "content": "Расскажи о Марсе."},
            {"role": "assistant", "content": "Марс — планета."},
            {"role": "user", "content": "Почему он красный?"},
            {"role": "assistant", "content": "Из-за оксидов железа."},
        ],
    ]
    role_ids = {
        tokenizer.token_id("<bos>"),
        tokenizer.token_id("<|system|>"),
        tokenizer.token_id("<|user|>"),
        tokenizer.token_id("<|assistant|>"),
    }

    for messages in conversations:
        full_ids, target_mask = encode_messages(messages, tokenizer)
        input_ids, labels = to_training_pair(messages, tokenizer)

        expected_target_ids: list[int] = []
        expected_target_positions: list[int] = []
        cursor = 1  # <bos>
        for message in messages:
            role_id = tokenizer.token_id(f"<|{message['role']}|>")
            assert full_ids[cursor] == role_id
            marker_position = cursor
            cursor += 1
            content_ids = tokenizer.encode(message["content"])
            if message["role"] == "assistant":
                expected_target_ids.extend(content_ids)
                expected_target_positions.extend(range(marker_position, marker_position + len(content_ids)))
            cursor += len(content_ids)
        expected_target_ids.append(tokenizer.token_id("<eos>"))
        expected_target_positions.append(len(full_ids) - 2)

        actual_positions = [index for index, label in enumerate(labels) if label != -100]
        actual_targets = [label for label in labels if label != -100]
        assert actual_targets == expected_target_ids
        assert actual_positions == expected_target_positions
        assert len(input_ids) == len(labels) == len(full_ids) - 1
        assert full_ids[-1] == tokenizer.token_id("<eos>")
        assert target_mask[-1] is True
        assert all(label == -100 or label not in role_ids for label in labels)
        assert labels[-1] == tokenizer.token_id("<eos>")

        history = messages[:-1]
        prompt_ids = encode_chat_prompt(history, tokenizer)
        last_assistant_marker = max(
            index for index, token_id in enumerate(full_ids) if token_id == tokenizer.token_id("<|assistant|>")
        )
        assert prompt_ids == full_ids[: last_assistant_marker + 1]


def test_example_serialization_special_tokens_and_inference_prefix() -> None:
    tokenizer = _tokenizer()
    messages = [
        {"role": "user", "content": "Привет"},
        {"role": "assistant", "content": "Привет! Чем могу помочь?"},
    ]
    full_ids, target_mask = encode_messages(messages, tokenizer)
    input_ids, labels = to_training_pair(messages, tokenizer)
    inference_ids = encode_chat_prompt(messages[:1], tokenizer)
    decoded = tokenizer.decode(full_ids, skip_special_tokens=False)

    assert decoded.startswith("<bos><|user|>Привет<|assistant|>Привет! Чем могу помочь?<eos>")
    assert tokenizer.encode(decoded) == full_ids
    assert full_ids.count(tokenizer.token_id("<bos>")) == 1
    assert inference_ids.count(tokenizer.token_id("<bos>")) == 1
    assert input_ids.count(tokenizer.token_id("<|user|>")) == 1
    assert labels.count(tokenizer.token_id("<eos>")) == 1
    assert target_mask[-1]
    assert inference_ids[-1] == tokenizer.token_id("<|assistant|>")
    assert inference_ids == full_ids[: len(inference_ids)]
    assert tokenizer.decode([tokenizer.token_id("<|assistant|>")], skip_special_tokens=False) == "<|assistant|>"


class _AlwaysEosModel(torch.nn.Module):
    def __init__(self, eos_id: int, vocab_size: int) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = type("Config", (), {"context_length": 16})()
        self.eos_id = eos_id
        self.vocab_size = vocab_size
        self.calls = 0

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        self.calls += 1
        logits = torch.full((*input_ids.shape, self.vocab_size), -100.0, device=input_ids.device)
        logits[..., self.eos_id] = 100.0
        cache = ((torch.zeros(1, 1, input_ids.shape[1], 1), torch.zeros(1, 1, input_ids.shape[1], 1)),)
        return type("Output", (), {"logits": logits, "past_key_values": cache if use_cache else None})()


def test_generation_stops_immediately_on_eos() -> None:
    model = _AlwaysEosModel(eos_id=2, vocab_size=8)
    output = generate_tokens(
        model,
        prompt_ids=[1],
        max_new_tokens=8,
        temperature=0,
        top_k=None,
        top_p=1.0,
        eos_token_id=2,
    )
    assert output == [1, 2]
    assert model.calls == 1
