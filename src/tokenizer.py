from __future__ import annotations

from pathlib import Path

from tokenizers import AddedToken, Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]


class ByteBPETokenizer:
    def __init__(self, tokenizer: HFTokenizer) -> None:
        self._tokenizer = tokenizer

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        return cls(HFTokenizer.from_file(str(path)))

    @classmethod
    def train(
        cls,
        files: list[str | Path],
        output_path: str | Path,
        vocab_size: int = 32000,
        min_frequency: int = 2,
    ) -> "ByteBPETokenizer":
        if vocab_size < 264:
            raise ValueError("Byte-level BPE vocab_size must be at least 264")
        tokenizer = HFTokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        special_tokens = [AddedToken(token, special=True, normalized=False) for token in SPECIAL_TOKENS]
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        tokenizer.train([str(path) for path in files], trainer)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(target))
        return cls(tokenizer)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    def token_id(self, token: str) -> int:
        value = self._tokenizer.token_to_id(token)
        if value is None:
            raise KeyError(f"Token {token!r} is not in the tokenizer vocabulary")
        return value

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self._tokenizer.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids.insert(0, self.token_id("<bos>"))
        if add_eos:
            ids.append(self.token_id("<eos>"))
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

