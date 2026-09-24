from src.tokenizer import ByteBPETokenizer


def test_train_and_round_trip_russian_and_english(tmp_path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "Привет, мир! Hello world.\n\n" * 40 + "Я обучаю свою модель. I train my own model.\n" * 40,
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = ByteBPETokenizer.train([corpus], tokenizer_path, vocab_size=300, min_frequency=1)
    sample = "Привет! Hello!"
    ids = tokenizer.encode(sample)
    assert tokenizer_path.exists()
    assert tokenizer.vocab_size >= 264
    assert tokenizer.decode(ids) == sample
    assert tokenizer.token_id("<bos>") >= 0
    assert tokenizer.token_id("<|assistant|>") >= 0

