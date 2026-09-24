import json

from src.dataset import TokenBlockDataset, prepare_dataset, read_records
from src.tokenizer import ByteBPETokenizer


def test_prepare_dataset_deduplicates_and_writes_splits(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "texts.txt"
    records = [f"Русский текст номер {index}. English sample number {index}." for index in range(20)]
    source.write_text("\n\n".join(records + [records[0]]), encoding="utf-8")
    tokenizer_path = tmp_path / "tokenizer.json"
    ByteBPETokenizer.train([source], tokenizer_path, vocab_size=300, min_frequency=1)
    output = tmp_path / "prepared"
    metadata = prepare_dataset(source, tokenizer_path, output, context_length=8, validation_fraction=0.2, seed=4)
    assert metadata["records"] == 20
    assert metadata["duplicates_removed"] == 1
    assert metadata["train_blocks"] > 0
    assert metadata["validation_blocks"] > 0
    train = TokenBlockDataset(output / "train.bin", metadata["train_blocks"], metadata["block_length"])
    input_ids, labels = train[0]
    assert input_ids.shape == labels.shape == (8,)
    assert json.loads((output / "metadata.json").read_text(encoding="utf-8"))["context_length"] == 8


def test_jsonl_records_are_read_from_text_field(tmp_path) -> None:
    source = tmp_path / "examples.jsonl"
    source.write_text(
        '{"text":"Привет"}\n{"text":"  Hello  "}\n"JSON string record"\n',
        encoding="utf-8",
    )
    records, duplicates = read_records([source], text_key="text")
    assert records == ["Привет", "Hello", "JSON string record"]
    assert duplicates == 0


def test_single_plain_text_record_is_split_without_overlap() -> None:
    from src.dataset import split_records

    train, validation = split_records(["one two three four five six seven eight nine ten"], 0.2, 42)
    assert len(train) == len(validation) == 1
    assert not set(train[0].split()).intersection(validation[0].split())
