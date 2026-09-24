# Data folders

Put your own `.txt` or `.jsonl` files in `data/raw/`. JSONL accepts either a
string on each line or an object with a `text` field (configurable with
`--text-key`). Keep private or licensed material out of public GitHub commits.

Tokenizer and preparation output goes into ignored local folders. The prepared
`.bin` files contain fixed-length `uint16` token blocks; `metadata.json` records
their sizes and context length.

