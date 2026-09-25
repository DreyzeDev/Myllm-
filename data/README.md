# Dataset V1

This repository contains the downloader and cleaner, source and rights manifest,
and a compact corpus report. Raw books and processed text stay on your machine;
they are ignored by Git and are not committed to GitHub.

## Sources

The first reproducible corpus combines a pinned selection from the
[Russian Stylometric Dataset](https://github.com/nevmenandr/RSD) (RSD) and two
English works by Charles Darwin from Project Gutenberg. See
[`source_manifest.yaml`](source_manifest.yaml) for exact paths, revision,
document counts, byte sizes, URLs, and rights notes. Source texts are not bundled
in this repository.

For V1 pretraining, the local dataset is expanded with pinned 2026-09-01
Wikimedia dumps: Russian Wikipedia, Simple English Wikipedia, and English
Wikibooks. Run `python scripts/download_wikimedia.py` and then convert each
dump with `scripts/extract_wikimedia.py`; the extractor keeps current main-
namespace pages and stores each original page URL for attribution. The Wikimedia
terms describe eligible text contributions under CC BY-SA 4.0 and GFDL, with
attribution requirements; see the source manifest for the terms link and dump
checksums. The original XML archives and extracted JSONL remain local and are
ignored by Git.

| Source | Extracted documents | Cleaned documents | Raw bytes | Rights summary |
|---|---:|---:|---:|---|
| RSD, six pinned text collections | 161 | 161 | 109,615,661 | Upstream says the underlying works are public domain under Russian law |
| Project Gutenberg #1228, *On the Origin of Species* | 1 | 1 | 970,612 | Project Gutenberg marks it public domain in the USA; Darwin died in 1882 |
| Project Gutenberg #944, *The Voyage of the Beagle* | 1 | 1 | 1,227,345 | Project Gutenberg marks it public domain in the USA; Darwin died in 1882 |
| Russian Wikipedia, pinned 2026-09-01 dump | 1,968,084 | 1,908,700 | 6,184,856,270 | Wikimedia text under CC BY-SA 4.0/GFDL terms; per-page URLs retained |
| Simple English Wikipedia, pinned 2026-09-01 dump | 188,016 | 186,600 | 385,846,687 | Wikimedia text under CC BY-SA 4.0/GFDL terms; per-page URLs retained |
| English Wikibooks, pinned 2026-09-01 dump | 72,754 | 70,600 | 207,339,557 | Wikimedia text under CC BY-SA 4.0/GFDL terms; per-page URLs retained |

The cleaned corpus has 2,166,063 documents and 1,101,292,823 words. The normalized text payload is 12,906,723,910 bytes; the on-disk JSONL is 13,599,656,626 bytes. Russian accounts for 87.6537% of counted words and English for 12.3463%. The cleaner removed 922 near-duplicates and 1,785 records matching email, phone-context, or credential patterns. See [`stats.json`](stats.json) for the complete local report.

RSD states that the works in its collection are public domain under Russian law.
Its dataset metadata and scripts are GPL-3.0; the collection's own README also
mentions CC BY-SA 3.0 for metadata. Only the works themselves are used here.
Project Gutenberg identifies the two eBooks as public domain in the USA and
requires readers outside the USA to check local copyright law. The downloaded
texts are cleaned to remove Project Gutenberg boilerplate before use. In Russia,
the default term is life plus 70 years; Darwin died in 1882 (see
[Article 1281, Part IV of the Civil Code](https://rospatent.gov.ru/ru/documents/grazhdanskiy-kodeks-rossiyskoy-federatsii-chast-chetvertaya)).
This legal screen does not substitute for checking another jurisdiction or a
special copyright exception.

## Download and clean

```bash
python scripts/download_dataset.py
python scripts/clean_dataset.py --input data/raw --output data/cleaned --workers 8
```

The downloader pins the RSD Git commit, verifies each Git blob SHA, verifies
Project Gutenberg file checksums, and writes `data/raw/raw_manifest.json`.
Downloads resume when the existing file matches its expected checksum. The
default total-size guard is 128 MiB and can be changed with
`--max-download-mb`.

The cleaner normalizes text to Unicode NFC, removes empty/corrupt records,
Project Gutenberg framing, HTML and common page-service footers, filters
short/noisy text and common exposed-secret/PII patterns, removes exact duplicates,
and drops near duplicates using 64-bit SimHash (Hamming distance at most 2).
It writes `data/cleaned/corpus.jsonl` with `text`, `source_id`, `source_path`,
and `language` fields, plus an ignored local `stats.json`. Basic pattern filtering cannot
identify every possible personal-data disclosure; review the resulting corpus
before use.

## Using this corpus with V1

The source manifest and measured statistics are versioned; the corpus itself is
not. Once you have downloaded and cleaned the source files, train a tokenizer on
your local cleaned JSONL and then prepare token blocks:

```bash
python scripts/train_tokenizer.py --input data/cleaned/corpus.jsonl --sample-fraction 0.1
python scripts/prepare_dataset.py \
  --input data/cleaned/corpus.jsonl \
  --tokenizer tokenizer/tokenizer.json \
  --output data/processed \
  --context-length 1024 \
  --validation-fraction 0.01
```

The BPE tokenizer is trained from scratch on a deterministic 10% document
sample to keep RAM use bounded; the complete cleaned corpus is used for token
blocks and pretraining. Prepared token blocks, datasets, and checkpoints stay
local and are excluded from Git.

## Prepared V1 corpus snapshot

For the local 2026-09-25 run, the 32,000-token tokenizer was trained on a
deterministic sample of 216,607 of 2,166,063 cleaned documents. The complete
corpus produced 2,097,957,189 estimated tokens. Token shares are 87.3961% Russian
and 12.6039% English; average tokens per counted word are 1.8994 for Russian
and 1.9447 for English. The seed-42 document-level 99/1 split contains
2,026,536 train blocks and 20,250 validation blocks at context length 1,024.
The block metadata and binary files are in ignored local directory
`data/processed/full_v1`; `source_manifest.yaml` stores these measurements.
