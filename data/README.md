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

| Source | Documents | Raw bytes | Rights summary |
|---|---:|---:|---|
| RSD, six pinned text collections | 161 | 109,615,661 | Upstream says the underlying works are public domain under Russian law |
| Project Gutenberg #1228, *On the Origin of Species* | 1 | 970,612 | Project Gutenberg marks it public domain in the USA; Darwin died in 1882 |
| Project Gutenberg #944, *The Voyage of the Beagle* | 1 | 1,227,345 | Project Gutenberg marks it public domain in the USA; Darwin died in 1882 |

The cleaned corpus has 163 documents and 9,651,853 words by the script's
Unicode-aware counter. The normalized text payload is 111,720,555
bytes. The on-disk JSONL is 112,092,080 bytes because each record carries source
provenance and JSON framing. Russian accounts for 96.2128% of counted words and
English for 3.7872%. See [`stats.json`](stats.json) for the full reproducible
summary.

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
python scripts/clean_dataset.py --input data/raw --output data/cleaned
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
python scripts/train_tokenizer.py --input data/cleaned/corpus.jsonl
python scripts/prepare_dataset.py \
  --input data/cleaned/corpus.jsonl \
  --tokenizer tokenizer/tokenizer.json \
  --output data/processed \
  --context-length 1024 \
  --validation-fraction 0.01
```

The tokenizer-training and model-pretraining commands are intentionally not run
as part of dataset preparation. Prepared token blocks and checkpoints also stay
local and are excluded from Git.
