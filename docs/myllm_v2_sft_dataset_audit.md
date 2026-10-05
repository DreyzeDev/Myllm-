# MyLLM V2 SFT dataset audit

Audit date: 2026-10-05. Dataset is prepared only; no SFT training has run.

## Dataset inventory

| Split | Conversations |
| --- | ---: |
| Train | 49,128 |
| Validation | 2,586 |
| **Train + validation** | **51,714** |
| Fixed evaluation (separate) | 13 |
| **All stored records** | **51,727** |

Category counts and share of all conversations:

| Category | Count | Share |
| --- | ---: | ---: |
| Conversation | 10,056 | 19.45% |
| Multi-turn | 3,889 | 7.52% |
| Instruction following | 10,507 | 20.32% |
| General knowledge / factual QA | 3,922 | 7.58% |
| Explanations | 3,548 | 6.86% |
| Language | 1,459 | 2.82% |
| Translation | 3,804 | 7.36% |
| Math | 7,570 | 14.64% |
| Programming | 6,959 | 13.46% |

Math and programming together are 28.09%, below the 30% cap. Numeric-normalized
family counts are interpreted separately for math and code: repeated arithmetic
forms or valid code syntax are not treated as evidence of low quality on their
own.

Declared user-prompt language is 92.85% Russian and 7.15% English. Character-based
automatic detection agrees to rounding: 48,018 Russian and 3,696 English prompts.
The tokenizer vocabulary is 32,000 and maximum sequence length is 598 tokens out
of the 1,024-token context. Mean encoded conversation length is 47.64 tokens;
mean assistant turn is 24.69 tokens.

| Encoded length | Assistant turns | Conversations |
| --- | ---: | ---: |
| 1–32 tokens | 42,870 | 14,009 |
| 33–149 tokens | 12,751 | 37,678 |
| 150–500 tokens | 21 | 26 |
| Over 500 tokens | 1 | 1 |

The longest assistant response is 579 tokens. Longer material includes authored
explanations and instruction sequences; the whole candidate remains within the
context limit. The long-answer share is intentionally small but nonzero.

## Response diversity and template audit

- Assistant turns: 55,643; normalized unique responses: 55,643; exact duplicate
  groups/excess: 0; unique share: 100%.
- Artificial preambles such as “можно ответить так” and “ответ получается таким”:
  0 remaining after the style filter.
- Largest numeric-normalized response family: 0.31% globally. Category maxima:
  general knowledge 3.80%, instruction following 1.67%, math 1.65%, programming
  1.15%, explanations 0.96%; conversation, multi-turn, language, and translation
  have no repeated exact response family after this normalization.
- Numeric-normalized excess is 97.38% in math because the value is the variable in
  a correctly repeated formula; the audit does not label that alone as bad.
  Programming is audited separately and contributes 13.46% of the candidate.

Most frequent assistant prefixes (3–10 words) are concentrated in the list
selection instructions: “вот три подходящих” (231), “из списка подойдут” (231),
“среди вариантов есть” (231), “возьму эти три” (229), and “в качестве ответа”
(229). Each is under 0.42% of assistant turns; none is one of the removed artificial
meta prefixes.

Most frequent 5-grams:

| N-gram | Count | Main source pattern |
| --- | ---: | --- |
| `функция будет такой python def` | 178 | Python function examples |
| `ряд по возрастанию выглядит так` | 175 | Numeric sorting answers |
| `вот короткое решение для числа` | 175 | Python unary examples |
| `от самого маленького до самого` | 175 | Numeric sorting answers |
| `вот последовательность от меньшего к` | 175 | Numeric sorting answers |

Most frequent 8-grams:

| N-gram | Count | Main source pattern |
| --- | ---: | --- |
| `чтобы найти процент от числа умножаем число на` | 170 | Percent calculation explanation |
| `найти процент от числа умножаем число на долю` | 170 | Percent calculation explanation |
| `сначала умножаем число на процент затем делим на` | 165 | Percent calculation explanation |
| `подготовь место и материалы затем выполни один простой` | 157 | Everyday multi-turn advice |
| `you do not need to finish everything in one` | 140 | English everyday dialogue |

These counts are retained in `data/sft_v2/audit_report.json`. Formula/code patterns
are grouped by skill; conversational advice patterns are a known synthetic-source
limitation and remain a minority of the full assistant turns. The multi-turn
planning generator was reduced from its earlier oversized share before this
candidate was finalized.

## Sources and RuBQ review

| Source type | Examples |
| --- | ---: |
| Project-authored curated synthetic | 21,319 |
| Deterministically computed/traced synthetic | 23,504 |
| Re-audited synthetic subset from old V1 SFT | 5,402 |
| External RuBQ 2.0 direct QA | 1,489 |

RuBQ 2.0 is attributed to its contributors and licensed CC BY-SA 4.0. The two
source-file SHA-256 values are in `data/sft_v2/dataset_manifest.yaml`. Only direct
question/answer fields were imported; context paragraphs and wiki prose were not
used. Three accepted answers received an explicit correction to clarify the
linked answer or place: Neo/Thomas Anderson, the ConsultantPlus developer, and
the city of Alexandria. Twenty-four additional source rows were excluded by the
manual fact-review rules, including false-premise or ambiguous questions.

Full automated source audit of 2,910 rows:

- Answer-array cardinality: 510 empty, 2,022 single-answer, and 378 with multiple
  answers. The input also contains multi-alias arrays with cardinalities above 2;
  these are outside the imported single-answer set.
- 5 exact duplicate question groups, 2 same-question answer conflicts, 53
  near-duplicate question pairs at the audit threshold, and 1 possible conflict
  on a property that may be multi-valued.
- Structural audit errors: 0. The importer retains 1,489 single-answer direct QA
  rows after automated filters, duplicate grouping, temporal/factual exclusions,
  and explicit corrections.
- Stratified manual review: 120 accepted candidate rows, 60 from each RuBQ file;
  120 pass, 0 rejected. The sample covers source file × property × answer kind.
  It is a representative sample, not a manual review of every imported row.

## Training-format and integrity gates

- `src.sft_data.encode_messages` and `encode_chat_prompt` provide the shared
  training/inference format: BOS, role token, content, assistant marker, EOS.
- Assistant response text and its terminal EOS have trainable labels. System,
  user, and role/separator tokens are masked with `-100`.
- Packing is off. EOS coverage failures: 0; assistant-mask failures: 0;
  train/inference template mismatches: 0. Unit tests cover label masking and EOS.
- Exact user-prompt overlap between train and validation: 0. Semantic group overlap:
  0. Fixed evaluation leakage: 0. PII/secret failures and invalid records: 0.
- Tokenizer compatibility: PASS. Context limit: PASS. V2 base lock for
  `checkpoints/v2-pretraining/step_126783`: PASS. Provenance/license audit: PASS.
- `pytest -q`: 28 passed. Dataset validator: PASS. RuBQ automated audit:
  PASS, including the 120-row human sample.

## Readiness

**READY FOR V2 SFT: YES.** The candidate meets the automated balance and integrity
gates and the stratified RuBQ sample passed. This means the dataset and pipeline
are prepared for a separately authorized SFT run; it does not mean training has
started. Base weights remain locked and unchanged.
