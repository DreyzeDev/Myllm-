# MyLLM V2 SFT dataset

**READY FOR V2 SFT: YES. Training has not been started.** This is a prepared
candidate, not a trained model. The starting point remains the immutable V2 base
`checkpoints/v2-pretraining/step_126783`; its lock passes validation.

## Final candidate

- 51,714 conversations: 49,128 train, 2,586 validation, 13 fixed evaluation prompts.
- User-prompt language estimate: 92.85% Russian and 7.15% English.
- Categories:

  | Category | Examples |
  | --- | ---: |
  | Conversation | 10,056 |
  | Multi-turn | 3,889 |
  | Instruction following | 10,507 |
  | Factual QA | 3,922 |
  | Explanations | 3,548 |
  | Language | 1,459 |
  | Translation | 3,804 |
  | Math | 7,570 |
  | Programming | 6,959 |

- Math and programming together are 28.09% of examples. Their answer structures
  are measured separately because correct calculations and code naturally reuse
  syntax.
- 32,000-token vocabulary; 1,024-token context. Longest encoded conversation is
  598 tokens. There are 21 assistant responses of 150–500 tokens and one above
  500 tokens.
- Assistant-only labels include EOS; user/system and role markers are masked.
  Packing is off. Training and inference use the same chat template.
- 55,643 assistant turns, 55,643 unique normalized responses, zero exact
  duplicate responses. Largest numeric-normalized template family is 0.31%.

## Provenance and RuBQ audit

- 1,489 direct single-answer examples from [RuBQ 2.0](https://github.com/vladislavneon/RuBQ),
  under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Attribution,
  URLs, checksums, normalization, corrections, and exclusions are in
  `dataset_manifest.yaml`.
- Automated audit covered all 2,910 source rows. It found 5 exact duplicate
  question groups, 2 answer conflicts, 53 near-duplicate question pairs, and one
  possible multi-valued fact conflict. Non-single-answer, outdated, ambiguous,
  false-premise, and other flagged rows are excluded or reviewed before import.
- A stratified manual sample of 120 accepted rows (60 from each source file)
  passed 120/120. This is a sample, not row-by-row human verification of all
  1,489 imported examples. No source paragraphs or Wikipedia completions were used.
- 5,402 selected V1 examples were re-audited; Wikimedia-derived rows were
  excluded. New arithmetic and Python examples are computed and checked. No
  teacher model or pretrained weights were used.

## Rebuild and verify

From the repository root:

```powershell
python scripts/build_sft_v2_109m_dataset.py
python scripts/audit_rubq_sft_v2.py --sample-size 120
python scripts/validate_sft_v2_109m_dataset.py
python -m pytest -q
```

After rebuilding, review the deterministic stratified RuBQ sample and update
`rubq_manual_review_log.json` before expecting the external fact-review gate to
pass. Dataset JSONL, source archives, audit reports, manual-review artifacts,
and checkpoints are ignored by Git; the compact manifest and audit documentation
are tracked.

See [`docs/myllm_v2_sft_dataset_audit.md`](../../docs/myllm_v2_sft_dataset_audit.md)
for category, language, length, diversity, masking, split-isolation, and source
audit details.

**SFT was not run.** A separate explicit instruction is required to start it.
