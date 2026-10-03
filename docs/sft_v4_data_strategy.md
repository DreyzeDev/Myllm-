# Future SFT data quality gate

Do not copy the V3 dataset into a new SFT run as-is. V1 is an experimental
release and SFT V3 is retained only as a comparison checkpoint.

Before any later SFT authorization, build a fresh, provenance-tracked dataset
and require these audits:

## Balance and conversational behavior

- Measure category counts and language shares after deduplication and after
  sampling. Keep ordinary conversation, instruction following, and multi-turn
  examples frequent enough that factual encyclopedic responses cannot dominate
  the batches.
- Include short, natural user questions and direct assistant answers, with
  multiple user phrasings and answer phrasings for common skills.
- Include actual multi-turn context references, not merely concatenated
  unrelated exchanges.
- Record source, URL, license, acquisition date, transformation, and review
  status for every externally derived group. Mark authored/synthetic examples
  and their review process explicitly.

## Repetition and template caps

- Report the number and percentage of unique normalized assistant responses,
  exact repeated responses, repeated normalized prefixes, and repeated 5- and
  8-grams. Normalize Unicode and whitespace before counting.
- Group examples into template families using both prompt and response
  structure. Report family counts and the largest family share overall and by
  category.
- Deduplicate identical examples and near-identical prompt/answer pairs. Cap
  any one generic response template at 0.5% of the corpus and any one template
  family at 2% of the corpus; exceptions for canonical short answers must be
  reviewed and reported separately.
- Require at least 80% unique normalized assistant responses overall, with a
  separate rate for each category. Add controlled paraphrases rather than
  cloning a single response under unrelated questions.

## Independent category checks

- **Math:** verify arithmetic answers with an independent deterministic
  evaluator; manually review word-problem interpretation and units.
- **Programming:** parse code, run small examples against expected outputs in a
  restricted test environment, and reject fabricated APIs or unrelated code.
- **Factual QA:** review against cited, reputable references; keep answer
  concise and directly responsive to the question.
- **Language and translation:** use bilingual review or a trusted reference;
  include direction labels so RU-to-EN and EN-to-RU are both checked.
- **Conversation/instructions:** manually inspect samples for naturalness,
  relevance, requested format, and absence of fabricated user context.

## Release gate for data

Reject empty or malformed records, mixed-up roles, broken Unicode, long
unnecessary answers, repeated assistant prefixes beyond the caps, PII/secrets,
train/validation/evaluation leakage, tokenizer incompatibility, context-length
overflow, and missing assistant EOS labels. Keep a fixed evaluation split that
is never added to training. No training should start until the complete report
passes and the user separately authorizes the run.
