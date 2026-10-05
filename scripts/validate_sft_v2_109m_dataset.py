from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sft_data import (
    encode_chat_prompt, encode_messages, is_obvious_contact_or_secret,
    normalize_for_dedup, to_training_pair, validate_turn_order,
)
from src.tokenizer import ByteBPETokenizer

REQUIRED_EVAL = {
    "Привет", "Как дела?", "Кто ты?", "Что ты умеешь?", "Столица России?",
    "Столица Франции?", "7 * 8?", "2 + 2?", "Что такое Солнце?",
    "Земля вращается вокруг чего?", "Переведи Hello на русский.",
    "Назови три планеты.", "Напиши Python функцию сложения двух чисел.",
}
ARTIFICIAL_PREFIXES = (
    "можно ответить так", "если тебе интересно", "если вам интересно",
    "можно сказать так", "можно написать так", "можно объяснить так",
    "ответ получается таким", "ответ получается следующим образом", "результат получается таким",
    "короткий ответ", "если кратко", "в двух словах", "запомни коротко",
    "основная идея такая", "для простого объяснения", "попробую объяснить проще",
    "попросту говоря", "проще говоря", "иными словами", "представь так", "это можно представить так",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def words(text: str) -> list[str]:
    return re.findall(r"[\w]+", normalize_for_dedup(text), flags=re.UNICODE)


def template(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    text = re.sub(r"(?<!\w)[+-]?\d+(?:[.,]\d+)?(?!\w)", "{N}", text)
    return re.sub(r"\s+", " ", text)


def external_answer_traces(source_answer: str, assistant_answer: str) -> bool:
    """Check that all informative source-answer words survive natural inflection/rephrasing."""
    source_words = words(source_answer)
    response_words = words(assistant_answer)
    informative = [word for word in source_words if len(word) >= 4]
    if not informative:
        informative = source_words
    if not informative:
        return False
    for source_word in informative:
        if not any(
            source_word == response_word
            or (min(len(source_word), len(response_word)) >= 5
                and (source_word.startswith(response_word[:5]) or response_word.startswith(source_word[:5])))
            for response_word in response_words
        ):
            return False
    return True


def detect_language(text: str) -> str:
    ru = sum("а" <= c.lower() <= "я" or c.lower() == "ё" for c in text)
    en = sum("a" <= c.lower() <= "z" for c in text)
    return "ru" if ru >= en else "en"


def read_jsonl(path: Path, errors: list[str]) -> list[dict]:
    rows = []
    if not path.is_file():
        errors.append(f"missing {path.name}")
        return rows
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                errors.append(f"{path.name}:{n}: invalid JSON: {exc}")
    return rows


def audit(root: Path, tokenizer_path: Path, context: int) -> tuple[dict, list[str]]:
    errors: list[str] = []
    try:
        manifest = yaml.safe_load((root / "dataset_manifest.yaml").read_text(encoding="utf-8"))
    except Exception as exc:
        manifest = {}
        errors.append(f"invalid manifest: {exc}")
    split_rows = {name: read_jsonl(root / f"{name}.jsonl", errors) for name in ("train", "validation")}
    evaluation = read_jsonl(root / "evaluation.jsonl", errors)
    rows = split_rows["train"] + split_rows["validation"]
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    categories, source_types, external = Counter(), Counter(), Counter()
    declared_lang, detected_lang = Counter(), Counter()
    response_counts, prefix_counts, template_counts = Counter(), Counter(), Counter()
    category_templates = defaultdict(Counter)
    category_prefixes = defaultdict(Counter)
    category_responses = defaultdict(Counter)
    category_grams = defaultdict(lambda: {5: Counter(), 8: Counter()})
    computational_checks = Counter()
    assistant_texts, group_splits = [], defaultdict(set)
    user_prompt_splits = defaultdict(set)
    assistant_length_buckets = Counter()
    sequence_length_buckets = Counter()
    category_length_buckets = defaultdict(Counter)
    artificial_prefix_counts = Counter()
    user_tokens = assistant_tokens = 0
    max_sequence = max_answer_tokens = max_answer_chars = 0
    eos_failures = mask_failures = template_failures = invalid = 0
    fixed_users = {normalize_for_dedup(r["messages"][0]["content"]) for r in evaluation if r.get("messages")}
    all_users = set()
    for split, data in split_rows.items():
        for line_no, row in enumerate(data, 1):
            messages = row.get("messages")
            issue = validate_turn_order(messages)
            if issue:
                invalid += 1
                errors.append(f"{split}:{line_no}: {issue}")
                continue
            category = row.get("category", "missing")
            categories[category] += 1
            if category not in {"conversation", "general_knowledge", "explanations", "math", "language",
                                "translation", "programming", "instruction_following", "multi_turn"}:
                errors.append(f"{split}:{line_no}: unexpected category {category!r}")
            source = row.get("source", {})
            source_types[source.get("type", "missing")] += 1
            if source.get("type") == "external_dataset":
                external[source.get("source_id", "missing")] += 1
            declared_lang[row.get("language", "missing")] += 1
            group_splits[str(row.get("group_id", row.get("id"))) ].add(split)
            combined = "\n".join(m["content"] for m in messages)
            if is_obvious_contact_or_secret(combined):
                errors.append(f"{split}:{line_no}: PII/secret pattern")
            if "\ufffd" in combined or "\x00" in combined or unicodedata.normalize("NFC", combined) != combined:
                errors.append(f"{split}:{line_no}: damaged or non-NFC Unicode")
            if any(mark in combined for mark in ("Ã", "Ð", "Ñ")):
                errors.append(f"{split}:{line_no}: possible mojibake")
            for msg in messages:
                text = msg["content"]
                token_count = len(tokenizer.encode(text))
                if msg["role"] == "user":
                    all_users.add(normalize_for_dedup(text))
                    user_prompt_splits[normalize_for_dedup(text)].add(split)
                    user_tokens += token_count
                elif msg["role"] == "assistant":
                    assistant_tokens += token_count
                    assistant_texts.append(text)
                    normalized_answer = normalize_for_dedup(text)
                    response_counts[normalized_answer] += 1
                    category_responses[row.get("category", "missing")][normalized_answer] += 1
                    response_shape = template(text)
                    template_counts[response_shape] += 1
                    category_templates[row.get("category", "missing")][response_shape] += 1
                    answer_words = words(text)
                    for n in range(3, min(10, len(answer_words)) + 1):
                        prefix = f"{n}:{' '.join(answer_words[:n])}"
                        prefix_counts[prefix] += 1
                        category_prefixes[row.get("category", "missing")][prefix] += 1
                    max_answer_tokens = max(max_answer_tokens, token_count)
                    max_answer_chars = max(max_answer_chars, len(text))
                    length_bucket = "short_1_32" if token_count <= 32 else "medium_33_149" if token_count <= 149 else "long_150_500" if token_count <= 500 else "extended_over_500"
                    assistant_length_buckets[length_bucket] += 1
                    category_length_buckets[row.get("category", "missing")][length_bucket] += 1
                    lowered = unicodedata.normalize("NFKC", text).casefold().lstrip()
                    for banned in ARTIFICIAL_PREFIXES:
                        if lowered.startswith(banned):
                            artificial_prefix_counts[banned] += 1
                    if len(text) > 3000:
                        errors.append(f"{split}:{line_no}: assistant response >3000 chars")
            prompt_text = " ".join(m["content"] for m in messages if m["role"] == "user")
            detected_lang[detect_language(prompt_text)] += 1
            try:
                ids, mask = encode_messages(messages, tokenizer)
                inputs, labels = to_training_pair(messages, tokenizer)
                max_sequence = max(max_sequence, len(ids))
                sequence_bucket = "short_1_32" if len(ids) <= 32 else "medium_33_149" if len(ids) <= 149 else "long_150_500" if len(ids) <= 500 else "extended_over_500"
                sequence_length_buckets[sequence_bucket] += 1
                if len(ids) > context:
                    errors.append(f"{split}:{line_no}: sequence {len(ids)} exceeds context {context}")
                eos = tokenizer.token_id("<eos>")
                if ids[-1] != eos or not mask[-1] or labels[-1] != eos:
                    eos_failures += 1
                expected = [ids[i+1] if mask[i+1] else -100 for i in range(len(ids)-1)]
                if labels != expected:
                    mask_failures += 1
                inference_ids = encode_chat_prompt(messages[:-1], tokenizer)
                marker = max(i for i, token in enumerate(ids) if token == tokenizer.token_id("<|assistant|>"))
                if inference_ids != ids[:marker+1]:
                    template_failures += 1
            except Exception as exc:
                errors.append(f"{split}:{line_no}: tokenizer/mask/template error: {exc}")
            try:
                source_data = row.get("source", {})
                generator = source_data.get("generator")
                assistant_answer = next(m["content"] for m in reversed(messages) if m["role"] == "assistant")
                if generator in {"integer_arithmetic_v2_109m", "integer_arithmetic_en_v1"}:
                    match = re.fullmatch(r"(-?\d+)([+−×÷])(-?\d+)", str(source_data.get("expression", "")))
                    if not match:
                        raise ValueError("malformed arithmetic expression metadata")
                    left, op, right = int(match[1]), match[2], int(match[3])
                    expected_value = {"+": left+right, "−": left-right, "×": left*right,
                                      "÷": left//right if right and left % right == 0 else None}.get(op)
                    if expected_value != source_data.get("expected") or str(expected_value) not in assistant_answer:
                        raise ValueError("arithmetic answer mismatch")
                    computational_checks[generator] += 1
                elif generator == "python_function_patterns_v2":
                    scope = {"__builtins__": {"max": max}}
                    exec(compile(source_data["code"], "<generated-python>", "exec"), scope, scope)
                    args = source_data["inputs"]
                    if args and isinstance(args[0], list):
                        result = scope[source_data["function_name"]](args[0])
                    else:
                        result = scope[source_data["function_name"]](*args)
                    if result != source_data["expected"]:
                        raise ValueError(f"Python result {result!r} != expected {source_data['expected']!r}")
                    computational_checks[generator] += 1
                elif generator == "rectangle_area_explanation_v1":
                    if source_data["inputs"][0] * source_data["inputs"][1] != source_data["expected"]:
                        raise ValueError("rectangle area mismatch")
                    if str(source_data["expected"]) not in assistant_answer:
                        raise ValueError("rectangle answer does not state expected area")
                    computational_checks[generator] += 1
                elif generator == "percent_explanation_v1":
                    p, whole = source_data["inputs"]
                    value = whole * p / 100
                    value = int(value) if value.is_integer() else round(value, 2)
                    if value != source_data["expected"] or str(value) not in assistant_answer:
                        raise ValueError("percentage explanation mismatch")
                    computational_checks[generator] += 1
                elif generator == "numeric_sort_instruction_v1":
                    if sorted(source_data["input"]) != source_data["expected"] or any(str(v) not in assistant_answer for v in source_data["expected"]):
                        raise ValueError("sort output mismatch")
                    computational_checks[generator] += 1
                elif generator == "bilingual_sentence_composition_v2":
                    expected_translation = source_data["russian_text"] if source_data["direction"] == "en_to_ru" else source_data["english_text"]
                    if assistant_answer != expected_translation:
                        raise ValueError("translation does not match its paired authored sentence")
                    computational_checks[generator] += 1
                elif source.get("type") == "external_dataset":
                    direct_answer = normalize_for_dedup(str(source_data.get("assistant_fact_answer") or source_data.get("source_answer", "")))
                    normalized_response = normalize_for_dedup(assistant_answer)
                    if source_data.get("paragraphs_used") is not False or not direct_answer or not external_answer_traces(direct_answer, normalized_response):
                        raise ValueError("external QA answer does not trace to direct answer_text")
                    computational_checks["rubq_direct_qa_trace"] += 1
            except Exception as exc:
                errors.append(f"{split}:{line_no}: computed/source QA audit failed: {exc}")

    required_actual = {r["messages"][0]["content"] for r in evaluation if r.get("messages")}
    if len(evaluation) != 13 or required_actual != REQUIRED_EVAL:
        errors.append(f"evaluation must contain the fixed 13 prompts; got {len(evaluation)}")
    eval_leaks = len(fixed_users & all_users)
    if eval_leaks:
        errors.append(f"{eval_leaks} fixed evaluation prompts occur in train/validation")
    split_leaks = sum(len(where) > 1 for where in group_splits.values())
    if split_leaks:
        errors.append(f"{split_leaks} semantic groups cross train/validation")
    exact_prompt_leaks = sum(len(where) > 1 for where in user_prompt_splits.values())
    if exact_prompt_leaks:
        errors.append(f"{exact_prompt_leaks} exact user prompts cross train/validation")

    exact_excess = sum(count-1 for count in response_counts.values() if count > 1)
    near_template_excess = sum(count-1 for count in template_counts.values() if count > 1)
    top_prefixes = [
        {"word_count": int(key.split(":", 1)[0]), "prefix": key.split(":", 1)[1], "count": count}
        for key, count in prefix_counts.most_common(50)
    ]
    top_templates = [
        {"template": key[:220], "count": count, "share_percent": round(100*count/max(1,len(assistant_texts)), 3)}
        for key, count in template_counts.most_common(50)
    ]
    category_template_summary = {}
    for category, counts in category_templates.items():
        total_category_responses = sum(counts.values())
        category_template_summary[category] = {
            "assistant_turns": total_category_responses,
            "template_family_count": len(counts),
            "largest_family_count": max(counts.values(), default=0),
            "largest_family_share_percent": round(100*max(counts.values(), default=0)/max(1,total_category_responses), 2),
            "numeric_normalized_template_excess": sum(n-1 for n in counts.values() if n > 1),
            "numeric_normalized_template_excess_share_percent": round(
                100*sum(n-1 for n in counts.values() if n > 1)/max(1,total_category_responses), 2),
            "top_10": [{"template": text[:180], "count": n} for text,n in counts.most_common(10)],
            "exact_duplicate_assistant_turns": sum(n-1 for n in category_responses[category].values() if n > 1),
            "unique_normalized_responses": len(category_responses[category]),
            "assistant_response_length_buckets": dict(sorted(category_length_buckets[category].items())),
        }
    category_prefix_summary = {
        category: [{"prefix": key.split(":",1)[1], "word_count": int(key.split(":",1)[0]), "count": n}
                   for key,n in counts.most_common(10)]
        for category, counts in category_prefixes.items()
    }
    grams = {}
    category_grams_report = {}
    for n in (5, 8):
        counts = Counter()
        for row in rows:
            for message in row.get("messages", []):
                if message.get("role") != "assistant":
                    continue
                answer = message["content"]
                w = words(answer)
                counts.update(" ".join(w[i:i+n]) for i in range(max(0, len(w)-n+1)))
                category_grams[row.get("category", "missing")][n].update(
                    " ".join(w[i:i+n]) for i in range(max(0, len(w)-n+1))
                )
        grams[str(n)] = [{"ngram": gram, "count": count} for gram, count in counts.most_common(50)]
    for category, values in category_grams.items():
        category_grams_report[category] = {
            str(n): [{"ngram": gram, "count": count} for gram, count in values[n].most_common(15)]
            for n in (5, 8)
        }

    long_answers = assistant_length_buckets.get("long_150_500", 0) + assistant_length_buckets.get("extended_over_500", 0)
    long_sequences = sequence_length_buckets.get("long_150_500", 0) + sequence_length_buckets.get("extended_over_500", 0)
    if long_answers < 20:
        errors.append(f"only {long_answers} assistant responses reach 150 tokens; at least 20 authored long examples required")
    if long_sequences < 20:
        errors.append(f"only {long_sequences} conversations reach 150 tokens; at least 20 long-context sequences required")
    if assistant_length_buckets.get("extended_over_500", 0) < 1 or sequence_length_buckets.get("extended_over_500", 0) < 1:
        errors.append("at least one authored assistant answer and sequence must exceed 500 tokens while remaining within context")
    if artificial_prefix_counts:
        errors.append(f"artificial response prefixes remain: {dict(artificial_prefix_counts)}")

    base_errors = []
    try:
        lock = json.loads((ROOT/"checkpoints"/"v2-pretraining"/"FINAL_BASE_LOCK.json").read_text(encoding="utf-8"))
        base = ROOT/"checkpoints"/"v2-pretraining"/"step_126783"
        if lock.get("status") != "immutable_final_base" or lock.get("step") != 126783:
            base_errors.append("lock metadata is not final step_126783")
        if manifest.get("base_checkpoint") != "checkpoints/v2-pretraining/step_126783":
            base_errors.append("dataset manifest names a different base")
        for entry in lock.get("sha256", []):
            path = base / entry["file"]
            if not path.is_file() or path.stat().st_size != entry["bytes"] or digest(path).lower() != entry["sha256"].lower():
                base_errors.append(f"base checkpoint changed: {entry['file']}")
    except Exception as exc:
        base_errors.append(f"cannot verify base lock: {exc}")
    errors.extend(base_errors)

    provenance_errors = []
    rubq_report = {}
    rubq_review_summary = {"status": "missing", "reviewed_count": 0, "passed_count": 0, "rejected_count": 0}
    if external.get("rubq_2_0", 0):
        expected_hashes = {
            "RuBQ_2.0_dev.json": "6CD20DCFAD3404736691C5DCEE271D19F579B6C5755A1AAD4C4DE644BC9DFD05",
            "RuBQ_2.0_test.json": "63D9E32A1BDEFB2DE713A9CCF3A0F9F0A4A48F2FDDDD35B3908D6846B63DC85B",
        }
        for name, expected_hash in expected_hashes.items():
            path = root/"sources"/"rubq"/name
            if not path.is_file() or digest(path).lower() != expected_hash.lower():
                provenance_errors.append(f"RuBQ source hash mismatch/missing: {name}")
        rubq = next((s for s in manifest.get("sources", []) if s.get("id") == "rubq_2_0"), {})
        if rubq.get("license") != "CC BY-SA 4.0" or not rubq.get("attribution"):
            provenance_errors.append("RuBQ license/attribution is missing")
        if manifest.get("external_dataset_audit", {}).get("paragraphs_loaded_or_used") is not False:
            provenance_errors.append("RuBQ paragraphs are not explicitly excluded")
        rubq_report_path = root / "rubq_automated_audit.json"
        if not rubq_report_path.is_file():
            provenance_errors.append("RuBQ full automated audit report is missing")
        else:
            try:
                rubq_report = json.loads(rubq_report_path.read_text(encoding="utf-8"))
                if rubq_report.get("errors"):
                    provenance_errors.append("RuBQ full automated audit reports source errors")
                rubq_review_summary = rubq_report.get("manual_review", rubq_review_summary)
                if (rubq_review_summary.get("review_status") != "PASS"
                        or rubq_review_summary.get("reviewed_count", 0) < 120
                        or rubq_review_summary.get("passed_count", 0) != rubq_review_summary.get("reviewed_count", 0)
                        or rubq_review_summary.get("rejected_count", 0) != 0):
                    provenance_errors.append("RuBQ final stratified sample has not passed 120 manual reviews")
            except Exception as exc:
                provenance_errors.append(f"cannot read RuBQ audit/manual review report: {exc}")
    errors.extend(provenance_errors)
    total = len(rows)
    lang_total = detected_lang["ru"] + detected_lang["en"]
    ru_percent = round(100*detected_lang["ru"]/max(1,lang_total), 2)
    en_percent = round(100*detected_lang["en"]/max(1,lang_total), 2)
    balance_minimums = {"conversation": 9000, "multi_turn": 3500, "explanations": 2500,
                        "general_knowledge": 2500, "language": 1200}
    for category, minimum in balance_minimums.items():
        if categories[category] < minimum:
            errors.append(f"assistant behavior balance: {category} has {categories[category]} examples; need at least {minimum}")
    compute_share = 100 * (categories["math"] + categories["programming"]) / max(1, total)
    if compute_share > 30:
        errors.append(f"math+programming are {compute_share:.2f}% of data; maximum allowed share is 30%")
    if not (85 <= ru_percent <= 95 and 5 <= en_percent <= 15):
        errors.append(f"language balance outside target: Russian {ru_percent:.2f}%, English {en_percent:.2f}%")
    report = {
        "dataset": {"total": total, "train": len(split_rows["train"]), "validation": len(split_rows["validation"]),
                    "evaluation": len(evaluation), "categories": dict(sorted(categories.items())),
                    "source_types": dict(sorted(source_types.items())), "external_source_examples": dict(sorted(external.items()))},
        "languages": {"declared": dict(sorted(declared_lang.items())), "detected_user_prompts": dict(sorted(detected_lang.items())),
                      "ru_percent": ru_percent, "en_percent": en_percent},
        "tokens": {"tokenizer_vocab": tokenizer.vocab_size, "context_length": context,
                   "max_sequence_tokens": max_sequence,
                   "mean_sequence_tokens": round((user_tokens+assistant_tokens)/max(1,total), 2),
                   "mean_user_tokens_per_turn": round(user_tokens/max(1,sum(sum(m["role"]=="user" for m in r["messages"]) for r in rows)), 2),
                   "mean_assistant_tokens_per_turn": round(assistant_tokens/max(1,len(assistant_texts)), 2),
                   "max_single_answer_chars": max_answer_chars, "max_single_answer_tokens": max_answer_tokens,
                   "assistant_response_length_buckets": dict(sorted(assistant_length_buckets.items())),
                   "conversation_sequence_length_buckets": dict(sorted(sequence_length_buckets.items()))},
        "diversity": {"assistant_turns": len(assistant_texts), "unique_normalized_assistant_responses": len(response_counts),
                      "exact_duplicate_groups": sum(count>1 for count in response_counts.values()),
                      "exact_duplicate_excess": exact_excess,
                      "exact_unique_share_percent": round(100*(len(assistant_texts)-exact_excess)/max(1,len(assistant_texts)),2),
                      "top_50_first_3_to_10_word_prefixes": top_prefixes,
                      "template_family_count_after_numeric_normalization": len(template_counts),
                      "largest_template_family_share_percent": round(100*max(template_counts.values(),default=0)/max(1,len(assistant_texts)),2),
                      "same_template_near_duplicate_excess_after_number_normalization": near_template_excess,
                      "same_template_near_duplicate_share_percent": round(100*near_template_excess/max(1,len(assistant_texts)),2),
                      "artificial_prefix_occurrences": dict(sorted(artificial_prefix_counts.items())),
                      "template_families_by_category": category_template_summary,
                      "top_prefixes_by_category": category_prefix_summary,
                      "top_50_template_families": top_templates,
                      "top_50_5grams": grams["5"], "top_50_8grams": grams["8"],
                      "top_15_5grams_and_8grams_by_category": category_grams_report},
        "integrity": {"invalid_records": invalid, "eos_failures": eos_failures, "assistant_mask_failures": mask_failures,
                      "train_inference_template_mismatches": template_failures, "fixed_eval_prompt_leaks": eval_leaks,
                      "exact_user_prompt_split_leaks": exact_prompt_leaks,
                      "semantic_group_split_leaks": split_leaks, "tokenizer_compatibility": tokenizer.vocab_size == 32000,
                      "context_compatibility": max_sequence <= context, "final_base_lock": "PASS" if not base_errors else "FAIL",
                      "license_provenance": "PASS" if not provenance_errors else "FAIL",
                      "math_programming_share_percent": round(compute_share, 2),
                      "category_balance_minimums": {key: {"actual": categories[key], "minimum": value, "pass": categories[key] >= value}
                                                     for key, value in balance_minimums.items()}},
        "semantic_checks": {"computed_or_traced_examples": dict(sorted(computational_checks.items()))},
        "external_qa_audit": {"source": "RuBQ 2.0", "raw_rows": rubq_report.get("raw_rows"),
                              "imported_single_answer_rows": external.get("rubq_2_0", 0),
                              "exact_duplicate_question_groups": rubq_report.get("questions", {}).get("exact_duplicate_groups"),
                              "same_question_answer_conflicts": len(rubq_report.get("questions", {}).get("same_question_answer_conflicts", [])),
                              "near_duplicate_question_pairs": len(rubq_report.get("questions", {}).get("near_duplicate_pairs_threshold_0_92", [])),
                              "possible_functional_fact_conflicts": len(rubq_report.get("entity_property_audit", {}).get("possible_conflicts_on_functional_properties", [])),
                              "manual_review": rubq_review_summary,
                              "paragraphs_used": False},
        "manifest": manifest, "errors": errors,
    }
    return report, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit MyLLM V2 SFT data without starting training.")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT/"data"/"sft_v2")
    parser.add_argument("--tokenizer", type=Path, default=ROOT/"tokenizer"/"tokenizer.json")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report, errors = audit(args.dataset_dir, args.tokenizer, args.context_length)
    output = args.report or args.dataset_dir/"audit_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    d, v, i = report["dataset"], report["diversity"], report["integrity"]
    print("MYLLM V2 SFT DATASET AUDIT")
    print(f"Total: {d['total']:,}; train={d['train']:,}; validation={d['validation']:,}; evaluation={d['evaluation']}")
    print(f"Categories: {d['categories']}")
    print(f"Sources: {d['source_types']}; external={d['external_source_examples']}")
    print(f"Language: RU {report['languages']['ru_percent']:.2f}% / EN {report['languages']['en_percent']:.2f}%")
    print(f"Exact assistant response excess: {v['exact_duplicate_excess']}; unique={v['exact_unique_share_percent']}%")
    print(f"Largest numeric-normalized template family: {v['largest_template_family_share_percent']}%")
    print(f"EOS failures={i['eos_failures']}; assistant-mask failures={i['assistant_mask_failures']}; template mismatches={i['train_inference_template_mismatches']}")
    print(f"Tokenizer={i['tokenizer_compatibility']}; context={i['context_compatibility']}; final base lock={i['final_base_lock']}; provenance={i['license_provenance']}")
    print(f"Report: {output}")
    if errors:
        print(f"Structural validation: FAIL ({len(errors)} issue(s))")
        for error in errors[:40]:
            print(f"- {error}")
        return 1
    print("Structural validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
