from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "sft_v2" / "sources" / "rubq"
SOURCE_FILES = ("RuBQ_2.0_dev.json", "RuBQ_2.0_test.json")
EXPECTED_SHA256 = {
    "RuBQ_2.0_dev.json": "6CD20DCFAD3404736691C5DCEE271D19F579B6C5755A1AAD4C4DE644BC9DFD05",
    "RuBQ_2.0_test.json": "63D9E32A1BDEFB2DE713A9CCF3A0F9F0A4A48F2FDDDD35B3908D6846B63DC85B",
}


def norm(text: Any) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return " ".join(re.findall(r"[\w]+", text, flags=re.UNICODE))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def answer_kind(row: dict[str, Any]) -> str:
    props = set(row.get("question_props") or [])
    answer = str(row.get("answer_text") or "")
    if props & {"wdt:P569", "wdt:P570", "wdt:P571", "wdt:P577", "wdt:P585"} or re.search(r"\b\d{3,4}\s*(?:г\.?|год|до н\.?\s*э\.?)", answer, re.I):
        return "date_or_year"
    if props & {"wdt:P36", "wdt:P17", "wdt:P19", "wdt:P20", "wdt:P276", "wdt:P131"}:
        return "place_or_country"
    if props & {"wdt:P50", "wdt:P57", "wdt:P86", "wdt:P170", "wdt:P112", "wdt:P61"}:
        return "person_or_creator"
    if re.fullmatch(r"[+−-]?\d+(?:[.,]\d+)?", answer.strip()):
        return "number"
    if props & {"wdt:P106", "wdt:P31", "wdt:P279", "wdt:P136", "wdt:P361", "wdt:P527"}:
        return "concept_or_type"
    return "other"


def source_audit() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    hashes: dict[str, dict[str, Any]] = {}
    file_counts: dict[str, int] = {}
    uid_locations: dict[str, list[str]] = defaultdict(list)
    questions: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    source_by_location: dict[tuple[str, int], dict[str, str]] = {}
    fact_values: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    formats: Counter[str] = Counter()
    props: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    suspect: Counter[str] = Counter()
    alias_consistency: Counter[str] = Counter()
    for filename in SOURCE_FILES:
        path = SOURCE_DIR / filename
        if not path.is_file():
            errors.append(f"missing source file: {filename}")
            continue
        actual_hash = sha256(path)
        hashes[filename] = {"sha256": actual_hash, "expected_sha256": EXPECTED_SHA256[filename], "matches": actual_hash == EXPECTED_SHA256[filename], "bytes": path.stat().st_size}
        if actual_hash != EXPECTED_SHA256[filename]:
            errors.append(f"source checksum mismatch: {filename}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid JSON in {filename}: {exc}")
            continue
        if not isinstance(raw, list):
            errors.append(f"root must be a JSON list: {filename}")
            continue
        file_counts[filename] = len(raw)
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                formats["non_object_record"] += 1
                suspect["non_object_record"] += 1
                continue
            required = {"uid", "question_text", "answer_text", "answers", "question_props", "question_uris"}
            missing = required - set(item)
            if missing:
                formats["missing_required_fields"] += 1
                suspect["missing_required_fields"] += 1
            if not isinstance(item.get("uid"), (int, str)):
                formats["invalid_uid_type"] += 1
            q = item.get("question_text")
            a = item.get("answer_text")
            if not isinstance(q, str) or not q.strip():
                suspect["empty_or_nonstring_question"] += 1
            if not isinstance(a, str) or not a.strip():
                suspect["empty_or_nonstring_answer"] += 1
            answers = item.get("answers")
            if not isinstance(answers, list):
                formats["answers_not_list"] += 1
                answers = []
            else:
                formats[f"answers_cardinality_{len(answers)}"] += 1
            if not isinstance(item.get("question_props"), (list, type(None))):
                formats["question_props_not_list"] += 1
            if not isinstance(item.get("question_uris"), (list, type(None))):
                formats["question_uris_not_list"] += 1
            uid = str(item.get("uid", ""))
            uid_locations[uid].append(filename)
            source_by_location[(filename, index)] = {"uid": uid, "question": str(q or ""), "answer": str(a or "")}
            qn, an = norm(q), norm(a)
            prop_list = item.get("question_props") if isinstance(item.get("question_props"), list) else []
            uri_list = item.get("question_uris") if isinstance(item.get("question_uris"), list) else []
            prop = prop_list[0] if prop_list else "no_property"
            props[prop] += 1
            kind = answer_kind(item)
            kinds[kind] += 1
            if qn:
                questions[qn].append((an, index, filename))
            if uri_list and prop != "no_property" and an:
                fact_values[f"{uri_list[0]}|{prop}"].append((an, index, filename))
            if len(answers) == 1 and isinstance(answers[0], dict):
                label_aliases = {norm(answers[0].get("label"))}
                ru_names = answers[0].get("wd_names", {}).get("ru", []) if isinstance(answers[0].get("wd_names"), dict) else []
                label_aliases.update(norm(alias) for alias in ru_names)
                if an and an in label_aliases:
                    alias_consistency["exact_alias"] += 1
                else:
                    alias_consistency["alias_not_exactly_equal"] += 1
            elif len(answers) != 1:
                alias_consistency["multi_or_zero_answer_not_compared"] += 1
            combined = f"{q or ''} {a or ''}"
            if any(unicodedata.category(ch) in {"Cf", "Cs"} or (unicodedata.category(ch) == "Cc" and ch not in "\t\r\n") for ch in combined):
                suspect["hidden_or_control_character"] += 1
            if "\ufffd" in combined or "\x00" in combined or unicodedata.normalize("NFC", combined) != combined:
                suspect["damaged_or_non_nfc_unicode"] += 1
            if len(str(q or "")) > 300:
                suspect["question_over_300_chars"] += 1
            if len(str(a or "")) > 100:
                suspect["answer_over_100_chars"] += 1
            if an and any(term in an for term in ("неизвестно", "нет данных", "не найдено", "не указано", "ошибка")):
                suspect["non_answer_or_uncertain_text"] += 1
            if an and re.search(r"\b(?:регион|провинц|координат|действующий|являющ(?:ийся|аяся))\b", an):
                suspect["article_like_or_qualified_answer"] += 1
            if an and re.search(r"[,;()]", str(a)):
                suspect["compound_or_editorial_answer"] += 1
            rows.append({"file": filename, "index": index, "uid": uid, "question": q or "", "answer": a or "", "question_norm": qn, "answer_norm": an, "props": prop_list, "uris": uri_list, "kind": kind, "answers_count": len(answers), "aliases_exact": bool(an and len(answers) == 1 and an in ({norm(answers[0].get("label"))} | {norm(x) for x in ((answers[0].get("wd_names") or {}).get("ru", [])) if isinstance(answers[0], dict)})) if len(answers) == 1 and isinstance(answers[0], dict) else None})

    uid_duplicates = {uid: locs for uid, locs in uid_locations.items() if len(locs) > 1}
    question_duplicates = []
    question_conflicts = []
    for question, group in questions.items():
        if len(group) > 1:
            record = {"question": question, "count": len(group),
                      "examples": [source_by_location.get((filename, index), {})
                                   for _, index, filename in group]}
            question_duplicates.append(record)
            unique_answers = {entry[0] for entry in group}
            if len(unique_answers) > 1:
                question_conflicts.append({"question": question, "answers": sorted(unique_answers), "occurrences": len(group)})
    fact_conflicts = []
    for fact_key, values in fact_values.items():
        distinct_answers = sorted({value[0] for value in values})
        if len(distinct_answers) > 1:
            prop = fact_key.rsplit("|", 1)[-1]
            fact_conflicts.append({"fact_key": fact_key, "property": prop, "distinct_answer_count": len(distinct_answers), "answers": distinct_answers[:10], "occurrences": len(values)})

    # Candidate generation via shared rare character trigrams; exact normalized duplicates are reported above.
    grams_by_row: list[set[str]] = []
    inverted: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        compact = re.sub(r"\s+", " ", row["question_norm"])
        grams = {compact[pos:pos+3] for pos in range(max(0, len(compact)-2))}
        grams_by_row.append(grams)
        for gram in grams:
            inverted[gram].append(idx)
    candidates: set[tuple[int, int]] = set()
    for positions in inverted.values():
        if len(positions) <= 80:
            for offset, left in enumerate(positions):
                candidates.update((left, right) for right in positions[offset+1:])
    near_pairs = []
    for left, right in candidates:
        a, b = rows[left], rows[right]
        if not a["question_norm"] or not b["question_norm"]:
            continue
        if abs(len(a["question_norm"]) - len(b["question_norm"])) > max(8, int(max(len(a["question_norm"]), len(b["question_norm"])) * 0.12)):
            continue
        ratio = difflib.SequenceMatcher(None, a["question_norm"], b["question_norm"], autojunk=False).ratio()
        if ratio >= 0.92 and a["question_norm"] != b["question_norm"]:
            near_pairs.append({"similarity": round(ratio, 4), "file_a": a["file"], "uid_a": a["uid"], "question_a": a["question"], "answer_a": a["answer"], "file_b": b["file"], "uid_b": b["uid"], "question_b": b["question"], "answer_b": b["answer"], "answers_agree": a["answer_norm"] == b["answer_norm"]})
    near_pairs.sort(key=lambda pair: (-pair["similarity"], pair["uid_a"], pair["uid_b"]))
    # Possible functional facts are a review queue, not automatic contradictions for non-functional properties.
    functional_props = {"wdt:P17", "wdt:P36", "wdt:P569", "wdt:P570", "wdt:P19", "wdt:P20"}
    functional_conflicts = [conflict for conflict in fact_conflicts if conflict["property"] in functional_props]
    result = {
        "files": hashes,
        "rows_by_file": file_counts,
        "raw_rows": len(rows),
        "schema": {"format_counts": dict(sorted(formats.items())), "issue_counts": dict(sorted(suspect.items())), "unique_uid_count": len(uid_locations), "duplicate_uids": uid_duplicates},
        "answers": {"kind_counts": dict(sorted(kinds.items())), "alias_match_counts": dict(sorted(alias_consistency.items())), "multi_or_zero_answer_rows": sum(n for key, n in formats.items() if key == "answers_cardinality_0" or key.startswith("answers_cardinality_") and key != "answers_cardinality_1")},
        "questions": {"unique_normalized": len(questions), "exact_duplicate_groups": len(question_duplicates), "exact_duplicates": question_duplicates, "same_question_answer_conflicts": question_conflicts, "near_duplicate_pairs_threshold_0_92": near_pairs},
        "entity_property_audit": {"question_uri_property_groups": len(fact_values), "multi_value_groups": len(fact_conflicts), "possible_conflicts_on_functional_properties": functional_conflicts, "note": "Multiple values are reported for manual interpretation; most Wikidata properties are not strictly single-valued."},
        "warnings": warnings,
        "errors": errors,
    }
    return result, rows


def make_stratified_sample(final_rows: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    import random

    rng = random.Random(seed)
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final_rows:
        src = row.get("source", {})
        if src.get("source_id") != "rubq_2_0":
            continue
        answer = str(src.get("source_answer") or "")
        props = src.get("source_properties") or []
        first_prop = props[0] if props else "no_property"
        kind = "date_or_year" if re.search(r"\b\d{3,4}\b", answer) else answer_kind({"answer_text": answer, "question_props": props})
        key = f"{src.get('source_file')}|{first_prop}|{kind}"
        strata[key].append({"split": row["split"], "id": row.get("id"), "group_id": row.get("group_id"), "uid": src.get("source_uid"), "source_file": src.get("source_file"), "property": first_prop, "answer_kind": kind, "question": row["messages"][0]["content"], "answer": row["messages"][-1]["content"], "source_answer": answer, "automatic_note": "Question/answer trace to direct RuBQ fields; formatter may normalize aliases/case or add audited units; paragraphs excluded."})
    for values in strata.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    source_files = sorted({key.split("|", 1)[0] for key in strata})
    base_quota, remainder = divmod(sample_size, max(1, len(source_files)))
    quotas = {filename: base_quota + int(index < remainder) for index, filename in enumerate(source_files)}
    for filename in source_files:
        keys = [key for key in sorted(strata) if key.startswith(filename + "|")]
        quota = min(quotas[filename], sum(len(strata[key]) for key in keys))
        picked = 0
        while keys and picked < quota:
            next_keys = []
            for key in keys:
                if strata[key]:
                    selected.append(strata[key].pop())
                    picked += 1
                    if strata[key]:
                        next_keys.append(key)
                    if picked >= quota:
                        break
            keys = next_keys
    if len(selected) < sample_size:
        keys = sorted(strata)
        while keys and len(selected) < sample_size:
            next_keys = []
            for key in keys:
                if strata[key]:
                    selected.append(strata[key].pop())
                    if strata[key]:
                        next_keys.append(key)
                    if len(selected) >= sample_size:
                        break
            keys = next_keys
    return selected


def main() -> int:
    # Windows PowerShell may keep cp1252 while the report contains Russian.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Full automated format/provenance audit plus a stratified RuBQ fact-review sample.")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "sft_v2")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    report, _ = source_audit()
    final_rows = []
    for split in ("train", "validation"):
        path = args.dataset_dir / f"{split}.jsonl"
        if not path.is_file():
            report["errors"].append(f"missing candidate split: {path.name}")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                row["split"] = split
                final_rows.append(row)
            except Exception as exc:
                report["errors"].append(f"invalid candidate jsonl row: {exc}")
    sample = make_stratified_sample(final_rows, args.sample_size, args.seed)
    sample_path = args.dataset_dir / "rubq_manual_review_sample.json"
    sample_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["final_candidate_imported_examples"] = sum(row.get("source", {}).get("source_id") == "rubq_2_0" for row in final_rows)
    review_log_path = args.dataset_dir / "rubq_manual_review_log.json"
    review_log = json.loads(review_log_path.read_text(encoding="utf-8")) if review_log_path.is_file() else {}
    reviewed = review_log.get("reviews", []) if isinstance(review_log, dict) else []
    review_key = lambda item: (str(item.get("source_file", "")), str(item.get("uid", "")))
    reviewed_by_key = {review_key(item): item for item in reviewed
                       if isinstance(item, dict) and item.get("uid") is not None}
    sampled_keys = {review_key(item) for item in sample}
    matched_reviews = [reviewed_by_key[key] for key in sampled_keys if key in reviewed_by_key]
    complete = bool(sample) and len(matched_reviews) == len(sample)
    all_pass = complete and all(item.get("decision") == "pass" for item in matched_reviews)
    report["manual_review"] = {"sample_file": sample_path.name, "sample_size": len(sample), "requested_size": args.sample_size, "seed": args.seed,
                               "stratification": "round-robin across source file × question property × answer kind",
                               "review_status": "PASS" if all_pass else "partial" if matched_reviews else "pending",
                               "reviewed_count": len(matched_reviews), "passed_count": sum(item.get("decision") == "pass" for item in matched_reviews),
                               "rejected_count": sum(item.get("decision") == "reject" for item in matched_reviews),
                               "sampled_uids_missing_from_log": sorted(f"{source_file}#{uid}" for source_file, uid in sampled_keys - set(reviewed_by_key)),
                               "note": "Stratified human review of the sampled accepted QA rows; not row-by-row review of the full source."}
    out = args.dataset_dir / "rubq_automated_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"RuBQ raw rows: {report['raw_rows']:,}; final imports: {report['final_candidate_imported_examples']:,}")
    print(f"Schema: {report['schema']['format_counts']}; suspect: {report['schema']['issue_counts']}")
    print(f"Exact duplicate question groups: {report['questions']['exact_duplicate_groups']}; same-question conflicts: {len(report['questions']['same_question_answer_conflicts'])}; near duplicates: {len(report['questions']['near_duplicate_pairs_threshold_0_92'])}")
    print(f"Possible functional-fact conflicts: {len(report['entity_property_audit']['possible_conflicts_on_functional_properties'])}; stratified review: {report['manual_review']['review_status']} {report['manual_review']['reviewed_count']}/{len(sample)}")
    print(f"Automated audit: {'PASS' if not report['errors'] else 'FAIL'}; report: {out}; sample: {sample_path}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
