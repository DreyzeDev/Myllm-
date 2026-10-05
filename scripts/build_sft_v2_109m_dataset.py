from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sft_data import is_obvious_contact_or_secret, normalize_for_dedup, validate_turn_order

SEED = 20261005
CURRENT_CAPITALS = {
    "москва", "париж", "рим", "мадрид", "берлин", "лиссабон", "осло", "стокгольм", "хельсинки", "варшава",
    "афины", "дублин", "токио", "пекин", "нью-дели", "сеул", "улан-батор", "бангкок", "ханой", "катманду",
    "каир", "найроби", "абуджа", "рабат", "аддис-абеба", "вашингтон", "оттава", "мехико", "гавана", "бразилиа",
    "буэнос-айрес", "сантьяго", "лима", "канберра", "веллингтон", "анкара", "тбилиси", "астана", "ташкент", "баку",
}
MANUAL_RUBQ_EXCLUSIONS = {
    ("RuBQ_2.0_dev.json", 6434): "Award fact could not be resolved confidently from the question wording.",
    ("RuBQ_2.0_dev.json", 6866): "Kuban Oblast formation year conflicts with the historical administrative timeline.",
    ("RuBQ_2.0_dev.json", 6232): "Person identity is ambiguous and the education answer is uncertain.",
    ("RuBQ_2.0_dev.json", 6369): "Diocesan jurisdiction is potentially stale and not sufficiently clear.",
    ("RuBQ_2.0_dev.json", 6387): "The title answer is inconsistent with Athos in The Three Musketeers.",
    ("RuBQ_2.0_dev.json", 90): "The question incorrectly places the Statue of Zeus at Olympia in Athens; reject the false-premise item.",
    ("RuBQ_2.0_dev.json", 3139): "Annual festival location statement has stale or oversimplified time framing.",
    ("RuBQ_2.0_dev.json", 6397): "The named figure is not a brother of Selene in Greek mythology.",
    ("RuBQ_2.0_dev.json", 2116): "The requested full name is incomplete in the source answer.",
    ("RuBQ_2.0_dev.json", 4012): "Sahara is the largest hot desert, not the largest desert overall.",
    ("RuBQ_2.0_dev.json", 280): "The question's claim of a legally official language in Australia is misleading.",
    ("RuBQ_2.0_dev.json", 3036): "The requested full character name is incomplete in the source answer.",
    ("RuBQ_2.0_dev.json", 6462): "The question conflates anthem lyrics with musical composition.",
    ("RuBQ_2.0_dev.json", 7166): "The source area is off by an order of magnitude for the Caspian Sea and fails factual review.",
    ("RuBQ_2.0_test.json", 120): "The wording treats a deceased party leader as currently leading the party.",
    ("RuBQ_2.0_test.json", 296): "The present-tense leadership question is stale after the person's death.",
    ("RuBQ_2.0_test.json", 6096): "The legal form answer is ambiguous for the international Mars company.",
    ("RuBQ_2.0_test.json", 199): "The answer gives a physical quantity rather than the SI unit named after Tesla.",
    ("RuBQ_2.0_test.json", 427): "The stadium's name has changed since the source record was collected.",
    ("RuBQ_2.0_test.json", 6496): "The source reverses the Jurassic and Triassic periods.",
    ("RuBQ_2.0_test.json", 233): "The present-tense theatre affiliation may be stale for the sampled actor.",
    ("RuBQ_2.0_test.json", 6479): "Ivanhoe is a historical novel; the source labels its genre as Romanticism.",
    ("RuBQ_2.0_test.json", 3133): "The source's first-host answer is inconsistent with the programme's presenter history.",
    ("RuBQ_2.0_test.json", 6559): "The question says Volvo generically; Volvo Cars and Volvo Group are distinct, and the source's Geely Automobile answer conflates ownership entities.",
}
MANUAL_RUBQ_ANSWER_CORRECTIONS = {
    # RuBQ's answer_text is a surname, while its linked entity label and
    # Russian aliases identify the character as Neo / Thomas Anderson.
    ("RuBQ_2.0_dev.json", 3125): "Нео (Томас Андерсон)",
    # Expand the product entity into an answer that actually says it is the
    # developer, matching the source organization's own description.
    ("RuBQ_2.0_dev.json", 6250): "Справочную правовую систему «КонсультантПлюс» разрабатывает компания «КонсультантПлюс»",
    ("RuBQ_2.0_dev.json", 571): "В египетском городе Александрия.",
}
EVAL = [
    ("Привет", "Привет! Чем могу помочь?"),
    ("Как дела?", "Всё хорошо, спасибо. Чем могу помочь?"),
    ("Кто ты?", "Я MyLLM — языковая модель, которая отвечает на вопросы и помогает с простыми задачами."),
    ("Что ты умеешь?", "Я могу объяснять простые темы, помогать с текстом, переводом, математикой и базовым Python."),
    ("Столица России?", "Столица России — Москва."),
    ("Столица Франции?", "Столица Франции — Париж."),
    ("7 * 8?", "7 × 8 = 56."),
    ("2 + 2?", "2 + 2 = 4."),
    ("Что такое Солнце?", "Солнце — звезда в центре Солнечной системы."),
    ("Земля вращается вокруг чего?", "Земля обращается вокруг Солнца."),
    ("Переведи Hello на русский.", "Hello по-русски — «Привет»."),
    ("Назови три планеты.", "Например: Меркурий, Венера и Земля."),
    ("Напиши Python функцию сложения двух чисел.", "def add(a, b):\n    return a + b"),
]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _lang(messages: list[dict[str, str]], declared: str | None = None) -> str:
    if declared in {"ru", "en"}:
        return declared
    text = " ".join(m["content"] for m in messages if m["role"] == "user")
    ru = sum("а" <= c.lower() <= "я" or c.lower() == "ё" for c in text)
    en = sum("a" <= c.lower() <= "z" for c in text)
    return "en" if en > ru else "ru"


class DatasetBuilder:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.seen_conversations: set[str] = set()
        self.seen_responses: set[str] = set()
        self.user_prompt_counts: Counter[str] = Counter()
        self.rejections: Counter[str] = Counter()
        self.legacy_kept = Counter()
        self.legacy_rejected = Counter()

    def add(
        self,
        category: str,
        language: str,
        group_id: str,
        messages: list[dict[str, str]],
        source: dict[str, Any],
    ) -> bool:
        error = validate_turn_order(messages)
        if error:
            self.rejections["invalid_turn_order"] += 1
            return False
        user_keys = [normalize_for_dedup(m["content"]) for m in messages if m["role"] == "user"]
        if any(not key or self.user_prompt_counts[key] >= 8 for key in user_keys):
            self.rejections["repeated_user_prompt_cap"] += 1
            return False
        joined = "\n".join(m["content"] for m in messages)
        if is_obvious_contact_or_secret(joined):
            self.rejections["pii_or_secret"] += 1
            return False
        if any(unicodedata.normalize("NFC", m["content"]) != m["content"] for m in messages):
            messages = [{**m, "content": unicodedata.normalize("NFC", m["content"])} for m in messages]
        style_repairs = 0
        for message in messages:
            if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                message["content"], changed = self.clean_assistant_style(message["content"])
                style_repairs += int(changed)
        if style_repairs:
            self.rejections["artificial_assistant_prefix_removed"] += style_repairs
        assistants = [m["content"] for m in messages if m["role"] == "assistant"]
        normalized = [normalize_for_dedup(text) for text in assistants]
        if any(not text for text in normalized):
            self.rejections["empty_assistant"] += 1
            return False
        if len(set(normalized)) != len(normalized):
            self.rejections["repeated_turn_in_conversation"] += 1
            return False
        conv_key = _sha("\n".join(f"{m['role']}:{normalize_for_dedup(m['content'])}" for m in messages))
        if conv_key in self.seen_conversations:
            self.rejections["duplicate_conversation"] += 1
            return False
        if any(text in self.seen_responses for text in normalized):
            self.rejections["duplicate_assistant_response"] += 1
            return False
        self.seen_conversations.add(conv_key)
        self.seen_responses.update(normalized)
        self.user_prompt_counts.update(user_keys)
        self.rows.append({
            "id": _sha(category + "\n" + group_id + "\n" + conv_key)[:20],
            "category": category,
            "language": language,
            "group_id": group_id,
            "messages": messages,
            "source": source,
        })
        return True

    @staticmethod
    def clean_assistant_style(text: str) -> tuple[str, bool]:
        """Remove generator narration that makes a direct answer sound templated."""
        original = text
        text = unicodedata.normalize("NFC", text).strip()
        patterns = (
            r"^(?:можно ответить так|можно сказать так|можно написать так|можно объяснить так|ответ получается таким(?:\s+образом)?|ответ получается следующим образом|результат получается таким(?:\s+образом)?)\s*[:,—-]?\s*",
            r"^(?:если тебе интересно|если вам интересно|короткий ответ|если кратко|в двух словах|запомни коротко|основная идея такая|для простого объяснения|попробую объяснить проще|попросту говоря|проще говоря|иными словами|представь так|это можно представить так)\s*[:,—-]?\s*",
        )
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        if text.startswith("iOS"):
            return text, text != original
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text, text != original

    def import_reaudited_synthetic_v1(self) -> None:
        """Reuse only synthetic V1 candidates; wiki-derived answers are categorically excluded."""
        old_dir = ROOT / "data" / "sft"
        for name in ("train.jsonl", "validation.jsonl"):
            path = old_dir / name
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    row = json.loads(line)
                    source = row.get("source", {})
                    kind = source.get("type")
                    if kind not in {"synthetic_curated", "synthetic_computed"}:
                        self.legacy_rejected["non_synthetic_including_wikimedia"] += 1
                        continue
                    if source.get("generator") == "curated_ru_smalltalk_v4":
                        self.legacy_rejected["low_diversity_smalltalk_v4"] += 1
                        continue
                    if source.get("generator") == "short_context_planning_v5":
                        self.legacy_rejected["awkward_english_multi_turn_planning_v5"] += 1
                        continue
                    if not isinstance(row.get("messages"), list):
                        self.legacy_rejected["bad_schema"] += 1
                        continue
                    # Repeat the key checks on every imported row. Never trust old PASS metadata.
                    style_repairs = 0
                    for message in row["messages"]:
                        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                            message["content"], changed = self.clean_assistant_style(message["content"])
                            style_repairs += int(changed)
                    responses = [m.get("content", "") for m in row["messages"] if m.get("role") == "assistant"]
                    if not responses or any(len(text) > 1600 for text in responses):
                        self.legacy_rejected["empty_or_long_answer"] += 1
                        continue
                    if str(row.get("category")) == "math" and all(re.fullmatch(r"\s*[-+]?\d+(?:[.,]\d+)?\s*", text) for text in responses):
                        self.legacy_rejected["bare_numeric_not_assistant_style"] += 1
                        continue
                    if str(row.get("category")) == "conversation":
                        self.legacy_rejected["conversation_examples_excluded_after_manual_spot_check"] += 1
                        continue
                    if source.get("generator") == "python_conditionals_v2" and len(responses) == 1:
                        match = re.search(r"result\s*=\s*(-?\d+)\s*>\s*(-?\d+)", responses[0])
                        if match:
                            left, right = int(match.group(1)), int(match.group(2))
                            result = left > right
                            notes = [
                                "Для этих значений сравнение даёт {result}.",
                                "Проверка показывает: {left} больше {right} — {result}.",
                                "В переменной result окажется значение {result}.",
                                "При такой паре условие истинно: {result}.",
                                "Результат выражения сравнения — {result}.",
                                "Число {left} больше числа {right}: {result}.",
                                "Если выполнить эту строку, Python получит {result}.",
                                "Для чисел {left} и {right} оператор > вернёт {result}.",
                                "Здесь результат проверки равен {result}.",
                                "Сравнение {left} > {right} даёт {result}.",
                                "Переменная result будет содержать {result}.",
                                "Ответ зависит от сравнения: {left} > {right} — {result}.",
                                "Для входных чисел результатом станет {result}.",
                                "Python сравнит {left} с {right} и вернёт {result}.",
                                "В этом случае условие «{left} > {right}» имеет значение {result}.",
                                "Код проверяет, больше ли {left} числа {right}; ответ — {result}.",
                                "Выражение {left} > {right} оценивается как {result}.",
                                "При запуске result получит {result}, так как числа сравниваются напрямую.",
                                "Проверка пары {left}, {right} возвращает {result}.",
                                "Коротко: результат сравнения равен {result}.",
                            ]
                            note = notes[int(hashlib.sha256(f"{left}:{right}:{line_no}".encode()).hexdigest(), 16) % len(notes)]
                            for message in row["messages"]:
                                if message.get("role") == "assistant":
                                    message["content"] = message["content"] + "\n" + note.format(left=left, right=right, result=result)
                            source = {**source, "assistant_style_repair": "added computed comparison explanation", "computed_result": result}
                    source = {
                        "type": "reaudited_synthetic",
                        "origin_dataset": "data/sft (V1; no Wikimedia rows imported)",
                        "origin_file": name,
                        "origin_line": line_no,
                        "origin_source": row.get("source", {}),
                        "audit": "schema, role order, PII/secrets, context, exact response dedup, Unicode",
                    }
                    if style_repairs:
                        source["assistant_style_repairs"] = style_repairs
                    accepted = self.add(
                        str(row.get("category", "")),
                        _lang(row["messages"], row.get("language")),
                        "reaudited-v1:" + str(row.get("group_id") or row.get("id") or line_no),
                        row["messages"],
                        source,
                    )
                    if accepted:
                        self.legacy_kept[str(row.get("category", "unknown"))] += 1
                    else:
                        self.legacy_rejected["quality_or_duplicate_gate"] += 1

    @staticmethod
    def format_rubq_response(question: str, answer: str, properties: list[str]) -> str:
        """Turn a short answer field into a direct assistant reply without canned narration."""
        answer = answer.strip().rstrip(" .!?«»\"'")
        qnorm = unicodedata.normalize("NFKC", question).casefold()
        if "wdt:P2047" in properties and "полет человека в космос" in qnorm and answer == "108":
            return "Первый полёт человека в космос длился 108 минут."
        if "wdt:P2101" in properties and answer == "0":
            return "Вода замерзает примерно при 0 °C при нормальном атмосферном давлении."
        if "wdt:P2102" in properties and answer == "187.4":
            return "Пропиленгликоль кипит примерно при 187,4 °C при нормальном атмосферном давлении."
        if "wdt:P38" in properties and "израиле" in qnorm and "шекель" in answer.casefold():
            return "Официальная валюта Израиля — новый израильский шекель."
        if "wdt:P403" in properties and "псекупс" in qnorm:
            return "Псекупс является притоком реки Кубань."
        if "трискайдекафобией" in qnorm:
            return "Люди с трискайдекафобией боятся числа 13."
        if "wdt:P50" in properties and "похождений бравого солдата швейка" in qnorm:
            return "Роман «Похождения бравого солдата Швейка» написал Ярослав Гашек."
        if "wdt:P138" in properties and "моне" in qnorm and "импрессионизм" in qnorm:
            return "Картина Клода Моне «Впечатление. Восход солнца» дала имя импрессионизму."
        if "wdt:P144" in properties and "травиата" in qnorm:
            return "Опера «Травиата» основана на романе Александра Дюма-сына «Дама с камелиями»."
        if "wdt:P1454" in properties and "тандер" in qnorm:
            return "Компания «Тандер» имеет форму акционерного общества (АО)."
        if "псевдоним" in qnorm and "зигги" in qnorm and "дэвид боуи" in answer.casefold():
            return "Под псевдонимом «Зигги Стардаст» выступал Дэвид Боуи."
        if "wdt:P86" in properties and "гимна футбольной лиги чемпионов" in qnorm and "гендел" in answer.casefold():
            return "Музыкальная тема гимна основана на коронационном гимне Георга Фридриха Генделя «Садок-священник»."
        if "wdt:P1376" in properties and "нижний новгород" in qnorm:
            return "Нижний Новгород — административный центр Приволжского федерального округа."
        if "wdt:P1477" in properties and "первого человека космонавта" in qnorm and "гагарин" in answer.casefold():
            return "Первым человеком, побывавшим в космосе, был Юрий Гагарин."
        if answer.casefold() == "ios":
            return "iOS."
        capital_prop = "wdt:P36" in properties
        if capital_prop:
            match = re.search(r"(?:столица(?:\s+у)?|столицей)\s+(?:страны\s+)?(.+?)(?:[?.!]|$)", question, re.IGNORECASE)
            if match:
                country = re.sub(r"^(?:у|страны)\s+", "", match.group(1).strip(), flags=re.IGNORECASE)
                if country:
                    return f"Столица {country} — {answer}."
        title = re.search(r"[«\"']([^»\"']{2,100})[»\"']", question)
        if title:
            work = title.group(1).strip()
            label_prefix = {
                "wdt:P50": "Автор произведения",
                "wdt:P57": "Режиссёр фильма",
                "wdt:P86": "Композитор произведения",
                "wdt:P170": "Создатель произведения",
            }
            for prop, label in label_prefix.items():
                if prop in properties:
                    return f"{label} «{work}» — {answer}."
        if answer:
            answer = answer[:1].upper() + answer[1:]
            return answer + ("." if answer[-1] not in ".!?" else "")
        return ""

    def import_rubq_qa(self) -> None:
        """Import direct RuBQ question/answer fields only; paragraph/context fields are never read."""
        source_dir = ROOT / "data" / "sft_v2" / "sources" / "rubq"
        attribution = "vladislavneon/RuBQ, RuBQ 2.0, CC BY-SA 4.0"
        seen_questions: dict[str, str] = {}
        for filename in ("RuBQ_2.0_dev.json", "RuBQ_2.0_test.json"):
            path = source_dir / filename
            if not path.is_file():
                self.rejections["rubq_source_missing"] += 1
                continue
            rows = json.loads(path.read_text(encoding="utf-8"))
            for item in rows:
                q = str(item.get("question_text") or "").strip()
                properties = item.get("question_props", []) if isinstance(item.get("question_props"), list) else []
                if (filename, item.get("uid")) in MANUAL_RUBQ_EXCLUSIONS:
                    self.rejections["rubq_manual_fact_review"] += 1
                    continue
                answer = str(item.get("answer_text") or "").strip().strip("'\"«» ")
                raw_answer = answer
                answers = item.get("answers")
                # Multi-answer and no-answer rows are excluded; no Wikipedia paragraph is loaded.
                if not isinstance(answers, list) or len(answers) != 1:
                    self.rejections["rubq_multi_or_missing_answer"] += 1
                    continue
                property_ids = {prop.rsplit(":", 1)[-1] for prop in properties}
                if property_ids & {"P54", "P118", "P2437", "P1113", "P509", "P1196"}:
                    self.rejections["rubq_time_sensitive_or_cause_of_death"] += 1
                    continue
                if "wdt:P2067" in properties and re.search(r"\b(сколько весит|весит|масса)\b", q, flags=re.IGNORECASE):
                    self.rejections["rubq_ambiguous_person_measurement"] += 1
                    continue
                if re.match(r"\s*(?:какие\b|перечислите\b|назовите\b)", q, flags=re.IGNORECASE):
                    self.rejections["rubq_plural_question_single_answer"] += 1
                    continue
                if "wdt:P36" in properties and answer.casefold().strip(" .") not in CURRENT_CAPITALS:
                    self.rejections["rubq_unverified_or_outdated_capital"] += 1
                    continue
                if not q or not answer or len(answer) > 100 or len(q) > 300:
                    self.rejections["rubq_empty_or_long"] += 1
                    continue
                if any(unicodedata.category(char) in {"Cf", "Cs"} or
                       (unicodedata.category(char) == "Cc" and char not in "\t\n\r")
                       for char in q + answer):
                    self.rejections["rubq_hidden_or_control_character"] += 1
                    continue
                if answer.casefold() == "дениел редклиф" or any(mark in answer for mark in (",", ";", "(", ")")) or re.search(
                        r"\b(регион|провинц|действующий|координат)\b", answer, flags=re.IGNORECASE):
                    self.rejections["rubq_article_like_or_non_atomic_answer"] += 1
                    continue
                if not re.search(r"[А-Яа-яЁё]", q) or is_obvious_contact_or_secret(q + " " + answer):
                    self.rejections["rubq_language_or_pii"] += 1
                    continue
                if any(token in answer.casefold() for token in ("неизвестно", "нет данных", "не найдено")):
                    self.rejections["rubq_no_answer_text"] += 1
                    continue
                q = unicodedata.normalize("NFC", q[:1].upper() + q[1:])
                if q[-1] not in ".?!":
                    q += "?"
                question_key = normalize_for_dedup(q)
                answer_key = normalize_for_dedup(answer)
                if question_key in seen_questions:
                    reason = "rubq_exact_question_answer_conflict" if seen_questions[question_key] != answer_key else "rubq_exact_duplicate_question"
                    self.rejections[reason] += 1
                    continue
                seen_questions[question_key] = answer_key
                answer_source_field = "answer_text"
                label_data = answers[0] if isinstance(answers[0], dict) else {}
                aliases = [str(label_data.get("label") or "")]
                names = label_data.get("wd_names", {}).get("ru", []) if isinstance(label_data.get("wd_names"), dict) else []
                aliases.extend(str(name) for name in names if isinstance(name, str))
                normalized_answer = normalize_for_dedup(answer)
                exact_alias = next((alias for alias in aliases if normalize_for_dedup(alias) == normalized_answer and normalize_for_dedup(alias)), None)
                if exact_alias and exact_alias != answer:
                    answer = exact_alias
                    answer_source_field = "answers[0].label/wd_names.ru (canonical capitalization)"
                    self.rejections["rubq_answer_capitalization_canonicalized_from_alias"] += 1
                elif aliases and not any(normalized_answer in normalize_for_dedup(alias) or normalize_for_dedup(alias) in normalized_answer for alias in aliases if normalize_for_dedup(alias)):
                    from difflib import SequenceMatcher
                    ranked = sorted(
                        ((SequenceMatcher(None, normalized_answer, normalize_for_dedup(alias)).ratio(), alias)
                         for alias in aliases if normalize_for_dedup(alias)), reverse=True
                    )
                    if ranked and ranked[0][0] >= 0.90:
                        answer = ranked[0][1]
                        answer_source_field = "answers[0].wd_names.ru (normalized typo against answer_text)"
                        self.rejections["rubq_answer_text_canonicalized_from_alias"] += 1
                correction = MANUAL_RUBQ_ANSWER_CORRECTIONS.get((filename, item.get("uid")))
                if correction:
                    answer = correction
                    answer_source_field = "manual_fact_correction (RuBQ entity label and Russian aliases)"
                    self.rejections["rubq_manual_fact_corrections"] += 1
                a = self.format_rubq_response(q, answer, item.get("question_props", []))
                uris = item.get("question_uris", [])
                props = item.get("question_props", [])
                # Hold together all paraphrases about one entity/property during the split.
                semantic_key = "|".join([*(str(uri) for uri in uris), *(str(prop) for prop in props)])
                semantic_key = semantic_key or normalize_for_dedup(q)
                self.add("general_knowledge", "ru", f"rubq-fact:{_sha(semantic_key)}", [
                    {"role": "user", "content": q}, {"role": "assistant", "content": a},
                ], {"type": "external_dataset", "source_id": "rubq_2_0", "source_file": filename,
                    "source_uid": item.get("uid"), "license": "CC BY-SA 4.0", "attribution": attribution,
                    "answer_source_field": answer_source_field, "source_answer": raw_answer,
                    "assistant_fact_answer": answer,
                    "source_properties": item.get("question_props", []), "source_uris": item.get("question_uris", []),
                    "paragraphs_used": False})

    def generate_math(self, target: int = 15000) -> None:
        rng = random.Random(SEED + 1)
        question_forms = [
            "Сколько будет {a} {op} {b}?", "Вычисли: {a} {op} {b}.",
            "Помоги посчитать {a} {op} {b}.", "Найди значение выражения {a} {op} {b}.",
            "Какой результат у примера {a} {op} {b}?", "Реши короткий пример: {a} {op} {b}.",
            "Посчитай, пожалуйста: {a} {op} {b}.", "Чему равно выражение {a} {op} {b}?",
        ]
        answer_forms = [
            "{a} {op} {b} = {r}, поэтому ответ — {r}.",
            "Получается {r}: {a} {word} {b}.",
            "Ответ: {r}. Проверка: {a} {op} {b} = {r}.",
            "Если выполнить {a} {word} {b}, выйдет {r}.",
            "Значение выражения — {r} ({a} {op} {b}).",
            "Результат вычисления: {r}; пример записывается как {a} {op} {b} = {r}.",
            "Верный результат — {r}: {a} {op} {b}.",
            "Считаем {a} {word} {b} и получаем {r}.",
            "Проверка простая: {a} {op} {b} даёт {r}.",
            "Здесь ответ {r}, потому что {a} {word} {b} = {r}.",
            "Итог: {r}. Само выражение равно {a} {op} {b}.",
            "Вычисление {a} {word} {b} заканчивается результатом {r}.",
            "Для примера {a} {op} {b} получаем число {r}.",
            "Результат примера — {r}; запись: {a} {op} {b}.",
            "Ответ {r}: это значение выражения {a} {op} {b}.",
            "Посчитай действие {a} {word} {b} — получится {r}.",
            "После вычисления {a} {op} {b} остаётся результат {r}.",
            "Значит, {a} {word} {b} равно {r}.",
            "В этом примере значение равно {r} ({a} {op} {b}).",
            "Коротко: {a} {op} {b} = {r}.",
            "Правильное значение — {r}; {a} {word} {b}.",
            "Я получаю {r}, если вычисляю {a} {word} {b}.",
            "Числовой результат: {r}. Выражение: {a} {op} {b}.",
            "Ответ {r} можно проверить равенством {a} {op} {b} = {r}.",
            "При вычислении {a} {op} {b} получается {r}.",
            "Равенство {a} {op} {b} верно, если справа поставить {r}.",
            "Здесь результат действия {a} {op} {b} — {r}.",
            "Вычислив {a} {word} {b}, получаем {r}.",
            "У выражения {a} {op} {b} значение {r}.",
            "Пример {a} {op} {b} решается так: ответ {r}.",
            "После вычисления выходит {r}: {a} {op} {b}.",
            "Для выражения {a} {op} {b} итоговое число — {r}.",
            "Полученный ответ {r} соответствует примеру {a} {op} {b}.",
            "Решение даёт {r}; проверка: {a} {op} {b} = {r}.",
            "Если посчитать {a} {word} {b}, результат будет {r}.",
            "В этом вычислении выходит {r} ({a} {op} {b}).",
            "Ответ к примеру {a} {op} {b}: {r}.",
            "Число {r} получается при действии {a} {word} {b}.",
            "Значит, для {a} {op} {b} верен результат {r}.",
            "Результат здесь равен {r}; исходное выражение — {a} {op} {b}.",
            "Посчитал: {a} {op} {b} = {r}.",
            "При таком действии {a} {word} {b} выходит {r}.",
            "Верный ответ: {r}. Его можно увидеть из равенства {a} {op} {b} = {r}.",
            "После подсчёта получаем {r}.",
            "Действие {a} {op} {b} даёт число {r}.",
            "Проверка показывает, что {a} {op} {b} равно {r}.",
            "Равенство записывается так: {a} {op} {b} = {r}.",
        ]
        ops = [("+", "плюс", lambda a, b: a + b), ("−", "минус", lambda a, b: a - b),
               ("×", "умножить на", lambda a, b: a * b), ("÷", "разделить на", lambda a, b: a // b)]
        candidates: list[tuple[int, int, str, str, int]] = []
        for a in range(1, 151):
            for b in range(1, 91):
                for op, word, fn in ops:
                    if op == "÷" and a % b:
                        continue
                    candidates.append((a, b, op, word, fn(a, b)))
        rng.shuffle(candidates)
        for index, (a, b, op, word, result) in enumerate(candidates[:target]):
            q = question_forms[index % len(question_forms)].format(a=a, op=op, b=b)
            answer = answer_forms[(index // len(question_forms) + index) % len(answer_forms)].format(
                a=a, op=op, b=b, word=word, r=result
            )
            group = f"math:{a}:{op}:{b}"
            self.add("math", "ru", group, [
                {"role": "user", "content": q}, {"role": "assistant", "content": answer}
            ], {"type": "synthetic_computed", "generator": "integer_arithmetic_v2_109m", "expression": f"{a}{op}{b}", "expected": result})
        english_questions = ["What is {a} {op} {b}?", "Calculate {a} {op} {b}.",
                             "Please work out {a} {op} {b}.", "Find the value of {a} {op} {b}."]
        english_answers = ["{a} {op} {b} = {r}, so the answer is {r}.",
                           "The result is {r}: {a} {word} {b}.",
                           "Answer: {r}. You can check it with {a} {op} {b} = {r}.",
                           "Evaluating {a} {word} {b} gives {r}."]
        for index, (a, b, op, word, result) in enumerate(candidates[target:target+600]):
            q = english_questions[index % len(english_questions)].format(a=a, op=op, b=b)
            answer = english_answers[(index // len(english_questions) + index) % len(english_answers)].format(
                a=a, op=op, b=b, word={"+":"plus", "−":"minus", "×":"times", "÷":"divided by"}[op], r=result)
            self.add("math", "en", f"math-en:{a}:{op}:{b}", [
                {"role": "user", "content": q}, {"role": "assistant", "content": answer}
            ], {"type": "synthetic_computed", "generator": "integer_arithmetic_en_v1", "expression": f"{a}{op}{b}", "expected": result})

    def generate_geography_and_science(self) -> None:
        # Compact, manually curated facts. No article paragraphs or scraped prose are used.
        countries = [
            ("Россия", "Москва", "Европа и Азия"), ("Франция", "Париж", "Европа"),
            ("Италия", "Рим", "Европа"), ("Испания", "Мадрид", "Европа"),
            ("Германия", "Берлин", "Европа"), ("Португалия", "Лиссабон", "Европа"),
            ("Норвегия", "Осло", "Европа"), ("Швеция", "Стокгольм", "Европа"),
            ("Финляндия", "Хельсинки", "Европа"), ("Польша", "Варшава", "Европа"),
            ("Греция", "Афины", "Европа"), ("Ирландия", "Дублин", "Европа"),
            ("Япония", "Токио", "Азия"), ("Китай", "Пекин", "Азия"),
            ("Индия", "Нью-Дели", "Азия"), ("Южная Корея", "Сеул", "Азия"),
            ("Монголия", "Улан-Батор", "Азия"), ("Таиланд", "Бангкок", "Азия"),
            ("Вьетнам", "Ханой", "Азия"), ("Непал", "Катманду", "Азия"),
            ("Египет", "Каир", "Африка"), ("Кения", "Найроби", "Африка"),
            ("Нигерия", "Абуджа", "Африка"), ("Марокко", "Рабат", "Африка"),
            ("Эфиопия", "Аддис-Абеба", "Африка"), ("США", "Вашингтон", "Северная Америка"),
            ("Канада", "Оттава", "Северная Америка"), ("Мексика", "Мехико", "Северная Америка"),
            ("Куба", "Гавана", "Карибский регион"), ("Бразилия", "Бразилиа", "Южная Америка"),
            ("Аргентина", "Буэнос-Айрес", "Южная Америка"), ("Чили", "Сантьяго", "Южная Америка"),
            ("Перу", "Лима", "Южная Америка"), ("Австралия", "Канберра", "Австралия и Океания"),
            ("Новая Зеландия", "Веллингтон", "Австралия и Океания"), ("Турция", "Анкара", "Европа и Азия"),
            ("Грузия", "Тбилиси", "Европа и Азия"), ("Казахстан", "Астана", "Европа и Азия"),
            ("Узбекистан", "Ташкент", "Азия"), ("Азербайджан", "Баку", "Европа и Азия"),
            ("Исландия", "Рейкьявик", "Европа"), ("Дания", "Копенгаген", "Европа"),
            ("Нидерланды", "Амстердам", "Европа"), ("Бельгия", "Брюссель", "Европа"),
            ("Австрия", "Вена", "Европа"), ("Швейцария", "Берн", "Европа"),
            ("Чехия", "Прага", "Европа"), ("Словакия", "Братислава", "Европа"),
            ("Венгрия", "Будапешт", "Европа"), ("Румыния", "Бухарест", "Европа"),
            ("Болгария", "София", "Европа"), ("Хорватия", "Загреб", "Европа"),
            ("Сербия", "Белград", "Европа"), ("Украина", "Киев", "Европа"),
            ("Беларусь", "Минск", "Европа"), ("Люксембург", "Люксембург", "Европа"),
            ("Мальта", "Валлетта", "Европа"), ("Кипр", "Никосия", "Европа и Азия"),
            ("Литва", "Вильнюс", "Европа"), ("Латвия", "Рига", "Европа"),
            ("Эстония", "Таллин", "Европа"), ("Саудовская Аравия", "Эр-Рияд", "Азия"),
            ("Объединённые Арабские Эмираты", "Абу-Даби", "Азия"), ("Катар", "Доха", "Азия"),
            ("Иран", "Тегеран", "Азия"), ("Ирак", "Багдад", "Азия"),
            ("Пакистан", "Исламабад", "Азия"), ("Бангладеш", "Дакка", "Азия"),
            ("Шри-Ланка", "Шри-Джаяварденепура-Котте", "Азия"), ("Сингапур", "Сингапур", "Азия"),
            ("Малайзия", "Куала-Лумпур", "Азия"), ("Филиппины", "Манила", "Азия"),
            ("Камбоджа", "Пномпень", "Азия"), ("Лаос", "Вьентьян", "Азия"),
            ("Мьянма", "Нейпьидо", "Азия"), ("Бутан", "Тхимпху", "Азия"),
            ("Афганистан", "Кабул", "Азия"), ("Кыргызстан", "Бишкек", "Азия"),
            ("Таджикистан", "Душанбе", "Азия"), ("Туркменистан", "Ашхабад", "Азия"),
            ("Гана", "Аккра", "Африка"), ("Сенегал", "Дакар", "Африка"),
            ("Танзания", "Додома", "Африка"), ("Уганда", "Кампала", "Африка"),
            ("Замбия", "Лусака", "Африка"), ("Ботсвана", "Габороне", "Африка"),
            ("Намибия", "Виндхук", "Африка"), ("Алжир", "Алжир", "Африка"),
            ("Тунис", "Тунис", "Африка"), ("Ливия", "Триполи", "Африка"),
            ("Мадагаскар", "Антананариву", "Африка"), ("Камерун", "Яунде", "Африка"),
            ("Кот-д'Ивуар", "Ямусукро", "Африка"), ("Южно-Африканская Республика", "Претория", "Африка"),
            ("Панама", "Панама", "Северная Америка"), ("Коста-Рика", "Сан-Хосе", "Центральная Америка"),
            ("Колумбия", "Богота", "Южная Америка"), ("Венесуэла", "Каракас", "Южная Америка"),
            ("Уругвай", "Монтевидео", "Южная Америка"), ("Парагвай", "Асунсьон", "Южная Америка"),
            ("Эквадор", "Кито", "Южная Америка"), ("Ямайка", "Кингстон", "Карибский регион"),
            ("Гаити", "Порт-о-Пренс", "Карибский регион"), ("Доминиканская Республика", "Санто-Доминго", "Карибский регион"),
            ("Гватемала", "Гватемала", "Центральная Америка"), ("Гондурас", "Тегусигальпа", "Центральная Америка"),
            ("Никарагуа", "Манагуа", "Центральная Америка"), ("Сальвадор", "Сан-Сальвадор", "Центральная Америка"),
            ("Фиджи", "Сува", "Австралия и Океания"), ("Папуа — Новая Гвинея", "Порт-Морсби", "Австралия и Океания"),
            ("Самоа", "Апиа", "Австралия и Океания"),
        ]
        country_genitive = {
            "Россия": "России", "Франция": "Франции", "Италия": "Италии", "Испания": "Испании",
            "Германия": "Германии", "Португалия": "Португалии", "Норвегия": "Норвегии", "Швеция": "Швеции",
            "Финляндия": "Финляндии", "Польша": "Польши", "Греция": "Греции", "Ирландия": "Ирландии",
            "Япония": "Японии", "Китай": "Китая", "Индия": "Индии", "Южная Корея": "Южной Кореи",
            "Монголия": "Монголии", "Таиланд": "Таиланда", "Вьетнам": "Вьетнама", "Непал": "Непала",
            "Египет": "Египта", "Кения": "Кении", "Нигерия": "Нигерии", "Марокко": "Марокко",
            "Эфиопия": "Эфиопии", "США": "Соединённых Штатов", "Канада": "Канады", "Мексика": "Мексики",
            "Куба": "Кубы", "Бразилия": "Бразилии", "Аргентина": "Аргентины", "Чили": "Чили",
            "Перу": "Перу", "Австралия": "Австралии", "Новая Зеландия": "Новой Зеландии", "Турция": "Турции",
            "Грузия": "Грузии", "Казахстан": "Казахстана", "Узбекистан": "Узбекистана", "Азербайджан": "Азербайджана",
            "Исландия": "Исландии", "Дания": "Дании", "Нидерланды": "Нидерландов", "Бельгия": "Бельгии",
            "Австрия": "Австрии", "Швейцария": "Швейцарии", "Чехия": "Чехии", "Словакия": "Словакии",
            "Венгрия": "Венгрии", "Румыния": "Румынии", "Болгария": "Болгарии", "Хорватия": "Хорватии",
            "Сербия": "Сербии", "Украина": "Украины", "Беларусь": "Беларуси", "Люксембург": "Люксембурга",
            "Мальта": "Мальты", "Кипр": "Кипра", "Литва": "Литвы", "Латвия": "Латвии", "Эстония": "Эстонии",
            "Саудовская Аравия": "Саудовской Аравии", "Объединённые Арабские Эмираты": "Объединённых Арабских Эмиратов",
            "Катар": "Катара", "Иран": "Ирана", "Ирак": "Ирака", "Пакистан": "Пакистана",
            "Бангладеш": "Бангладеш", "Шри-Ланка": "Шри-Ланки", "Сингапур": "Сингапура", "Малайзия": "Малайзии",
            "Филиппины": "Филиппин", "Камбоджа": "Камбоджи", "Лаос": "Лаоса", "Мьянма": "Мьянмы",
            "Бутан": "Бутана", "Афганистан": "Афганистана", "Кыргызстан": "Кыргызстана", "Таджикистан": "Таджикистана",
            "Туркменистан": "Туркменистана", "Гана": "Ганы", "Сенегал": "Сенегала", "Танзания": "Танзании",
            "Уганда": "Уганды", "Замбия": "Замбии", "Ботсвана": "Ботсваны", "Намибия": "Намибии",
            "Алжир": "Алжира", "Тунис": "Туниса", "Ливия": "Ливии", "Мадагаскар": "Мадагаскара",
            "Камерун": "Камеруна", "Кот-д'Ивуар": "Кот-д'Ивуара", "Южно-Африканская Республика": "Южно-Африканской Республики",
            "Панама": "Панамы", "Коста-Рика": "Коста-Рики", "Колумбия": "Колумбии", "Венесуэла": "Венесуэлы",
            "Уругвай": "Уругвая", "Парагвай": "Парагвая", "Эквадор": "Эквадора", "Ямайка": "Ямайки",
            "Гаити": "Гаити", "Доминиканская Республика": "Доминиканской Республики", "Гватемала": "Гватемалы",
            "Гондурас": "Гондураса", "Никарагуа": "Никарагуа", "Сальвадор": "Сальвадора", "Фиджи": "Фиджи",
            "Папуа — Новая Гвинея": "Папуа — Новой Гвинеи", "Самоа": "Самоа",
        }
        q_forms = [
            "Какая столица у государства «{country}»?", "Назови столицу страны «{country}».",
            "Как называется столица {country_gen}?", "Какой город — столица {country_gen}?",
            "У государства «{country}» какой город является столицей?", "Мне нужен город-столица государства «{country}».",
            "В каком городе расположена столица {country_gen}?", "Какой город служит столицей страны «{country}»?",
        ]
        a_forms = [
            "Столица {country_gen} — {capital}.", "Для государства «{country}» столица — {capital}.",
            "{capital} — столица {country_gen}.", "Главный город страны «{country}» — {capital}.",
            "Столицей государства «{country}» является {capital}.", "У государства «{country}» столица — {capital}.",
            "Столица государства «{country}» называется {capital}.", "Это {capital}, столица государства «{country}».",
        ]
        for ci, (country, capital, continent) in enumerate(countries):
            country_gen = country_genitive.get(country, country)
            for k in range(8):
                self.add("general_knowledge", "ru", f"geo:capital:{country}", [
                    {"role": "user", "content": q_forms[k].format(country=country, country_gen=country_gen)},
                    {"role": "assistant", "content": a_forms[k].format(country=country, country_gen=country_gen, capital=capital)},
                ], {"type": "synthetic_curated", "generator": "curated_capitals_v1", "fact_id": f"capital:{country}", "manual_fact": f"{capital} is capital of {country}"})
            for k in range(4):
                prompts = [f"На каком континенте находится {country}?", f"К какой части света относится государство «{country}»?",
                           f"Где географически расположена страна «{country}»?", f"Укажи часть света для государства «{country}»."]
                answers = [f"{country} находится в регионе {continent}.", f"{country} относится к региону {continent}.",
                           f"Географический регион {country} — {continent}.", f"Это {continent}."]
                self.add("general_knowledge", "ru", f"geo:continent:{country}", [
                    {"role": "user", "content": prompts[k]}, {"role": "assistant", "content": answers[k]},
                ], {"type": "synthetic_curated", "generator": "curated_geography_v1", "fact_id": f"continent:{country}", "manual_fact": f"{country}: {continent}"})

        # Add natural question paraphrases for the same curated geography facts.
        # This expands factual QA without increasing the math/code share or
        # recycling the assistant reply verbatim.
        extra_capital_questions = [
            "Напомни, какой город является столицей {country_gen}?",
            "Какой город называют столицей {country_gen}?",
            "Столичный город страны «{country}» — какой?",
            "Мне нужно проверить столицу: какой город у {country_gen}?",
            "Где находится столица государства «{country}»?",
            "Какой город указан столицей страны «{country}»?",
            "Назови столицу государства «{country}».",
            "Как называется главный город страны «{country}»?",
            "Как называется город, который выполняет роль столицы {country_gen}?",
            "Если говорить о столице {country_gen}, какой город нужно назвать?",
        ]
        extra_capital_answers = [
            "Это {capital}.",
            "У государства «{country}» столицей служит город {capital}.",
            "Столичный город {country_gen} — {capital}.",
            "Столица государства «{country}» — город {capital}.",
            "В качестве столицы {country_gen} называют {capital}.",
            "Город {capital} является столицей {country_gen}.",
            "Для страны «{country}» столицей считается {capital}.",
            "Столицей страны «{country}» считается город {capital}.",
            "В этом случае ответ — {capital}, столица {country_gen}.",
            "Государственная столица {country_gen} — {capital}.",
        ]
        for country, capital, _ in countries:
            country_gen = country_genitive.get(country, country)
            for k, (question_form, answer_form) in enumerate(zip(extra_capital_questions, extra_capital_answers)):
                self.add("general_knowledge", "ru", f"geo:capital:{country}", [
                    {"role": "user", "content": question_form.format(country=country, country_gen=country_gen)},
                    {"role": "assistant", "content": answer_form.format(country=country, country_gen=country_gen, capital=capital)},
                ], {"type": "synthetic_curated", "generator": "curated_capitals_paraphrases_v2",
                    "fact_id": f"capital:{country}:{k}", "manual_fact": f"{capital} is capital of {country}"})

        moscow_facts = [
            ("В какой части света находится Москва?", "Москва расположена в европейской части России.", "Moscow is in European Russia"),
            ("Какая река протекает через Москву?", "Через Москву протекает Москва-река.", "Moskva River flows through Moscow"),
            ("Где находится Красная площадь?", "Красная площадь расположена в центре Москвы, рядом с Кремлём.", "Red Square is in central Moscow by the Kremlin"),
            ("В каком городе находится Московский Кремль?", "Московский Кремль находится в центре Москвы.", "Moscow Kremlin is in central Moscow"),
            ("Москва находится в Европе или Азии?", "Москва находится в европейской части России.", "Moscow is in Europe"),
            ("На какой реке стоит Москва?", "Москва расположена на берегах Москвы-реки.", "Moscow lies on Moskva River"),
            ("В какой стране находится город Москва?", "Москва находится в России и является её столицей.", "Moscow is in Russia and is its capital"),
            ("Где расположен Кремль в Москве?", "Кремль расположен в историческом центре Москвы.", "Kremlin is in Moscow's historic center"),
            ("Какой город крупнейший в России по населению?", "Крупнейший по населению город России — Москва.", "Moscow is Russia's most populous city"),
            ("С каким городом связана Красная площадь?", "Красная площадь — одна из главных площадей Москвы.", "Red Square is a major Moscow square"),
            ("В какой части России расположена столица?", "Столица России, Москва, находится на западе страны, в её европейской части.", "Moscow is in western European Russia"),
        ]
        for index, (question, answer, fact) in enumerate(moscow_facts):
            self.add("general_knowledge", "ru", f"geo:moscow:fact:{index}", [
                {"role": "user", "content": question}, {"role": "assistant", "content": answer}
            ], {"type": "synthetic_curated", "generator": "curated_moscow_facts_v1", "manual_fact": fact})

        planet_facts = [
            ("Меркурий", "ближе всех к Солнцу", "Какая планета ближе всего к Солнцу?"),
            ("Венера", "самая горячая планета Солнечной системы", "Какая планета считается самой горячей?"),
            ("Земля", "третья планета от Солнца", "Какое место Земля занимает от Солнца?"),
            ("Марс", "четвёртая планета от Солнца", "Какую планету называют четвёртой от Солнца?"),
            ("Юпитер", "крупнейшая планета Солнечной системы", "Какая планета крупнейшая?"),
            ("Сатурн", "известен заметной системой колец", "У какой планеты особенно заметны кольца?"),
            ("Уран", "вращается с очень большим наклоном оси", "У какой планеты необычно сильно наклонена ось вращения?"),
            ("Нептун", "восьмая планета от Солнца", "Какая планета находится восьмой от Солнца?"),
        ]
        for planet, fact, question in planet_facts:
            answer_variants = [f"{planet} — это планета, которая {fact}.", f"Короткий ответ: {planet} {fact}.",
                               f"Речь о {planet}: она {fact}.", f"Это {planet}; она {fact}."]
            for k, answer in enumerate(answer_variants):
                self.add("general_knowledge", "ru", f"solar:{planet}:{fact}", [
                    {"role": "user", "content": question if k == 0 else f"А что известно про {planet}: {fact}?"},
                    {"role": "assistant", "content": answer},
                ], {"type": "synthetic_curated", "generator": "curated_solar_system_v1", "fact_id": f"planet:{planet}:{k}", "manual_fact": fact})

        planet_order = ["Меркурий", "Венера", "Земля", "Марс", "Юпитер", "Сатурн", "Уран", "Нептун"]
        planet_types = ["планета земной группы", "планета земной группы", "планета земной группы", "планета земной группы",
                        "газовый гигант", "газовый гигант", "ледяной гигант", "ледяной гигант"]
        ordinal = ["первой", "второй", "третьей", "четвёртой", "пятой", "шестой", "седьмой", "восьмой"]
        for index, planet in enumerate(planet_order):
            order_answer = f"{planet} — {ordinal[index]} планета от Солнца."
            self.add("general_knowledge", "ru", f"solar:order:{planet}", [
                {"role": "user", "content": f"Какое место {planet} занимает от Солнца?"},
                {"role": "assistant", "content": order_answer},
            ], {"type": "synthetic_curated", "generator": "curated_solar_system_v2", "manual_fact": order_answer})
            type_answer = f"{planet} относится к типу «{planet_types[index]}»."
            self.add("general_knowledge", "ru", f"solar:type:{planet}", [
                {"role": "user", "content": f"Какого типа планета {planet}?"},
                {"role": "assistant", "content": type_answer},
            ], {"type": "synthetic_curated", "generator": "curated_solar_system_v2", "manual_fact": type_answer})
            if index < len(planet_order) - 1:
                next_planet = planet_order[index + 1]
                next_answer = f"После {planet} от Солнца идёт {next_planet}."
                self.add("general_knowledge", "ru", f"solar:next:{planet}", [
                    {"role": "user", "content": f"Какая планета следует за {planet} от Солнца?"},
                    {"role": "assistant", "content": next_answer},
                ], {"type": "synthetic_curated", "generator": "curated_solar_system_v2", "manual_fact": next_answer})

        core_facts = [
            ("Солнце", "звезда в центре Солнечной системы", "Что такое Солнце?"),
            ("Луна", "естественный спутник Земли", "Что такое Луна?"),
            ("Земля", "вращается вокруг своей оси", "Что Земля делает вокруг своей оси?"),
            ("Земля", "обращается вокруг Солнца", "Вокруг чего обращается Земля?"),
            ("вода при обычном давлении", "замерзает примерно при 0 °C", "При какой температуре замерзает вода?"),
            ("вода при обычном давлении", "кипит примерно при 100 °C", "При какой температуре кипит вода?"),
            ("растения", "используют свет в процессе фотосинтеза", "Зачем растениям нужен свет?"),
            ("звук", "распространяется в среде как колебания", "Как распространяется звук?"),
            ("интернет", "соединяет устройства и сети для обмена данными", "Что такое интернет простыми словами?"),
            ("компас", "помогает определять стороны света", "Для чего нужен компас?"),
        ]
        for subject, fact, question in core_facts:
            variants = [f"{subject.capitalize()} {fact}.", f"Коротко: {subject} {fact}.",
                        f"Можно сказать так: {subject} {fact}.", f"Это связано с тем, что {subject} {fact}."]
            for i, answer in enumerate(variants):
                self.add("general_knowledge", "ru", f"core:{subject}:{fact}", [
                    {"role": "user", "content": question if i == 0 else f"Объясни кратко: {question}"},
                    {"role": "assistant", "content": answer},
                ], {"type": "synthetic_curated", "generator": "curated_core_facts_v1", "fact_id": f"core:{subject}:{i}", "manual_fact": fact})

    def generate_explanations(self, target: int = 1800) -> None:
        concepts = [
            ("переменная в Python", "имя для значения, которое программа использует или меняет"),
            ("цикл", "конструкция для повторения набора команд"),
            ("условие if", "проверка, от результата которой зависит выполнение кода"),
            ("функция", "именованный фрагмент кода, который можно вызвать с нужными данными"),
            ("список в Python", "упорядоченная коллекция, в которой хранят несколько элементов"),
            ("процент", "доля, выраженная в сотых частях целого"),
            ("дробь", "запись части целого с числителем и знаменателем"),
            ("гравитация", "взаимное притяжение тел, обладающих массой"),
            ("испарение", "переход вещества из жидкости в газ с её поверхности"),
            ("конденсация", "переход пара или газа в жидкость при подходящих условиях"),
            ("электрический ток", "направленное движение электрических зарядов"),
            ("алгоритм", "точная последовательность шагов для решения задачи"),
            ("интернет", "глобальная система соединённых компьютерных сетей и устройств"),
            ("перевод", "передача смысла сообщения средствами другого языка"),
            ("подлежащее", "главный член предложения, который называет предмет речи"),
            ("сказуемое", "главный член предложения, который сообщает о действии или состоянии"),
            ("окисление железа", "химический процесс, при котором железо взаимодействует с кислородом и может образоваться ржавчина"),
            ("площадь прямоугольника", "величина, равная произведению его длины на ширину"),
            ("периметр", "сумма длин границ плоской фигуры"),
            ("планета", "небесное тело, которое обращается вокруг звезды и не светит как звезда"),
            ("атом", "наименьшая частица химического элемента, сохраняющая его свойства"),
            ("молекула", "частица вещества, состоящая из связанных атомов"),
            ("масса", "физическая величина, связанная с количеством вещества и инертностью тела"),
            ("объём", "величина, показывающая, сколько пространства занимает тело"),
            ("плотность", "отношение массы вещества к занимаемому им объёму"),
            ("сила", "величина, которая описывает взаимодействие тел и может менять их движение или форму"),
            ("энергия", "физическая величина, связанная со способностью системы совершать работу или передавать тепло"),
            ("температура", "величина, характеризующая тепловое состояние тела"),
            ("давление", "отношение силы, действующей перпендикулярно поверхности, к площади этой поверхности"),
            ("скорость", "отношение пройденного пути ко времени движения"),
            ("средняя скорость", "отношение всего пройденного пути ко всему времени движения"),
            ("трение", "взаимодействие поверхностей, которое мешает их относительному скольжению"),
            ("инерция", "свойство тела сохранять покой или равномерное движение, если внешнее воздействие не меняет его"),
            ("звук", "колебания, которые распространяются в упругой среде и могут восприниматься слухом"),
            ("отражение света", "изменение направления светового луча на границе поверхностей"),
            ("магнит", "тело, создающее магнитное поле и взаимодействующее с некоторыми материалами и другими магнитами"),
            ("проводник", "материал, в котором электрический заряд может перемещаться сравнительно свободно"),
            ("изолятор", "материал, который сильно затрудняет движение электрических зарядов"),
            ("среднее арифметическое", "сумма чисел, разделённая на их количество"),
            ("уравнение", "равенство с неизвестным значением, которое нужно найти"),
            ("координаты", "числа, которые задают положение точки относительно выбранной системы отсчёта"),
            ("масштаб карты", "отношение расстояния на карте к соответствующему расстоянию на местности"),
            ("веб-браузер", "программа для открытия и просмотра страниц и других материалов в интернете"),
            ("сервер", "компьютер или программа, которая предоставляет данные или услуги другим устройствам"),
            ("база данных", "организованное хранилище сведений, которым удобно управлять и из которого можно получать нужные записи"),
            ("параметр функции", "имя для входного значения, которое функция получает при вызове"),
            ("аргумент функции", "конкретное значение, переданное параметру при вызове функции"),
            ("возвращаемое значение", "результат, который функция передаёт обратно в вызвавший её код"),
            ("логическое значение", "один из двух результатов проверки: истина или ложь"),
            ("словарь в Python", "структура данных, которая связывает ключи со значениями"),
            ("строка в Python", "последовательность символов, используемая для хранения текста"),
            ("индекс списка", "номер позиции элемента; в Python отсчёт начинается с нуля"),
            ("ошибка времени выполнения", "ошибка, которая возникает во время исполнения программы"),
            ("отладчик", "инструмент для пошагового выполнения кода и поиска ошибок"),
            ("переменная в алгебре", "обозначение величины, значение которой может меняться или быть неизвестным"),
            ("пропорция", "равенство двух отношений"),
            ("угол", "фигура, образованная двумя лучами с общим началом"),
            ("диаметр", "хорда окружности, проходящая через её центр"),
            ("радиус", "отрезок от центра окружности до точки на ней"),
            ("экосистема", "сообщество организмов вместе со средой их обитания и связями между ними"),
            ("пищевая цепь", "последовательность организмов, в которой каждый предыдущий служит пищей следующему"),
            ("опыление", "перенос пыльцы к части цветка, где может произойти оплодотворение"),
            ("испарение воды", "переход жидкой воды в водяной пар с поверхности"),
            ("круговорот воды", "непрерывное движение воды между атмосферой, сушей и водоёмами"),
            ("климат", "характерный для региона режим погоды, рассматриваемый за длительный период"),
            ("погода", "состояние атмосферы в определённом месте и в конкретное время"),
            ("орбита", "траектория движения одного тела вокруг другого под действием притяжения"),
            ("спутник планеты", "небесное тело, которое обращается вокруг планеты"),
            ("галактика", "система из звёзд, газа, пыли и других компонентов, связанных гравитацией"),
            ("фотосинтез", "процесс, в котором растения используют энергию света для создания органических веществ"),
            ("хлорофилл", "зелёный пигмент растений, участвующий в поглощении света при фотосинтезе"),
            ("клеточная мембрана", "оболочка клетки, которая отделяет её содержимое и регулирует обмен веществ"),
            ("ДНК", "молекула, в которой хранится наследственная информация организма"),
            ("микроорганизм", "маленький организм, который обычно можно увидеть только с помощью увеличения"),
            ("эксперимент", "проверка предположения в заранее организованных условиях"),
            ("наблюдение", "сбор сведений о явлении без намеренного изменения его условий"),
            ("гипотеза", "проверяемое предположение, которое объясняет наблюдение или предсказывает результат"),
            ("давление воздуха", "действие массы воздуха на поверхность; оно меняется с высотой и погодными условиями"),
            ("точка кипения", "температура, при которой жидкость начинает интенсивно переходить в пар при заданном давлении"),
            ("теплопроводность", "передача тепловой энергии между частями вещества или соприкасающимися телами"),
            ("возобновляемый источник энергии", "источник, который естественно пополняется и не исчерпывается при обычном использовании в коротком масштабе времени"),
            ("солнечная панель", "устройство, преобразующее энергию солнечного света в электричество"),
            ("парниковый эффект", "удержание частью атмосферы теплового излучения, из-за которого поверхность планеты теплее"),
            ("тектоническая плита", "крупный подвижный фрагмент внешней твёрдой оболочки Земли"),
            ("вулкан", "геологическое образование, через которое на поверхность могут выходить магма, газы и пепел"),
            ("эрозия", "разрушение и перенос почвы или горных пород водой, ветром, льдом и другими воздействиями"),
            ("демократия", "форма управления, в которой граждане участвуют в принятии общественных решений напрямую или через представителей"),
            ("конституция", "основной закон государства, который задаёт основы его устройства и права граждан"),
            ("парламент", "представительный орган, который принимает законы и выполняет другие установленные законом функции"),
            ("инфляция", "устойчивый рост общего уровня цен, из-за которого на одну и ту же сумму можно купить меньше товаров и услуг"),
            ("налог", "обязательный платёж государству, установленный законом для финансирования общественных расходов"),
            ("медиана", "серединное значение упорядоченного набора чисел; при чётном количестве берут среднее двух центральных"),
            ("вероятность", "числовая оценка того, насколько возможно событие, обычно от нуля до единицы"),
            ("API", "описание правил, по которым программы обмениваются запросами и данными"),
            ("JSON", "текстовый формат записи структурированных данных в виде объектов, массивов и простых значений"),
            ("HTML", "язык разметки, который задаёт структуру содержимого веб-страницы"),
            ("CSS", "язык правил, который описывает внешний вид элементов веб-страницы"),
            ("IP-адрес", "числовой или буквенно-числовой сетевой адрес, используемый для идентификации устройства или интерфейса в сети"),
            ("маршрутизатор", "сетевое устройство, которое пересылает пакеты данных между сетями"),
            ("кэш браузера", "временное хранилище копий ресурсов, помогающее быстрее открывать уже посещённые страницы"),
            ("облачное хранилище", "сервис, который хранит файлы на удалённых серверах и позволяет получать к ним доступ через сеть"),
            ("открытая лицензия", "условия, которые заранее разрешают определённые способы использования произведения при соблюдении правил лицензии"),
            ("общественное достояние", "статус произведения, для которого исключительные авторские права истекли или не действуют в данной юрисдикции"),
            ("проверка источника", "оценка происхождения сведений, компетентности автора, даты и подтверждений"),
            ("цифровая приватность", "контроль человека над сбором, хранением и использованием сведений о нём в цифровой среде"),
            ("резервная копия", "отдельно сохранённая копия данных, помогающая восстановить их после потери или повреждения"),
            ("версия программы", "зафиксированный выпуск кода с определённым набором изменений и номером или меткой"),
            ("система контроля версий", "инструмент и набор правил для записи истории изменений файлов и совместной работы над кодом"),
            ("коммит в Git", "сохранённый снимок выбранных изменений в истории репозитория"),
            ("ветка в Git", "отдельная линия изменений, позволяющая работать над задачей независимо от других веток"),
            ("интерпретатор", "программа, которая читает исходный код и выполняет его команды"),
            ("компилятор", "программа, которая переводит исходный код или его часть в другую форму, например машинный код"),
            ("тип данных", "категория значений, которая определяет допустимые операции над ними"),
            ("булево значение", "логическое значение, которое принимает одно из двух состояний: истина или ложь"),
            ("рекурсия", "способ решения задачи, при котором функция вызывает сама себя для более простой подзадачи"),
            ("итерация", "один повтор цикла или отдельный шаг последовательного процесса"),
            ("исключение в программе", "событие во время выполнения, которое сигнализирует о проблеме и может прервать обычный ход программы"),
            ("тест программы", "проверка, которая сравнивает фактическую работу кода с ожидаемым результатом"),
            ("модуль программы", "файл или компонент с кодом, который можно подключать и использовать в другой части проекта"),
            ("библиотека программирования", "набор готовых функций и компонентов, которые можно подключить к своей программе"),
            ("запрос к базе данных", "инструкция, которая просит систему найти, добавить, изменить или удалить записи"),
            ("таблица базы данных", "структура для хранения однотипных записей в строках и полях"),
            ("сетевой протокол", "набор правил, по которым устройства обмениваются данными"),
            ("доменное имя", "удобное для чтения имя сайта или сетевого узла, которое сопоставляется с техническим адресом"),
            ("шифрование", "преобразование данных в вид, который трудно прочитать без нужного ключа"),
            ("многофакторная аутентификация", "проверка входа с помощью двух или более разных подтверждений личности"),
            ("фишинг", "обманная попытка выманить у человека данные, выдавая сообщение или сайт за надёжный источник"),
            ("цифровой след", "совокупность данных, которые остаются после действий человека в цифровых сервисах"),
            ("авторское право", "правовая охрана произведения и установленных законом интересов его автора или правообладателя"),
            ("цитирование", "указание использованного фрагмента или источника, чтобы показать, откуда взяты сведения"),
            ("предвзятость источника", "устойчивый уклон в отборе или подаче сведений, который может влиять на вывод"),
            ("первичный источник", "непосредственное свидетельство события или оригинальный материал, созданный его участником либо исследователем"),
            ("вторичный источник", "материал, который анализирует, пересказывает или сопоставляет первичные свидетельства"),
            ("корреляция", "статистическая связь, при которой изменения одной величины сопровождаются изменениями другой"),
            ("причинность", "отношение, в котором одно событие или условие влияет на возникновение другого"),
            ("выборка", "часть объектов или наблюдений, выбранная для изучения более широкой совокупности"),
            ("среднее значение", "результат, полученный делением суммы чисел на их количество"),
            ("мода в статистике", "значение, которое встречается в наборе данных чаще остальных"),
            ("диапазон значений", "разность между наибольшим и наименьшим значениями набора"),
            ("координатная ось", "прямая с выбранным началом и единицей отсчёта, на которой задают координаты точек"),
            ("масштаб", "отношение размера изображения, карты или модели к размеру соответствующего объекта"),
            ("градус Цельсия", "единица температуры шкалы, в которой при нормальном давлении вода замерзает около нуля градусов"),
            ("электрическая цепь", "соединение элементов, по которому при замкнутом пути может течь электрический ток"),
            ("тень", "область, куда свет не попадает напрямую из-за преграды"),
            ("преломление света", "изменение направления света при переходе между средами с разными оптическими свойствами"),
            ("звёздная система", "группа звёзд, связанных гравитацией и обращающихся вокруг общего центра масс"),
            ("карликовая планета", "небесное тело, обращающееся вокруг Солнца, почти округлое, но не очистившее окрестность своей орбиты"),
            ("естественный отбор", "процесс, при котором наследуемые особенности, помогающие размножению, со временем чаще сохраняются в популяции"),
            ("пищевая сеть", "система связанных пищевых цепей, показывающая, кто кем питается в экосистеме"),
            ("адаптация организма", "наследуемая особенность или процесс, помогающий организму выживать и размножаться в определённых условиях"),
            ("миграция животных", "регулярное перемещение животных между районами, часто связанное с сезонами, пищей или размножением"),
            ("кровеносная система", "система органов и сосудов, которая переносит кровь и вещества по организму"),
            ("иммунная система", "совокупность клеток, тканей и процессов, помогающих организму распознавать и ограничивать инфекции и другие угрозы"),
            ("питательное вещество", "компонент пищи, который организм использует для энергии, роста или поддержания функций"),
            ("государственная граница", "линия и вертикальная поверхность, определяющие пределы территории государства"),
            ("часовой пояс", "область, в которой для гражданского времени используют согласованное смещение относительно UTC"),
            ("полушарие Земли", "одна из двух половин планеты, условно разделённых экватором или выбранным меридианом"),
            ("природный ресурс", "компонент природы, который люди могут использовать для жизни и хозяйственной деятельности"),
            ("водосборный бассейн", "территория, с которой вода стекает в одну реку, озеро или другую общую водную систему"),
            ("дельта реки", "участок у устья, где река разделяется на рукава и откладывает наносы"),
            ("ледник", "медленно движущаяся масса многолетнего льда, образующаяся на суше из накопленного снега"),
            ("цунами", "длинная серия морских волн, часто возникающая из-за подводного землетрясения или другого резкого смещения воды"),
            ("воздушная масса", "крупный объём воздуха с относительно сходными температурой и влажностью"),
            ("водораздел", "граница между соседними территориями стока воды в разные бассейны"),
        ]
        prompts = ["Объясни простыми словами, что такое {topic}.", "Что означает {topic}?", "Помоги понять: {topic} — это что?",
                   "Дай короткое объяснение понятия «{topic}».", "Расскажи без сложных терминов про {topic}.",
                   "Как объяснить новичку, что такое {topic}?"]
        styles = ["{topic_cap} — {fact}.",
                  "Термин «{topic}» означает следующее: {fact}.",
                  "Под «{topic}» понимают {fact}.",
                  "Словом «{topic}» называют {fact}.",
                  "В учебном контексте {topic} — {fact}.",
                  "Здесь {topic} — {fact}.",
                  "Это называют «{topic}»: {fact}.",
                  "Если говорить о {topic}, речь идёт про {fact}.",
                  "Основной смысл термина «{topic}» — {fact}.",
                  "В данном случае {topic} — это {fact}."]
        for idx in range(target):
            topic, fact = concepts[idx % len(concepts)]
            cycle = idx // len(concepts)
            p = prompts[cycle % len(prompts)].format(topic=topic)
            a = styles[(cycle // len(prompts) + idx) % len(styles)].format(topic=topic, topic_cap=topic.capitalize(), fact=fact)
            self.add("explanations", "ru", f"explain:{topic}", [
                {"role": "user", "content": p}, {"role": "assistant", "content": a}
            ], {"type": "synthetic_curated", "generator": "short_explanation_cards_v1", "concept": topic})
        # Numerically grounded explanations provide distinct calculations with many answer forms.
        area_prefixes = [
            "Площадь — это длина, умноженная на ширину:",
            "Чтобы найти площадь, перемножаем стороны:",
            "У прямоугольника площадь вычисляется так:",
            "Внутреннюю площадь считаем произведением сторон:",
            "Формула площади прямоугольника — длина × ширина:",
            "Для площади прямоугольника нужно умножить стороны:",
        ]
        area_endings = [
            "значит, это {area} квадратных единиц.", "получается {area} квадратных единиц.",
            "то есть площадь равна {area}.", "итоговая площадь — {area} квадратных единиц.",
            "поэтому ответ: {area} квадратных единиц.",
        ]
        area_prompts = [
            "Как найти площадь прямоугольника со сторонами {length} и {width}?",
            "Объясни коротко, как вычислить площадь прямоугольника {length} на {width}.",
            "У прямоугольника длина {length}, ширина {width}. Какова площадь?",
            "Помоги разобраться с площадью: стороны прямоугольника {length} и {width}.",
            "Покажи расчёт площади прямоугольника размером {length} × {width}.",
            "Сколько квадратных единиц занимает прямоугольник {length} на {width}?",
        ]
        area_styles = [f"{prefix} {{length}} × {{width}} = {{area}}, {{ending}}"
                       for prefix in area_prefixes for ending in area_endings]
        added = 0
        for length in range(2, 32):
            for width in range(2, 32):
                area = length * width
                self.add("explanations", "ru", f"explain:rectangle-area:{length}:{width}", [
                    {"role": "user", "content": area_prompts[added % len(area_prompts)].format(length=length, width=width)},
                    {"role": "assistant", "content": area_styles[added % len(area_styles)].format(length=length, width=width, area=area, ending=area_endings[added % len(area_endings)].format(area=area))},
                ], {"type": "synthetic_computed", "generator": "rectangle_area_explanation_v1", "inputs": [length, width], "expected": area})
                added += 1
                if added >= 900:
                    break
            if added >= 900:
                break
        percent_prefixes = [
            "Чтобы найти процент от числа, умножаем число на долю:",
            "Процент означает долю из ста; расчёт такой:",
            "Сначала умножаем число на процент, затем делим на сто:",
            "Для вычисления берём {percent} сотых от числа {whole}:",
            "Запишем процент как дробь со знаменателем сто:",
            "Можно посчитать по формуле «число × процент ÷ 100»:",
        ]
        percent_endings = [
            "значит, ответ — {result}.", "поэтому получается {result}.",
            "итого: {result}.", "следовательно, результат равен {result}.",
            "это и есть {result}.",
        ]
        percent_prompts = [
            "Сколько будет {percent}% от {whole}? Объясни расчёт.",
            "Покажи простое вычисление: {percent}% от числа {whole}.",
            "Как найти {percent} процентов от {whole}?",
            "Объясни на примере, чему равны {percent}% числа {whole}.",
            "Вычисли {percent}% от {whole} и коротко поясни.",
            "Мне нужно узнать {percent}% от {whole}; покажи формулу.",
        ]
        percent_styles = [f"{prefix} {{whole}} × {{percent}} ÷ 100 = {{result}}, {{ending}}"
                          for prefix in percent_prefixes for ending in percent_endings]
        added = 0
        for percent in (5, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90):
            for whole in range(10, 201):
                raw = whole * percent / 100
                result = int(raw) if raw.is_integer() else round(raw, 2)
                self.add("explanations", "ru", f"explain:percent:{percent}:{whole}", [
                    {"role": "user", "content": percent_prompts[added % len(percent_prompts)].format(percent=percent, whole=whole)},
                    {"role": "assistant", "content": percent_styles[added % len(percent_styles)].format(
                        whole=whole, percent=percent, result=result,
                        ending=percent_endings[added % len(percent_endings)].format(result=result))},
                ], {"type": "synthetic_computed", "generator": "percent_explanation_v1", "inputs": [percent, whole], "expected": result})
                added += 1
                if added >= 1000:
                    break
            if added >= 1000:
                break

    def generate_programming(self, target: int = 10000) -> None:
        rng = random.Random(SEED + 2)
        names = ["solve", "calculate", "result_for", "run_task", "get_value", "compute", "answer", "process"]
        parameters = ["a", "b", "left", "right", "first", "second", "value", "number", "x", "y",
                      "base", "power", "amount", "count", "width", "height", "low", "high", "start", "end",
                      "price", "quantity", "score", "limit", "item"]
        qforms = ["Напиши функцию Python и покажи результат для {a} и {b}: {task}.",
                  "Как решить задачу на Python: {task} для значений {a} и {b}? Покажи короткий пример.",
                  "Составь простую функцию Python: {task}. Проверь её на числах {a} и {b}.",
                  "Нужен короткий пример на Python для задачи «{task}», входные числа — {a} и {b}."]
        tasks = ["сложить два числа", "найти разность первого и второго", "умножить числа", "вернуть большее число",
                 "вернуть True, если число чётное", "возвести число в квадрат",
                 "вычислить среднее арифметическое двух чисел", "вернуть остаток от деления первого числа на второе",
                 "проверить, равны ли два числа", "возвести первое число в степень второго",
                 "сложить числа из списка циклом", "посчитать положительные числа в списке условием"]
        intros = ["Для {a} и {b} функция может выглядеть так:", "Пара чисел {a}, {b}; вот короткий код:",
                  "С аргументами {a} и {b} можно написать такую функцию:", "Проверим значения {a} и {b} на этом примере:",
                  "Числа {a} и {b} подойдут для такой проверки:", "Вызов с числами {a}, {b} можно оформить так:",
                  "На входе {a} и {b}; реализация функции:", "Аргументы {a} и {b} обработает такой код:",
                  "Для пары {a}/{b} можно взять эту функцию:", "Пример решения для чисел {a}, {b}:",
                  "Возьмём {a} и {b} и проверим результат функцией:", "Здесь входные числа — {a} и {b}; код такой:",
                  "Функция для значений {a}, {b} может быть записана так:", "Сначала зададим код для пары {a}, {b}:",
                  "Короткий вариант с аргументами {a} и {b}:", "Вот реализация, которую можно проверить на {a} и {b}:"]
        intros.extend([
            "Один из вариантов для чисел {a} и {b}:", "Вот функция, проверенная на {a} и {b}:",
            "Для входов {a} и {b} подойдёт такая реализация:", "Пример функции с аргументами {a} и {b}:",
            "Можно обработать пару {a}, {b} вот так:", "Зададим функцию и проверим числа {a}, {b}:",
            "Для проверки возьмём {a} и {b}:", "Ниже функция для входов {a} и {b}:",
            "Так можно решить задачу для значений {a} и {b}:", "Посмотрим на работу кода с числами {a} и {b}:",
            "Запишем решение и подставим {a} и {b}:", "Эти значения можно передать функции так: {a}, {b}.",
            "Подойдёт такой небольшой пример для {a} и {b}:", "Функция принимает числа {a} и {b} и выполняет задачу:",
            "Применим короткую функцию к значениям {a}, {b}:", "Например, числа {a} и {b} можно обработать так:",
        ])
        unary_intros = ["Для числа {a} функция будет такой:", "Проверим значение {a}:",
                        "Вот короткое решение для числа {a}:", "При аргументе {a} код выглядит так:",
                        "Для входа {a} можно написать:", "Возьмём число {a} и проверим его функцией:"]
        examples = []
        for a in range(-120, 121):
            for b in range(-120, 121):
                examples.append((a, b))
        rng.shuffle(examples)
        count = 0
        for a, b in examples:
            for kind, task in enumerate(tasks):
                if kind == 7 and b == 0:
                    continue
                if kind == 9 and not 0 <= b <= 5:
                    continue
                fn = names[(count + kind) % len(names)]
                x_index = count % len(parameters)
                y_index = (count // len(parameters) + 7) % len(parameters)
                if x_index == y_index:
                    y_index = (y_index + 1) % len(parameters)
                x, y = parameters[x_index], parameters[y_index]
                example = f"{fn}({a}, {b})" if kind not in {4, 5} else f"{fn}({a})"
                if kind == 0:
                    body, result = f"return {x} + {y}", a + b
                elif kind == 1:
                    body, result = f"return {x} - {y}", a - b
                elif kind == 2:
                    body, result = f"return {x} * {y}", a * b
                elif kind == 3:
                    body, result = f"if {x} > {y}:\n        return {x}\n    return {y}", max(a, b)
                elif kind == 4:
                    body, result = f"return {x} % 2 == 0", a % 2 == 0
                elif kind == 5:
                    body, result = f"return {x} ** 2", a * a
                elif kind == 6:
                    body, result = f"return ({x} + {y}) / 2", (a + b) / 2
                elif kind == 7:
                    body, result = f"return {x} % {y}", a % b
                elif kind == 8:
                    body, result = f"return {x} == {y}", a == b
                elif kind == 9:
                    body, result = f"return {x} ** {y}", a ** b
                elif kind == 10:
                    x = ["numbers", "values", "items", "entries", "data", "measurements", "scores", "parts", "amounts", "samples"][count % 10]
                    loop_var = ["item", "value", "number", "element", "entry", "sample", "part", "score", "amount", "datum"][count // 10 % 10]
                    accumulator = ["total", "sum_value", "result", "combined", "answer", "running_sum", "value_sum", "collected", "subtotal", "aggregate"][count // 100 % 10]
                    body, result = f"{accumulator} = 0\n    for {loop_var} in {x}:\n        {accumulator} += {loop_var}\n    return {accumulator}", a + b
                    example = f"{fn}([{a}, {b}])"
                else:
                    x = ["numbers", "values", "items", "entries", "data", "measurements", "scores", "parts", "amounts", "samples"][count % 10]
                    loop_var = ["item", "value", "number", "element", "entry", "sample", "part", "score", "amount", "datum"][count // 10 % 10]
                    accumulator = ["positive_count", "count", "matches", "found", "total_positive", "n_positive", "result_count", "qualifying", "positive_total", "counter"][count // 100 % 10]
                    body, result = f"{accumulator} = 0\n    for {loop_var} in {x}:\n        if {loop_var} > 0:\n            {accumulator} += 1\n    return {accumulator}", int(a > 0) + int(b > 0)
                    example = f"{fn}([{a}, {b}])"
                code = f"def {fn}({x}{', ' + y if kind not in {4, 5, 10, 11} else ''}):\n    {body}"
                intro = (unary_intros[count % len(unary_intros)].format(a=a) if kind in {4, 5}
                         else intros[count % len(intros)].format(a=a,b=b))
                answer = f"{intro}\n\n```python\n{code}\n```\nВызов {example} вернёт {result}."
                if kind == 4:
                    q = f"Напиши функцию Python, которая проверяет чётность числа {a}, и покажи результат."
                elif kind == 5:
                    q = f"Как написать функцию Python для квадрата числа {a}? Покажи проверку."
                elif kind == 10:
                    q = f"Покажи функцию Python, которая складывает числа в списке [{a}, {b}] с помощью цикла."
                elif kind == 11:
                    q = f"Напиши функцию Python, которая считает количество положительных чисел в списке [{a}, {b}]."
                else:
                    q = qforms[count % len(qforms)].format(task=task, a=a, b=b)
                self.add("programming", "ru", f"py:{kind}:{a}:{b}", [
                    {"role": "user", "content": q}, {"role": "assistant", "content": answer},
                ], {"type": "synthetic_computed", "generator": "python_function_patterns_v2", "task": task,
                    "expected": result, "inputs": [a] if kind in {4, 5} else [[a, b]] if kind in {10, 11} else [a, b],
                    "code": code, "function_name": fn})
                count += 1
                if count >= target:
                    return

    def generate_translation(self, target: int = 3200) -> None:
        # A small authored bilingual lexicon is composed only into simple, checked sentences.
        subjects = [("I", "Я", 1), ("You", "Ты", 2), ("We", "Мы", 3), ("They", "Они", 4)]
        verbs = [
            ("read", "читаю", "читаешь", "читаем", "читают"),
            ("open", "открываю", "открываешь", "открываем", "открывают"),
            ("see", "вижу", "видишь", "видим", "видят"),
            ("buy", "покупаю", "покупаешь", "покупаем", "покупают"),
            ("carry", "несу", "несёшь", "несём", "несут"),
            ("find", "нахожу", "находишь", "находим", "находят"),
            ("draw", "рисую", "рисуешь", "рисуем", "рисуют"),
            ("write", "пишу", "пишешь", "пишем", "пишут"),
        ]
        objects = [
            ("a book", "книгу"), ("a map", "карту"), ("a letter", "письмо"), ("a red apple", "красное яблоко"),
            ("a small house", "небольшой дом"), ("a new notebook", "новую тетрадь"), ("a green bag", "зелёную сумку"),
            ("a blue cup", "синюю чашку"), ("a short story", "короткий рассказ"), ("a paper plane", "бумажный самолётик"),
            ("a warm scarf", "тёплый шарф"), ("a clean window", "чистое окно"), ("a simple picture", "простую картину"),
            ("a wooden chair", "деревянный стул"), ("a small key", "маленький ключ"), ("a fresh tomato", "свежий помидор"),
            ("a useful guide", "полезный справочник"), ("a yellow flower", "жёлтый цветок"), ("a glass bottle", "стеклянную бутылку"),
            ("a warm sweater", "тёплый свитер"),
        ]
        locations = [("at home", "дома"), ("in the park", "в парке"), ("at the library", "в библиотеке"),
                     ("near the station", "возле станции"), ("in the garden", "в саду"), ("at school", "в школе")]
        time_phrases = [("in the morning", "утром"), ("in the evening", "вечером"),
                        ("after lunch", "после обеда"), ("on weekends", "по выходным")]
        compatible_objects = {
            0: [0, 1, 2, 8, 16], 1: [0, 5, 6, 11, 14, 18], 2: [0, 1, 3, 5, 6, 8, 11, 14, 17],
            3: [3, 5, 6, 9, 10, 14, 15, 17, 18, 19], 4: [5, 6, 9, 10, 13, 14, 15, 18, 19],
            5: [0, 1, 3, 5, 6, 8, 14, 15, 17], 6: [1, 4, 8, 12, 17], 7: [0, 2, 5, 8, 16],
        }
        styles = ["Translate into Russian: \"{en}.\"", "Переведи на русский: «{en}.»",
                  "Как по-русски сказать: \"{en}.\"", "Переведи эту фразу на английский: «{ru}.»",
                  "Translate into English: \"{ru}.\"", "Как сказать по-английски: «{ru}.»"]
        count = 0
        for si, (subj_en, subj_ru, person) in enumerate(subjects):
            verb_idx = person
            for vi, verb in enumerate(verbs):
                for oi in compatible_objects[vi]:
                    obj_en, obj_ru = objects[oi]
                    for li, (loc_en, loc_ru) in enumerate(locations):
                        for ti, (time_en, time_ru) in enumerate(time_phrases):
                            en = f"{subj_en} {verb[0]} {obj_en} {loc_en} {time_en}"
                            ru = f"{subj_ru} {verb[verb_idx]} {obj_ru} {loc_ru} {time_ru}"
                            style = styles[(si + vi + oi + li + ti) % len(styles)]
                            direction_en_to_ru = style in styles[0:3]
                            question = style.format(en=en, ru=ru)
                            answer = (ru if direction_en_to_ru else en) + "."
                            messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
                            if not self.add("translation", _lang(messages), f"translation:{si}:{vi}:{oi}:{li}:{ti}", messages,
                                            {"type": "synthetic_curated", "generator": "bilingual_sentence_composition_v2",
                                             "direction": "en_to_ru" if direction_en_to_ru else "ru_to_en",
                                             "pair_id": f"{si}-{vi}-{oi}-{li}-{ti}", "english_text": en + ".", "russian_text": ru + "."}):
                                continue
                            count += 1
                            if count >= target:
                                return

    def generate_instruction_following(self, target: int = 4200) -> None:
        rng = random.Random(SEED + 3)
        groups = [
            ("планеты", ["Меркурий", "Венера", "Земля", "Марс", "Юпитер", "Сатурн", "Уран", "Нептун"]),
            ("предметы для школы", ["тетрадь", "линейка", "ручка", "альбом", "ластик", "карандаш", "папка", "циркуль"]),
            ("фрукты", ["яблоко", "груша", "слива", "персик", "абрикос", "вишня", "банан", "апельсин"]),
            ("инструменты", ["молоток", "отвёртка", "плоскогубцы", "пила", "рулетка", "дрель", "ключ", "напильник"]),
            ("виды транспорта", ["автобус", "поезд", "велосипед", "самолёт", "трамвай", "корабль", "метро", "такси"]),
            ("животные", ["лиса", "ёж", "заяц", "медведь", "бобр", "волк", "белка", "барсук"]),
            ("музыкальные инструменты", ["флейта", "скрипка", "виолончель", "гитара", "кларнет", "труба", "арфа", "барабан"]),
            ("профессии", ["переводчик", "инженер", "врач", "садовник", "повар", "строитель", "водитель", "учитель"]),
        ]
        qforms = ["Назови три элемента из списка: {items}.", "Выбери первые три пункта и перечисли их через запятую: {items}.",
                  "Из этого набора выпиши три предмета: {items}.", "Коротко перечисли три позиции из списка: {items}."]
        answer_forms = ["{chosen}", "Вот три подходящих: {chosen}.", "Три пункта: {chosen}.", "Можно назвать такие: {chosen}.",
                        "Например: {chosen}.", "Из списка подойдут {chosen}.", "Возьму эти три: {chosen}.",
                        "Подходят, например, {chosen}.", "В качестве ответа: {chosen}.", "Выбираю {chosen}.",
                        "Три варианта — {chosen}.", "Можно выбрать {chosen}.", "Подойдут такие пункты: {chosen}.",
                        "Мой выбор: {chosen}.", "Один из возможных наборов: {chosen}.", "Вот выбранные элементы: {chosen}.",
                        "Среди вариантов есть {chosen}.", "Назову {chosen}.", "Подходящий набор: {chosen}.",
                        "Три из перечисленных: {chosen}."]
        for idx in range(target):
            label, pool = groups[idx % len(groups)]
            chosen = rng.sample(pool, k=6)
            chosen_order = chosen[:3]
            items = ", ".join(chosen)
            selection = ", ".join(chosen_order)
            q = qforms[idx % len(qforms)].format(items=items)
            a = answer_forms[(idx // len(qforms)) % len(answer_forms)].format(chosen=selection)
            # Change the user constraint when the output says exactly which items to use.
            if idx % 4 == 1:
                q = f"Перечисли ровно первые три из этих {label}: {items}. Ответь одной строкой."
            self.add("instruction_following", "ru", f"instruction:list:{idx}:{_sha(items)[:8]}", [
                {"role": "user", "content": q}, {"role": "assistant", "content": a},
            ], {"type": "synthetic_computed", "generator": "list_selection_v2", "expected_items": chosen_order})

        # Distinct sorting requests: outputs are computed, and every prompt carries its own input list.
        sorted_rng = random.Random(SEED + 30)
        seen_lists: set[tuple[int, ...]] = set()
        sort_q = ["Упорядочи эти числа по возрастанию: {values}.", "Расставь числа от меньшего к большему: {values}.",
                  "Отсортируй список чисел по возрастанию: {values}.", "Запиши числа в порядке возрастания: {values}.",
                  "Поставь эти значения по порядку, начиная с наименьшего: {values}.", "Переставь числа от минимального к максимальному: {values}.",
                  "Расположи числа по возрастанию: {values}.", "Как будет выглядеть список после сортировки по возрастанию: {values}?",
                  "Отсортируй значения по возрастанию и запиши результат: {values}.", "Сделай сортировку чисел по возрастанию: {values}.",
                  "Напиши этот набор в возрастающем порядке: {values}.", "Расставь значения от самого маленького до самого большого: {values}.",
                  "От малого к большому перечисли: {values}.", "Укажи возрастающую последовательность из этих чисел: {values}.",
                  "Приведи список к возрастающему порядку: {values}.", "Сначала меньшее число, затем большее: отсортируй {values}.",
                  "Расположи элементы числового списка по возрастанию: {values}.", "Каков результат сортировки {values} по возрастанию?",
                  "Перечисли числа в порядке увеличения: {values}.", "Сортируй набор от наименьшего значения к наибольшему: {values}."]
        sort_a = ["По возрастанию: {result}.", "От меньшего к большему: {result}.",
                  "Получится такой порядок: {result}.", "Отсортированный список: {result}.",
                  "Сначала идут меньшие числа: {result}.", "Правильная последовательность — {result}.",
                  "В возрастающем порядке: {result}.", "Итог сортировки: {result}.",
                  "Я бы записал их так: {result}.", "Числа после сортировки: {result}.",
                  "От самого маленького до самого большого: {result}.", "Вот последовательность от меньшего к большему: {result}.",
                  "После перестановки получится: {result}.", "Нужный порядок такой: {result}.",
                  "Ряд по возрастанию выглядит так: {result}.", "Результат: {result}.",
                  "Возрастает так: {result}.", "Готовый отсортированный ряд: {result}.",
                  "Запись в нужном порядке: {result}.", "От минимального к максимальному: {result}."]
        for idx in range(3500):
            values = tuple(sorted_rng.sample(range(-500, 1001), k=6))
            if values in seen_lists:
                continue
            seen_lists.add(values)
            shown = list(values)
            sorted_rng.shuffle(shown)
            sorted_values = tuple(sorted(values))
            shown_text = ", ".join(map(str, shown))
            result_text = ", ".join(map(str, sorted_values))
            self.add("instruction_following", "ru", f"instruction:sort:{','.join(map(str, values))}", [
                {"role": "user", "content": sort_q[idx % len(sort_q)].format(values=shown_text)},
                {"role": "assistant", "content": sort_a[idx % len(sort_a)].format(result=result_text)},
            ], {"type": "synthetic_computed", "generator": "numeric_sort_instruction_v1", "input": shown, "expected": list(sorted_values)})

    def generate_smalltalk(self) -> None:
        # Hand-authored short exchanges; each reply is phrased for its own user intent.
        pairs = [
            ("Здравствуй!", "Здравствуйте! Что обсудим?"),
            ("Привет, ты тут?", "Да, я здесь. Чем могу помочь?"),
            ("Доброе утро!", "Доброе утро! Что хочешь узнать сегодня?"),
            ("Добрый вечер.", "Добрый вечер! Как прошёл твой день?"),
            ("Рад снова написать.", "И я рад продолжить разговор. Что у тебя на уме?"),
            ("Как проходит твой день?", "Спасибо, я готов общаться и помогать. А у тебя как?"),
            ("Как сам?", "Я на связи и готов отвечать. Что нового у тебя?"),
            ("Чем занят?", "Сейчас отвечаю на твои сообщения. О чём поговорим?"),
            ("Ты можешь немного помочь?", "Конечно. Расскажи, что нужно сделать."),
            ("Мне нужен совет по учёбе.", "Давай разберёмся. С каким предметом или заданием возникла трудность?"),
            ("Можно задать короткий вопрос?", "Конечно, задавай. Постараюсь ответить ясно и коротко."),
            ("Я не знаю, с чего начать.", "Расскажи, чего хочешь добиться, и мы выберем первый небольшой шаг."),
            ("У меня есть идея для рассказа.", "Здорово! Опиши её в паре предложений, и я помогу развить сюжет."),
            ("Я немного устал.", "Тогда можно сделать короткую паузу. Хочешь пока поговорить о чём-нибудь лёгком?"),
            ("Сегодня хорошая погода.", "Похоже на приятный день. Ты уже успел немного прогуляться?"),
            ("Спасибо за объяснение.", "Пожалуйста. Если останется непонятный момент, разберём его отдельно."),
            ("Благодарю, этого достаточно.", "Хорошо, рад, что ответ пригодился. Обращайся, если появится ещё вопрос."),
            ("Ты здорово выручил.", "Рад помочь. Если понадобится продолжить, просто напиши."),
            ("Ладно, до связи.", "До связи! Пусть остаток дня пройдёт спокойно."),
            ("Мне пора идти.", "Понял. Хорошего дня, будем на связи."),
            ("Пожелай мне удачи на экзамене.", "Удачи на экзамене! Внимательно читай задания и не торопись."),
            ("Я волнуюсь перед выступлением.", "Это вполне понятно. Сделай спокойный вдох и начни с первой знакомой мысли."),
            ("Можешь ответить одним предложением?", "Да, постараюсь уложить ответ в одно ясное предложение."),
            ("Расскажи что-нибудь интересное.", "Ось Земли немного наклонена, поэтому в течение года в разных полушариях меняются времена года."),
            ("Ты запомнишь тему нашего разговора?", "В рамках этой беседы я могу учитывать уже написанные сообщения."),
            ("Ты настоящий человек?", "Нет, я языковая модель MyLLM и создаю текстовые ответы на сообщения."),
            ("Чем ты можешь быть полезен?", "Могу помочь с короткими объяснениями, текстом, переводом, простой математикой и базовым Python."),
            ("Можно попросить тебя проверить фразу?", "Да. Пришли предложение и скажи, что проверить: грамматику, орфографию или ясность."),
            ("Я хочу потренировать английский.", "Давай! Напиши фразу по-английски, а я помогу проверить её."),
            ("Давай поговорим о космосе.", "Хорошо. Тебе интереснее планеты, звёзды или полёты в космос?"),
            ("Мне скучно, предложи простое занятие.", "Можно выбрать короткое дело: прочитать рассказ, прогуляться или собрать небольшой пазл."),
            ("Хочу написать короткое поздравление.", "Кому оно адресовано и по какому поводу? Я помогу подобрать тёплые слова."),
            ("Hi!", "Hello! What would you like to talk about?"),
            ("How is your day going?", "I am here and ready to help. How is your day?"),
            ("Could you help me with a short translation?", "Sure. Send me the sentence and tell me which language you need."),
            ("Thanks, that answered my question.", "You are welcome. Feel free to ask if another question comes up."),
            ("I need a simple Python example.", "Of course. Tell me what the code should do, and I will keep the example simple."),
            ("I am not sure how to begin.", "Tell me what you want to accomplish, and we can choose a manageable first step."),
            ("Good night.", "Good night! I hope you get some good rest."),
            ("See you later.", "See you later. Take care!")
        ]
        for index, (question, answer) in enumerate(pairs):
            lang = "en" if any("a" <= char.lower() <= "z" for char in question) and not any("а" <= char.lower() <= "я" for char in question) else "ru"
            self.add("conversation", lang, f"smalltalk-v2:{index}", [
                {"role": "user", "content": question}, {"role": "assistant", "content": answer}
            ], {"type": "synthetic_curated", "generator": "hand_authored_smalltalk_v2"})

        dialogues = [
            [
                {"role": "user", "content": "Привет, можно спросить про Москву?"},
                {"role": "assistant", "content": "Конечно. Что именно тебя интересует?"},
                {"role": "user", "content": "Почему её называют столицей России?"},
                {"role": "assistant", "content": "Москва — столица России; в городе находятся федеральные органы власти страны."},
            ],
            [
                {"role": "user", "content": "Помоги с простым Python."},
                {"role": "assistant", "content": "Хорошо. Какую задачу должна решать программа?"},
                {"role": "user", "content": "Мне нужна функция, которая складывает два числа."},
                {"role": "assistant", "content": "Функция `def add(a, b): return a + b` принимает два числа и возвращает их сумму."},
            ],
            [
                {"role": "user", "content": "Переведи короткую фразу на русский."},
                {"role": "assistant", "content": "Пришли фразу, и я переведу её."},
                {"role": "user", "content": "Thank you for your help."},
                {"role": "assistant", "content": "Это значит: «Спасибо за помощь»."},
            ],
            [
                {"role": "user", "content": "Назови несколько планет."},
                {"role": "assistant", "content": "Например, Меркурий, Земля и Юпитер."},
                {"role": "user", "content": "Какая из них самая большая?"},
                {"role": "assistant", "content": "Юпитер — крупнейшая планета Солнечной системы."},
            ],
            [
                {"role": "user", "content": "Сколько будет 8 умножить на 7?"},
                {"role": "assistant", "content": "8 × 7 = 56."},
                {"role": "user", "content": "Как это быстро проверить?"},
                {"role": "assistant", "content": "Можно сложить 8 семь раз: 8 + 8 + 8 + 8 + 8 + 8 + 8 = 56."},
            ],
            [
                {"role": "user", "content": "Проверь предложение: «Мальчик читает книга»."},
                {"role": "assistant", "content": "В конце нужно изменить форму слова «книга»."},
                {"role": "user", "content": "Как будет правильно?"},
                {"role": "assistant", "content": "Правильно: «Мальчик читает книгу»."},
            ],
        ]
        for index, messages in enumerate(dialogues):
            self.add("multi_turn", "ru", f"hand-authored-smalltalk-multiturn-v2:{index}", messages,
                     {"type": "synthetic_curated", "generator": "hand_authored_smalltalk_multiturn_v2"})

    def generate_multiturn_and_conversation(self, target: int = 6400) -> None:
        activities = ["почитать книгу", "погулять в парке", "заняться рисованием", "приготовить простой ужин", "разобрать фотографии",
                      "выучить несколько новых слов", "посадить цветы", "собрать модель", "покататься на велосипеде", "послушать музыку",
                      "навести порядок на столе", "сходить в библиотеку", "попробовать новый рецепт", "сделать зарядку", "написать открытку",
                      "посмотреть научный фильм", "позвонить старому другу", "начать вести дневник", "собрать пазл", "пройтись у реки",
                      "выбрать книгу для чтения", "потренировать английские слова", "испечь печенье", "посетить музей", "сделать бумажную поделку",
                      "обновить плейлист", "разобрать рабочие заметки", "поиграть в настольную игру", "посмотреть на звёзды", "помыть велосипед",
                      "составить список дел", "посадить зелень в горшок", "порисовать акварелью", "попрактиковать Python", "приготовить чай",
                      "сходить за продуктами", "проветрить комнату", "сделать фотографии", "прочитать короткий рассказ", "попробовать вязание"]
        time_words = ["сегодня", "завтра", "в выходной", "после занятий", "вечером", "в субботу", "на каникулах", "после работы"]
        openers = ["Мне хочется {activity} {when}.", "Я подумываю {activity} {when}. Как тебе идея?", "Как тебе идея {activity} {when}?",
                   "Планирую {activity} {when}, но пока сомневаюсь."]
        replies = ["Звучит неплохо. Попробуй {activity} {when}, если будет желание.",
                   "Если тебе интересно, найди удобный момент и попробуй {activity} {when}.",
                   "Можно начать с малого: попробуй {activity} {when} ненадолго, а потом реши, хочется ли продолжать.",
                   "Если расписание позволяет, выдели время, чтобы {activity} {when}.",
                   "Неплохая идея, если тебе этого хочется. Можно {activity} {when} без спешки.",
                   "Звучит приятно. Для начала попробуй {activity} {when} в спокойном темпе.",
                   "Если сомневаешься, запланируй {activity} {when} на короткое время.",
                   "Попробуй {activity} {when}; это может стать хорошей небольшой паузой.",
                   "Не нужно строить большой план: попробуй {activity} {when} просто для удовольствия.",
                   "Если хочется переключиться, можно {activity} {when} и посмотреть, понравится ли.",
                   "Звучит выполнимо. Начни с короткого шага, чтобы {activity} {when}.",
                   "Похоже на спокойную идею: попробуй {activity} {when} и не торопись.",
                   "Можно попробовать {activity} {when} в удобном темпе и прислушаться к своим ощущениям.",
                   "Если найдётся свободное время, попробуй {activity} {when}; планы всегда можно скорректировать.",
                   "Для начала достаточно {activity} {when} без завышенных ожиданий.",
                   "Главное — выбрать удобный момент. Попробуй {activity} {when} и оцени, понравилось ли.",
                   "Если тебе интересно, попробуй {activity} {when} и посмотри, понравится ли.",
                   "Сделай план посильным: попробуй {activity} {when} совсем недолго.",
                   "Можно заранее решить, сколько времени уделить тому, чтобы {activity} {when}.",
                   "Поставь небольшую цель: попробуй {activity} {when} в комфортном для тебя темпе.",
                   "Если хочется, попробуй {activity} {when}; не обязательно добиваться идеального результата.",
                   "Можно проверить эту идею на практике: {activity} {when} короткое время.",
                   "Если день будет свободным, попробуй {activity} {when} и посмотри, понравится ли.",
                   "Выбери подходящий момент и попробуй {activity} {when} без спешки.",
                   "Сначала оцени, сколько у тебя времени, а затем попробуй {activity} {when}.",
                   "План можно упростить: попробуй {activity} {when} только один раз.",
                   "Если хочется нового занятия, {activity} {when} может оказаться хорошим вариантом.",
                   "Дай себе возможность попробовать: {activity} {when} без строгих ожиданий.",
                   "Начни с небольшого шага и попробуй {activity} {when}, если будет настроение.",
                   "Можно заранее подготовиться, чтобы спокойно {activity} {when}.",
                   "Если планы не изменятся, попробуй {activity} {when} и оцени результат.",
                   "Для начала выбери короткий промежуток и попробуй {activity} {when}.",
                   "Хороший способ проверить идею — попробовать {activity} {when} ненадолго.",
                   "Необязательно решать всё заранее: попробуй {activity} {when} и выбери дальше по ощущениям.",
                   "Если тебе будет удобно, попробуй {activity} {when}; при желании продолжишь позже.",
                   "Можно отвести на это немного времени и попробовать {activity} {when}.",
                   "Пусть это будет простой план: попробуй {activity} {when} и не торопись.",
                   "Попробуй {activity} {when}, а после оцени, хочется ли повторить.",
                   "Если есть сомнения, начни с самого лёгкого варианта: {activity} {when}.",
                   "Подумай, что тебе понадобится, и попробуй {activity} {when} в удобное время."]
        followups = ["С чего начать?", "А как сделать это проще?", "Что подготовить заранее?", "Как не бросить на полпути?"]
        suggestions = ["Если хочешь {activity} {when}, заранее подготовь всё необходимое.",
                       "Чтобы {activity} {when}, выбери удобное место и выдели достаточно времени.",
                       "Начни с короткого шага: попробуй {activity} {when} и посмотри, как пойдёт.",
                       "Попробуй {activity} {when} в спокойном темпе; спешить не обязательно.",
                       "Если сомневаешься, попробуй {activity} {when} ненадолго и оцени, хочется ли продолжить.",
                       "Перед тем как {activity} {when}, приготовь всё, что может понадобиться.",
                       "Можно {activity} {when} понемногу, а затем решить, хочется ли продолжать.",
                       "Поставь небольшую цель: попробуй {activity} {when} в течение короткого времени.",
                       "Чтобы было проще {activity}, раздели подготовку на несколько небольших шагов.",
                       "Постарайся {activity} {when} без завышенных ожиданий.",
                       "Если будешь заниматься этим {when}, начни с самого лёгкого шага для плана «{activity}».",
                       "Выдели немного времени, чтобы {activity} {when}, и оставь запас на отдых.",
                       "Подготовь место для занятия «{activity}», а потом попробуй {activity} {when}.",
                       "Если планы на {when} изменятся, перенеси занятие «{activity}» на другой день.",
                       "Не перегружай план: попробуй {activity} {when} и остановись, если устанешь.",
                       "Подготовь первый шаг для того, чтобы {activity} {when}.",
                       "Чтобы не откладывать, выбери удобное время и попробуй {activity} {when}.",
                       "Договорись с собой попробовать {activity} {when} всего несколько минут.",
                       "Попробуй {activity} {when}; если времени не хватит, перенесёшь остальное.",
                       "Включи в план немного времени, чтобы {activity} {when}, и не перегружай себя.",
                       "Выбери удобное время и попробуй {activity} {when} без спешки.",
                       "Сначала подготовь всё необходимое, затем попробуй {activity} {when}.",
                       "Если планы изменятся, перенеси {activity} на другое время, не отменяя идею совсем.",
                       "Начни с самого простого: попробуй {activity} {when}, а дальше решишь по настроению.",
                       "Чтобы {activity} {when}, заранее убери одно отвлечение и подготовь материалы.",
                       "Необязательно делать всё сразу: попробуй {activity} {when} в удобном объёме.",
                       "Сделай план гибким: занятие «{activity}» можно сократить или перенести.",
                       "Если не знаешь, с чего начать, выбери один простой шаг для того, чтобы {activity} {when}.",
                       "Убери лишние отвлечения перед тем, как {activity} {when}.",
                       "Не требуй от себя многого: попробуй {activity} {when} столько, сколько будет комфортно.",
                       "Чтобы {activity} {when}, заранее проверь, всё ли нужное у тебя под рукой.",
                       "Можно договориться с кем-нибудь о компании, если хочется {activity} {when}.",
                       "Чтобы начать {activity} {when}, выбери один простой шаг и не спеши.",
                       "Если появится препятствие, измени план, но оставь возможность {activity} {when}.",
                       "Выбери реалистичный объём: попробуй {activity} {when} без лишней нагрузки.",
                       "Поставь напоминание о том, чтобы {activity} {when}, если боишься забыть.",
                       "Можно сначала подготовить материалы, а потом спокойно {activity} {when}.",
                       "Если устанешь, сделай паузу и вернись к плану «{activity}» позже.",
                       "Заранее выбери время для того, чтобы {activity} {when}, и оставь немного свободного времени.",
                       "Для комфортного старта подготовь всё для занятия «{activity}» {when}."]
        starts = ["Привет! Мне хочется {activity} {when}. Что скажешь?",
                  "Подскажи, пожалуйста: я планирую {activity} {when}.",
                  "Как тебе идея {activity} {when}?",
                  "Я подумываю {activity} {when}, но пока сомневаюсь. Поможешь оценить план?"]
        for idx in range(target):
            activity = activities[idx % len(activities)]
            when = time_words[(idx // len(activities)) % len(time_words)]
            variant = (idx // (len(activities) * len(time_words) // 2)) % 40
            prompt = openers[variant % len(openers)].format(activity=activity, when=when)
            response = replies[variant % len(replies)].format(activity=activity, when=when)
            if idx % 2 == 0:
                suggestion = suggestions[variant % len(suggestions)].format(activity=activity, when=when)
                messages = [
                    {"role": "user", "content": starts[variant % len(starts)].format(activity=activity, when=when)},
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": followups[variant % len(followups)]},
                    {"role": "assistant", "content": suggestion},
                ]
                category = "multi_turn"
            else:
                messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
                category = "conversation"
            self.add(category, "ru", f"chat:{idx}:{activity}:{when}:{variant}", messages,
                     {"type": "synthetic_curated", "generator": "contextual_everyday_dialogue_v3", "scenario": activity})

    def finish(self, out_dir: Path) -> dict[str, Any]:
        rng = random.Random(SEED)
        evaluation_users = {normalize_for_dedup(question) for question, _ in EVAL}
        before = len(self.rows)
        self.rows = [row for row in self.rows if not any(
            message["role"] == "user" and normalize_for_dedup(message["content"]) in evaluation_users
            for message in row["messages"]
        )]
        self.rejections["fixed_evaluation_prompt_leak"] += before - len(self.rows)
        rng.shuffle(self.rows)
        # Merge semantic groups that reuse the same exact user prompt so no prompt leaks across splits.
        parents = {str(row["group_id"]): str(row["group_id"]) for row in self.rows}

        def find(group: str) -> str:
            while parents[group] != group:
                parents[group] = parents[parents[group]]
                group = parents[group]
            return group

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[max(left_root, right_root)] = min(left_root, right_root)

        prompt_owner: dict[str, str] = {}
        for row in self.rows:
            group = str(row["group_id"])
            for message in row["messages"]:
                if message["role"] != "user":
                    continue
                prompt_key = normalize_for_dedup(message["content"])
                previous = prompt_owner.setdefault(prompt_key, group)
                union(group, previous)
        group_rows: dict[str, list[dict[str, Any]]] = {}
        for row in self.rows:
            group_rows.setdefault(find(str(row["group_id"])), []).append(row)
        group_order = list(group_rows)
        rng.shuffle(group_order)
        train: list[dict[str, Any]] = []
        validation: list[dict[str, Any]] = []
        total = len(self.rows)
        validation_limit = max(1, round(total * 0.05))
        for group in group_order:
            target = validation if len(validation) < validation_limit else train
            target.extend(group_rows[group])
        rng.shuffle(train)
        rng.shuffle(validation)
        out_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        for split, rows in (("train", train), ("validation", validation)):
            path = out_dir / f"{split}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            files[split] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "examples": len(rows), "bytes": path.stat().st_size}
        eval_path = out_dir / "evaluation.jsonl"
        with eval_path.open("w", encoding="utf-8", newline="\n") as handle:
            for i, (question, answer) in enumerate(EVAL):
                row = {"id": f"eval-{i+1:02d}", "category": "evaluation", "language": "ru",
                       "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
                       "source": {"type": "fixed_internal_evaluation", "source_id": "sft-v2-fixed-13"}}
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        files["evaluation"] = {"path": eval_path.name, "sha256": hashlib.sha256(eval_path.read_bytes()).hexdigest(), "examples": len(EVAL), "bytes": eval_path.stat().st_size}
        source_counts = Counter(row["source"]["type"] for row in self.rows)
        category_counts = Counter(row["category"] for row in self.rows)
        external_counts = Counter(row["source"].get("source_id") for row in self.rows if row["source"].get("type") == "external_dataset")
        legacy_inputs = []
        for name in ("train.jsonl", "validation.jsonl"):
            path = ROOT / "data" / "sft" / name
            if path.is_file():
                legacy_inputs.append({"path": f"data/sft/{name}", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
        manifest = {
            "dataset_name": "MyLLM V2 109M SFT candidate",
            "version": "sft-v2-109m-2026-10-05",
            "status": "prepared_not_trained",
            "training_started": False,
            "base_checkpoint": "checkpoints/v2-pretraining/step_126783",
            "base_checkpoint_immutable": True,
            "base_checkpoint_lock": "checkpoints/v2-pretraining/FINAL_BASE_LOCK.json",
            "pretrained_weights_used": False,
            "tokenizer": {"path": "tokenizer/tokenizer.json", "vocab_size": 32000},
            "chat_template": "src.sft_data.encode_messages / encode_chat_prompt; <bos><|user|>...<|assistant|>...<eos>",
            "loss_policy": "assistant message text plus terminal EOS only; system/user/role markers masked to -100",
            "packing": False,
            "context_length": 1024,
            "split_strategy": "stable group_id grouping; 95/5 target; fixed evaluation held out",
            "seed": SEED,
            "dataset": {"conversations": len(self.rows), "categories": dict(sorted(category_counts.items())),
                        "source_types": dict(sorted(source_counts.items())),
                        "external_source_examples": dict(sorted(external_counts.items()))},
            "files": files,
            "sources": [
                {"id": "rubq_2_0", "name": "RuBQ 2.0 Russian question answering dataset",
                 "url": "https://github.com/vladislavneon/RuBQ", "license": "CC BY-SA 4.0",
                 "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                 "attribution": "vladislavneon/RuBQ contributors",
                 "scope": "Only question_text and single-answer answer_text imported; answers[0] Russian aliases are used for spelling/capitalization normalization, and the formatter clarifies units or repairs audited malformed phrasing. Paragraphs, queries, and contexts were not used.",
                 "examples_in_dataset": external_counts.get("rubq_2_0", 0),
                 "source_files": [
                     {"path": "sources/rubq/RuBQ_2.0_dev.json", "sha256": "6CD20DCFAD3404736691C5DCEE271D19F579B6C5755A1AAD4C4DE644BC9DFD05", "bytes": 1232673},
                     {"path": "sources/rubq/RuBQ_2.0_test.json", "sha256": "63D9E32A1BDEFB2DE713A9CCF3A0F9F0A4A48F2FDDDD35B3908D6846B63DC85B", "bytes": 4980094}
                 ], "paragraphs_imported": 0},
                {"id": "reaudited_synthetic_v1_subset", "name": "Selected synthetic examples from prior MyLLM V1 SFT data",
                 "url": None, "license": "Original project-authored/computed examples; no external text imported",
                 "scope": "Each row rechecked; all Wikimedia rows excluded; exact assistant response and conversation dedup repeated.",
                 "examples_in_dataset": dict(self.legacy_kept), "input_file_checksums": legacy_inputs},
                {"id": "v2_authored_computed", "name": "New V2 task-specific authored/computed data",
                 "url": None, "license": "Project-authored synthetic examples; arithmetic and code-trace outputs computed deterministically",
                 "scope": "No teacher model; no raw web text; no personal conversations; no secrets.",
                 "examples_in_dataset": len(self.rows) - sum(self.legacy_kept.values())},
            ],
            "legacy_v1_audit": {"used": True, "wikimedia_examples_imported": 0,
                                "kept_by_category": dict(self.legacy_kept),
                                "rejected_reasons": dict(self.legacy_rejected)},
            "generation": {"script": "scripts/build_sft_v2_109m_dataset.py", "synthetic_teacher_model": False},
            "build_rejections": dict(self.rejections),
            "external_dataset_audit": {"rubq_files_loaded": ["RuBQ_2.0_dev.json", "RuBQ_2.0_test.json"],
                                       "only_direct_qa_fields": True, "paragraphs_loaded_or_used": False,
                                       "single_answer_rows_only": True,
                                       "manual_fact_review": "pending stratified final-candidate sample review; audit tool records sampled UID decisions in rubq_manual_review_log.json"},
            "quality_gates": {"min_total_examples": 50000, "target_ru_percent": "85-95", "target_en_percent": "5-15",
                              "exact_assistant_duplicate_policy": "globally rejected at build time",
                              "manual_sample_review": "pending; review the deterministic stratified RuBQ sample generated by scripts/audit_rubq_sft_v2.py; full row-by-row review is not required",
                              "editorial_review_required_before_training": True},
        }
        manifest_path = out_dir / "dataset_manifest.yaml"
        manifest_path.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False, width=110), encoding="utf-8")
        (out_dir / "build_rejections.json").write_text(json.dumps({"legacy": dict(self.legacy_rejected), "new": dict(self.rejections)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a separate V2 SFT candidate without training the model.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "sft_v2")
    parser.add_argument("--math-examples", type=int, default=7000)
    parser.add_argument("--explanation-examples", type=int, default=4500)
    parser.add_argument("--programming-examples", type=int, default=6000)
    parser.add_argument("--translation-examples", type=int, default=2800)
    parser.add_argument("--instruction-examples", type=int, default=5000)
    # Keep the daily-planning generator as a supporting source, not the
    # dominant voice of the assistant. The target retains a broad set of
    # varied scenarios while reducing repeated advice phrasing.
    parser.add_argument("--single-turn-examples", type=int, default=9000)
    parser.add_argument("--multi-turn-examples", type=int, default=5000)
    args = parser.parse_args()
    builder = DatasetBuilder()
    builder.import_reaudited_synthetic_v1()
    builder.import_rubq_qa()
    builder.generate_geography_and_science()
    builder.generate_math(args.math_examples)
    builder.generate_explanations(args.explanation_examples)
    builder.generate_programming(args.programming_examples)
    builder.generate_translation(args.translation_examples)
    builder.generate_instruction_following(args.instruction_examples)
    builder.generate_smalltalk()
    from src.sft_v2_curated import add_english_conversation, add_language_skill, add_long_context_examples, add_natural_dialogues
    add_natural_dialogues(builder, args.single_turn_examples, args.multi_turn_examples)
    add_long_context_examples(builder)
    add_language_skill(builder)
    add_english_conversation(builder)
    manifest = builder.finish(args.output_dir)
    print(f"Dataset directory: {args.output_dir}")
    print(f"Total: {manifest['dataset']['conversations']}")
    print(f"Categories: {manifest['dataset']['categories']}")
    print(f"Sources: {manifest['dataset']['source_types']}")
    print(f"Legacy V1 synthetic kept: {dict(builder.legacy_kept)}")
    print(f"Legacy rejected: {dict(builder.legacy_rejected)}")
    print(f"Build rejections: {dict(builder.rejections)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
