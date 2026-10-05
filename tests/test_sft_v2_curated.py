import re

from src.sft_v2_curated import (
    FOLLOWUP_CONTEXT_TIES,
    add_natural_dialogues,
)


class _RecordingBuilder:
    def __init__(self):
        self.rows = []

    def add(self, category, language, group_id, messages, source):
        self.rows.append({
            "category": category,
            "language": language,
            "group_id": group_id,
            "messages": messages,
            "source": source,
        })
        return True


def test_curated_planning_dialogues_keep_turns_contextual_and_grammatical():
    builder = _RecordingBuilder()
    add_natural_dialogues(builder, single_target=128, multi_target=256)

    assert len(FOLLOWUP_CONTEXT_TIES) == 16
    assert all(len(variants) == 4 for variants in FOLLOWUP_CONTEXT_TIES)
    assert len(builder.rows) == 384
    for row in builder.rows:
        messages = row["messages"]
        expected_roles = ["user", "assistant"] * (2 if row["category"] == "multi_turn" else 1)
        assert [message["role"] for message in messages] == expected_roles
        text = "\n".join(message["content"].casefold() for message in messages)
        assert not re.search(r"\bна\s+в\s+выходной\b", text)
        assert not re.search(r"\bв\s+в\s+субботу\b", text)
        assert "для этого выбери время, чтобы" not in text
