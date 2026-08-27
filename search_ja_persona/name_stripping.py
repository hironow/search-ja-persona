"""Person-name removal for embedding inputs.

Vector search chases the persona's own name (geographic surnames pull
queries toward the wrong prefecture, a Mr. 温泉 tops hot-spring queries),
so names are stripped from the text that gets embedded. Stored text is
untouched: BM25 name lookup and display keep working.

The dataset has no name column; every persona field opens with the name
("姓 名は、…" or unspaced "西口甲一は…"), so the name is detected from
those openings and accepted only when at least two fields agree — a
single-field candidate ("東京は…"-style subjects) is never trusted, and
conflicting candidates make the whole record a no-op.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

_STOPWORD_SUBJECTS = frozenset(
    {"彼", "彼女", "彼ら", "彼女ら", "私", "本人", "同氏", "自身", "誰も"}
)

_SPACED_OPENING = re.compile(
    r"^([^\s、。「」]{1,6}[ 　][^\s、。「」]{1,8})(（[^）]{0,15}）)?は"
)
_UNSPACED_OPENING = re.compile(r"^([^\s、。「」]{2,8})(（[^）]{0,15}）)?は")

# Removal spans: the subject form takes the optional age parenthetical,
# the particle, and a trailing comma with it; bare mentions need a
# following particle/punctuation boundary so an unspaced name cannot be
# carved out of the middle of a longer word.
_BOUNDARY = "はがのをにへとでもや、。\\s氏さ"


def detect_person_name(field_texts: Sequence[str]) -> str | None:
    """Return the persona's full name, or None when not confidently found."""

    counts: Counter[str] = Counter()
    for text in field_texts:
        if not text:
            continue
        match = _SPACED_OPENING.match(text) or _UNSPACED_OPENING.match(text)
        if match and match.group(1) not in _STOPWORD_SUBJECTS:
            counts[match.group(1)] += 1
    if not counts:
        return None
    (top, top_count), *_rest = counts.most_common(1)
    if top_count < 2:
        return None
    if len([c for c, n in counts.items() if n == top_count]) > 1:
        return None
    return top


def strip_person_names(field_texts: Sequence[str]) -> list[str]:
    """Return copies of the texts with the detected name removed."""

    name = detect_person_name(field_texts)
    if name is None:
        return list(field_texts)

    escaped = re.escape(name)
    subject = re.compile(escaped + r"(（[^）]{0,15}）)?(は|が)、?")
    bare = re.compile(escaped + rf"(?=[{_BOUNDARY}]|$)")

    stripped: list[str] = []
    for text in field_texts:
        if not text:
            stripped.append(text)
            continue
        cleaned = subject.sub("", text)
        cleaned = bare.sub("", cleaned)
        stripped.append(cleaned)
    return stripped
