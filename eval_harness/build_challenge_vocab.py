#!/usr/bin/env python3
"""Mine the challenge's object vocabulary from all 75 training questions.

Why: SysNav's semantic_mapping/config/objects.yaml is tuned to its own
benchmarks and *comments out* exactly the classes the challenge leans on
(pillow, window, door, refrigerator, table, ...). The open-vocab detector
(YOLO-World/YOLOe) only finds what it is prompted with, so we rebuild the
prompt vocabulary from the actual questions.

What it does (pure text-mining, offline):
  1. Read every numerical / object_reference / instruction_following question.
  2. spaCy noun-chunk extraction -> object phrases. We KEEP noun-noun
     compound modifiers that name the object kind ("coffee table",
     "potted plant", "door frame", "wall lamp") and DROP queried attributes
     (colors/sizes/materials, reusing reasoning.decomposition._ATTRIBUTES),
     determiners, and numerals. Head nouns are singularized.
  3. Filter out non-detectable abstract/structural nouns ("number", "path",
     "side", ...).
  4. Emit:
       - eval_harness/challenge_vocab_report.txt   (ranked phrases + counts)
       - ai_module/src/vln_orchestrator/config/challenge_classes.yaml
         (same schema as objects.yaml; point detection_node's `object_file`
          param at this on the Jazzy box).

Run: python3 eval_harness/build_challenge_vocab.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import spacy

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "questions" / "questions.json"
REPORT = ROOT / "eval_harness" / "challenge_vocab_report.txt"
YAML_OUT = (
    ROOT / "ai_module" / "src" / "vln_orchestrator" / "config" / "challenge_classes.yaml"
)

# Reuse the attribute set from the reasoning module so "red pillow" -> pillow
# while "potted plant" / "folding screen" keep their object-kind modifier.
sys.path.insert(0, str(ROOT / "ai_module" / "src" / "vln_orchestrator"))
from vln_orchestrator.reasoning.decomposition import _ATTRIBUTES  # noqa: E402

# Nouns that are abstract / relational / structural-but-not-an-instance — never
# useful as detector prompts.
NON_OBJECT_NOUNS = {
    "number", "path", "front", "back", "side", "top", "bottom", "middle",
    "center", "centre", "left", "right", "way", "room", "area", "corner",
    "end", "edge", "row", "pair", "group", "one", "ones", "thing", "things",
    "side", "part", "place", "amount", "set",
}

# Quantifiers/determiners spaCy sometimes tags as amod ("how many pillows");
# never part of an object name, so strip them from phrases.
QUANTIFIERS = {
    "many", "several", "few", "some", "all", "both", "each", "every", "much",
    "more", "most", "any", "no", "another", "other", "such",
}

# Extra prompts to append for a given mined phrase. Open-vocab detectors match
# on literal prompt text, so we add correct spellings for typos that appear in
# the source questions (kept alongside the original to match either labeling).
EXTRA_PROMPTS = {
    "refridgerator": ["refrigerator"],
}


def load_questions(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    qs: list[str] = []
    for scene in data:
        for qlist in scene["questions"].values():
            qs.extend(qlist)
    return qs


def singularize(word: str) -> str:
    w = word.lower()
    if w.endswith("ies") and len(w) > 3:
        return w[:-3] + "y"
    if w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def phrase_from_chunk(chunk) -> str | None:
    """Reduce a spaCy noun chunk to a normalized object phrase, or None."""
    root = chunk.root
    if root.pos_ not in ("NOUN", "PROPN"):
        return None
    head = singularize(root.lemma_.lower() if root.pos_ == "NOUN" else root.text.lower())
    if not head.isalpha() or head in NON_OBJECT_NOUNS:
        return None

    # collect object-kind modifiers that sit before the head: compound nouns and
    # non-attribute adjectives (e.g. "folding", "dressing"), preserving order.
    mods: list[str] = []
    for tok in chunk:
        if tok is root:
            break
        if tok.pos_ in ("DET", "NUM", "PRON"):
            continue
        if tok.dep_ in ("det", "nummod", "poss", "predet"):
            continue
        low = tok.lemma_.lower() if tok.pos_ == "NOUN" else tok.text.lower()
        if not low.isalpha():
            continue
        if low in _ATTRIBUTES or low in QUANTIFIERS:  # not part of the object kind
            continue
        if tok.dep_ in ("compound", "amod", "nmod"):
            mods.append(low)
    phrase = " ".join(mods + [head]).strip()
    return phrase or None


def main() -> int:
    nlp = spacy.load("en_core_web_sm")
    questions = load_questions(QUESTIONS)

    phrase_counts: Counter[str] = Counter()
    head_to_phrases: dict[str, set[str]] = defaultdict(set)
    dropped: Counter[str] = Counter()

    for q in questions:
        doc = nlp(q)
        for chunk in doc.noun_chunks:
            phrase = phrase_from_chunk(chunk)
            if phrase is None:
                root = chunk.root
                if root.pos_ in ("NOUN", "PROPN"):
                    dropped[singularize(root.lemma_.lower())] += 1
                continue
            phrase_counts[phrase] += 1
            head_to_phrases[phrase.split()[-1]].add(phrase)

    # --- report ---------------------------------------------------------- #
    lines = [
        f"Challenge object vocabulary — mined from {len(questions)} questions",
        "=" * 60,
        f"{len(phrase_counts)} distinct object phrases, "
        f"{len(head_to_phrases)} head nouns.",
        "",
        "Ranked object phrases (phrase: count):",
    ]
    for phrase, n in phrase_counts.most_common():
        lines.append(f"  {n:2d}  {phrase}")
    lines += ["", "Grouped by head noun:"]
    for head in sorted(head_to_phrases):
        variants = sorted(head_to_phrases[head])
        lines.append(f"  {head}: {', '.join(variants)}")
    lines += ["", "Dropped (abstract/structural/attribute-only heads):"]
    for w, n in dropped.most_common():
        lines.append(f"  {n:2d}  {w}")
    REPORT.write_text("\n".join(lines) + "\n")

    # --- yaml (objects.yaml schema) -------------------------------------- #
    # One class per head noun, prompts = all phrase variants seen (most specific
    # first). All are countable instances.
    yaml_lines = [
        "# Challenge detection vocabulary — AUTO-GENERATED by",
        "# eval_harness/build_challenge_vocab.py from questions/questions.json.",
        "# Schema matches SysNav semantic_mapping/config/objects.yaml. On the",
        "# Jazzy box, point detection_node's `object_file` parameter here.",
        "prompts:",
    ]
    # order classes by total frequency (most common first), then name.
    head_freq = {
        h: sum(phrase_counts[p] for p in ps) for h, ps in head_to_phrases.items()
    }
    for head in sorted(head_to_phrases, key=lambda h: (-head_freq[h], h)):
        variants = sorted(
            head_to_phrases[head], key=lambda p: (-phrase_counts[p], -len(p))
        )
        key = head.replace(" ", "_")
        for v in list(variants):
            variants += [e for e in EXTRA_PROMPTS.get(v, []) if e not in variants]
        yaml_lines.append(f"  {key}:")
        yaml_lines.append("    prompts:")
        for v in variants:
            yaml_lines.append(f"      - {v}")
        yaml_lines.append("    is_instance: true")
    yaml_lines += [
        "  unknown:",
        "    prompts:",
        "      - unknown",
        "    is_instance: false",
    ]
    YAML_OUT.parent.mkdir(parents=True, exist_ok=True)
    YAML_OUT.write_text("\n".join(yaml_lines) + "\n")

    print(f"questions mined:        {len(questions)}")
    print(f"distinct object phrases:{len(phrase_counts):>4}")
    print(f"head-noun classes:      {len(head_to_phrases):>4}")
    print(f"report -> {REPORT.relative_to(ROOT)}")
    print(f"yaml   -> {YAML_OUT.relative_to(ROOT)}")
    print()
    print("Top 15 object phrases:")
    for phrase, n in phrase_counts.most_common(15):
        print(f"  {n:2d}  {phrase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
