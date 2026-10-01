"""Procedural generator of TRAINING tasks for self-training.

A small local code model is fine-tuned on its own successful attempts on these
tasks and then evaluated on the hand-written EVAL suites (``tasks/tasks.json``
and ``tasks/tasks_hard.json``). Training tasks therefore have to be:

- plentiful and varied: ``FAMILIES`` holds many problem families, and every
  family varies its function name, constants, conventions (inclusive vs
  exclusive bounds, tie rules, case sensitivity, ...) and input types from a
  seeded RNG, so two tasks of one family need genuinely different code;
- deterministic: the same ``(n, seed, families)`` always yields identical
  output (every task gets its own ``random.Random`` seeded from a string);
- trustworthy: expected values are computed here by running the canonical
  solution, and every mutant (plausible buggy variant) is checked to fail at
  least one test before the task is emitted; ``validate_generated`` re-checks
  all of that in the real sandbox;
- disjoint from the eval suites (see ``EXCLUDED_EVAL_TOPICS``).

Each family function returns a ``_Spec`` (signature, docstring, body, test
inputs, mutations); ``_finish`` turns it into a tasks.json-style dict.
"""

from __future__ import annotations

import ast
import copy
import json
import random
import re
import textwrap
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

# EXCLUDED_EVAL_TOPICS -- problems already in the eval suites; no family here is
# equivalent to any of them:
#   easy suite (tasks/tasks.json): caesar shift, run-length encoding, second
#     largest distinct value, rounded mean, word counts, clamp all, merge
#     intervals, "1h30m" duration parsing, leaderboard ranking (rank players),
#     sliding-window maxima, add days to a date, greedy word wrap, summarize
#     integer ranges ("1-3,5"), min coins, grid shortest path S->E, course
#     order / topological sort, LRU cache simulation, infix expression
#     evaluation with precedence, quoted CSV line parsing, weighted interval
#     scheduling (max booking value).
#   hard suite (tasks/tasks_hard.json): cent allocation / refunds, keyset
#     pagination, semver range matching, SLA business-minute deadlines, config
#     deep merge, token-bucket rate limiter, limit order book, nested
#     transactional KV store, undo/redo text buffer, card-hold ledger, template
#     rendering, JSON Patch (RFC 6902), cron next fire time, TOML-subset
#     parser, priority cluster job dispatch, earliest connection time
#     (offline union-find), total subarray spread, k-th smallest pair
#     distance, offline range "count at most" queries.
# Near neighbours were deliberately steered away from those: window stats are
# sums/counts (not maxima), the interval family measures overlap depth (not
# merging), the DP coin family counts ways (not the minimum), the stack
# machine runs explicit instructions (no infix parsing), key=value parsing is
# single-line and quote-free, and the text family pads/aligns a single string
# (no wrapping).

FAMILIES: dict[str, Callable[[random.Random, int], dict]] = {}

WORDS = [
    "apple", "river", "stone", "cloud", "tiger", "maple", "ocean", "pixel", "lemon", "quartz",
    "delta", "ember", "forest", "glider", "harbor", "island", "jungle", "kettle", "lantern",
    "meadow", "nectar", "orbit", "prairie", "quiver", "rocket", "saddle", "timber", "umbra",
    "velvet", "willow", "yonder", "zephyr",
]
NAMES = ["amy", "Bob", "carl", "Dana", "eve", "Finn", "gus", "Hana", "ivy", "Jon", "kim", "Lea", "max", "Nia"]


class _Reject(Exception):
    """The drawn parameters/inputs did not give a good task; the builder retries."""


@dataclass
class _Spec:
    name: str
    sig: str
    ret: str
    doc: str
    body: str
    cases: list[tuple[str, list[tuple]]]
    examples: list[tuple]
    mutations: list[tuple[str | None, str]]
    difficulty: str = "easy"
    extra_tags: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- framework


def _dd(text: str) -> str:
    return textwrap.dedent(text).strip("\n") + "\n"


def _doc_text(doc: str) -> str:
    """Normalise a docstring body built from f-string pieces: every line is
    re-indented from scratch, and continuation lines of a '- ' / '1. ' list item
    are indented under the item's text."""
    out: list[str] = []
    hang = 0
    for raw in doc.strip().splitlines():
        line = raw.strip()
        if not line:
            out.append("")
            hang = 0
        elif line.startswith("- ") or re.match(r"\d+\. ", line):
            out.append(line)
            hang = 2 if line.startswith("- ") else line.index(" ") + 1
        else:
            out.append(" " * hang + line)
    return "\n".join(out) + "\n"


def _indent(code: str) -> str:
    return textwrap.indent(_dd(code), "    ")


def _lit(value) -> str:
    text = repr(value)
    if ast.literal_eval(text) != value:
        raise RuntimeError(f"value does not round-trip through repr: {text}")
    return text


def _args(args: tuple) -> str:
    return ", ".join(_lit(a) for a in args)


def _load(source: str, name: str):
    namespace: dict = {"__name__": "generated"}
    exec(compile(source, "<generated>", "exec"), namespace)  # noqa: S102 -- our own code
    return namespace[name]


def _one(*values) -> list[tuple]:
    return [(v,) for v in values]


def _outputs(fn, calls: list[tuple]) -> list:
    out = []
    for args in calls:
        try:
            out.append(("ok", fn(*copy.deepcopy(args))))
        except Exception as exc:  # a crashing mutant is a killed mutant
            out.append(("err", type(exc).__name__))
    return out


def _same(a: list, b: list) -> bool:
    for (ka, va), (kb, vb) in zip(a, b, strict=True):
        if ka != kb:
            return False
        if ka == "ok" and not (va == vb):  # the harness compares with ==
            return False
    return True


def _finish(family: str, index: int, spec: _Spec) -> dict:
    if not 5 <= len(spec.cases) <= 8:
        raise RuntimeError(f"{family}: need 5-8 tests, got {len(spec.cases)}")
    header = f"def {spec.name}({spec.sig}) -> {spec.ret}:\n"
    plain = _load(header + _indent(spec.body), spec.name)
    examples = [f">>> {spec.name}({_args(a)})\n{_lit(plain(*copy.deepcopy(a)))}" for a in spec.examples]
    doc = _doc_text(spec.doc) + "\n" + "\n".join(examples)
    if '"""' in doc:
        raise RuntimeError(f"{family}: docstring contains triple quotes")
    prefix = "r" if "\\" in doc else ""
    prompt = header + textwrap.indent(f'{prefix}"""{doc}\n"""', "    ") + "\n"
    canonical = prompt + _indent(spec.body)
    fn = _load(canonical, spec.name)

    calls = [args for _, group in spec.cases for args in group]
    reference = _outputs(fn, calls)
    for args, (kind, value) in zip(calls, reference, strict=True):
        if kind != "ok":
            raise RuntimeError(f"{family}: canonical raised {value} on {args!r}")

    lines = []
    for test_name, group in spec.cases:
        lines.append(f"def {test_name}():")
        for args in group:
            lines.append(f"    assert {spec.name}({_args(args)}) == {_lit(fn(*copy.deepcopy(args)))}")
        lines.append("")
    test_code = "\n".join(lines)

    body = _dd(spec.body)
    mutants: list[str] = []
    for old, new in spec.mutations:
        if old is None:
            mutated = _dd(new)
        else:
            if old not in body:
                raise RuntimeError(f"{family}: mutation target not found: {old!r}")
            mutated = body.replace(old, new, 1)
        if mutated == body:
            continue
        source = prompt + textwrap.indent(mutated, "    ")
        if source in mutants:
            continue
        mutant_fn = _load(source, spec.name)
        if not _same(_outputs(mutant_fn, calls), reference):
            mutants.append(source)
    if len(mutants) < 2:
        raise _Reject(f"only {len(mutants)} mutants killed")

    return {
        "task_id": f"gen_{family}_{index:04d}",
        "category": "function",
        "difficulty": spec.difficulty,
        "tags": ["generated", family, *spec.extra_tags],
        "prompt": prompt,
        "entry_point": spec.name,
        "test_code": test_code,
        "canonical_solution": canonical,
        "mutants": mutants[:4],
    }


def _family(name: str):
    def deco(fn: Callable[[random.Random], _Spec]):
        def builder(rng: random.Random, index: int) -> dict:
            last: Exception | None = None
            for _ in range(40):
                try:
                    return _finish(name, index, fn(rng))
                except _Reject as exc:
                    last = exc
            raise RuntimeError(f"family {name!r} could not build a task: {last}")

        builder.__name__ = f"build_{name}"
        builder.__doc__ = fn.__doc__
        FAMILIES[name] = builder
        return fn

    return deco


def _ints(rng: random.Random, n: int, lo: int, hi: int) -> list[int]:
    return [rng.randint(lo, hi) for _ in range(n)]


def _mixcase(rng: random.Random, word: str) -> str:
    return "".join(c.upper() if rng.random() < 0.5 else c for c in word)


def _phrase(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n))


# --------------------------------------------------------------------------- string transforms


@_family("alternate_case")
def _alternate_case(rng: random.Random) -> _Spec:
    """Alternate upper/lower case of ASCII letters (start case and counting scope vary)."""
    name = rng.choice(["alternate_case", "zigzag_case", "sponge_case", "alternating_caps"])
    first_upper = rng.random() < 0.5
    letters_only = rng.random() < 0.5
    p = 0 if first_upper else 1
    first, second = ("upper", "lower") if first_upper else ("lower", "upper")
    if letters_only:
        rule = (f"Only ASCII letters are counted: the 1st, 3rd, 5th, ... ASCII letter of `text`\n"
                f"becomes {first}case and the 2nd, 4th, ... becomes {second}case. Other\n"
                f"characters do not advance the count.")
        body = f"""
        out = []
        i = 0
        for ch in text:
            if ch.isascii() and ch.isalpha():
                out.append(ch.upper() if i % 2 == {p} else ch.lower())
                i += 1
            else:
                out.append(ch)
        return "".join(out)
        """
        mutations = [
            (f"i % 2 == {p}", f"i % 2 == {1 - p}"),
            ("ch.isascii() and ch.isalpha()", "ch.isalpha()"),
            ("        i += 1\n    else:\n        out.append(ch)\n", "    else:\n        out.append(ch)\n    i += 1\n"),
        ]
    else:
        rule = (f"Position decides: an ASCII letter at an even index of `text` (0, 2, 4, ...)\n"
                f"becomes {first}case and one at an odd index becomes {second}case. Every\n"
                f"character, letter or not, occupies an index.")
        body = f"""
        out = []
        for i, ch in enumerate(text):
            if ch.isascii() and ch.isalpha():
                out.append(ch.upper() if i % 2 == {p} else ch.lower())
            else:
                out.append(ch)
        return "".join(out)
        """
        mutations = [
            (f"i % 2 == {p}", f"i % 2 == {1 - p}"),
            ("ch.isascii() and ch.isalpha()", "ch.isalpha()"),
            ("enumerate(text)", "enumerate(text.strip())"),
        ]
    doc = f"""
    Return a copy of `text` whose ASCII letters alternate between cases.

    {rule}
    Non-letters, and non-ASCII letters such as 'é', are copied unchanged.
    The empty string returns the empty string.
    """
    w = rng.choice(WORDS)
    cases = [
        ("test_single_word", _one(w)),
        ("test_words_with_space", _one(_phrase(rng, 2))),
        ("test_digits_and_punctuation", _one(f"{w[:2]}1{w[2:4]}2-x!")),
        ("test_non_ascii_letters_unchanged", _one("éa éb", "ÉcolE")),
        ("test_leading_spaces", _one("  ab cd", " " + _mixcase(rng, rng.choice(WORDS)))),
        ("test_mixed_case_input", _one(_mixcase(rng, _phrase(rng, 2)))),
        ("test_empty", _one("")),
    ]
    return _Spec(name, "text: str", "str", doc, body, cases, [(f"{w} 42",)], mutations,
                 difficulty="easy" if not letters_only else "medium")


@_family("strip_vowels")
def _strip_vowels(rng: random.Random) -> _Spec:
    """Remove or mask vowels (y-rule, case scope and remove-vs-mask vary)."""
    name = rng.choice(["strip_vowels", "drop_vowels", "devowel", "mask_vowels"])
    with_y = rng.random() < 0.5
    both_cases = rng.random() < 0.6
    mask = rng.choice([None, None, "*", "_", "#"])
    vowels = "aeiou" + ("y" if with_y else "")
    vow_lit = vowels + vowels.upper() if both_cases else vowels
    alt_case = vowels if both_cases else vowels + vowels.upper()
    alt_y = ("aeiou" if with_y else "aeiouy")
    alt_y = alt_y + alt_y.upper() if both_cases else alt_y
    what = f"the letters {', '.join(vowels)}" + (" in either case" if both_cases else
                                                   " (lowercase only; uppercase vowels are NOT affected)")
    action = "is removed" if mask is None else f"is replaced by {mask!r}"
    doc = f"""
    Return `text` where every vowel {action}.

    Vowels are {what}.
    {"'y' counts as a vowel." if with_y else "'y' is NOT a vowel."} All other characters are kept unchanged,
    in their original order. The empty string returns the empty string.
    """
    if mask is None:
        body = f"""
        vowels = {vow_lit!r}
        return "".join(ch for ch in text if ch not in vowels)
        """
        loop = "for ch in text if"
        mut_loop = "for ch in text.lower() if"
    else:
        body = f"""
        vowels = {vow_lit!r}
        return "".join({mask!r} if ch in vowels else ch for ch in text)
        """
        loop = "for ch in text)"
        mut_loop = "for ch in text.lower())"
    mutations = [
        (f"vowels = {vow_lit!r}", f"vowels = {alt_case!r}"),
        (f"vowels = {vow_lit!r}", f"vowels = {alt_y!r}"),
        (loop, mut_loop),
    ]
    cases = [
        ("test_simple_word", _one(rng.choice(WORDS))),
        ("test_y_handling", _one("Yearly rhythm", "y")),
        ("test_upper_and_lower", _one("AEIOU aeiou", _mixcase(rng, rng.choice(WORDS)))),
        ("test_sentence", _one(_phrase(rng, 3).capitalize() + "!")),
        ("test_no_vowels", _one("xyz 123", "Shh.")),
        ("test_empty", _one("")),
    ]
    return _Spec(name, "text: str", "str", doc, body, cases, [("Python Is Fun",)], mutations,
                 difficulty="easy")


@_family("slugify")
def _slugify(rng: random.Random) -> _Spec:
    """URL slug with a chosen separator and optional maximum length."""
    name = rng.choice(["slugify", "make_slug", "to_slug", "url_slug"])
    sep = rng.choice(["-", "_", "."])
    maxlen = rng.choice([None, rng.randint(8, 14)])
    trunc_doc = (f"- If the slug is longer than {maxlen} characters, keep only its first {maxlen}\n"
                 f"      characters, then remove any '{sep}' left at the end.\n" if maxlen else "")
    doc = f"""
    Turn `text` into a URL slug.

    - The text is lowercased first; ASCII letters a-z and digits 0-9 are kept.
    - Every maximal run of other characters (spaces, punctuation, non-ASCII
      letters such as 'é', ...) becomes ONE '{sep}' between kept characters.
    - The slug never starts or ends with '{sep}'.
    {trunc_doc}- If nothing is kept, return the empty string.
    """
    tail = f'slug = slug[:{maxlen}].rstrip({sep!r})\n' if maxlen else ""
    body = _dd(f"""
    out = []
    pending = False
    for ch in text.lower():
        if ch.isascii() and ch.isalnum():
            if pending and out:
                out.append({sep!r})
            out.append(ch)
            pending = False
        else:
            pending = True
    slug = "".join(out)
    """) + tail + "return slug\n"
    mutations = [
        ("if pending and out:", "if pending:"),
        ("ch.isascii() and ch.isalnum()", "ch.isalnum()"),
        ("text.lower()", "text"),
    ]
    if maxlen:
        mutations.append((f".rstrip({sep!r})", ""))
        mutations.append((f"slug[:{maxlen}]", f"slug[:{maxlen - 1}]"))
    w1, w2 = rng.sample(WORDS, 2)
    cases = [
        ("test_basic_title", _one(f"{w1.title()}, {w2.title()}!")),
        ("test_collapses_and_strips", _one("  --Already--slugged--  ", "A  B   C")),
        ("test_non_ascii_become_separators", _one("Café au lait")),
        ("test_digits_kept", _one(f"Top {rng.randint(2, 99)} {w1} of {rng.randint(1990, 2030)}")),
        ("test_nothing_kept", _one("!!!", "")),
        ("test_mixed_case", _one(_mixcase(rng, f"{w2} {w1}"))),
    ]
    if maxlen:
        cases.append(("test_truncation_strips_trailing_separator",
                      _one("x" * (maxlen - 1) + " " + w1, _phrase(rng, 5))))
    return _Spec(name, "text: str", "str", doc, body, cases, [("Hello, World!",)], mutations,
                 difficulty="medium" if maxlen else "easy")


@_family("pad_text")
def _pad_text(rng: random.Random) -> _Spec:
    """Align a string inside a fixed width (alignment, fill, tie side and overflow rule vary)."""
    align = rng.choice(["left", "right", "center", "center"])
    fill = rng.choice([".", "*", "-", " "])
    extra_left = rng.random() < 0.5
    overflow = rng.choice(["cut", "ellipsis", "keep"])
    name = {"left": ["pad_right", "left_align", "fill_after"],
            "right": ["pad_left", "right_align", "fill_before"],
            "center": ["center_text", "center_in", "centered"]}[align]
    name = rng.choice(name)
    if align == "left":
        place = f"`text` followed by {fill!r} characters"
        build = f"return text + {fill!r} * total"
    elif align == "right":
        place = f"{fill!r} characters followed by `text`"
        build = f"return {fill!r} * total + text"
    else:
        side = "LEFT" if extra_left else "RIGHT"
        place = (f"`text` centered between runs of {fill!r}; when the padding cannot be split\n"
                 f"    evenly, the extra character goes on the {side}")
        left = "(total + 1) // 2" if extra_left else "total // 2"
        build = f"left = {left}\nreturn {fill!r} * left + text + {fill!r} * (total - left)"
    over_doc = {
        "cut": "If `text` is longer than `width`, return its first `width` characters.",
        "ellipsis": ("If `text` is longer than `width`, return its first `width - 3` characters\n"
                     "    followed by '...' (when width < 3, just its first `width` characters)."),
        "keep": "If `text` is longer than `width`, return it unchanged (never cut).",
    }[overflow]
    over_code = {
        "cut": "return text[:width]",
        "ellipsis": 'return text[:width - 3] + "..." if width >= 3 else text[:width]',
        "keep": "return text",
    }[overflow]
    doc = f"""
    Return a string of length exactly `width` consisting of {place}.

    {over_doc}
    If `width` <= 0, return '' (this check comes first).
    """
    body = f'if width <= 0:\n    return ""\nif len(text) > width:\n    {over_code}\ntotal = width - len(text)\n{build}\n'
    mutations = [("if width <= 0:\n    return \"\"\n", ""), ("len(text) > width", "len(text) >= width")]
    if align == "center":
        alt = "total // 2" if extra_left else "(total + 1) // 2"
        mutations.append((f"left = {'(total + 1) // 2' if extra_left else 'total // 2'}", f"left = {alt}"))
    if overflow == "ellipsis":
        mutations.append(("text[:width - 3]", "text[:width - 2]"))
    if overflow == "keep":
        mutations.append(("return text\n", "return text[:width]\n"))
    w = rng.choice(WORDS)
    cases = [
        ("test_odd_padding", [(w, len(w) + 3)]),
        ("test_even_padding", [(w, len(w) + 4), ("ab", 6)]),
        ("test_exact_width", [(w, len(w))]),
        ("test_too_long", [(w + "xyz", len(w)), ("abcdef", 2)]),
        ("test_zero_or_negative_width", [(w, 0), ("abc", -2)]),
        ("test_empty_text", [("", 3)]),
    ]
    diff = "medium" if align == "center" or overflow == "ellipsis" else "easy"
    return _Spec(name, "text: str, width: int", "str", doc, body, cases, [("cat", 7)], mutations,
                 difficulty=diff)


# --------------------------------------------------------------------------- parsing


@_family("parse_pairs")
def _parse_pairs(rng: random.Random) -> _Spec:
    """Parse a 'k=v;k2=v2' settings string (separators, duplicate rule, int conversion vary)."""
    name = rng.choice(["parse_pairs", "parse_kv", "parse_settings", "parse_options"])
    ps = rng.choice([";", ",", "&", "|"])
    ks = rng.choice(["=", ":"])
    first_wins = rng.random() < 0.5
    convert = rng.random() < 0.5
    conv_doc = ("- A value made only of ASCII digits, optionally preceded by ONE '-', is\n"
                "      converted to int; every other value (including '' and '-') stays a str."
                if convert else "- Values always stay strings, even if they look like numbers.")
    doc = f"""
    Parse a settings string such as 'a{ks}1{ps} b{ks}x' into a dict.

    - Entries are separated by '{ps}'. Each entry is stripped of surrounding
      whitespace; empty entries are ignored.
    - An entry is split at its FIRST '{ks}' into key and value (any later '{ks}'
      belongs to the value); key and value are each stripped of whitespace.
    - Entries without '{ks}', or whose key is empty after stripping, are ignored.
    - If a key occurs more than once, the {"FIRST" if first_wins else "LAST"} occurrence wins.
    {conv_doc}
    """
    dup = "    if key in result:\n        continue\n" if first_wins else ""
    conv = ("    digits = value[1:] if value.startswith(\"-\") else value\n"
            "    if digits.isascii() and digits.isdigit():\n"
            "        value = int(value)\n") if convert else ""
    body = (f"result = {{}}\nfor part in text.split({ps!r}):\n    part = part.strip()\n"
            f"    if {ks!r} not in part:\n        continue\n    key, value = part.split({ks!r}, 1)\n"
            f"    key = key.strip()\n    value = value.strip()\n    if not key:\n        continue\n"
            f"{dup}{conv}    result[key] = value\nreturn result\n")
    mutations = [
        (f"key, value = part.split({ks!r}, 1)", f"key, value = part.split({ks!r})[:2]"),
        ("    key = key.strip()\n", ""),
    ]
    if first_wins:
        mutations.append((dup, ""))
    else:
        mutations.append(("    result[key] = value\n", "    if key in result:\n        continue\n    result[key] = value\n"))
    if convert:
        mutations.append(('digits = value[1:] if value.startswith("-") else value', "digits = value"))
    else:
        mutations.append(("    result[key] = value\n", "    result[key] = int(value) if value.isdigit() else value\n"))
    k1, k2, k3 = rng.sample(["host", "port", "mode", "user", "level", "retries", "name", "path"], 3)
    n1 = rng.randint(1, 999)
    cases = [
        ("test_basic", _one(f"{k1}{ks}{rng.choice(WORDS)}{ps}{k2}{ks}{n1}")),
        ("test_whitespace_is_stripped", _one(f"  {k1} {ks} a b {ps}   {k2}{ks}  {n1}  ")),
        ("test_duplicate_keys", _one(f"{k1}{ks}1{ps}{k2}{ks}x{ps}{k1}{ks}2")),
        ("test_separator_inside_value", _one(f"{k3}{ks}a{ks}b{ks}c{ps}{k1}{ks}{ks}")),
        ("test_ignored_entries", _one(f"{ps}{ps}{k1}{ps}{ks}orphan{ps}  {ps}{k2}{ks}ok{ps}")),
        ("test_number_like_values", _one(f"a{ks}-{n1}{ps}b{ks}007{ps}c{ks}-{ps}d{ks}1.5{ps}e{ks}--3")),
        ("test_empty_string", _one("")),
    ]
    diff = "medium" if convert or not first_wins else "easy"
    return _Spec(name, "text: str", "dict", doc, body, cases, [(f"x{ks}1{ps} y{ks}hi",)], mutations, diff)


@_family("parse_table")
def _parse_table(rng: random.Random) -> _Spec:
    """Parse delimited lines with a header into dicts (delimiter, short-row rule, int cells vary)."""
    name = rng.choice(["parse_table", "read_rows", "rows_to_records", "table_records"])
    sep = rng.choice(["|", ";", "\t"])
    sep_doc = "a tab character" if sep == "\t" else f"'{sep}'"
    pad = rng.random() < 0.5
    convert = rng.random() < 0.5
    doc = f"""
    Parse a small delimited table into a list of dicts.

    - `text` is made of lines separated by newlines. Lines that are empty or
      contain only whitespace are ignored entirely.
    - The first remaining line is the header: splitting it on {sep_doc} gives the
      column names, each stripped of surrounding whitespace.
    - Every later line is a data row: split it on {sep_doc} and strip each cell.
    - {"A row with fewer cells than the header is padded with None for the missing columns." if pad else
       "A row with fewer cells than the header is skipped (left out of the result)."}
      Cells beyond the number of header columns are ignored.
    - {"Cells made only of ASCII digits become ints; all other cells stay strings." if convert else
       "All cells stay strings, even if they look like numbers."}

    Return one dict per data row, in input order, mapping column name -> cell.
    Return [] when there is no data row (no non-blank lines, or only a header).
    """
    short = ("        cells += [None] * (len(header) - len(cells))\n" if pad else "        continue\n")
    conv = ("        if isinstance(cell, str) and cell.isascii() and cell.isdigit():\n"
            "            cell = int(cell)\n") if convert else ""
    body = (f"lines = [ln for ln in text.splitlines() if ln.strip()]\nif not lines:\n    return []\n"
            f"header = [h.strip() for h in lines[0].split({sep!r})]\nrecords = []\nfor ln in lines[1:]:\n"
            f"    cells = [c.strip() for c in ln.split({sep!r})]\n    if len(cells) < len(header):\n{short}"
            f"    row = {{}}\n    for key, cell in zip(header, cells):\n{conv}        row[key] = cell\n"
            f"    records.append(row)\nreturn records\n")
    other = ("        continue\n" if pad else "        cells += [None] * (len(header) - len(cells))\n")
    mutations = [
        ("if not lines:\n    return []\n", ""),
        (f"cells = [c.strip() for c in ln.split({sep!r})]", f"cells = ln.split({sep!r})"),
        ("if ln.strip()]", "if ln]"),
        (short, other),
        (f"header = [h.strip() for h in lines[0].split({sep!r})]", f"header = lines[0].split({sep!r})"),
    ]
    cols = rng.sample(["id", "name", "qty", "city", "code", "age"], 3)

    def row(cells: list[str]) -> str:
        return sep.join(cells)

    def rand_cells() -> list[str]:
        return [rng.choice([str(rng.randint(0, 500)), rng.choice(WORDS)]) for _ in cols]

    head = row([f" {c} " for c in cols])
    basic = "\n".join([head, row(rand_cells()), row([f"  {c} " for c in rand_cells()])])
    cases = [
        ("test_basic_table", _one(basic)),
        ("test_blank_lines_ignored", _one("\n" + head + "\n\n   \n" + row(rand_cells()) + "\n")),
        ("test_short_row", _one("\n".join([head, row(rand_cells()[:2]), row(rand_cells())]))),
        ("test_extra_cells_ignored", _one("\n".join([head, row([*rand_cells(), "extra", "9"])]))),
        ("test_header_only_or_empty", _one(head, "", "  \n \n")),
        ("test_number_like_cells", _one("\n".join([head, row(["007", "12a", " -3 "])]))),
    ]
    diff = "medium" if pad or convert else "easy"
    return _Spec(name, "text: str", "list", doc, body, cases, [(f"a{sep}b\n1{sep}x",)], mutations, diff)


# --------------------------------------------------------------------------- list / array ops


@_family("chunk_list")
def _chunk_list(rng: random.Random) -> _Spec:
    """Split a list into fixed-size chunks (keep / drop / pad / right-aligned remainder)."""
    k = rng.randint(2, 5)
    mode = rng.choice(["keep", "drop", "pad", "right"])
    fill = rng.choice([0, None, -1])
    name = rng.choice({"keep": ["chunked", "split_into_chunks", "batch_items"],
                       "drop": ["full_chunks", "complete_batches", "whole_groups"],
                       "pad": ["padded_chunks", "chunk_and_pad", "fixed_batches"],
                       "right": ["chunk_from_end", "right_aligned_chunks", "tail_chunks"]}[mode])
    keep = f"chunks = [items[i:i + {k}] for i in range(0, len(items), {k})]\n"
    if mode == "keep":
        rule = (f"from left to right. The last chunk holds whatever remains and may be\n"
                f"shorter than {k} (it is never empty).")
        body = keep + "return chunks\n"
        mutations = [(f"range(0, len(items), {k})", f"range(0, len(items) - 1, {k})"),
                     (f"range(0, len(items), {k})", f"range(0, len(items) - {k} + 1, {k})"),
                     (f"items[i:i + {k}]", f"items[i:i + {k} - 1]")]
    elif mode == "drop":
        rule = (f"from left to right. Leftover elements that do not fill a complete final\n"
                f"chunk are discarded, so every chunk has exactly {k} elements.")
        body = f"return [items[i:i + {k}] for i in range(0, len(items) - {k} + 1, {k})]\n"
        mutations = [(f"len(items) - {k} + 1", f"len(items) - {k}"),
                     (f"len(items) - {k} + 1", "len(items)"),
                     (f"range(0, len(items) - {k} + 1, {k})", f"range(0, len(items) // {k}, {k})")]
    elif mode == "pad":
        rule = (f"from left to right. If the last chunk is shorter than {k}, it is padded on the\n"
                f"right with {fill!r} until it has {k} elements.")
        body = keep + (f"if chunks and len(chunks[-1]) < {k}:\n"
                       f"    chunks[-1] = chunks[-1] + [{fill!r}] * ({k} - len(chunks[-1]))\nreturn chunks\n")
        mutations = [(f"if chunks and len(chunks[-1]) < {k}:", f"if len(chunks[-1]) < {k}:"),
                     (f"[{fill!r}] * ({k} - len(chunks[-1]))", f"[{fill!r}] * ({k} - len(chunks[-1]) - 1)"),
                     (f"range(0, len(items), {k})", f"range(0, len(items) - {k} + 1, {k})")]
    else:
        rule = (f"aligned to the END of the list: every chunk has exactly {k} elements except\n"
                f"possibly the FIRST one, which holds the len(items) % {k} leftover elements\n"
                f"(and is omitted when there is no leftover).")
        body = (f"r = len(items) % {k}\nchunks = [items[:r]] if r else []\n"
                f"chunks += [items[i:i + {k}] for i in range(r, len(items), {k})]\nreturn chunks\n")
        mutations = [("[items[:r]] if r else []", "[items[:r]]"),
                     (None, keep + "return chunks\n"),
                     (f"range(r, len(items), {k})", f"range(r + 1, len(items), {k})")]
    doc = f"""
    Split `items` into consecutive chunks of {k} elements, {rule}

    Element order is preserved and the input list is not modified.
    An empty list gives [].
    """
    vals = _ints(rng, 4 * k + 3, -9, 50)
    cases = [
        ("test_exact_multiple", _one(vals[:2 * k])),
        ("test_remainder_of_one", _one(vals[:2 * k + 1])),
        ("test_remainder_of_k_minus_one", _one(vals[:3 * k - 1])),
        ("test_shorter_than_chunk", _one(vals[:k - 1])),
        ("test_single_element", _one([vals[0]])),
        ("test_empty", _one([])),
        ("test_long_list", _one(vals)),
    ]
    diff = "easy" if mode in ("keep", "drop") else "medium"
    return _Spec(name, "items: list[int]", "list[list[int]]", doc, body, cases,
                 [(list(range(1, k + 3)),)], mutations, diff)


@_family("rotate_seq")
def _rotate_seq(rng: random.Random) -> _Spec:
    """Rotate a list or string left/right by k with wrap-around and negative k."""
    right = rng.random() < 0.5
    is_str = rng.random() < 0.5
    d, o = ("right", "left") if right else ("left", "right")
    name = rng.choice([f"rotate_{d}", f"shift_{d}", f"cycle_{d}"]) + ("_text" if is_str else "")
    var = "text" if is_str else "items"
    sig = "text: str, k: int" if is_str else "items: list[int], k: int"
    kind = "string" if is_str else "list"
    move = ("moves the last element to the front" if right else "moves the first element to the end")
    doc = f"""
    Return a new {kind} with the elements of `{var}` rotated {d} by `k` positions.

    Rotating {d} by 1 {move}. `k` may exceed the length (it wraps around)
    or be negative: rotating {d} by -k is the same as rotating {o} by k.
    An empty {kind} returns an empty {kind} for every k.
    """
    formula = f"{var}[n - k:] + {var}[:n - k]" if right else f"{var}[k:] + {var}[:k]"
    other = f"{var}[k:] + {var}[:k]" if right else f"{var}[n - k:] + {var}[:n - k]"
    body = f"""
    n = len({var})
    if n == 0:
        return {var}[:]
    k %= n
    return {formula}
    """
    mutations = [
        (f"if n == 0:\n    return {var}[:]\n", ""),
        ("k %= n", "k = min(k, n)"),
        (f"return {formula}", f"return {other}"),
    ]
    if is_str:
        seq = "".join(rng.choice("abcdefghij") for _ in range(rng.randint(5, 8)))
    else:
        seq = _ints(rng, rng.randint(5, 8), 0, 40)
    n = len(seq)
    cases = [
        ("test_by_one", [(seq, 1)]),
        ("test_by_length_minus_one", [(seq, n - 1)]),
        ("test_zero_and_full_length", [(seq, 0), (seq, n)]),
        ("test_larger_than_length", [(seq, n + 2), (seq, 3 * n + 1)]),
        ("test_negative_k", [(seq, -1), (seq, -(n + 2))]),
        ("test_empty", [(seq[:0], 3), (seq[:0], 0)]),
        ("test_single_element", [(seq[:1], 5)]),
    ]
    ex = ("abcde", 2) if is_str else ([1, 2, 3, 4, 5], 2)
    return _Spec(name, sig, "str" if is_str else "list[int]", doc, body, cases, [ex], mutations, "easy")


@_family("window_stat")
def _window_stat(rng: random.Random) -> _Spec:
    """Statistic over every length-k window (sum / evens / distinct / floor mean; list or count)."""
    stat = rng.choice(["sum", "evens", "distinct", "mean"])
    as_count = rng.random() < 0.4
    k_hint = rng.randint(2, 4)
    t = {"sum": rng.randint(5, 40), "evens": rng.randint(1, 2), "distinct": rng.randint(2, 3),
         "mean": rng.randint(2, 12)}[stat]
    expr = {"sum": "sum(w)", "evens": "sum(1 for x in w if x % 2 == 0)", "distinct": "len(set(w))",
            "mean": "sum(w) // k"}[stat]
    stat_doc = {"sum": "the sum of the window",
                "evens": "how many elements of the window are even (0 and negative even numbers count)",
                "distinct": "the number of distinct values in the window",
                "mean": "the window sum floor-divided by k (rounded DOWN, so -7 // 2 == -4)"}[stat]
    base = {"sum": "window_sums", "evens": "window_even_counts", "distinct": "window_distinct",
            "mean": "window_floor_means"}[stat]
    name = base if not as_count else rng.choice(["count_windows_at_least", "windows_reaching", "count_strong_windows"])
    if as_count:
        what = f"Return how many windows have {stat_doc} >= {t}."
        ret, empty, final = "int", "0", f"return sum(1 for s in stats if s >= {t})\n"
    else:
        what = f"Return, for each window from left to right, {stat_doc}."
        ret, empty, final = "list[int]", "[]", "return stats\n"
    doc = f"""
    Slide a window of length `k` over `nums`.

    The windows are nums[i:i + k] for i = 0, 1, ..., len(nums) - k.
    {what}
    If k <= 0 or k > len(nums) there are no windows: return {empty}.
    """
    body = (f"if k <= 0 or k > len(nums):\n    return {empty}\nstats = []\n"
            f"for i in range(len(nums) - k + 1):\n    w = nums[i:i + k]\n    stats.append({expr})\n" + final)
    stat_mut = {"sum": ("sum(w)", "sum(w[1:])"), "evens": ("x % 2 == 0", "x % 2 == 0 and x != 0"),
                "distinct": ("len(set(w))", "len(w)"), "mean": ("sum(w) // k", "int(sum(w) / k)")}[stat]
    mutations = [("range(len(nums) - k + 1)", "range(len(nums) - k)"),
                 ("if k <= 0 or k > len(nums):", "if k > len(nums):"),
                 stat_mut]
    if as_count:
        mutations.insert(0, (f"s >= {t}", f"s > {t}"))
    lo = -10 if stat == "mean" else 0
    nums = _ints(rng, 9, lo, 12)
    nums2 = [rng.choice([0, 2, 3, 3, 4, -4, 7]) for _ in range(7)]
    cases = [
        ("test_typical", [(nums, k_hint)]),
        ("test_window_of_one", [(nums[:5], 1)]),
        ("test_whole_list_is_one_window", [(nums[:4], 4)]),
        ("test_repeats_zero_and_negatives", [(nums2, 2), (nums2, 3)]),
        ("test_no_windows", [(nums[:3], 4), (nums[:3], 0), ([], 1)]),
        ("test_longer_window", [(nums, k_hint + 2)]),
    ]
    diff = "medium" if as_count or stat == "mean" else "easy"
    return _Spec(name, "nums: list[int], k: int", ret, doc, body, cases, [([1, 2, 3, 4, 6], 2)],
                 mutations, diff)


@_family("dedupe_ordered")
def _dedupe_ordered(rng: random.Random) -> _Spec:
    """Remove duplicates keeping first or last occurrence, case-sensitive or not."""
    keep_first = rng.random() < 0.5
    ci = rng.random() < 0.5
    name = rng.choice(["unique_in_order", "remove_duplicates", "distinct_items", "dedupe"])
    if not keep_first:
        name = rng.choice(["keep_last_unique", "dedupe_keep_last", "latest_distinct"])
    key = "item.lower()" if ci else "item"
    alt_key = "item" if ci else "item.lower()"
    cmp_doc = ("compared case-insensitively (two strings are duplicates when their .lower()\n"
               "    forms are equal)" if ci else "compared exactly (case-sensitive: 'a' and 'A' differ)")
    order_doc = ("Kept elements stay in the order of their positions in `items`."
                 if keep_first else
                 "Kept elements appear in the order of the positions of those last occurrences.")
    doc = f"""
    Return the strings of `items` with duplicates removed; strings are {cmp_doc}.

    For every group of duplicates only the {"FIRST" if keep_first else "LAST"} occurrence is kept, exactly as
    it is spelled there. {order_doc}
    The input list is not modified; an empty list gives [].
    """
    loop = "items" if keep_first else "reversed(items)"
    ret = "out" if keep_first else "out[::-1]"
    body = (f"seen = set()\nout = []\nfor item in {loop}:\n    key = {key}\n    if key in seen:\n"
            f"        continue\n    seen.add(key)\n    out.append(item)\nreturn {ret}\n")
    mutations = [(f"key = {key}", f"key = {alt_key}"), (f"return {ret}", "return sorted(out)")]
    if keep_first:
        mutations.append(("for item in items:", "for item in reversed(items):"))
    else:
        mutations.append(("return out[::-1]", "return out"))
        mutations.append(("for item in reversed(items):", "for item in items:"))
    ws = rng.sample(WORDS, 4)
    a, b, c, d = ws
    cases = [
        ("test_no_duplicates", _one([a, b, c])),
        ("test_exact_duplicates", _one([a, b, a, c, b, a])),
        ("test_case_variants", _one([a, a.upper(), b, a.title(), b.upper()])),
        ("test_all_same", _one([d, d, d])),
        ("test_mixed", _one([c, d.upper(), a, d, c.title(), b, a])),
        ("test_empty", _one([])),
    ]
    diff = "easy" if keep_first and not ci else "medium"
    return _Spec(name, "items: list[str]", "list[str]", doc, body, cases, [(["x", "Y", "x", "y"],)],
                 mutations, diff)


@_family("group_runs")
def _group_runs(rng: random.Random) -> _Spec:
    """Group consecutive numbers sharing a key (parity / sign / tens / mod 3), optional min run length."""
    key = rng.choice(["parity", "sign", "tens", "mod3"])
    m = rng.choice([1, 1, 2, 3])
    expr = {"parity": "x % 2", "sign": "(x > 0) - (x < 0)", "tens": "x // 10", "mod3": "x % 3"}[key]
    key_doc = {"parity": "parity (even vs odd)",
               "sign": "sign (negative, zero and positive are three different keys)",
               "tens": "x // 10 (floor division, so -1 // 10 == -1 and 9 // 10 == 0)",
               "mod3": "x % 3 (Python modulo: always 0, 1 or 2, so -1 % 3 == 2)"}[key]
    name = rng.choice({"parity": ["parity_runs", "group_by_parity"], "sign": ["sign_runs", "group_by_sign"],
                       "tens": ["decade_runs", "group_by_tens"], "mod3": ["mod3_runs", "group_by_mod3"]}[key])
    filt = f"Runs shorter than {m} elements are left out of the result." if m > 1 else "Every run is kept."
    doc = f"""
    Split `nums` into maximal runs of CONSECUTIVE elements that share the same
    {key_doc}.

    Return the runs as lists, in order; together (before filtering) they contain
    every element exactly once. {filt}
    An empty list gives [].
    """
    final = f"return [g for g in groups if len(g) >= {m}]\n" if m > 1 else "return groups\n"
    body = (f"groups = []\nprev = None\nfor x in nums:\n    key = {expr}\n    if groups and key == prev:\n"
            f"        groups[-1].append(x)\n    else:\n        groups.append([x])\n    prev = key\n" + final)
    mutations = [("prev = key", "prev = x"),
                 ("for x in nums:", f"for x in sorted(nums, key=lambda x: {expr}):")]
    key_mut = {"sign": ("(x > 0) - (x < 0)", "x > 0"), "tens": ("x // 10", "abs(x) // 10"),
               "mod3": ("x % 3", "abs(x) % 3"), "parity": ("x % 2", "x % 2 == 0 and x != 0")}[key]
    mutations.append(key_mut)
    if m > 1:
        mutations.insert(0, (f"len(g) >= {m}", f"len(g) > {m}"))
    nums = _ints(rng, 12, -15, 30)
    runs = [x for v in _ints(rng, 4, -12, 25) for x in [v, v + 2, v + 4][: rng.randint(1, 3)]]
    cases = [
        ("test_mixed_values", _one(nums)),
        ("test_constructed_runs", _one(runs)),
        ("test_negatives_and_zero", _one([-1, -3, 0, 0, 2, -10, -11, 9, 10, 4])),
        ("test_all_one_run", _one([6, 6, 6])),
        ("test_single_element", _one([rng.randint(-5, 5)])),
        ("test_empty", _one([])),
    ]
    return _Spec(name, "nums: list[int]", "list[list[int]]", doc, body, cases, [([1, 3, 2, 4, 5],)],
                 mutations, "medium" if m > 1 or key in ("tens", "mod3") else "easy")


# --------------------------------------------------------------------------- counting / frequency


@_family("top_k_frequent")
def _top_k_frequent(rng: random.Random) -> _Spec:
    """k most frequent values with a chosen tie rule."""
    tie = rng.choice(["small", "large", "first"])
    name = rng.choice(["top_k_frequent", "most_frequent", "frequent_values", "commonest"])
    tie_doc = {"small": "the smaller value first", "large": "the larger value first",
               "first": "earlier first appearance in `nums` first"}[tie]
    sort = {"small": "sorted(counts, key=lambda v: (-counts[v], v))",
            "large": "sorted(counts, key=lambda v: (-counts[v], -v))",
            "first": "sorted(counts, key=lambda v: -counts[v])"}[tie]
    doc = f"""
    Return the `k` most frequent values of `nums`, most frequent first.

    Values with equal frequency are ordered with {tie_doc}.
    If k exceeds the number of distinct values, return all distinct values in
    that order. If k <= 0 or nums is empty, return [].
    """
    body = (f"counts = {{}}\nfor x in nums:\n    counts[x] = counts.get(x, 0) + 1\norder = {sort}\n"
            f"return order[:k] if k > 0 else []\n")
    alt = {"small": "sorted(counts, key=lambda v: (-counts[v], -v))",
           "large": "sorted(counts, key=lambda v: (-counts[v], v))",
           "first": "sorted(counts, key=lambda v: (-counts[v], v))"}[tie]
    mutations = [(sort, alt), ("order[:k] if k > 0 else []", "order[:k]"),
                 ("-counts[v]", "counts[v]")]
    if tie == "first":
        mutations.append((sort, "sorted(counts, key=lambda v: (-counts[v], -v))"))
    nums = [rng.choice([3, 9, 1, 7, 4, 4, 9]) for _ in range(9)] + [rng.randint(10, 20)]
    tied = [5, 2, 8, 2, 5, 8, 1]
    cases = [
        ("test_clear_winner", [([4, 1, 4, 2, 4, 1], 1)]),
        ("test_ties", [(tied, 2), (tied, 3)]),
        ("test_random_list", [(nums, 2), (nums, 3)]),
        ("test_k_larger_than_distinct", [(tied, 10)]),
        ("test_non_positive_k", [(nums, 0), (nums, -1)]),
        ("test_empty", [([], 3)]),
        ("test_negative_values", [([-1, -2, -2, -1, 0], 2)]),
    ]
    return _Spec(name, "nums: list[int], k: int", "list[int]", doc, body, cases, [([3, 1, 3, 2, 1], 2)],
                 mutations, "medium" if tie == "first" else "easy")


@_family("modal_char")
def _modal_char(rng: random.Random) -> _Spec:
    """Most common character with case folding, filter and tie rule varying."""
    ci = rng.random() < 0.5
    letters = rng.random() < 0.5
    tie = rng.choice(["min", "max", "first"])
    name = rng.choice(["most_common_char", "dominant_letter", "modal_char", "top_symbol"])
    filt_doc = ("Only ASCII letters (a-z, A-Z) are counted; every other character is ignored."
                if letters else "Every character that is not whitespace (str.isspace) is counted.")
    case_doc = ("Counting is case-insensitive ('A' and 'a' are the same character) and the\n"
                "    answer is returned in lowercase." if ci else
                "Counting is case-sensitive: 'A' and 'a' are different characters.")
    tie_doc = {"min": "the smallest one in ordinary string order (code point)",
               "max": "the largest one in ordinary string order (code point)",
               "first": "the one whose first counted occurrence comes earliest in `text`"}[tie]
    doc = f"""
    Return the character that occurs most often in `text`.

    {filt_doc}
    {case_doc}
    If several characters share the highest count, return {tie_doc}.
    If no character is counted, return ''.
    """
    filt = "not (ch.isascii() and ch.isalpha())" if letters else "ch.isspace()"
    fold = "    ch = ch.lower()\n" if ci else ""
    pick = {"min": "min(cands)", "max": "max(cands)", "first": "cands[0]"}[tie]
    body = (f"counts = {{}}\nfor ch in text:\n    if {filt}:\n        continue\n{fold}"
            f"    counts[ch] = counts.get(ch, 0) + 1\nif not counts:\n    return \"\"\nbest = max(counts.values())\n"
            f"cands = [c for c in counts if counts[c] == best]\nreturn {pick}\n")
    alt_pick = {"min": "max(cands)", "max": "min(cands)", "first": "min(cands)"}[tie]
    mutations = [(f"return {pick}", f"return {alt_pick}"), ("if not counts:\n    return \"\"\n", "")]
    if ci:
        mutations.append((fold, ""))
    else:
        mutations.append(("    counts[ch] = counts.get(ch, 0) + 1\n",
                          "    ch = ch.lower()\n    counts[ch] = counts.get(ch, 0) + 1\n"))
    if letters:
        mutations.append(("not (ch.isascii() and ch.isalpha())", "not ch.isalpha()"))
    else:
        mutations.append(("ch.isspace()", 'ch == " "'))
    w = rng.choice(WORDS)
    cases = [
        ("test_simple_word", _one(w + w[-1])),
        ("test_ties", _one("zzaabb", "baab")),
        ("test_case_matters_or_not", _one("aAAbb", "BbbAa")),
        ("test_digits_punctuation_whitespace", _one("1!1! a\t\t\t\n\n", "éé e ..")),
        ("test_phrase", _one(_mixcase(rng, _phrase(rng, 3)))),
        ("test_nothing_counted", _one("", "   ")),
    ]
    return _Spec(name, "text: str", "str", doc, body, cases, [("Mississippi",)], mutations,
                 "medium" if tie == "first" or ci else "easy")


# --------------------------------------------------------------------------- number theory


@_family("digit_ops")
def _digit_ops(rng: random.Random) -> _Spec:
    """Digit arithmetic on abs(n): sum / digital root / nonzero product / alternating sum / count of d."""
    op = rng.choice(["sum", "root", "product", "alt", "count"])
    d = rng.randint(0, 9)
    words = ["zeros", "ones", "twos", "threes", "fours", "fives", "sixes", "sevens", "eights", "nines"]
    name = rng.choice({"sum": ["digit_sum", "sum_of_digits", "add_digits"],
                       "root": ["digital_root", "repeated_digit_sum"],
                       "product": ["nonzero_digit_product", "digit_product"],
                       "alt": ["alternating_digit_sum", "alt_digit_sum"],
                       "count": [f"count_{words[d]}", f"how_many_{words[d]}"]}[op])
    doc = {
        "sum": "Return the sum of the decimal digits of `n`. The sign is ignored (-123 -> 6); 0 -> 0.",
        "root": ("Repeatedly replace abs(n) by the sum of its decimal digits until a single\n"
                 "digit remains, and return that digit. 0 -> 0; the sign of n is ignored."),
        "product": ("Return the product of the NON-ZERO decimal digits of abs(n). Zero digits are\n"
                    "skipped; when there is no non-zero digit (n == 0) the result is 1."),
        "alt": ("Return the alternating sum of the decimal digits of abs(n), starting from the\n"
                "LAST (units) digit with a plus sign: units - tens + hundreds - ...\n"
                "For example 1234 -> 4 - 3 + 2 - 1 = 2. 0 -> 0."),
        "count": (f"Return how many times the digit {d} occurs in the decimal representation\n"
                  f"of abs(n) (written without leading zeros; 0 is written '0')."),
    }[op]
    body = {
        "sum": "return sum(int(c) for c in str(abs(n)))\n",
        "root": "n = abs(n)\nwhile n >= 10:\n    n = sum(int(c) for c in str(n))\nreturn n\n",
        "product": 'p = 1\nfor c in str(abs(n)):\n    if c != "0":\n        p *= int(c)\nreturn p\n',
        "alt": ("digits = str(abs(n))[::-1]\n"
                "return sum(int(c) if i % 2 == 0 else -int(c) for i, c in enumerate(digits))\n"),
        "count": f"return str(abs(n)).count({str(d)!r})\n",
    }[op]
    mutations = {
        "sum": [("str(abs(n))", "str(n)"), ("str(abs(n))", "str(abs(n))[1:]")],
        "root": [("n = abs(n)\n", ""), ("while n >= 10:", "while n > 10:"), ("while n >= 10:", "if n >= 10:")],
        "product": [('    if c != "0":\n        p *= int(c)', "    p *= int(c)"), ("p = 1", "p = 0"),
                    ("str(abs(n))", "str(n)")],
        "alt": [("str(abs(n))[::-1]", "str(abs(n))"), ("i % 2 == 0", "i % 2 == 1"), ("str(abs(n))", "str(n)")],
        "count": [("str(abs(n))", "str(abs(n))[:-1]"), ("str(abs(n))", "str(abs(n))[1:]"),
                  (f".count({str(d)!r})", f".count({str((d + 1) % 10)!r})")],
    }[op]
    big = int(f"{d}{rng.randint(10, 99)}{d}0{rng.randint(1, 9)}{d}")
    cases = [
        ("test_small", _one(rng.randint(1, 9), rng.randint(10, 99))),
        ("test_with_zeros", _one(1000 + rng.randint(0, 9) * 10, 10, 19)),
        ("test_negative", _one(-rng.randint(100, 9999))),
        ("test_zero", _one(0)),
        ("test_large", _one(big, rng.randint(10 ** 8, 10 ** 9))),
        ("test_repeated_digit", _one(int(str(d or 7) * 5), d)),
    ]
    return _Spec(name, "n: int", "int", doc, body, cases, [(4096,)], mutations,
                 "easy" if op in ("sum", "count") else "medium")


@_family("base_convert")
def _base_convert(rng: random.Random) -> _Spec:
    """Integer <-> base-B string, B in 2..36, sign and digit case conventions."""
    b = rng.choice([2, 3, 5, 7, 8, 12, 16, 20, 36])
    to_str = rng.random() < 0.5
    upper = rng.random() < 0.5
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"[:b]
    if upper:
        alphabet = alphabet.upper()
    letters = b > 10
    if to_str:
        name = rng.choice([f"to_base{b}", f"int_to_base{b}", f"encode_base{b}"])
        letter_doc = (f" and values 10..{b - 1} as the {'UPPERCASE' if upper else 'lowercase'} letters "
                      f"{alphabet[10]}..{alphabet[-1]}" if letters else "")
        doc = f"""
        Return the base-{b} representation of the integer `n` as a string.

        Digit values 0-9 are written '0'-'9'{letter_doc}.
        No prefix and no leading zeros; 0 is '0'. Negative numbers get a leading '-'
        (for example -{b} is '-10').
        """
        body = (f"digits = {alphabet!r}\nif n == 0:\n    return \"0\"\nsign = \"-\" if n < 0 else \"\"\n"
                f"n = abs(n)\nout = []\nwhile n:\n    n, r = divmod(n, {b})\n    out.append(digits[r])\n"
                f"return sign + \"\".join(reversed(out))\n")
        mutations = [("if n == 0:\n    return \"0\"\n", ""), ('"".join(reversed(out))', '"".join(out)'),
                     ('sign = "-" if n < 0 else ""', 'sign = ""')]
        if letters:
            mutations.append((f"digits = {alphabet!r}", f"digits = {alphabet.swapcase()!r}"))
        cases = [
            ("test_small_values", _one(1, b - 1, b)),
            ("test_zero", _one(0)),
            ("test_larger_value", _one(rng.randint(b ** 3, b ** 5))),
            ("test_negative", _one(-rng.randint(b + 1, b ** 3))),
            ("test_powers", _one(b ** 4, b ** 2 - 1)),
            ("test_random_values", _one(rng.randint(100, 5000), rng.randint(5000, 10 ** 6))),
        ]
        sig, ret, ex = "n: int", "str", [(b * b + 1,)]
    else:
        name = rng.choice([f"from_base{b}", f"parse_base{b}", f"decode_base{b}"])
        letter_doc = (f"; values 10..{b - 1} are letters, accepted in upper OR lower case" if letters else "")
        doc = f"""
        Parse `text`, an integer written in base {b}, and return its value as an int.

        Digits are '0'-'9'{letter_doc}.
        Surrounding whitespace is ignored and an optional leading '-' makes the value
        negative. You may assume the rest is valid: at least one digit, each < {b}.
        """
        lower = alphabet.lower()
        body = (f"digits = {lower!r}\ns = text.strip().lower()\nneg = s.startswith(\"-\")\nif neg:\n    s = s[1:]\n"
                f"value = 0\nfor ch in s:\n    value = value * {b} + digits.index(ch)\nreturn -value if neg else value\n")
        mutations = [("return -value if neg else value", "return value"), ("text.strip().lower()", "text.lower()"),
                     (f"value * {b} + digits.index(ch)", f"value + {b} * digits.index(ch)")]
        if letters:
            mutations.insert(0, ("text.strip().lower()", "text.strip()"))

        def enc(v: int) -> str:
            out = ""
            while v:
                v, r = divmod(v, b)
                out = alphabet[r] + out
            return out or "0"

        v1, v2 = rng.randint(b ** 2, b ** 4), rng.randint(10 ** 4, 10 ** 6)
        cases = [
            ("test_single_digit", _one(alphabet[1], alphabet[-1])),
            ("test_zero", _one("0", "-0")),
            ("test_multi_digit", _one(enc(v1), enc(v2))),
            ("test_negative", _one("-" + enc(v1))),
            ("test_whitespace", _one(f"  {enc(v2)}\n")),
            ("test_other_case", _one(enc(v2).swapcase(), enc(b ** 3 + b - 1).swapcase())),
        ]
        sig, ret, ex = "text: str", "int", [("10",), ("-" + enc(b * 3 + 1),)]
    return _Spec(name, sig, ret, doc, body, cases, ex, mutations, "medium" if letters else "easy")


@_family("gcd_lcm")
def _gcd_lcm(rng: random.Random) -> _Spec:
    """gcd of a list / lcm of a list / count of coprime pairs, with zero and sign rules."""
    op = rng.choice(["gcd", "lcm", "coprime"])
    if op == "gcd":
        name = rng.choice(["gcd_all", "list_gcd", "common_divisor"])
        doc = """
        Return the greatest common divisor of all integers in `nums`, as a
        non-negative int. Signs are ignored. gcd(0, x) == abs(x), so zeros do not
        change the result; an empty list, or a list of only zeros, gives 0.
        """
        body = "import math\nresult = 0\nfor x in nums:\n    result = math.gcd(result, x)\nreturn result\n"
        mutations = [("result = 0", "result = 1"), ("result = 0", "result = nums[0]"),
                     ("    result = math.gcd(result, x)", "    result = math.gcd(result, x) if x else 1")]
    elif op == "lcm":
        name = rng.choice(["lcm_all", "list_lcm", "common_multiple"])
        doc = """
        Return the least common multiple of all integers in `nums`, as a
        non-negative int. Signs are ignored. If any element is 0 the result is 0.
        The lcm of an empty list is 1.
        """
        body = ("import math\nresult = 1\nfor x in nums:\n    if x == 0:\n        return 0\n"
                "    result = abs(result * x) // math.gcd(result, x)\nreturn result\n")
        mutations = [("abs(result * x)", "result * x"),
                     ("result = abs(result * x) // math.gcd(result, x)", "result = abs(result * x)"),
                     ("result = 1\n", "result = 0 if not nums else 1\n")]
    else:
        name = rng.choice(["count_coprime_pairs", "coprime_pairs", "relatively_prime_pairs"])
        doc = """
        Return the number of index pairs (i, j) with i < j such that
        gcd(nums[i], nums[j]) == 1. Signs are ignored; gcd(0, 0) == 0 and
        gcd(0, x) == abs(x), so 0 is coprime only with 1 and -1.
        """
        body = ("import math\ncount = 0\nfor i in range(len(nums)):\n    for j in range(i + 1, len(nums)):\n"
                "        if math.gcd(nums[i], nums[j]) == 1:\n            count += 1\nreturn count\n")
        mutations = [("range(i + 1, len(nums))", "range(i, len(nums))"), ("== 1:", "<= 1:"),
                     ("range(i + 1, len(nums))", "range(len(nums))")]
    f = rng.choice([2, 3, 5, 6])
    multiples = [f * rng.randint(1, 9) for _ in range(4)]
    cases = [
        ("test_common_factor", _one(multiples)),
        ("test_coprime_values", _one([rng.choice([7, 11, 13]), 4, 9])),
        ("test_with_zero", _one([0, multiples[0]], [0, 0], [0, 1, 5])),
        ("test_negative_values", _one([-multiples[1], multiples[2]], [-1, 1, 6])),
        ("test_single_and_empty", _one([multiples[3]], [1], [])),
        ("test_random", _one(_ints(rng, 6, 1, 30))),
    ]
    return _Spec(name, "nums: list[int]", "int", doc, body, cases, [([12, 18, 30],)], mutations,
                 "medium" if op != "gcd" else "easy")


@_family("primes_in_range")
def _primes_in_range(rng: random.Random) -> _Spec:
    """Primes in a range: list / count / sum, inclusive or half-open bounds."""
    out = rng.choice(["list", "count", "sum"])
    inclusive = rng.random() < 0.5
    name = rng.choice({"list": ["primes_between", "list_primes", "primes_in_range"],
                       "count": ["count_primes", "prime_count"], "sum": ["prime_sum", "sum_primes"]}[out])
    bound = "lo <= m <= hi (both ends included)" if inclusive else "lo <= m < hi (hi excluded)"
    out_doc = {"list": "a list of the primes among them, in increasing order",
               "count": "how many of them are prime", "sum": "the sum of the primes among them"}[out]
    empty = "[]" if out == "list" else "0"
    doc = f"""
    Consider the integers m with {bound}.
    Return {out_doc}.

    A prime is an integer greater than 1 whose only positive divisors are 1 and
    itself, so 0, 1 and negative numbers are never prime. If the range is empty
    (for example lo > hi) the result is {empty}.
    """
    stop = "hi + 1" if inclusive else "hi"
    final = {"list": "return primes\n", "count": "return len(primes)\n", "sum": "return sum(primes)\n"}[out]
    body = (f"def is_prime(m):\n    if m < 2:\n        return False\n    i = 2\n    while i * i <= m:\n"
            f"        if m % i == 0:\n            return False\n        i += 1\n    return True\n"
            f"primes = [m for m in range(lo, {stop}) if is_prime(m)]\n" + final)
    mutations = [(f"range(lo, {stop})", f"range(lo, {'hi' if inclusive else 'hi + 1'})"),
                 ("while i * i <= m:", "while i * i < m:"), ("if m < 2:", "if m < 1:"),
                 (f"range(lo, {stop})", f"range(lo + 1, {stop})")]
    lo = rng.randint(10, 60)
    cases = [
        ("test_small_range", [(1, 10)]),
        ("test_bounds_are_primes", [(2, 13), (lo, lo + 1)]),
        ("test_squares_are_not_prime", [(24, 26), (48, 50)]),
        ("test_negative_start", [(-10, 3)]),
        ("test_empty_range", [(20, 10), (14, 14)]),
        ("test_random_range", [(lo, lo + rng.randint(10, 40))]),
        ("test_exact_prime_bounds", [(11, 11), (3, 7)]),
    ]
    return _Spec(name, "lo: int, hi: int", "list[int]" if out == "list" else "int", doc, body, cases,
                 [(10, 20)], mutations, "easy" if inclusive else "medium")


@_family("bit_ops")
def _bit_ops(rng: random.Random) -> _Spec:
    """Bit tricks: popcount / next power of two / reverse within width / gray encode/decode / hamming."""
    op = rng.choice(["popcount", "nextpow", "reverse", "gray", "ungray", "hamming"])
    w = rng.choice([4, 8, 12, 16])
    if op == "popcount":
        name = rng.choice(["count_set_bits", "popcount", "bit_weight"])
        doc = "Return the number of 1 bits in the binary representation of `n` (n >= 0)."
        body = "count = 0\nwhile n:\n    n &= n - 1\n    count += 1\nreturn count\n"
        mutations = [("n &= n - 1", "n >>= 1"), ("while n:", "while n > 1:")]
        cases = [("test_zero", _one(0)), ("test_powers_of_two", _one(1, 64, 1024)),
                 ("test_all_ones", _one(255, 7)), ("test_mixed", _one(rng.randint(100, 10 ** 6))),
                 ("test_large", _one(2 ** 40 + 5))]
        sig = "n: int"
    elif op == "nextpow":
        name = rng.choice(["next_power_of_two", "round_up_pow2", "ceil_pow2"])
        doc = ("Return the smallest power of two that is >= `n`.\n"
               "For n <= 1 the answer is 1 (2 ** 0). Exact powers of two return themselves.")
        body = "p = 1\nwhile p < n:\n    p <<= 1\nreturn p\n"
        mutations = [("while p < n:", "while p <= n:"), ("p = 1", "p = 2")]
        cases = [("test_exact_powers", _one(1, 8, 1024)), ("test_between_powers", _one(3, 1000, 129)),
                 ("test_small_and_negative", _one(0, -5)), ("test_random", _one(rng.randint(10, 10 ** 6))),
                 ("test_large", _one(2 ** 33 + 1))]
        sig = "n: int"
    elif op == "reverse":
        name = rng.choice([f"reverse_bits{w}", f"mirror_bits{w}", f"bit_reverse{w}"])
        doc = (f"Reverse the lowest {w} bits of `n` (0 <= n < 2 ** {w}): bit 0 becomes bit {w - 1},\n"
               f"bit 1 becomes bit {w - 2}, and so on. Return the result as an int.")
        body = f"out = 0\nfor _ in range({w}):\n    out = (out << 1) | (n & 1)\n    n >>= 1\nreturn out\n"
        mutations = [(f"range({w})", f"range({w - 1})"), ("(out << 1) | (n & 1)", "(out << 1) | n")]
        cases = [("test_zero", _one(0)), ("test_lowest_bit", _one(1)), ("test_highest_bit", _one(2 ** (w - 1))),
                 ("test_all_ones", _one(2 ** w - 1)), ("test_random", _one(rng.randint(2, 2 ** w - 2))),
                 ("test_pattern", _one(int("10" * (w // 2), 2)))]
        sig = "n: int"
    elif op == "gray":
        name = rng.choice(["to_gray_code", "gray_encode", "binary_to_gray"])
        doc = "Return the reflected binary Gray code of `n` (n >= 0): n XOR (n shifted right by 1)."
        body = "return n ^ (n >> 1)\n"
        mutations = [("n ^ (n >> 1)", "n ^ (n << 1)"), ("n ^ (n >> 1)", "n ^ (n >> 2)"), ("n ^ (n >> 1)", "n | (n >> 1)")]
        cases = [("test_zero_and_one", _one(0, 1)), ("test_small", _one(2, 3, 4)), ("test_seven", _one(7)),
                 ("test_random", _one(rng.randint(10, 10 ** 5))), ("test_power_of_two", _one(256))]
        sig = "n: int"
    elif op == "ungray":
        name = rng.choice(["from_gray_code", "gray_decode", "gray_to_binary"])
        doc = ("Inverse of the reflected binary Gray code: return the m >= 0 with\n"
               "m ^ (m >> 1) == g. (Equivalently XOR together g, g >> 1, g >> 2, ...)")
        body = "result = 0\nwhile g:\n    result ^= g\n    g >>= 1\nreturn result\n"
        mutations = [("g >>= 1", "g >>= 2"), ("result ^= g", "result |= g")]
        cases = [("test_zero_and_one", _one(0, 1)), ("test_small", _one(2, 3, 6)), ("test_four", _one(4)),
                 ("test_random", _one(rng.randint(10, 10 ** 5))), ("test_all_ones", _one(255))]
        sig = "g: int"
    else:
        name = rng.choice(["hamming_distance", "bit_difference", "differing_bits"])
        doc = "Return how many bit positions differ between the non-negative ints `a` and `b`."
        body = 'return bin(a ^ b).count("1")\n'
        mutations = [("a ^ b", "a & b"), ("a ^ b", "a | b"), ('bin(a ^ b).count("1")', 'len(bin(a ^ b)) - 2')]
        x = rng.randint(0, 1000)
        cases = [("test_equal", [(x, x)]), ("test_zero", [(0, 0), (0, 5)]), ("test_one_bit", [(8, 0), (6, 7)]),
                 ("test_random", [(rng.randint(0, 10 ** 6), rng.randint(0, 10 ** 6))]),
                 ("test_overlapping", [(12, 10), (255, 1)])]
        sig = "a: int, b: int"
    ex = [(1, 4)] if op == "hamming" else [(6,)]
    return _Spec(name, sig, "int", doc, body, cases, ex, mutations,
                 "easy" if op in ("popcount", "gray", "hamming") else "medium")


# --------------------------------------------------------------------------- sorting


@_family("sort_records")
def _sort_records(rng: random.Random) -> _Spec:
    """Order (name, age, score) records by a composite key (field, direction, tie rule vary)."""
    primary = rng.choice(["age", "score"])
    desc = rng.random() < 0.5
    other = "score" if primary == "age" else "age"
    second = rng.choice(["name", "name_ci", "other_asc", "other_desc"])
    idx = {"age": 1, "score": 2}
    k1 = f"{'-' if desc else ''}r[{idx[primary]}]"
    k1_flip = f"{'' if desc else '-'}r[{idx[primary]}]"
    k2 = {"name": "r[0]", "name_ci": "r[0].lower()", "other_asc": f"r[{idx[other]}]",
          "other_desc": f"-r[{idx[other]}]"}[second]
    second_doc = {"name": "name in ascending string order (case-sensitive, so 'Zed' sorts before 'amy')",
                  "name_ci": "name in ascending order compared case-insensitively",
                  "other_asc": f"{other} ascending", "other_desc": f"{other} descending"}[second]
    name = rng.choice(["sort_people", "order_records", "arrange_rows", "sort_entries"])
    doc = f"""
    Each record is a (name, age, score) tuple. Return the names ordered by:

    1. {primary} {"descending" if desc else "ascending"};
    2. then, among equal {primary}, by {second_doc};
    3. records equal on both keys keep their original relative order.

    An empty list gives [].
    """
    body = f"return [r[0] for r in sorted(records, key=lambda r: ({k1}, {k2}))]\n"
    k2_alt = {"name": "r[0].lower()", "name_ci": "r[0]", "other_asc": f"-r[{idx[other]}]",
              "other_desc": f"r[{idx[other]}]"}[second]
    mutations = [(f"({k1}, {k2})", f"({k1_flip}, {k2})"), (f"({k1}, {k2})", f"{k1}"),
                 (f"({k1}, {k2})", f"({k2}, {k1})"), (f"({k1}, {k2})", f"({k1}, {k2_alt})")]
    people = rng.sample(NAMES, 8)

    def rec(nm: str) -> tuple:
        return (nm, rng.choice([20, 30, 30, 41]), rng.choice([5, 7, 7, 9]))

    recs = [rec(nm) for nm in people[:6]]
    tie = [(people[0], 30, 7), (people[1], 30, 5), (people[2], 30, 9), (people[3].lower(), 25, 7),
           (people[4].upper(), 25, 7)]
    cases = [
        ("test_random_records", _one(recs)),
        ("test_ties_on_primary", _one(tie)),
        ("test_case_of_names", _one([("bob", 30, 7), ("Amy", 30, 7), ("carl", 30, 7), ("Dan", 30, 7)])),
        ("test_duplicates_keep_order", _one([("x", 1, 1), ("y", 2, 2), ("x", 1, 1)])),
        ("test_single", _one([recs[0]])),
        ("test_empty", _one([])),
    ]
    return _Spec(name, "records: list[tuple[str, int, int]]", "list[str]", doc, body, cases,
                 [([("ann", 30, 5), ("bo", 25, 9), ("cy", 30, 9)],)], mutations,
                 "medium" if second in ("name_ci", "other_desc") or desc else "easy")


# --------------------------------------------------------------------------- matrices / grids


@_family("matrix_transform")
def _matrix_transform(rng: random.Random) -> _Spec:
    """Transpose / rotate cw / rotate ccw / anti-transpose a rectangular matrix."""
    op = rng.choice(["transpose", "cw", "ccw", "anti"])
    name = rng.choice({"transpose": ["transpose", "swap_axes"], "cw": ["rotate_clockwise", "turn_right"],
                       "ccw": ["rotate_counterclockwise", "turn_left"],
                       "anti": ["anti_transpose", "reflect_anti_diagonal"]}[op])
    op_doc = {
        "transpose": "the transpose: result[i][j] == grid[j][i] (C rows of R columns)",
        "cw": "the grid rotated 90 degrees CLOCKWISE: the first column, read bottom to top, becomes the first row",
        "ccw": ("the grid rotated 90 degrees COUNTER-clockwise: the last column, read top to\n"
                "    bottom, becomes the first row"),
        "anti": ("the grid reflected across its anti-diagonal: result[i][j] == grid[R-1-j][C-1-i]\n"
                 "    (C rows of R columns)"),
    }[op]
    comps = {
        "transpose": "[[grid[r][c] for r in range(len(grid))] for c in range(len(grid[0]))]",
        "cw": "[[grid[r][c] for r in range(len(grid) - 1, -1, -1)] for c in range(len(grid[0]))]",
        "ccw": "[[grid[r][c] for r in range(len(grid))] for c in range(len(grid[0]) - 1, -1, -1)]",
        "anti": "[[grid[R - 1 - j][C - 1 - i] for j in range(R)] for i in range(C)]",
    }
    comp = comps[op]
    pre = "R, C = len(grid), len(grid[0])\n" if op == "anti" else ""
    doc = f"""
    `grid` is a rectangular matrix: R rows, each with the same C >= 0 entries.
    Return a NEW matrix that is {op_doc}.

    If R == 0 or C == 0, return []. The input is not modified.
    """
    body = f"if not grid or not grid[0]:\n    return []\n{pre}return {comp}\n"
    swap = {"cw": "ccw", "ccw": "cw", "transpose": "cw", "anti": "transpose"}[op]
    mutations = [("if not grid or not grid[0]:\n    return []\n", ""),
                 (None, f"if not grid or not grid[0]:\n    return []\nreturn {comps[swap]}\n")]
    if op == "anti":
        mutations.append(("grid[R - 1 - j][C - 1 - i]", "grid[C - 1 - j][R - 1 - i]"))
        mutations.append(("for j in range(R)] for i in range(C)", "for j in range(C)] for i in range(R)"))
    else:
        mutations.append(("for c in range(len(grid[0]))", "for c in range(len(grid))")
                         if "for c in range(len(grid[0]))" in comp else
                         ("range(len(grid[0]) - 1, -1, -1)", "range(len(grid) - 1, -1, -1)"))

    def mat(r: int, c: int) -> list[list[int]]:
        return [_ints(rng, c, 0, 9) for _ in range(r)]

    cases = [
        ("test_square", _one(mat(3, 3))),
        ("test_wide", _one(mat(2, 4))),
        ("test_tall", _one(mat(4, 2))),
        ("test_single_row_and_column", _one(mat(1, 3), mat(3, 1))),
        ("test_one_by_one", _one([[rng.randint(0, 9)]])),
        ("test_empty", _one([], [[]])),
    ]
    return _Spec(name, "grid: list[list[int]]", "list[list[int]]", doc, body, cases,
                 [([[1, 2, 3], [4, 5, 6]],)], mutations, "medium" if op == "anti" else "easy")


def _grid(rng: random.Random, rows: int, cols: int, mark: str, density: float) -> list[str]:
    return ["".join(mark if rng.random() < density else "." for _ in range(cols)) for _ in range(rows)]


@_family("neighbor_counts")
def _neighbor_counts(rng: random.Random) -> _Spec:
    """Per-cell count of marked neighbours (4/8-connectivity, mark char, include-self vary)."""
    mark = rng.choice(["#", "*", "1", "@"])
    eight = rng.random() < 0.5
    self_too = rng.random() < 0.3
    name = rng.choice(["neighbor_counts", "count_neighbors", "adjacent_counts", "mark_density"])
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if eight:
        dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    if self_too:
        dirs.append((0, 0))
    alt = dirs[:4] + ([(0, 0)] if self_too else []) if eight else dirs + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    conn = ("the 8 surrounding cells (orthogonal and diagonal)" if eight else
            "the 4 orthogonally adjacent cells (up, down, left, right)")
    self_doc = (" plus the cell itself" if self_too else " (the cell itself is NOT counted)")
    doc = f"""
    `grid` is a list of equal-length strings; a cell is marked when it is '{mark}'.
    Return a matrix of the same shape where each entry is the number of marked
    cells among {conn}{self_doc}.

    Cells outside the grid count as unmarked. An empty grid gives [].
    """
    body = (f"dirs = {dirs!r}\nrows = len(grid)\nout = []\nfor r in range(rows):\n    row = []\n"
            f"    for c in range(len(grid[r])):\n        n = 0\n        for dr, dc in dirs:\n"
            f"            rr, cc = r + dr, c + dc\n"
            f"            if 0 <= rr < rows and 0 <= cc < len(grid[rr]) and grid[rr][cc] == {mark!r}:\n"
            f"                n += 1\n        row.append(n)\n    out.append(row)\nreturn out\n")
    mutations = [("0 <= rr < rows and 0 <= cc < len(grid[rr])", "rr < rows and cc < len(grid[rr])"),
                 (f"dirs = {dirs!r}", f"dirs = {alt!r}")]
    if self_too:
        mutations.append((f"dirs = {dirs!r}", f"dirs = {dirs[:-1]!r}"))
    else:
        mutations.append((f"dirs = {dirs!r}", f"dirs = {[*dirs, (0, 0)]!r}"))
    cases = [
        ("test_small_grid", _one(_grid(rng, 3, 3, mark, 0.5))),
        ("test_wide_grid", _one(_grid(rng, 2, 5, mark, 0.5))),
        ("test_edges_do_not_wrap", _one([mark + "." * 3 + mark, "." * 5, mark + "...."])),
        ("test_all_marked", _one([mark * 3] * 3)),
        ("test_none_marked", _one(["...", "..."])),
        ("test_single_cell_and_empty", _one([mark], [])),
    ]
    return _Spec(name, "grid: list[str]", "list[list[int]]", doc, body, cases, [([f"{mark}.", f".{mark}"],)],
                 mutations, "medium" if eight else "easy")


@_family("count_islands")
def _count_islands(rng: random.Random) -> _Spec:
    """Flood-fill region count (land char, 4/8 connectivity, minimum island size vary)."""
    land = rng.choice(["#", "X", "1", "@"])
    eight = rng.random() < 0.5
    min_size = rng.choice([1, 1, 2, 3])
    name = rng.choice(["count_islands", "count_regions", "count_blobs", "land_masses"])
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)] + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if eight else [])
    alt = dirs[:4] if eight else dirs + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    conn = "8 neighbours (diagonal contact also connects)" if eight else "4 orthogonal neighbours (no diagonals)"
    size_doc = (f"Only islands made of at least {min_size} cells are counted." if min_size > 1 else
                "Every island counts, even a single cell.")
    doc = f"""
    `grid` is a list of equal-length strings where '{land}' is land and any other
    character is water. An island is a maximal group of land cells connected
    through their {conn}.

    Return the number of islands. {size_doc}
    An empty grid has 0 islands.
    """
    body = (f"dirs = {dirs!r}\nseen = set()\ncount = 0\nfor r in range(len(grid)):\n"
            f"    for c in range(len(grid[r])):\n        if grid[r][c] != {land!r} or (r, c) in seen:\n"
            f"            continue\n        seen.add((r, c))\n        stack = [(r, c)]\n        size = 0\n"
            f"        while stack:\n            cr, cc = stack.pop()\n            size += 1\n"
            f"            for dr, dc in dirs:\n                nr, nc = cr + dr, cc + dc\n"
            f"                if 0 <= nr < len(grid) and 0 <= nc < len(grid[nr]) and grid[nr][nc] == {land!r} "
            f"and (nr, nc) not in seen:\n                    seen.add((nr, nc))\n"
            f"                    stack.append((nr, nc))\n        if size >= {min_size}:\n            count += 1\n"
            f"return count\n")
    mutations = [(f"dirs = {dirs!r}", f"dirs = {alt!r}"),
                 ("0 <= nr < len(grid) and 0 <= nc < len(grid[nr])", "nr < len(grid) and nc < len(grid[nr])"),
                 (f"if size >= {min_size}:", f"if size > {min_size}:")]
    if min_size > 1:
        mutations.append((f"if size >= {min_size}:", "if size >= 1:"))
    cases = [
        ("test_random_grid", _one(_grid(rng, 4, 5, land, 0.45))),
        ("test_diagonal_contact", _one([f"{land}..", f".{land}{land}", f"..{land}"])),
        ("test_wrap_around_is_not_contact", _one([f"{land}{land}..{land}{land}", "......", f"{land}....{land}"])),
        ("test_another_random_grid", _one(_grid(rng, 5, 4, land, 0.5))),
        ("test_all_land_or_water", _one([land * 3] * 2, ["..."])),
        ("test_empty", _one([])),
    ]
    return _Spec(name, "grid: list[str]", "int", doc, body, cases, [([f"{land}{land}.", f"..{land}"],)],
                 mutations, "hard" if eight and min_size > 1 else "medium")


@_family("spiral_order")
def _spiral_order(rng: random.Random) -> _Spec:
    """Spiral traversal: clockwise, counter-clockwise, or inside-out."""
    mode = rng.choice(["cw", "ccw", "inward_reversed"])
    name = rng.choice({"cw": ["spiral_order", "spiral_walk"], "ccw": ["spiral_order_ccw", "counter_spiral"],
                       "inward_reversed": ["spiral_inside_out", "unwind_spiral"]}[mode])
    mode_doc = {
        "cw": ("Start at the top-left corner and walk CLOCKWISE: right along the top row,\n"
               "down the right column, left along the bottom row, up the left column, then\n"
               "continue with the next inner ring."),
        "ccw": ("Start at the top-left corner and walk COUNTER-clockwise: down the left column,\n"
                "right along the bottom row, up the right column, left along the top row, then\n"
                "continue with the next inner ring."),
        "inward_reversed": ("Take the clockwise spiral (start top-left, go right along the top row, down the\n"
                            "right column, left along the bottom, up the left side, then inner rings) and\n"
                            "return it in REVERSE order, so the list ends at the top-left element."),
    }[mode]
    doc = f"""
    Return every element of the rectangular matrix `grid` exactly once, in spiral order.

    {mode_doc}
    An empty matrix ([] or rows with no entries) gives [].
    """
    pre = "grid = [list(col) for col in zip(*grid)]\n" if mode == "ccw" else ""
    post = "return result[::-1]\n" if mode == "inward_reversed" else "return result\n"
    body = (f"result = []\nif not grid or not grid[0]:\n    return result\n{pre}"
            "top, bottom, left, right = 0, len(grid) - 1, 0, len(grid[0]) - 1\n"
            "while top <= bottom and left <= right:\n"
            "    for c in range(left, right + 1):\n        result.append(grid[top][c])\n"
            "    for r in range(top + 1, bottom + 1):\n        result.append(grid[r][right])\n"
            "    if top < bottom:\n        for c in range(right - 1, left - 1, -1):\n"
            "            result.append(grid[bottom][c])\n"
            "    if left < right:\n        for r in range(bottom - 1, top, -1):\n            result.append(grid[r][left])\n"
            "    top += 1\n    bottom -= 1\n    left += 1\n    right -= 1\n" + post)
    mutations = [("    if top < bottom:\n        for c", "    if True:\n        for c"),
                 ("    if left < right:\n        for r", "    if True:\n        for r"),
                 ("if not grid or not grid[0]:", "if not grid:")]
    if mode == "ccw":
        mutations.append((pre, ""))
    elif mode == "inward_reversed":
        mutations.append(("return result[::-1]", "return result"))

    def mat(r: int, c: int) -> list[list[int]]:
        start = rng.randint(0, 20)
        return [[start + i * c + j for j in range(c)] for i in range(r)]

    cases = [
        ("test_square", _one(mat(3, 3))),
        ("test_wide", _one(mat(3, 4))),
        ("test_tall", _one(mat(4, 2))),
        ("test_single_row", _one(mat(1, 4))),
        ("test_single_column", _one(mat(3, 1))),
        ("test_empty", _one([], [[]])),
    ]
    return _Spec(name, "grid: list[list[int]]", "list[int]", doc, body, cases, [([[1, 2], [3, 4]],)],
                 mutations, "medium" if mode == "cw" else "hard")


@_family("grid_paths")
def _grid_paths(rng: random.Random) -> _Spec:
    """Count monotone paths through a grid with blocked cells (blocked char and diagonal moves vary)."""
    block = rng.choice(["#", "X", "*"])
    diag = rng.random() < 0.4
    name = rng.choice(["count_paths", "grid_routes", "lattice_paths", "path_count"])
    moves = "one step right, one step down, or one step diagonally down-right" if diag else \
        "one step right or one step down"
    doc = f"""
    `grid` is a list of equal-length strings; '{block}' cells are blocked and every
    other character is open. Count the paths from the top-left cell to the
    bottom-right cell that only visit open cells, where each move is {moves}.

    If the start or end cell is blocked the answer is 0; a 1x1 open grid has
    exactly 1 path. An empty grid (or empty rows) gives 0.
    """
    extra = "        if r > 0 and c > 0:\n            total += ways[r - 1][c - 1]\n" if diag else ""
    body = ("if not grid or not grid[0]:\n    return 0\nrows, cols = len(grid), len(grid[0])\n"
            "ways = [[0] * cols for _ in range(rows)]\nfor r in range(rows):\n    for c in range(cols):\n"
            f"        if grid[r][c] == {block!r}:\n            continue\n        if r == 0 and c == 0:\n"
            "            ways[r][c] = 1\n            continue\n        total = 0\n"
            "        if r > 0:\n            total += ways[r - 1][c]\n        if c > 0:\n            total += ways[r][c - 1]\n"
            f"{extra}        ways[r][c] = total\nreturn ways[rows - 1][cols - 1]\n")
    mutations = [(f"        if grid[r][c] == {block!r}:\n            continue\n        if r == 0 and c == 0:\n"
                  "            ways[r][c] = 1\n            continue\n",
                  "        if r == 0 and c == 0:\n            ways[r][c] = 1\n            continue\n"
                  f"        if grid[r][c] == {block!r}:\n            continue\n"),
                 (f"if grid[r][c] == {block!r}:", "if False:"),
                 ("if not grid or not grid[0]:\n    return 0\n", "")]
    if diag:
        mutations.append((extra, ""))
    else:
        mutations.append(("        ways[r][c] = total\n",
                          "        if r > 0 and c > 0:\n            total += ways[r - 1][c - 1]\n        ways[r][c] = total\n"))

    def g(rows: int, cols: int) -> list[str]:
        cells = [[block if rng.random() < 0.2 else "." for _ in range(cols)] for _ in range(rows)]
        cells[0][0] = cells[-1][-1] = "."
        return ["".join(row) for row in cells]

    cases = [
        ("test_open_grid", _one(["...", "...", "..."], ["....", "...."])),
        ("test_random_blocks", _one(g(4, 4), g(3, 5))),
        ("test_wall_with_gap", _one(["....", f"{block}{block}.{block}", "...."])),
        ("test_blocked_start_or_end", _one([f"{block}..", "..."], ["...", f"..{block}"])),
        ("test_single_cell", _one(["."], [block])),
        ("test_empty", _one([], [""])),
    ]
    return _Spec(name, "grid: list[str]", "int", doc, body, cases, [(["..", ".."],)], mutations,
                 "hard" if diag else "medium")


# --------------------------------------------------------------------------- intervals


@_family("interval_depth")
def _interval_depth(rng: random.Random) -> _Spec:
    """Maximum overlap depth of integer intervals (closed vs half-open; depth vs earliest peak point)."""
    closed = rng.random() < 0.5
    point = rng.random() < 0.5
    name = rng.choice(["busiest_point", "peak_time", "first_peak"] if point else
                      ["max_overlap", "peak_concurrency", "max_depth"])
    cover = ("start <= t <= end (both ends included)" if closed else
             "start <= t < end (end excluded, so start == end covers nothing)")
    if point:
        what = ("Return the SMALLEST point covered by the largest number of intervals, or\n"
                "    None if no point is covered at all (for example when intervals is empty).")
        ret, final = "int | None", "return when\n"
    else:
        what = "Return the largest number of intervals that cover one common point (0 if none)."
        ret, final = "int", "return best\n"
    doc = f"""
    Each (start, end) pair with start <= end covers the integer points t with
    {cover}. Intervals may repeat and are given in any order.

    {what}
    """
    end = "e + 1" if closed else "e"
    body = (f"events = []\nfor s, e in intervals:\n    events.append((s, 1))\n    events.append(({end}, -1))\n"
            "events.sort(key=lambda ev: (ev[0], ev[1]))\nbest = 0\ncur = 0\nwhen = None\nfor t, d in events:\n"
            "    cur += d\n    if cur > best:\n        best = cur\n        when = t\n" + final)
    mutations = [("events.sort(key=lambda ev: (ev[0], ev[1]))", "events.sort(key=lambda ev: (ev[0], -ev[1]))")]
    if closed:
        mutations.append(("events.append((e + 1, -1))", "events.append((e, -1))"))
    else:
        mutations.append(("events.append((e, -1))", "events.append((e + 1, -1))"))
    if point:
        mutations.append(("if cur > best:", "if cur >= best:"))
    else:
        mutations.append(("best = 0\n", "best = 1\n"))
    base = rng.randint(0, 20)
    rand = []
    for _ in range(6):
        s = rng.randint(base, base + 15)
        rand.append((s, s + rng.randint(0, 6)))
    cases = [
        ("test_nested", _one([(base, base + 10), (base + 2, base + 5), (base + 3, base + 4)])),
        ("test_touching_endpoints", _one([(1, 3), (3, 5), (5, 7)])),
        ("test_random_intervals", _one(rand)),
        ("test_two_peaks", _one([(0, 2), (1, 3), (10, 12), (11, 13), (11, 11)])),
        ("test_zero_length", _one([(4, 4), (4, 4)], [(2, 2), (0, 5)])),
        ("test_empty", _one([])),
    ]
    return _Spec(name, "intervals: list[tuple[int, int]]", ret, doc, body, cases, [([(1, 4), (2, 6), (5, 8)],)],
                 mutations, "hard" if point else "medium")


# --------------------------------------------------------------------------- simulations


@_family("stack_machine")
def _stack_machine(rng: random.Random) -> _Spec:
    """Instruction-list stack machine (sub operand order, underflow policy, result form vary)."""
    a_minus_b = rng.random() < 0.5
    stop_on_error = rng.random() < 0.5
    top_only = rng.random() < 0.5
    name = rng.choice(["run_stack", "stack_machine", "execute_ops", "run_program"])
    sub = "a - b" if a_minus_b else "b - a"
    ret_doc = ("the value on top of the final stack, or None if the final stack is empty" if top_only else
               "the final stack as a list, bottom first")
    err_doc = ("execution stops immediately and the function returns None" if stop_on_error else
               "that instruction is skipped (the stack is unchanged) and execution continues")
    doc = f"""
    Run `program`, a list of instruction strings, on an initially empty stack and
    return {ret_doc}.

    Instructions:
    - 'push N' pushes the integer N (N may be negative).
    - 'pop' removes the top value; 'dup' pushes a copy of the top value.
    - 'swap' exchanges the top two values.
    - 'add', 'sub' and 'mul' pop the top value b, then the next value a, and push
      a + b, {sub}, and a * b respectively.

    If an instruction needs more values than the stack holds (one for pop/dup,
    two for swap/add/sub/mul), {err_doc}.
    """
    on_err = "return None" if stop_on_error else "continue"
    final = "return stack[-1] if stack else None\n" if top_only else "return stack\n"
    body = ("stack = []\nneed = {\"pop\": 1, \"dup\": 1, \"swap\": 2, \"add\": 2, \"sub\": 2, \"mul\": 2}\n"
            "for ins in program:\n    parts = ins.split()\n    op = parts[0]\n    if op == \"push\":\n"
            "        stack.append(int(parts[1]))\n        continue\n    if len(stack) < need[op]:\n"
            f"        {on_err}\n    if op == \"pop\":\n        stack.pop()\n    elif op == \"dup\":\n"
            "        stack.append(stack[-1])\n    elif op == \"swap\":\n        stack[-1], stack[-2] = stack[-2], stack[-1]\n"
            "    else:\n        b = stack.pop()\n        a = stack.pop()\n        if op == \"add\":\n"
            "            stack.append(a + b)\n        elif op == \"sub\":\n"
            f"            stack.append({sub})\n        else:\n            stack.append(a * b)\n" + final)
    mutations = [(f"stack.append({sub})", f"stack.append({'b - a' if a_minus_b else 'a - b'})"),
                 ("if len(stack) < need[op]:", "if not stack:"),
                 (f"        {on_err}\n", f"        {'continue' if stop_on_error else 'return None'}\n")]
    if top_only:
        mutations.append(("return stack[-1] if stack else None", "return stack[-1]"))
    else:
        mutations.append(("stack[-1], stack[-2] = stack[-2], stack[-1]", "stack[-1], stack[-2] = stack[-1], stack[-2]"))
    x, y = rng.randint(2, 20), rng.randint(-9, 30)
    cases = [
        ("test_arithmetic", _one([f"push {x}", f"push {y}", "sub", "push 3", "mul"])),
        ("test_dup_and_swap", _one([f"push {y}", f"push {x}", "swap", "dup", "add", "sub"])),
        ("test_underflow_on_binary_op", _one([f"push {x}", "add", f"push {y}", "sub"])),
        ("test_underflow_on_pop", _one(["pop", f"push {x}", "dup", "mul"])),
        ("test_negative_push", _one(["push -4", f"push {x}", "add", f"push {y}", "swap"])),
        ("test_empty_program_and_empty_end", _one([], [f"push {x}", "pop"])),
    ]
    return _Spec(name, "program: list[str]", "int | None" if top_only else "list[int] | None", doc, body, cases,
                 [(["push 2", "push 5", "sub"],)], mutations, "medium" if stop_on_error else "hard")


@_family("round_robin")
def _round_robin(rng: random.Random) -> _Spec:
    """Round-robin CPU scheduling with quantum Q (finish order / finish times / total waiting)."""
    q = rng.randint(1, 4)
    out = rng.choice(["order", "times", "wait"])
    name = rng.choice({"order": ["rr_finish_order", "round_robin_order"],
                       "times": ["rr_finish_times", "round_robin_times"],
                       "wait": ["rr_total_wait", "round_robin_waiting"]}[out])
    out_doc = {"order": "the job names in the order they finish",
               "times": "a dict mapping each job name to its finish time",
               "wait": "the total waiting time: the sum over all jobs of (finish time - burst)"}[out]
    empty = {"order": "[]", "times": "{}", "wait": "0"}[out]
    doc = f"""
    Simulate round-robin scheduling with a time quantum of {q}.

    `jobs` is a list of (name, burst) pairs: all jobs arrive at time 0 and are
    queued in list order; bursts are positive ints and names are unique. Time
    starts at 0. Repeatedly take the job at the FRONT of the queue and run it for
    min({q}, remaining work) time units; if work remains it goes to the BACK of the
    queue, otherwise it finishes at the current time. Switching costs nothing.

    Return {out_doc}. No jobs gives {empty}.
    """
    final = {"order": "return order\n", "times": "return finish\n",
             "wait": "return sum(finish[n] - b for n, b in jobs)\n"}[out]
    body = ("from collections import deque\nqueue = deque([n, b] for n, b in jobs)\nt = 0\norder = []\nfinish = {}\n"
            f"while queue:\n    job = queue.popleft()\n    run = min({q}, job[1])\n    t += run\n    job[1] -= run\n"
            "    if job[1] == 0:\n        order.append(job[0])\n        finish[job[0]] = t\n    else:\n"
            "        queue.append(job)\n" + final)
    mutations = [("        queue.append(job)\n", "        queue.appendleft(job)\n"),
                 (f"run = min({q}, job[1])", f"run = min({q + 1}, job[1])"),
                 ("    t += run\n", "    t += 1\n")]
    if out == "wait":
        mutations.append(("finish[n] - b", "finish[n]"))
    names = rng.sample(["A", "B", "C", "D", "E", "F"], 4)
    jobs = [(nm, rng.randint(1, 7)) for nm in names]
    cases = [
        ("test_random_jobs", _one(jobs)),
        ("test_short_jobs_finish_in_first_round", _one([(names[0], 5), (names[1], 1), (names[2], q)])),
        ("test_single_job", _one([(names[3], 7)])),
        ("test_equal_bursts", _one([(nm, q + 1) for nm in names[:3]])),
        ("test_long_and_short", _one([("long", 3 * q + 2), ("short", 1), ("mid", q + 1)])),
        ("test_no_jobs", _one([])),
    ]
    return _Spec(name, "jobs: list[tuple[str, int]]", {"order": "list[str]", "times": "dict[str, int]",
                                                         "wait": "int"}[out],
                 doc, body, cases, [([("a", 3), ("b", 1)],)], mutations, "medium" if out == "order" else "hard")


@_family("elevator")
def _elevator(rng: random.Random) -> _Spec:
    """Elevator serving requests: FCFS or nearest-first with a tie rule; distance or visit order."""
    policy = rng.choice(["fcfs", "nearest_low", "nearest_high"])
    out = rng.choice(["distance", "order"])
    name = rng.choice(["elevator_distance", "lift_travel", "total_floors"] if out == "distance" else
                      ["elevator_stops", "lift_visit_order", "service_order"])
    pol_doc = {
        "fcfs": "It serves the requests strictly in the given order.",
        "nearest_low": ("At every step it goes to the pending request closest to its current floor;\n"
                        "    if two pending floors are equally close it picks the LOWER one."),
        "nearest_high": ("At every step it goes to the pending request closest to its current floor;\n"
                         "    if two pending floors are equally close it picks the HIGHER one."),
    }[policy]
    out_doc = ("the total number of floors travelled (sum of the absolute floor differences of all moves)"
               if out == "distance" else "the list of floors in the order they are served")
    doc = f"""
    An elevator starts at floor `start` and must serve every floor in `requests`
    (integers, possibly negative; repeated floors are served once per occurrence).
    {pol_doc}

    Return {out_doc}. No requests gives {"0" if out == "distance" else "[]"}.
    """
    final = "return total\n" if out == "distance" else "return visited\n"
    if policy == "fcfs":
        body = ("cur = start\ntotal = 0\nvisited = []\nfor f in requests:\n    total += abs(f - cur)\n"
                "    cur = f\n    visited.append(f)\n" + final)
        mutations = [("    cur = f\n", ""), ("for f in requests:", "for f in sorted(requests):"),
                     ("abs(f - cur)", "(f - cur)")]
        if out == "order":
            mutations = [("for f in requests:", "for f in sorted(requests):"),
                         ("for f in requests:", "for f in sorted(requests, key=lambda x: abs(x - start)):")]
    else:
        key = "(abs(f - cur), f)" if policy == "nearest_low" else "(abs(f - cur), -f)"
        alt = "(abs(f - cur), -f)" if policy == "nearest_low" else "(abs(f - cur), f)"
        body = ("pending = list(requests)\ncur = start\ntotal = 0\nvisited = []\nwhile pending:\n"
                f"    nxt = min(pending, key=lambda f: {key})\n    pending.remove(nxt)\n    total += abs(nxt - cur)\n"
                "    cur = nxt\n    visited.append(nxt)\n" + final)
        mutations = [(key, alt), ("    cur = nxt\n", ""), (None, "cur = start\ntotal = 0\nvisited = []\n"
                     "for f in requests:\n    total += abs(f - cur)\n    cur = f\n    visited.append(f)\n" + final)]
        if out == "distance":
            mutations.append(("total += abs(nxt - cur)", "total += abs(nxt - start)"))
    s = rng.randint(0, 10)
    reqs = _ints(rng, 6, -3, 15)
    cases = [
        ("test_random_requests", [(s, reqs)]),
        ("test_equal_distance_tie", [(5, [3, 7, 6]), (0, [2, -2])]),
        ("test_repeated_floor", [(4, [4, 9, 4, 2])]),
        ("test_negative_floors", [(0, [-5, 3, -1])]),
        ("test_single_request", [(s, [s + 3])]),
        ("test_no_requests", [(s, [])]),
    ]
    return _Spec(name, "start: int, requests: list[int]", "int" if out == "distance" else "list[int]", doc, body,
                 cases, [(5, [8, 2, 6])], mutations, "easy" if policy == "fcfs" else "medium")


# --------------------------------------------------------------------------- dict / nested data


@_family("flatten_dict")
def _flatten_dict(rng: random.Random) -> _Spec:
    """Flatten nested dicts to joined key paths (separator and empty-dict rule vary)."""
    sep = rng.choice([".", "/", "_", ":"])
    keep_empty = rng.random() < 0.5
    name = rng.choice(["flatten_dict", "flatten_keys", "path_keys", "flatten_config"])
    empty_doc = ("An empty nested dict is kept as a value: its path maps to {}." if keep_empty else
                 "An empty nested dict contributes no entry at all.")
    doc = f"""
    Flatten the nested dict `data` into a single-level dict.

    Every value that is not a dict is stored under the path of keys leading to
    it, joined with '{sep}'. Keys are converted with str(), so the int key 3
    contributes '3'. Lists and other non-dict values are kept as they are (not
    descended into). {empty_doc}
    An empty `data` gives {{}}. Keys are never empty and never contain '{sep}'.
    """
    empty = "            out[path] = {}\n" if keep_empty else "            pass\n"
    body = ("out = {}\n\ndef walk(prefix, node):\n    for key, value in node.items():\n"
            f"        path = prefix + {sep!r} + str(key) if prefix else str(key)\n"
            "        if isinstance(value, dict) and value:\n            walk(path, value)\n"
            f"        elif isinstance(value, dict):\n{empty}        else:\n            out[path] = value\n\n"
            "walk(\"\", data)\nreturn out\n")
    mutations = [(f"path = prefix + {sep!r} + str(key) if prefix else str(key)", f"path = prefix + {sep!r} + str(key)"),
                 (empty, "            pass\n" if keep_empty else "            out[path] = {}\n"),
                 (f"prefix + {sep!r} + str(key) if", "prefix + str(key) if"),
                 ("if isinstance(value, dict) and value:", "if isinstance(value, dict) and len(value) > 1:")]
    k = rng.sample(["db", "host", "port", "auth", "user", "mode", "cache", "ttl"], 6)
    v = rng.randint(1, 9999)
    cases = [
        ("test_two_levels", _one({k[0]: {k[1]: "localhost", k[2]: v}, k[3]: True})),
        ("test_three_levels", _one({k[0]: {k[1]: {k[2]: 1, k[3]: 2}}, k[4]: {k[5]: {"x": None}}})),
        ("test_lists_are_leaves", _one({k[0]: [1, {"a": 2}], k[1]: {k[2]: [3]}})),
        ("test_empty_nested_dicts", _one({k[0]: {}, k[1]: {k[2]: {}}, k[3]: 0})),
        ("test_int_keys", _one({1: {2: "a"}, k[4]: {3: {k[5]: "b"}}})),
        ("test_flat_and_empty", _one({k[0]: 1, k[1]: 2}, {})),
    ]
    return _Spec(name, "data: dict", "dict", doc, body, cases, [({"a": {"b": 1, "c": {"d": 2}}, "e": 3},)],
                 mutations, "medium")


@_family("invert_mapping")
def _invert_mapping(rng: random.Random) -> _Spec:
    """Invert a dict with a collision policy (list / sorted list / first / last / smallest key)."""
    policy = rng.choice(["list", "sorted", "first", "last", "smallest"])
    name = rng.choice(["invert_mapping", "invert_dict", "reverse_lookup", "flip_mapping"])
    pol_doc = {
        "list": "maps to the LIST of all keys that had it, in the dict's insertion order",
        "sorted": "maps to the list of all keys that had it, sorted in ascending order",
        "first": "maps to the FIRST key (in insertion order) that had it",
        "last": "maps to the LAST key (in insertion order) that had it",
        "smallest": "maps to the smallest key (ordinary string order) that had it",
    }[policy]
    doc = f"""
    Invert `mapping` (str keys, int values): each distinct value becomes a key that
    {pol_doc}.

    {"Values that occur only once still map to a one-element list." if policy in ("list", "sorted") else
     "Every distinct value appears exactly once in the result."}
    An empty mapping gives {{}}.
    """
    body = {
        "list": "out = {}\nfor k, v in mapping.items():\n    out.setdefault(v, []).append(k)\nreturn out\n",
        "sorted": ("out = {}\nfor k, v in mapping.items():\n    out.setdefault(v, []).append(k)\n"
                   "return {v: sorted(ks) for v, ks in out.items()}\n"),
        "first": "out = {}\nfor k, v in mapping.items():\n    if v not in out:\n        out[v] = k\nreturn out\n",
        "last": "out = {}\nfor k, v in mapping.items():\n    out[v] = k\nreturn out\n",
        "smallest": ("out = {}\nfor k, v in mapping.items():\n    if v not in out or k < out[v]:\n"
                     "        out[v] = k\nreturn out\n"),
    }[policy]
    mutations = {
        "list": [("append(k)", "insert(0, k)"), ("    out.setdefault(v, []).append(k)\n", "    out[v] = [k]\n")],
        "sorted": [("return {v: sorted(ks) for v, ks in out.items()}", "return out"),
                   ("sorted(ks)", "sorted(ks, reverse=True)"), ("    out.setdefault(v, []).append(k)\n", "    out[v] = [k]\n")],
        "first": [("    if v not in out:\n        out[v] = k\n", "    out[v] = k\n"), ("out[v] = k", "out[v] = [k]")],
        "last": [("    out[v] = k\n", "    if v not in out:\n        out[v] = k\n"), ("out[v] = k", "out[v] = [k]")],
        "smallest": [("k < out[v]", "k > out[v]"), ("if v not in out or k < out[v]:", "if v not in out:")],
    }[policy]
    keys = rng.sample(WORDS, 6)
    vals = [rng.randint(1, 3) for _ in keys]
    cases = [
        ("test_no_collisions", _one({keys[0]: 1, keys[1]: 2})),
        ("test_collisions", _one(dict(zip(keys, vals, strict=True)))),
        ("test_reverse_alphabetical_insertion", _one({"zeta": 5, "beta": 5, "mu": 5, "alpha": 6})),
        ("test_single", _one({keys[2]: 0})),
        ("test_all_same_value", _one({k: 7 for k in keys[:4]})),
        ("test_empty", _one({})),
    ]
    return _Spec(name, "mapping: dict[str, int]", "dict", doc, body, cases, [({"a": 1, "b": 2, "c": 1},)],
                 mutations, "easy" if policy in ("list", "last") else "medium")


@_family("group_words")
def _group_words(rng: random.Random) -> _Spec:
    """Group words by length / first letter / last letter with an ordering rule for members."""
    key = rng.choice(["len", "first", "last"])
    order = rng.choice(["appearance", "sorted", "unique"])
    name = rng.choice(["group_words", "bucket_words", "index_words", "words_by_key"])
    key_doc = {"len": "their length", "first": "their first character, lowercased",
               "last": "their last character, lowercased"}[key]
    order_doc = {"appearance": "in the order they appear in `words` (duplicates kept)",
                 "sorted": "sorted in ascending string order (case-sensitive; duplicates kept)",
                 "unique": "sorted in ascending string order with exact duplicates removed"}[order]
    doc = f"""
    Group the non-empty strings in `words` by {key_doc}.

    Return a dict mapping each key to the list of words having that key,
    {order_doc}. Words keep their original spelling. Empty strings are ignored;
    an empty input gives {{}}.
    """
    expr = {"len": "len(w)", "first": "w[0].lower()", "last": "w[-1].lower()"}[key]
    post = {"appearance": "", "sorted": "for k in groups:\n    groups[k] = sorted(groups[k])\n",
            "unique": "for k in groups:\n    groups[k] = sorted(set(groups[k]))\n"}[order]
    body = (f"groups = {{}}\nfor w in words:\n    if not w:\n        continue\n    key = {expr}\n"
            f"    groups.setdefault(key, []).append(w)\n{post}return groups\n")
    mutations = [("    if not w:\n        continue\n", "")]
    if key != "len":
        mutations.append((expr, expr.replace(".lower()", "")))
    else:
        mutations.append(("key = len(w)", "key = len(w.strip())"))
    if order == "appearance":
        mutations.append(("    groups.setdefault(key, []).append(w)\n",
                          "    if w not in groups.get(key, []):\n        groups.setdefault(key, []).append(w)\n"))
    elif order == "sorted":
        mutations += [("sorted(groups[k])", "sorted(set(groups[k]))"), (post, "")]
    else:
        mutations += [("sorted(set(groups[k]))", "sorted(groups[k])"), ("sorted(set(groups[k]))", "list(set(groups[k]))")]
    ws = rng.sample(WORDS, 6)
    cases = [
        ("test_basic", _one(ws)),
        ("test_mixed_case", _one([ws[0].title(), ws[0], ws[1].upper(), ws[1], "Zed", "zoo"])),
        ("test_duplicates", _one([ws[2], ws[3], ws[2], ws[2]])),
        ("test_empty_strings_ignored", _one(["", ws[4], "", " a"])),
        ("test_single_letters", _one(["b", "a", "B", "ab", "ba"])),
        ("test_empty", _one([])),
    ]
    return _Spec(name, "words: list[str]", "dict", doc, body, cases, [(["ant", "Bee", "art", "bat"],)],
                 mutations, "easy" if order == "appearance" else "medium")


# --------------------------------------------------------------------------- calendar math


@_family("calendar_day")
def _calendar_day(rng: random.Random) -> _Spec:
    """Day-of-year <-> (month, day) under Gregorian or Julian leap rules, with validation."""
    gregorian = rng.random() < 0.6
    forward = rng.random() < 0.5
    leap_doc = ("Gregorian rule: a year is a leap year if it is divisible by 4, except years\n"
                "    divisible by 100 that are not divisible by 400 (2000 is leap, 1900 is not)" if gregorian else
                "Julian rule: a year is a leap year exactly when it is divisible by 4 (so 1900\n"
                "    IS a leap year)")
    leap = ("leap = (year % 4 == 0 and year % 100 != 0) or year % 400 == 0" if gregorian else
            "leap = year % 4 == 0")
    alt_leap = "leap = year % 4 == 0" if gregorian else "leap = (year % 4 == 0 and year % 100 != 0) or year % 400 == 0"
    days = "days = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]\n"
    if forward:
        name = rng.choice(["day_of_year", "ordinal_day", "day_number"])
        doc = f"""
        Return the day of the year (1 for January 1st) of the date year-month-day.

        Leap years use the {leap_doc}.
        February has 29 days in a leap year and 28 otherwise.
        If month is not in 1..12, or day is not a valid day of that month, return -1.
        """
        body = (f"{leap}\n{days}if not 1 <= month <= 12 or not 1 <= day <= days[month - 1]:\n    return -1\n"
                "return sum(days[:month - 1]) + day\n")
        mutations = [(leap, alt_leap), ("sum(days[:month - 1])", "sum(days[:month])"),
                     ("if not 1 <= month <= 12 or not 1 <= day <= days[month - 1]:", "if not 1 <= month <= 12:"),
                     ("if not 1 <= month <= 12 or ", "if ")]
        cases = [
            ("test_january_first", [(2023, 1, 1)]),
            ("test_end_of_year", [(2023, 12, 31), (2024, 12, 31)]),
            ("test_century_years", [(1900, 3, 1), (2000, 3, 1)]),
            ("test_february_29", [(2024, 2, 29), (2023, 2, 29), (1900, 2, 29)]),
            ("test_invalid_month_or_day", [(2024, 13, 1), (2024, 0, 5), (2024, 4, 31), (2024, 6, 0)]),
            ("test_random_date", [(rng.randint(1800, 2400), rng.randint(3, 12), rng.randint(1, 28))]),
        ]
        sig, ret, ex = "year: int, month: int, day: int", "int", [(2024, 3, 1)]
    else:
        name = rng.choice(["date_from_ordinal", "ordinal_to_date", "month_and_day"])
        doc = f"""
        Given a `year` and a day number `n` (1 = January 1st), return the date as a
        (month, day) tuple.

        Leap years use the {leap_doc}.
        February has 29 days in a leap year and 28 otherwise.
        If n is not between 1 and the number of days in that year (365 or 366),
        return None.
        """
        body = (f"{leap}\n{days}if not 1 <= n <= (366 if leap else 365):\n    return None\n"
                "for month, length in enumerate(days, start=1):\n    if n <= length:\n        return (month, n)\n"
                "    n -= length\nreturn None\n")
        mutations = [(leap, alt_leap), ("if n <= length:", "if n < length:"),
                     ("if not 1 <= n <= (366 if leap else 365):", "if not 1 <= n <= 366:"),
                     ("enumerate(days, start=1)", "enumerate(days)")]
        cases = [
            ("test_first_and_second_day", [(2023, 1), (2023, 2)]),
            ("test_month_boundaries", [(2023, 31), (2023, 32), (2023, 59), (2023, 60)]),
            ("test_leap_year", [(2024, 60), (2024, 366), (2000, 60)]),
            ("test_century_year", [(1900, 60), (1900, 366)]),
            ("test_out_of_range", [(2023, 0), (2023, 366), (2024, 367), (2024, -3)]),
            ("test_random_day", [(rng.randint(1800, 2400), rng.randint(61, 360))]),
        ]
        sig, ret, ex = "year: int, n: int", "tuple[int, int] | None", [(2024, 61)]
    return _Spec(name, sig, ret, doc, body, cases, ex, mutations, "medium" if gregorian else "easy")


# --------------------------------------------------------------------------- validation


@_family("bracket_check")
def _bracket_check(rng: random.Random) -> _Spec:
    """Bracket balance over a chosen bracket set: bool / first error index / max depth."""
    pairs = rng.sample(["()", "[]", "{}", "<>"], rng.randint(2, 4))
    out = rng.choice(["bool", "index", "depth"])
    name = rng.choice({"bool": ["brackets_balanced", "is_well_nested"],
                       "index": ["first_bracket_error", "bracket_error_position"],
                       "depth": ["max_nesting", "bracket_depth"]}[out])
    closers = {p[1]: p[0] for p in pairs}
    openers = "".join(p[0] for p in pairs)
    pair_doc = ", ".join(f"'{p}'" for p in pairs)
    out_doc = {
        "bool": "Return True if the brackets are balanced and properly nested, else False.",
        "index": ("Return -1 if the brackets are balanced and properly nested. Otherwise return\n"
                  "    the index of the first closing bracket that does not match the most recent\n"
                  "    unclosed opening bracket (or has none); if every closer matches but some\n"
                  "    brackets are left open at the end, return len(text)."),
        "depth": ("If the brackets are balanced and properly nested, return the maximum nesting\n"
                  "    depth (0 when there are no brackets); otherwise return -1."),
    }[out]
    doc = f"""
    Check the brackets in `text`. The bracket pairs are {pair_doc}; every other
    character (including any other bracket-like character) is ignored.

    {out_doc}
    """
    fail = {"bool": "return False", "index": "return i", "depth": "return -1"}[out]
    final = {"bool": "return not stack\n", "index": "return len(text) if stack else -1\n",
             "depth": "return -1 if stack else depth\n"}[out]
    track = "        depth = max(depth, len(stack))\n" if out == "depth" else ""
    body = (f"pairs = {closers!r}\nstack = []\ndepth = 0\nfor i, ch in enumerate(text):\n    if ch in {openers!r}:\n"
            f"        stack.append(ch)\n{track}    elif ch in pairs:\n        if not stack or stack[-1] != pairs[ch]:\n"
            f"            {fail}\n        stack.pop()\n" + final)
    mutations = [("if not stack or stack[-1] != pairs[ch]:", "if stack[-1] != pairs[ch]:"),
                 ("if not stack or stack[-1] != pairs[ch]:", "if not stack:")]
    mutations.append({"bool": ("return not stack", "return True"),
                      "index": ("return len(text) if stack else -1", "return -1"),
                      "depth": ("        stack.append(ch)\n        depth = max(depth, len(stack))\n",
                                "        depth = max(depth, len(stack))\n        stack.append(ch)\n")}[out])
    o1, c1 = pairs[0]
    o2, c2 = pairs[1]
    unused = [p for p in ["()", "[]", "{}", "<>"] if p not in pairs]
    noise = unused[0] if unused else "ab"
    cases = [
        ("test_balanced_nested", _one(f"{o1}x{o2}{o1}{c1}{c2}{c1}", f"{o2}{c2}{o1}{c1}")),
        ("test_wrong_type", _one(f"{o1}{o2}{c1}{c2}", f"a{o2}{c1}")),
        ("test_stray_closer", _one(f"{c2}{o1}{c1}", f"{o1}{c1}{c1}")),
        ("test_unclosed_at_end", _one(f"{o1}{o2}{c2}", f"{o2}")),
        ("test_other_characters_ignored", _one(f"{noise[1]}{o1}{noise[0]}{c1}", "no brackets")),
        ("test_empty", _one("")),
        ("test_deep", _one(o1 * 3 + o2 + c2 + c1 * 3 + o2 + c2)),
    ]
    return _Spec(name, "text: str", {"bool": "bool", "index": "int", "depth": "int"}[out], doc, body, cases,
                 [(f"{o1}{o2}{c2}{c1}",)], mutations, "easy" if out == "bool" else "medium")


@_family("checksum")
def _checksum(rng: random.Random) -> _Spec:
    """ID checksum validation: Luhn, ISBN-10 or cyclic-weight mod-10 schemes."""
    scheme = rng.choice(["luhn", "isbn10", "weighted"])
    weights = rng.choice([[3, 1], [1, 3], [7, 3, 1], [2, 1], [1, 2, 3]])
    clean = ('chars = [c for c in code if c not in " -"]\n')
    if scheme == "luhn":
        name = rng.choice(["luhn_valid", "check_card_number", "passes_luhn"])
        doc = """
        Return True if `code` passes the Luhn checksum, else False.

        Spaces and hyphens are ignored. After removing them, the code must be
        non-empty and consist only of ASCII digits, otherwise return False.
        Luhn: starting from the RIGHTMOST digit (position 1), double every digit
        in an even position (2nd, 4th, ... from the right); if doubling gives more
        than 9, subtract 9. The code is valid when the sum of all digits obtained
        this way is divisible by 10.
        """
        body = (clean + "if not chars or not all(c.isascii() and c.isdigit() for c in chars):\n    return False\n"
                "total = 0\nfor i, c in enumerate(reversed(chars)):\n    d = int(c)\n    if i % 2 == 1:\n"
                "        d *= 2\n        if d > 9:\n            d -= 9\n    total += d\nreturn total % 10 == 0\n")
        mutations = [("if i % 2 == 1:", "if i % 2 == 0:"), ("        if d > 9:\n            d -= 9\n", ""),
                     ("enumerate(reversed(chars))", "enumerate(chars)"), ("if not chars or not all", "if not all")]
        length = rng.choice([8, 11, 15, 16])
    elif scheme == "isbn10":
        name = rng.choice(["isbn10_valid", "check_isbn10", "is_isbn10"])
        doc = """
        Return True if `code` is a valid ISBN-10, else False.

        Spaces and hyphens are ignored. What remains must be exactly 10 characters:
        nine ASCII digits followed by a final digit or an uppercase 'X' (worth 10).
        A lowercase 'x' is NOT accepted. With values v1..v10 from left to right the
        code is valid when 10*v1 + 9*v2 + ... + 2*v9 + 1*v10 is divisible by 11.
        """
        body = (clean + "if len(chars) != 10:\n    return False\ntotal = 0\nfor i, c in enumerate(chars):\n"
                "    if c.isascii() and c.isdigit():\n        v = int(c)\n    elif c == \"X\" and i == 9:\n        v = 10\n"
                "    else:\n        return False\n    total += (10 - i) * v\nreturn total % 11 == 0\n")
        mutations = [("total % 11 == 0", "total % 10 == 0"), ('c == "X" and i == 9', 'c in "Xx" and i == 9'),
                     ("if len(chars) != 10:", "if len(chars) > 10:"), ('c == "X" and i == 9', 'c == "X"')]
        length = 10
    else:
        name = rng.choice(["weighted_check_valid", "check_code", "verify_check_digit"])
        wdoc = ", ".join(map(str, weights))
        doc = f"""
        Return True if `code` passes a weighted mod-10 check, else False.

        Spaces and hyphens are ignored. After removing them the code must be
        non-empty and consist only of ASCII digits, otherwise return False.
        Multiply the digits, from LEFT to right, by the weights {wdoc}, repeating
        that weight pattern as often as needed. The code is valid when the sum of
        the products is divisible by 10.
        """
        body = (clean + "if not chars or not all(c.isascii() and c.isdigit() for c in chars):\n    return False\n"
                f"weights = {weights!r}\n"
                "total = sum(int(c) * weights[i % len(weights)] for i, c in enumerate(chars))\nreturn total % 10 == 0\n")
        mutations = [("enumerate(chars)", "enumerate(reversed(chars))"), ("if not chars or not all", "if not all"),
                     ("weights[i % len(weights)]", "weights[i % len(weights) - 1]"),
                     ('c not in " -"', 'c != " "')]
        length = rng.choice([8, 12, 13])
    checker = _load(f"def f(code):\n{_indent(body)}", "f")

    def valid_code() -> str:
        stem = "".join(str(rng.randint(0, 9)) for _ in range(length - 1))
        options = [stem + str(d) for d in range(10)] + ([stem + "X"] if scheme == "isbn10" else [])
        good = [c for c in options if checker(c)]
        if not good:
            raise _Reject("no valid check digit")
        return rng.choice(good)

    v1, v2, v3 = valid_code(), valid_code(), valid_code()
    bad = v1[:-1] + str((int(v1[-1]) + 1) % 10) if v1[-1].isdigit() else v1[:-1] + "0"
    swapped = v2[1] + v2[0] + v2[2:]
    grouped = "-".join([v3[:3], v3[3:6], v3[6:]])
    cases = [
        ("test_valid_codes", _one(v1, v2)),
        ("test_one_digit_changed", _one(bad)),
        ("test_adjacent_swap_and_reversal", _one(swapped, v1[::-1])),
        ("test_separators_ignored", _one(grouped, " ".join(v3))),
        ("test_invalid_characters", _one(v2[:-1] + "a", v2.replace(v2[0], "x", 1).lower(), v1 + "x")),
        ("test_empty_or_only_separators", _one("", " - ")),
        ("test_known_values", _one("0306406152", "79927398713", "036000291452", "0-306-40615-X")),
    ]
    return _Spec(name, "code: str", "bool", doc, body, cases, [(v1,)], mutations,
                 "medium" if scheme != "weighted" else "easy")


# --------------------------------------------------------------------------- small DP


@_family("count_change")
def _count_change(rng: random.Random) -> _Spec:
    """Count ways to make an amount: unlimited combinations / each item once / ordered sequences."""
    mode = rng.choice(["unlimited", "once", "ordered"])
    name = rng.choice({"unlimited": ["count_ways", "change_combinations", "ways_to_pay"],
                       "once": ["subset_sum_ways", "ways_each_once", "pick_items_ways"],
                       "ordered": ["ordered_ways", "count_sequences", "step_combinations"]}[mode])
    mode_doc = {
        "unlimited": ("`coins` holds DISTINCT positive denominations, each usable any number of times.\n"
                      "Two ways are the same if they use the same number of each coin (order\n"
                      "does not matter)."),
        "once": ("`coins` holds positive values; each ELEMENT can be used at most once, and equal\n"
                 "values at different positions count as different elements. A way is a\n"
                 "set of positions (order does not matter)."),
        "ordered": ("`coins` holds DISTINCT positive values, each usable any number of times, and\n"
                    "ORDER MATTERS: 1+2 and 2+1 are different ways."),
    }[mode]
    doc = f"""
    Return the number of ways to make exactly `amount` from `coins`.

    {mode_doc}
    The amount 0 can be made in exactly 1 way (use nothing); a negative amount in 0 ways.
    """
    loops = {
        "unlimited": "for c in coins:\n    for a in range(c, amount + 1):\n        ways[a] += ways[a - c]\n",
        "once": "for c in coins:\n    for a in range(amount, c - 1, -1):\n        ways[a] += ways[a - c]\n",
        "ordered": "for a in range(1, amount + 1):\n    for c in coins:\n        if c <= a:\n            ways[a] += ways[a - c]\n",
    }
    body = "if amount < 0:\n    return 0\nways = [1] + [0] * amount\n" + loops[mode] + "return ways[amount]\n"
    others = [m for m in loops if m != mode]
    mutations = [(loops[mode], loops[others[0]]), (loops[mode], loops[others[1]]),
                 ("if amount < 0:\n    return 0\n", ""), ("ways = [1] + [0] * amount", "ways = [0] * (amount + 1)")]
    coins = sorted(rng.sample([1, 2, 3, 5, 7, 10], 3))
    dup = [*coins, coins[0]] if mode == "once" else coins
    cases = [
        ("test_small_amount", [(coins, rng.randint(5, 9))]),
        ("test_larger_amount", [(coins, rng.randint(12, 20))]),
        ("test_zero_amount", [(coins, 0)]),
        ("test_negative_amount", [(coins, -3)]),
        ("test_unreachable", [([4, 6], 7), ([], 3)]),
        ("test_repeated_or_classic", [(dup, coins[0] * 2), ([1, 2, 5], 5)]),
    ]
    return _Spec(name, "coins: list[int], amount: int", "int", doc, body, cases, [([1, 2, 3], 4)], mutations,
                 "medium" if mode == "unlimited" else "hard")


@_family("longest_monotone")
def _longest_monotone(rng: random.Random) -> _Spec:
    """Longest monotone subsequence or contiguous run (strict/non-strict, up/down)."""
    cmp = rng.choice(["<", "<=", ">", ">="])
    contiguous = rng.random() < 0.4
    word = {"<": "strictly increasing", "<=": "non-decreasing", ">": "strictly decreasing",
            ">=": "non-increasing"}[cmp]
    name = rng.choice(["longest_run", "longest_streak", "max_monotone_run"] if contiguous else
                      ["longest_subsequence", "lis_length", "longest_chain"])
    what = (f"the longest CONTIGUOUS run (slice nums[i:j]) that is {word}" if contiguous else
            f"the longest {word} SUBSEQUENCE (elements kept in order, not necessarily adjacent)")
    doc = f"""
    Return the length of {what}.

    "{word}" compares each element with the previous one using {cmp!r}.
    A single element counts as length 1; an empty list gives 0.
    """
    flip = {"<": "<=", "<=": "<", ">": ">=", ">=": ">"}[cmp]
    if contiguous:
        body = ("if not nums:\n    return 0\nbest = cur = 1\nfor i in range(1, len(nums)):\n"
                f"    cur = cur + 1 if nums[i - 1] {cmp} nums[i] else 1\n    best = max(best, cur)\nreturn best\n")
        mutations = [(f"nums[i - 1] {cmp} nums[i]", f"nums[i - 1] {flip} nums[i]"),
                     ("if not nums:\n    return 0\n", "if not nums:\n    return 1\n"),
                     ("    best = max(best, cur)\n", "")]
        mutations.append((None, "if not nums:\n    return 0\nbest = cur = 1\nfor i in range(1, len(nums)):\n"
                          f"    cur = cur + 1 if nums[i - 1] {cmp} nums[i] else 0\n    best = max(best, cur)\nreturn best\n"))
    else:
        body = ("if not nums:\n    return 0\nbest = [1] * len(nums)\nfor i in range(len(nums)):\n    for j in range(i):\n"
                f"        if nums[j] {cmp} nums[i]:\n            best[i] = max(best[i], best[j] + 1)\nreturn max(best)\n")
        mutations = [(f"nums[j] {cmp} nums[i]", f"nums[j] {flip} nums[i]"), ("if not nums:\n    return 0\n", ""),
                     ("return max(best)", "return best[-1]"), ("for j in range(i):", "for j in range(i - 1):")]
    nums = _ints(rng, 10, 0, 9)
    cases = [
        ("test_random", _one(nums, _ints(rng, 8, -5, 5))),
        ("test_sorted_ascending", _one([1, 2, 3, 4, 5])),
        ("test_sorted_descending", _one([9, 7, 4, 2])),
        ("test_with_equal_values", _one([3, 3, 3, 1, 1, 5, 5])),
        ("test_single_and_empty", _one([rng.randint(0, 9)], [])),
        ("test_zigzag", _one([1, 5, 2, 6, 3, 7, 4, 4])),
    ]
    return _Spec(name, "nums: list[int]", "int", doc, body, cases, [([2, 2, 1, 3, 4],)], mutations,
                 "easy" if contiguous else "hard")


# --------------------------------------------------------------------------- graphs


@_family("bfs_hops")
def _bfs_hops(rng: random.Random) -> _Spec:
    """BFS on an edge list: directed or undirected; one distance or all distances."""
    directed = rng.random() < 0.5
    all_dist = rng.random() < 0.4
    name = rng.choice(["hop_distances", "bfs_levels", "reach_distances"] if all_dist else
                      ["hop_distance", "min_hops", "degrees_apart"])
    edge_doc = ("Each edge (a, b) is DIRECTED: it can only be travelled from a to b." if directed else
                "Each edge (a, b) is UNDIRECTED: it can be travelled both ways.")
    if all_dist:
        what = ("Return a dict mapping every node reachable from `src` (including `src` itself, at\n"
                "    distance 0, even if it appears in no edge) to its distance in edges.")
        sig, ret, final = "edges: list[tuple[str, str]], src: str", "dict[str, int]", "return dist\n"
    else:
        what = ("Return the minimum number of edges on a path from `src` to `dst`: 0 when they\n"
                "    are the same node (even if it appears in no edge), -1 if `dst` is unreachable.")
        sig, ret, final = "edges: list[tuple[str, str]], src: str, dst: str", "int", "return dist.get(dst, -1)\n"
    doc = f"""
    `edges` lists the edges of a graph whose nodes are strings. {edge_doc}
    {what}
    Duplicate edges and self-loops may occur.
    """
    back = "" if directed else "    adj.setdefault(b, []).append(a)\n"
    body = ("from collections import deque\nadj = {}\nfor a, b in edges:\n    adj.setdefault(a, []).append(b)\n"
            f"{back}dist = {{src: 0}}\nqueue = deque([src])\nwhile queue:\n    node = queue.popleft()\n"
            "    for nxt in adj.get(node, []):\n        if nxt not in dist:\n            dist[nxt] = dist[node] + 1\n"
            "            queue.append(nxt)\n" + final)
    mutations = [("adj.get(node, [])", "adj[node]"), ("node = queue.popleft()", "node = queue.pop()")]
    if directed:
        mutations.append(("    adj.setdefault(a, []).append(b)\n",
                          "    adj.setdefault(a, []).append(b)\n    adj.setdefault(b, []).append(a)\n"))
    else:
        mutations.append((back, ""))
    if not all_dist:
        mutations.append(("dist.get(dst, -1)", "dist.get(dst, 0)"))
    nodes = rng.sample(["a", "b", "c", "d", "e", "f", "g", "h"], 7)
    a, b, c, d, e, f, g = nodes
    chain = [(a, b), (b, c), (c, d), (a, e), (e, d), (d, f)]
    shortcut = [(a, b), (a, c), (b, d), (c, d), (d, e), (b, e), (a, b)]
    extra = [(f, a), (g, g)]

    def q(es, s, t):
        return (es, s) if all_dist else (es, s, t)

    cases = [
        ("test_chain_with_shortcut", [q(chain, a, f)]),
        ("test_dfs_order_trap", [q([(a, b), (b, c), (c, d), (a, d), (d, e)], a, e),
                                 q([(a, b), (a, c), (b, d), (d, e), (c, e)], a, e)]),
        ("test_direction_matters", [q(chain + extra, f, a), q(chain, d, a)]),
        ("test_unreachable_node", [q(shortcut, a, g)]),
        ("test_same_node", [q(shortcut, g, g), q([], a, a)]),
        ("test_duplicates_and_self_loops", [q(shortcut + extra, e, c)]),
    ]
    ex = ([("x", "y"), ("y", "z")], "x") if all_dist else ([("x", "y"), ("y", "z")], "x", "z")
    return _Spec(name, sig, ret, doc, body, cases, [ex], mutations, "medium" if not directed else "hard")


@_family("components")
def _components(rng: random.Random) -> _Spec:
    """Connected components of an undirected graph: count / sorted sizes / largest / count of size >= s."""
    out = rng.choice(["count", "sizes", "largest", "atleast"])
    s = rng.randint(2, 3)
    name = rng.choice({"count": ["count_components", "count_groups"], "sizes": ["component_sizes", "group_sizes"],
                       "largest": ["largest_component", "biggest_group"],
                       "atleast": ["count_big_components", "groups_of_size_at_least"]}[out])
    out_doc = {"count": "the number of connected components",
               "sizes": "the sizes of all connected components, largest first",
               "largest": "the size of the largest connected component (0 when n == 0)",
               "atleast": f"how many connected components have at least {s} nodes"}[out]
    doc = f"""
    A graph has nodes 0 .. n-1 and the undirected `edges` (pairs of node indices;
    duplicates and self-loops may occur). Every node belongs to exactly one
    connected component, so isolated nodes are components of size 1.

    Return {out_doc}.
    """
    final = {"count": "return len(sizes)\n", "sizes": "return sorted(sizes.values(), reverse=True)\n",
             "largest": "return max(sizes.values(), default=0)\n",
             "atleast": f"return sum(1 for v in sizes.values() if v >= {s})\n"}[out]
    body = ("parent = list(range(n))\n\ndef find(x):\n    while parent[x] != x:\n        parent[x] = parent[parent[x]]\n"
            "        x = parent[x]\n    return x\n\nfor a, b in edges:\n    ra, rb = find(a), find(b)\n    if ra != rb:\n"
            "        parent[ra] = rb\nsizes = {}\nfor v in range(n):\n    r = find(v)\n    sizes[r] = sizes.get(r, 0) + 1\n"
            + final)
    mutations = [("parent[ra] = rb", "parent[a] = b"), ("for v in range(n):", "for v in range(1, n):")]
    mutations.append({"count": ("return len(sizes)", "return n - len(edges)"),
                      "sizes": ("sorted(sizes.values(), reverse=True)", "sorted(sizes.values())"),
                      "largest": ("max(sizes.values(), default=0)", "max(sizes.values())"),
                      "atleast": (f"v >= {s}", f"v > {s}")}[out])
    if out == "largest":
        mutations.append(("sizes[r] = sizes.get(r, 0) + 1", "sizes[r] = sizes.get(r, 1) + 1"))
    n = rng.randint(7, 10)
    edges = [(rng.randrange(n), rng.randrange(n)) for _ in range(rng.randint(4, 7))]
    cases = [
        ("test_random_graph", [(n, edges)]),
        ("test_chain_and_isolated", [(6, [(0, 1), (2, 1), (3, 2)])]),
        ("test_cycle_and_duplicates", [(5, [(0, 1), (1, 2), (2, 0), (0, 1), (3, 3)])]),
        ("test_union_order_trap", [(6, [(0, 1), (2, 3), (1, 3), (4, 5), (5, 0)])]),
        ("test_no_edges", [(4, [])]),
        ("test_no_nodes", [(0, [])]),
        ("test_two_pairs", [(s + 3, [(0, 1), (2, 3)] + [(3, i) for i in range(4, s + 3)])]),
    ]
    return _Spec(name, "n: int, edges: list[tuple[int, int]]", "list[int]" if out == "sizes" else "int", doc, body,
                 cases, [(5, [(0, 1), (1, 2), (3, 4)])], mutations, "medium" if out == "count" else "hard")


# --------------------------------------------------------------------------- public API


def generate_training_tasks(n: int, seed: int = 0, families: list[str] | None = None) -> list[dict]:
    """Deterministic: same (n, seed, families) -> identical output. Round-robins over families so
    every family is represented; task_id = f"gen_{family}_{i:04d}" (unique)."""
    names = list(families) if families else list(FAMILIES)
    unknown = [f for f in names if f not in FAMILIES]
    if unknown:
        raise ValueError(f"unknown families: {unknown}; choose from {sorted(FAMILIES)}")
    if not names:
        return []
    tasks = []
    for i in range(n):
        family = names[i % len(names)]
        index = i // len(names)
        rng = random.Random(f"agent-eval-training/{seed}/{family}/{index}")
        task = FAMILIES[family](rng, index)
        task["task_id"] = f"gen_{family}_{i:04d}"
        tasks.append(task)
    return tasks


def _to_task(entry: dict):
    from agent_eval.models import Task

    return Task(
        task_id=entry["task_id"],
        prompt=entry["prompt"],
        entry_point=entry["entry_point"],
        test_code=entry["test_code"],
        difficulty=entry.get("difficulty", "medium"),
        tags=list(entry.get("tags", [])),
        canonical_solution=entry.get("canonical_solution"),
        mutants=list(entry.get("mutants", [])),
        category=entry.get("category", "function"),
        per_test_timeout=entry.get("per_test_timeout"),
    )


def _problems_by_task(tasks: list[dict], workers: int = 4) -> dict[str, list[str]]:
    from agent_eval.sandbox import run_tests

    jobs = []
    for entry in tasks:
        task = _to_task(entry)
        if not task.canonical_solution:
            jobs.append((task.task_id, "missing", None, task))
        else:
            jobs.append((task.task_id, "canonical", task.canonical_solution, task))
        for i, mutant in enumerate(task.mutants):
            jobs.append((task.task_id, f"mutant {i}", mutant, task))

    def run(job):
        task_id, label, code, task = job
        if code is None:
            return task_id, "no canonical solution"
        result = run_tests(code, task)
        if label == "canonical":
            if not result.passed:
                detail = "; ".join(f"{c.name}: {c.error}" for c in result.failing) or result.error or "failed"
                return task_id, f"canonical failed: {detail[:300]}"
            if not 5 <= result.num_total <= 8:
                return task_id, f"expected 5-8 tests, ran {result.num_total}"
        elif result.passed:
            return task_id, f"{label} survived"
        return task_id, None

    problems: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for task_id, problem in pool.map(run, jobs):
            if problem:
                problems.setdefault(task_id, []).append(problem)
    for entry in tasks:
        if len(entry.get("mutants", [])) < 2:
            problems.setdefault(entry["task_id"], []).append("fewer than 2 mutants")
    return problems


def validate_generated(tasks: list[dict], workers: int = 4) -> list[str]:
    """Run each task's canonical + mutants in the sandbox (agent_eval.sandbox.run_tests on
    agent_eval.models.Task built from the dict). Return a list of problems
    (task_id + reason) -- canonical failing, or a mutant surviving. Empty list = all good.
    Use a ThreadPoolExecutor (each sandbox run is its own subprocess)."""
    problems = _problems_by_task(tasks, workers)
    return [f"{entry['task_id']}: {p}" for entry in tasks for p in problems.get(entry["task_id"], [])]


def write_training_suite(path, n, seed=0, families=None, validate=True) -> dict:
    """Generate, optionally validate (DROP any task that fails validation rather than raising),
    write JSON (indent=2) to path, return stats {"written", "dropped", "families": {name: count}}."""
    tasks = generate_training_tasks(n, seed=seed, families=families)
    dropped = 0
    if validate:
        bad = _problems_by_task(tasks)
        dropped = sum(1 for t in tasks if t["task_id"] in bad)
        tasks = [t for t in tasks if t["task_id"] not in bad]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for t in tasks:
        family = t["tags"][1]
        counts[family] = counts.get(family, 0) + 1
    return {"written": len(tasks), "dropped": dropped, "families": counts}


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    import argparse

    parser = argparse.ArgumentParser(description="Write a procedurally generated training suite.")
    parser.add_argument("path")
    parser.add_argument("-n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()
    print(json.dumps(write_training_suite(args.path, args.n, args.seed, validate=not args.no_validate), indent=2))
