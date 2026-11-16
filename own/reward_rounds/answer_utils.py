"""Answer extraction, format checks, and equivalence helpers.

The functions here are deliberately independent from the reward formula.  They
are used both by online reward computation and by offline Reward QA.
"""

from __future__ import annotations

import math
import re
from typing import Any

try:
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify
except Exception:  # pragma: no cover - optional dependency
    NormalizationConfig = None
    LatexExtractionConfig = None
    parse = None
    verify = None

try:
    import sympy as sp
except Exception:  # pragma: no cover - optional dependency
    sp = None


ANSWER_ANCHOR_RE = re.compile(
    r"(?:answer|final answer|"
    r"\u7b54\u6848|\u6700\u7ec8\u7b54\u6848|\u6700\u540e\u7b54\u6848|\u6700\u7ec8\u7ed3\u679c)"
    r"\s*[:：]\s*(.+)",
    flags=re.IGNORECASE,
)

ANSWER_ANCHOR_COUNT_RE = re.compile(
    r"(?:answer|final answer|"
    r"\u7b54\u6848|\u6700\u7ec8\u7b54\u6848|\u6700\u540e\u7b54\u6848|\u6700\u7ec8\u7ed3\u679c)"
    r"\s*[:：]",
    flags=re.IGNORECASE,
)


def extract_answer(response: str) -> str:
    """Extract the final answer from common math-response formats."""

    if not response:
        return ""

    boxed = extract_last_boxed(response)
    if boxed:
        return boxed.strip()

    answer_tag = re.findall(r"<answer>\s*(.*?)\s*</answer>", response, flags=re.IGNORECASE | re.DOTALL)
    if answer_tag:
        return clean_answer_text(answer_tag[-1])

    answer_line = ANSWER_ANCHOR_RE.findall(response)
    if answer_line:
        return clean_answer_text(answer_line[-1])

    short_last_line = ""
    for line in reversed(response.strip().splitlines()):
        line = line.strip()
        if line:
            short_last_line = line
            break
    if short_last_line and len(short_last_line) <= 80:
        return clean_answer_text(short_last_line)

    return ""


def count_answer_anchors(response: str) -> int:
    return len(ANSWER_ANCHOR_COUNT_RE.findall(response or ""))


def passes_format(response: str) -> bool:
    has_answer_line = count_answer_anchors(response) > 0
    has_boxed = re.search(r"\\boxed\s*\{", response or "") is not None
    has_answer_tag = re.search(r"<answer>.*?</answer>", response or "", re.IGNORECASE | re.DOTALL) is not None
    return (has_answer_line and has_boxed) or has_answer_tag


def is_correct_answer(response: str, extracted: str, label: Any) -> tuple[bool, bool]:
    """Return (is_correct, parse_ok).

    ``parse_ok`` means a structured verifier or symbolic equivalence check was
    used.  Plain normalized string and numeric fallback matches return
    ``parse_ok=False`` so Reward QA can spot fragile cases.
    """

    label_text = "" if label is None else str(label)
    if not label_text.strip():
        return False, False

    parse_result = _math_verify_correct(response=response, extracted=extracted, label=label_text)
    if parse_result is not None:
        return parse_result, True

    for candidate in _answer_variants(extracted):
        for label_candidate in _answer_variants(label_text):
            symbolic_result = _symbolic_equivalent(candidate, label_candidate)
            if symbolic_result is not None:
                return symbolic_result, True

            collection_result = _collection_equivalent(candidate, label_candidate)
            if collection_result is not None:
                return collection_result, True

            normalized_label = normalize_answer(label_candidate)
            normalized_pred = normalize_answer(candidate)
            if normalized_pred and normalized_label and normalized_pred == normalized_label:
                return True, False

            pred_number = to_float_or_none(normalized_pred)
            label_number = to_float_or_none(normalized_label)
            if pred_number is not None and label_number is not None:
                return math.isclose(pred_number, label_number, rel_tol=1e-9, abs_tol=1e-9), False

    return False, False


def normalize_answer(text: str) -> str:
    text = clean_answer_text(text)
    text = text.lower()
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\dfrac": "\\frac",
        "\\tfrac": "\\frac",
        "\\!": "",
        "\\,": "",
        "\\;": "",
        "\\ ": "",
        "\\infty": "infty",
        "+\\infty": "+infty",
        "-\\infty": "-infty",
        "\u221e": "infty",
        "\u222a": "\\cup",
        "\u2212": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\(?:text|mathrm)\s*\{([^{}]*)\}", r"\1", text)
    text = text.replace("$", "")
    text = text.replace(" ", "")
    text = text.strip("。,:;，；")
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    letter_choice = re.fullmatch(r"\(([a-z])\)", text)
    if letter_choice:
        text = letter_choice.group(1)
    text = _normalize_latex_fraction_shorthand(text)
    text = re.sub(r"(?<![\d])([+-]?)\.(\d+)", r"\g<1>0.\2", text)
    text = re.sub(r"([_^])\{([^{}]+)\}", r"\1\2", text)
    text = re.sub(r"(?<=\d)\.+$", "", text)
    return text


def clean_answer_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"^[:：\s]+", "", text)
    text = re.sub(r"\s*(?:<\|im_end\|>|</s>)\s*$", "", text)
    return text.strip()


def _answer_variants(text: str) -> list[str]:
    """Return conservative final-answer variants for common surface forms."""

    variants: list[str] = []

    def add(value: str) -> None:
        value = clean_answer_text(value)
        if value and value not in variants:
            variants.append(value)

    add(text)
    cleaned = clean_answer_text(text)
    if not cleaned:
        return variants

    unwrapped = _strip_math_wrappers(cleaned)
    add(unwrapped)

    first_math = _extract_leading_math_span(cleaned)
    if first_math:
        add(_strip_math_wrappers(first_math))

    embedded_math = _extract_unique_embedded_math_span(cleaned)
    if embedded_math:
        add(embedded_math)

    latex_token = _extract_unique_latex_token(cleaned)
    if latex_token:
        add(latex_token)

    rhs_match = re.fullmatch(r"\s*[a-zA-Z]\s*=\s*(.+?)\s*", unwrapped)
    if rhs_match:
        add(rhs_match.group(1))

    if not _is_structured_final_answer(unwrapped):
        leading_number = re.match(r"\s*([-+]?\d+(?:\.\d+)?)\b", unwrapped)
        if leading_number:
            add(leading_number.group(1))

        numeric_tokens = re.findall(r"[-+]?\d+(?:\.\d+)?", unwrapped)
        if len(numeric_tokens) == 1:
            add(numeric_tokens[0])

    return variants


def _is_structured_final_answer(text: str) -> bool:
    normalized = normalize_answer(text)
    if not normalized:
        return False
    if "infty" in normalized or "\\cup" in normalized:
        return True
    return bool(re.fullmatch(r"[\[(].*,.*[\])]", normalized))


def _normalize_latex_fraction_shorthand(text: str) -> str:
    text = re.sub(r"\\frac\s*([+-]?\d+)\s*\{([^{}]+)\}", r"\\frac{\1}{\2}", text)
    text = re.sub(r"\\frac\s*([+-]?\d)\s*([+-]?\d)(?![\d{}])", r"\\frac{\1}{\2}", text)
    return text


def _extract_leading_math_span(text: str) -> str:
    text = clean_answer_text(text).strip()
    patterns = [
        r"^(\$\$.*?\$\$)",
        r"^(\$.*?\$)",
        r"^(\\\[.*?\\\])",
        r"^(\\\(.*?\\\))",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.DOTALL)
        if match:
            return match.group(1)
    return ""


def _extract_unique_embedded_math_span(text: str) -> str:
    text = clean_answer_text(text).strip()
    spans: list[str] = []
    patterns = [
        r"\$\$(.*?)\$\$",
        r"\$(.*?)\$",
        r"\\\[(.*?)\\\]",
        r"\\\((.*?)\\\)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.DOTALL):
            candidate = match.group(1).strip()
            if _is_substantial_math_candidate(candidate):
                spans.append(candidate)
    return _unique_candidate(spans)


def _extract_unique_latex_token(text: str) -> str:
    text = clean_answer_text(text).strip()
    patterns = [
        r"(?<![A-Za-z0-9}])[-+]?\s*\\(?:dfrac|tfrac|frac)\s*\{[^{}]+\}\s*\{[^{}]+\}",
        r"(?<![A-Za-z0-9}])[-+]?\s*\\sqrt\s*\{[^{}]+\}",
    ]
    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(match.group(0) for match in re.finditer(pattern, text))
    return _unique_candidate(candidates)


def _is_substantial_math_candidate(text: str) -> bool:
    normalized = normalize_answer(text)
    if not normalized:
        return False
    if re.fullmatch(r"[a-z]", normalized):
        return False
    return bool(re.search(r"\d|\\frac|\\sqrt|[(){}]", normalized))


def _unique_candidate(candidates: list[str]) -> str:
    unique: list[str] = []
    for candidate in candidates:
        candidate = clean_answer_text(candidate)
        if candidate and candidate not in unique:
            unique.append(candidate)
    return unique[0] if len(unique) == 1 else ""


def _strip_math_wrappers(text: str) -> str:
    text = clean_answer_text(text).strip()
    pairs = [
        (r"\\\[\s*", r"\s*\\\]"),
        (r"\\\(\s*", r"\s*\\\)"),
        (r"\$\$\s*", r"\s*\$\$"),
        (r"\$\s*", r"\s*\$"),
    ]
    changed = True
    while changed:
        changed = False
        for left, right in pairs:
            pattern = rf"^{left}(.+?){right}$"
            match = re.fullmatch(pattern, text, flags=re.DOTALL)
            if match:
                text = match.group(1).strip()
                changed = True
    return text


def extract_last_boxed(text: str) -> str:
    matches: list[str] = []
    for match in re.finditer(r"\\boxed\s*\{", text or ""):
        start = match.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            matches.append(text[start : i - 1])
    return matches[-1] if matches else ""


def to_float_or_none(text: str) -> float | None:
    try:
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
            return float(text)
    except Exception:
        return None
    return None


def _math_verify_correct(response: str, extracted: str, label: str) -> bool | None:
    if parse is None or verify is None:
        return None

    try:
        gold = parse(label, extraction_mode="first_match")
        if len(gold) == 0:
            return None

        candidates = [extracted, response]
        for candidate in candidates:
            if not candidate:
                continue
            parsed = parse(
                candidate,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            if len(parsed) != 0 and bool(verify(gold, parsed)):
                return True
        return False
    except Exception:
        return None


def _symbolic_equivalent(pred: str, label: str) -> bool | None:
    if sp is None:
        return None

    pred_expr = _to_sympy_expr(pred)
    label_expr = _to_sympy_expr(label)
    if pred_expr is None or label_expr is None:
        return None

    try:
        return bool(sp.simplify(pred_expr - label_expr) == 0)
    except Exception:
        try:
            return bool(sp.equals(pred_expr, label_expr))
        except Exception:
            return None


def _collection_equivalent(pred: str, label: str) -> bool | None:
    pred_items = _split_collection(pred)
    label_items = _split_collection(label)
    if pred_items is None or label_items is None:
        return None
    if len(pred_items) != len(label_items):
        return False

    unmatched = list(label_items)
    for pred_item in pred_items:
        matched_index = None
        for idx, label_item in enumerate(unmatched):
            symbolic = _symbolic_equivalent(pred_item, label_item)
            if symbolic is True or normalize_answer(pred_item) == normalize_answer(label_item):
                matched_index = idx
                break
        if matched_index is None:
            return False
        unmatched.pop(matched_index)
    return True


def _split_collection(text: str) -> list[str] | None:
    normalized = clean_answer_text(text).strip()
    if not normalized:
        return None
    if normalized.startswith("\\{") and normalized.endswith("\\}"):
        normalized = normalized[2:-2]
    elif normalized.startswith("{") and normalized.endswith("}"):
        normalized = normalized[1:-1]
    elif not ("," in normalized or "\uff0c" in normalized):
        return None

    parts = _split_top_level_commas(normalized)
    parts = [part.strip() for part in parts if part.strip()]
    return parts if len(parts) > 1 else None


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for idx, char in enumerate(text):
        if char in "({[":
            depth += 1
        elif char in ")}]":
            depth = max(0, depth - 1)
        elif char in ",\uff0c" and depth == 0:
            parts.append(text[start:idx])
            start = idx + 1
    parts.append(text[start:])
    return parts


def _to_sympy_expr(text: str):
    expr_text = _latexish_to_sympy_text(text)
    if not expr_text:
        return None
    if re.search(r"[a-zA-Z]", expr_text) and not re.search(r"(sqrt|pi|E|oo)", expr_text):
        return None
    try:
        return sp.sympify(expr_text, locals={"sqrt": sp.sqrt, "pi": sp.pi, "E": sp.E, "oo": sp.oo})
    except Exception:
        return None


def _latexish_to_sympy_text(text: str) -> str:
    text = normalize_answer(text)
    if not text:
        return ""

    text = re.sub(r"\\boxed\s*\{(.+)\}$", r"\1", text)
    text = text.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    text = text.replace("^", "**")

    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", text)
        text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)

    text = text.replace("{", "(").replace("}", ")")
    text = re.sub(r"(?<=\d)(?=sqrt\()", "*", text)
    text = re.sub(r"(?<=\))(?=sqrt\()", "*", text)
    text = re.sub(r"(?<=\d)(?=\()", "*", text)
    text = re.sub(r"(?<=\))(?=\d)", "*", text)
    text = text.strip(".")
    return text
