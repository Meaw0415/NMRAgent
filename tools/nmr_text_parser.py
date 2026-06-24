"""Utilities for parsing compact NMR text into model-ready shift lists."""

from __future__ import annotations

import re
from typing import List


def parse_numeric_shifts(text: str) -> List[float]:
    vals: List[float] = []
    for token in re.findall(r"-?\d+(?:\.\d+)?", str(text or "")):
        vals.append(float(token))
    return vals


def _representative_shift(segment: str) -> float | None:
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", segment)]
    if not nums:
        return None
    # For ranges such as 3.19-2.98, use the midpoint for integration expansion.
    if re.search(r"\d+(?:\.\d+)?\s*[–-]\s*\d+(?:\.\d+)?", segment) and len(nums) >= 2:
        return round((nums[0] + nums[1]) / 2.0, 4)
    return nums[0]


def _integral(segment: str) -> int:
    match = re.search(r"(\d+(?:\.\d+)?)\s*H\b", segment, flags=re.IGNORECASE)
    if not match:
        return 1
    value = float(match.group(1))
    return max(1, int(round(value)))


def parse_h_nmr_text(text: str) -> List[float]:
    """Parse 1H NMR text and expand shifts by integration.

    Examples:
        1.11 (dd, J = 6.9, 2.9 Hz, 6H) -> six copies of 1.11
        3.19-2.98 (m, 2H) -> two copies of midpoint 3.085
    """
    raw = str(text or "")
    raw = raw.split("δ", 1)[1] if "δ" in raw else raw
    segments = re.split(r"(?<=\))\s*,|;", raw)
    out: List[float] = []
    for segment in segments:
        if not segment.strip():
            continue
        shift = _representative_shift(segment)
        if shift is None:
            continue
        out.extend([shift] * _integral(segment))
    return out


def parse_c_nmr_text(text: str) -> List[float]:
    raw = str(text or "")
    if "δ" in raw:
        raw = raw.split("δ", 1)[1]
    # Drop common instrument/solvent metadata before parsing shifts.
    raw = re.sub(r"\b(13C|101|100|125|150|MHz|CDCl3|Chloroform-d|NMR)\b", " ", raw, flags=re.IGNORECASE)
    return parse_numeric_shifts(raw)
