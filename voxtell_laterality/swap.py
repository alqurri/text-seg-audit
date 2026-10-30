"""Text transforms for the laterality audit.

For each free-text finding we build three prompts:
  orig   - the finding text as written
  swap   - laterality swapped (right<->left, RUL<->LUL, RLL<->LLL, RML<->lingula)
  strip  - laterality removed (side words deleted), only when nothing else implies a side

A finding is eligible only if its text names exactly one side.
"""
import re

BILATERAL = re.compile(r"\b(bilateral(ly)?|both|bibasal|bibasilar|biapical|bilobar\s+bilateral)\b", re.I)
SIDE_WORD = re.compile(r"\b(right|left)\b", re.I)
ABBREV = re.compile(r"\b(RUL|RML|RLL|LUL|LLL)\b", re.I)
RML_PHRASE = re.compile(r"\b(right\s+)?middle\s+lobe\b", re.I)
LINGULA = re.compile(r"\blingula(r)?(\s+segment(s)?)?\b", re.I)

ABBREV_SIDE = {"RUL": "right", "RML": "right", "RLL": "right", "LUL": "left", "LLL": "left"}
ABBREV_SWAP = {"RUL": "LUL", "LUL": "RUL", "RLL": "LLL", "LLL": "RLL"}  # RML handled separately
ABBREV_STRIP = {"RUL": "upper lobe", "LUL": "upper lobe", "RLL": "lower lobe", "LLL": "lower lobe"}


def _match_case(src: str, dst: str) -> str:
    if src.isupper():
        return dst.upper()
    if src[0].isupper():
        return dst[0].upper() + dst[1:]
    return dst


def stated_sides(text: str) -> set:
    """Sides named or implied by the text."""
    sides = {m.group(1).lower() for m in SIDE_WORD.finditer(text)}
    sides |= {ABBREV_SIDE[m.group(1).upper()] for m in ABBREV.finditer(text)}
    if RML_PHRASE.search(text):
        sides.add("right")
    if LINGULA.search(text):
        sides.add("left")
    return sides


def eligibility(text: str):
    """Return (side, reason). side is 'left'/'right' if eligible, else None."""
    if BILATERAL.search(text):
        return None, "bilateral"
    sides = stated_sides(text)
    if not sides:
        return None, "no_side"
    if len(sides) > 1:
        return None, "both_sides"
    return sides.pop(), "ok"


# "lingula of the left upper lobe": the lingula IS part of the LUL, so the whole phrase maps to the RML
LINGULA_OF_LUL = re.compile(r"\blingula(r)?(\s+segment(s)?)?\s+of\s+the\s+(left\s+upper\s+lobe|LUL)\b", re.I)
# placeholders contain no letters, so later word-boundary regexes cannot match inside them
_P_LING, _P_RML, _P_RMLADJ = "\x01", "\x02", "\x03"


def swap_side(text: str) -> str:
    """Swap laterality. Lobe anatomy is mapped to its contralateral counterpart."""
    out = text
    out = LINGULA_OF_LUL.sub(lambda m: _P_RML, out)
    out = RML_PHRASE.sub(lambda m: _P_LING, out)
    out = LINGULA.sub(lambda m: _P_RML if not m.group(1) else _P_RMLADJ, out)
    out = ABBREV.sub(lambda m: _match_case(m.group(1), ABBREV_SWAP[m.group(1).upper()].lower())
                     if m.group(1).upper() in ABBREV_SWAP else _P_LING, out)
    out = SIDE_WORD.sub(lambda m: _match_case(m.group(1), "left" if m.group(1).lower() == "right" else "right"), out)
    out = (out.replace(_P_LING, "lingula")
              .replace(_P_RMLADJ, "right middle lobe")
              .replace(_P_RML, "right middle lobe"))
    return out


def swap_ok(text: str, swapped: str, side: str) -> bool:
    """Sanity check on a swap: printable, and it names exactly the opposite side."""
    return swapped.isprintable() and stated_sides(swapped) == {opposite(side)} and swapped != text


def strip_side(text: str):
    """Remove laterality. Returns None if a side would still be implied (middle lobe / lingula)."""
    if RML_PHRASE.search(text) or LINGULA.search(text) or re.search(r"\bRML\b", text, re.I):
        return None
    out = ABBREV.sub(lambda m: ABBREV_STRIP[m.group(1).upper()], text)
    out = re.sub(r"\b(right|left)\s+", "", out, flags=re.I)
    out = SIDE_WORD.sub("", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    out = re.sub(r"\s+([,.;:])", r"\1", out)
    if stated_sides(out):
        return None
    return out


def opposite(side: str) -> str:
    return "left" if side == "right" else "right"
