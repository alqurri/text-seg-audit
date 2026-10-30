"""Meaning-preserving paraphrase of the LViT report template (wording changes, no attribute changes)."""
import re

RULES = [
    (r"\bBilateral pulmonary infection\b", "Bilateral lung infection"),
    (r"\bUnilateral pulmonary infection\b", "Infection of one lung"),
    (r"\binfected areas\b", "lesions"),
    (r"\binfected area\b", "lesion"),
    (r"\ball (left|right) lung\b", r"whole \1 lung"),
    (r"\b(upper|middle|lower) (left|right) lung\b", r"\2 \1 lung"),
]


def paraphrase(t):
    s = str(t)
    for pat, rep in RULES:
        s = re.sub(pat, rep, s, flags=re.I)
    return s


def para_build(df, seed=0):
    own = dict(zip(df.Image, df.Description))
    d = {k: paraphrase(t) for k, t in own.items()}
    d = {k: v for k, v in d.items() if v.strip().lower() != str(own[k]).strip().lower()}
    return own, {"para": d}
