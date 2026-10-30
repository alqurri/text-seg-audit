"""Paraphrase effect for RecLMIS restricted to reports that fit in 18 CLIP tokens before AND after rewriting."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import para
import reclmis_slots as S
from clip.simple_tokenizer import SimpleTokenizer
tok = SimpleTokenizer()
pad = lambda t: t + " EOF XXX" * (9 - len(t.split())) if len(t.split()) < 9 else t
n = lambda t: len(tok.encode(pad(t))) + 2
def build(df, seed=0):
    own, conds = para.para_build(df)
    keep = {k: v for k, v in conds["para"].items() if n(own[k]) <= 18 and n(v) <= 18}
    print(f"untruncated before and after: {len(keep)} of {len(conds['para'])} paraphrased reports")
    return own, {"para_untrunc": keep}
S.build = build
S.main()
