"""Truncation-controlled paraphrase for RecLMIS (CLIP, token_len 18): pick a rewording of the opening phrase
that changes NO report's CLIP token count, then rerun (a) that rule alone and (b) the full paraphrase with it.
This documents how the default rule 0 in para.py ("Bilateral lung infection") was chosen.
Run from the RecLMIS repo root:
  python $AUDIT/para_lenmatch.py --cfg Config_covid19 --ckpt ckpt/qata/best_model-RecLMIS.pth.tar --outbase $OUT/para_lenmatch
"""
import os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import para
from clip.simple_tokenizer import SimpleTokenizer
tok = SimpleTokenizer()
pad = lambda t: t + " EOF XXX" * (9 - len(t.split())) if len(t.split()) < 9 else t
n = lambda t: len(tok.encode(pad(t))) + 2
df = pd.read_excel(os.path.join("datasets", "Covid19", "Test_Folder", "Test_text.xlsx"))   # relative to the RecLMIS repo
CANDS = ["Bilateral lung infection", "Both lungs infected", "Infection both lungs", "Pulmonary infection, bilateral",
         "Bilateral infection of lungs", "Lung infection, bilateral", "Infection, both lungs"]
ORIG = list(para.RULES)
chosen = None
for c in CANDS:
    para.RULES[:] = [(r"\bBilateral pulmonary infection\b", c)]
    d = [n(para.paraphrase(t)) - n(t) for t in df.Description if para.paraphrase(t) != t]
    print(f"{c!r:35s} token delta: min {min(d)}, max {max(d)}, nonzero {np.mean(np.array(d) != 0):.1%}")
    if chosen is None and min(d) == 0 and max(d) == 0:
        chosen = c
print("CHOSEN:", chosen)
if chosen is None:
    sys.exit("no length-matched candidate; paste this output")
full = [(r"\bBilateral pulmonary infection\b", chosen)] + ORIG[1:]
para.RULES[:] = full
d = np.array([n(para.paraphrase(t)) - n(t) for t in df.Description])
print(f"full length-matched paraphrase: token delta min {d.min()} max {d.max()}, reports pushed past 18: "
      f"{np.mean([(n(para.paraphrase(t)) > 18) and (n(t) <= 18) for t in df.Description]):.1%}")
import reclmis_slots as R
R.build = para.para_build
args = sys.argv[1:]
i = args.index("--outbase"); outbase = args[i + 1]; base = args[:i] + args[i + 2:]
for tag, rules in (("rule0_lenmatched", [full[0]]), ("all_lenmatched", full)):
    para.RULES[:] = rules
    print(f"\n### {tag}", flush=True)
    sys.argv = [sys.argv[0]] + base + ["--out", os.path.join(outbase, tag)]
    R.main()
