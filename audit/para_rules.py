"""Apply ONE paraphrase rule at a time to localise which rewording breaks the model.
usage: python para_rules.py reclmis|ariadne <runner args without --out> --outbase DIR"""
import os, sys
which = sys.argv[1]; args = sys.argv[2:]
i = args.index("--outbase"); outbase = args[i + 1]; args = args[:i] + args[i + 2:]
sys.path.insert(0, os.getcwd()); sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import para
ALL = list(para.RULES)
if which == "reclmis":
    import reclmis_slots as R
    R.build = para.para_build
    run = R.main
else:
    import ariadne_cf as R
    R.build_conditions = lambda df, n, seed=0: ({"matched": dict(zip(df.Image, df.Description))}, set(), "n/a")
    R.slots.build = para.para_build
    run = R.main
for j, rule in enumerate(ALL):
    para.RULES[:] = [rule]
    print(f"\n### rule {j}: {rule[0]!r} -> {rule[1]!r}", flush=True)
    sys.argv = [sys.argv[0]] + args + ["--out", f"{outbase}/rule{j}"]
    run()
