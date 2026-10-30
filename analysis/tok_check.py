"""Share of reports past each model's text limit before vs after paraphrase
(RecLMIS: 18 CLIP tokens after its 'EOF XXX' padding; Ariadne's Thread: 24 CXR-BERT tokens).
Run once in an env with OpenAI CLIP and once in an env with transformers (each part is skipped if missing):
  python tok_check.py --xlsx QaTa=$EXT/RecLMIS/datasets/Covid19/Test_Folder/Test_text.xlsx \
                      --xlsx MosMed=$EXT/RecLMIS/datasets/MosMedplus/Test_Folder/Test_text.xlsx
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "audit"))
import para  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--xlsx", action="append", required=True, help="NAME=path")
a = ap.parse_args()
counters = {}
try:
    from clip.simple_tokenizer import SimpleTokenizer
    st = SimpleTokenizer()
    pad = lambda t: t + " EOF XXX" * (9 - len(t.split())) if len(t.split()) < 9 else t
    counters["RecLMIS/CLIP (limit 18)"] = (lambda t: len(st.encode(pad(t))) + 2, 18)
except ImportError:
    pass
try:
    from transformers import AutoTokenizer
    bt = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True)
    counters["Ariadne/CXR-BERT (limit 24)"] = (lambda t: len(bt.encode(t)), 24)
except Exception as e:
    print("CXR-BERT tokenizer unavailable:", e)
for spec in a.xlsx:
    dname, path = spec.split("=", 1)
    df = pd.read_excel(path); col = "Description" if "Description" in df else df.columns[1]
    texts = [str(t) for t in df[col]]
    for cname, (f, lim) in counters.items():
        o = np.array([f(t) for t in texts]); p = np.array([f(para.paraphrase(t)) for t in texts])
        print(f"{dname:6s} {cname:28s} orig >{lim}: {np.mean(o > lim):5.1%} | para >{lim}: {np.mean(p > lim):5.1%} "
              f"| newly truncated {np.mean((p > lim) & (o <= lim)):5.1%} | max len {o.max()}->{p.max()}")
