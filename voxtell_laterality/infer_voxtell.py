"""Step 3 (VoxTell env): run VoxTell on every scan with
  controls: "left lung", "right lung"
  per eligible finding: orig / swap / strip prompts
and store the binary masks (in the CT's original voxel grid) as packed npz in work/preds/.

Prompts are embedded once for the whole set; the orig-vs-swap embedding cosine is saved too
(the "lexical cliff" check: are swapped prompts nearly identical for the text encoder?).

  python infer_voxtell.py --work work [--model-dir /path/voxtell_v1.1] [--max-prompts 6]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_jsonl, reader_to_original, save_packed  # noqa: E402

CONTROLS = {"ctrl_left": "left lung", "ctrl_right": "right lung"}


def scan_prompts(case, findings):
    out = dict(CONTROLS)
    for f in findings:
        if f["case"] != case or not f["side"]:
            continue
        out[f"f{f['key']}_orig"] = f["text"]
        out[f"f{f['key']}_swap"] = f["swap"]
        if f.get("strip"):
            out[f"f{f['key']}_strip"] = f["strip"]
    return out


def get_predictor(a):
    if os.environ.get("LATKT_MOCK"):  # tests only
        from tests.mock_predictor import MockPredictor
        return MockPredictor(os.environ["LATKT_MOCK"], a.work)
    import torch
    from voxtell.inference.predictor import VoxTellPredictor
    dev = torch.device(a.device)
    pred = VoxTellPredictor(model_dir=a.model_dir, device=dev)
    record_model_identity(a, pred)
    return pred


def record_model_identity(a, pred):
    """Write the VoxTell package version and the resolved weights (path, size, sha256) to <work>/model_info.json."""
    import hashlib, json
    from importlib.metadata import version
    info = {"voxtell_version": version("voxtell"), "model_dir_arg": a.model_dir,
            "VOXTELL_MODEL": os.environ.get("VOXTELL_MODEL")}
    md = a.model_dir or os.environ.get("VOXTELL_MODEL") or getattr(pred, "model_dir", None)
    files = []
    if md and os.path.isdir(md):
        for dp, _, fns in os.walk(md):
            for fn in sorted(fns):
                if fn.endswith((".pth", ".pt", ".json")):
                    p = os.path.join(dp, fn); h = hashlib.sha256()
                    with open(p, "rb") as fh:
                        for chunk in iter(lambda: fh.read(1 << 20), b""):
                            h.update(chunk)
                    files.append({"file": os.path.relpath(p, md), "bytes": os.path.getsize(p), "sha256": h.hexdigest()})
    info["weights"] = files
    with open(os.path.join(a.work, "model_info.json"), "w") as fh:
        json.dump(info, fh, indent=1)


def read_image(path):
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    return NibabelIOWithReorient().read_images([path])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--model-dir", default=None, help="default: $VOXTELL_MODEL or HF download")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-prompts", type=int, default=8, help="prompts per forward pass (memory)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    scans = read_jsonl(os.path.join(a.work, "scans.jsonl"))
    findings = read_jsonl(os.path.join(a.work, "findings.jsonl"))
    if a.limit:
        scans = scans[: a.limit]
    pred = get_predictor(a)

    # ---- embed the union of prompts once
    per_scan = {s["case"]: scan_prompts(s["case"], findings) for s in scans}
    union = sorted({p.lower() for d in per_scan.values() for p in d.values()})
    t = time.time()
    emb = {}
    for i in range(0, len(union), 32):
        chunk = union[i:i + 32]
        e = pred.embed_text_prompts(chunk)[0].float().cpu()
        emb.update({p: e[j] for j, p in enumerate(chunk)})
    print(f"embedded {len(union)} prompts in {time.time() - t:.0f}s")

    cos_path = os.path.join(a.work, "embed_cos.jsonl")
    with open(cos_path, "w") as fh:
        for f in findings:
            if not f["side"]:
                continue
            o = emb.get(f["text"].lower())
            if o is None:
                continue
            row = {"case": f["case"], "key": f["key"]}
            for kind in ("swap", "strip"):
                if f.get(kind):
                    v = emb[f[kind].lower()]
                    row[f"cos_{kind}"] = float((o @ v) / (o.norm() * v.norm()))
            fh.write(json.dumps(row) + "\n")

    # ---- predict
    out_dir = os.path.join(a.work, "preds")
    os.makedirs(out_dir, exist_ok=True)
    for i, s in enumerate(scans, 1):
        out = os.path.join(out_dir, f"{s['case']}.npz")
        if os.path.exists(out):
            continue
        t = time.time()
        img, props = read_image(s["image"])
        prompts = per_scan[s["case"]]
        keys = list(prompts)
        masks = {}
        for j in range(0, len(keys), a.max_prompts):
            ks = keys[j:j + a.max_prompts]
            vecs = np.stack([emb[prompts[k].lower()].numpy() for k in ks])[None]
            te = pred.as_embedding_tensor(vecs) if hasattr(pred, "as_embedding_tensor") else _to_dev(vecs, pred)
            seg = pred.predict_single_image(img, text_embeddings=te)   # (n, X, Y, Z) reader space
            seg = reader_to_original(seg, props)
            if tuple(seg.shape[1:]) != tuple(s["shape"]):
                raise RuntimeError(f"{s['case']}: prediction {seg.shape[1:]} != CT {s['shape']} after reorientation")
            masks.update({k: seg[n] > 0 for n, k in enumerate(ks)})
        save_packed(out, masks)
        json.dump(prompts, open(out.replace(".npz", ".prompts.json"), "w"))
        print(f"[{i}/{len(scans)}] {s['case']} {len(keys)} prompts {time.time() - t:.0f}s", flush=True)


def _to_dev(vecs, pred):
    import torch
    return torch.from_numpy(vecs).to(pred.device)


if __name__ == "__main__":
    main()
