"""Stand-in for VoxTellPredictor used by the end-to-end test.

Works in nnU-Net reader space exactly like VoxTell, so the test exercises the orientation round trip.
mode 'follow': returns lesions on the side the prompt names (what a location-faithful model does).
mode 'ignore': returns the finding's original lesion whatever side the prompt names.
"""
import hashlib
import os

import numpy as np
import torch
from nibabel.orientations import apply_orientation, io_orientation

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import LEFT_LABELS, RIGHT_LABELS, load_nifti, load_rex_mask, read_jsonl  # noqa: E402
from swap import stated_sides  # noqa: E402


def to_reader(arr, affine):
    return apply_orientation(arr, io_orientation(affine)).transpose(2, 1, 0)


class MockPredictor:
    def __init__(self, mode, work):
        self.mode, self.device = mode, torch.device("cpu")
        self.dim = 16
        self.vec2prompt = {}
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
        self.cases = {}
        findings = read_jsonl(os.path.join(work, "findings.jsonl"))
        for s in read_jsonl(os.path.join(work, "scans.jsonl")):
            img, _ = NibabelIOWithReorient().read_images([s["image"]])
            nimg, _ = load_nifti(s["image"])
            _, lobes = load_nifti(os.path.join(work, "lungs", f"{s['case']}.nii.gz"))
            gt = load_rex_mask(s["mask"], s["n_findings"], s["shape"])
            self.cases[hashlib.md5(img.tobytes()).hexdigest()] = dict(
                aff=nimg.affine, lobes=lobes, gt=gt, findings=[f for f in findings if f["case"] == s["case"]])

    def embed_text_prompts(self, prompts):
        out = []
        for p in prompts:
            h = np.frombuffer(hashlib.sha256(p.lower().encode()).digest()[: self.dim * 2], np.uint16)
            v = torch.tensor(h, dtype=torch.float32)
            self.vec2prompt[tuple(v.tolist())] = p.lower()
            out.append(v)
        return torch.stack(out)[None]

    def _side_entities(self, c, side, category):
        lab = LEFT_LABELS if side == "left" else RIGHT_LABELS
        lung_side = np.isin(c["lobes"], lab)
        out = np.zeros(c["lobes"].shape, bool)
        for f in c["findings"]:
            if f.get("category") != category:
                continue
            ch = c["gt"][f["ch"]]
            for v in np.unique(ch[ch > 0]):
                e = ch == v
                if (e & lung_side).sum() >= 0.5 * e.sum() or self._near(e, lung_side):
                    out |= e
        return out

    @staticmethod
    def _near(e, lung):
        from scipy.ndimage import binary_dilation
        return (binary_dilation(lung, iterations=3) & e).sum() >= 0.5 * e.sum()

    def predict_single_image(self, img, text_embeddings):
        c = self.cases[hashlib.md5(img.tobytes()).hexdigest()]
        outs = []
        for v in text_embeddings[0]:
            p = self.vec2prompt[tuple(v.float().tolist())]
            if p in ("left lung", "right lung"):
                m = np.isin(c["lobes"], LEFT_LABELS if p.startswith("left") else RIGHT_LABELS)
            else:
                f = next(f for f in c["findings"] if p in {f["text"].lower(), (f.get("swap") or "").lower(),
                                                           (f.get("strip") or "").lower()})
                own = c["gt"][f["ch"]] > 0
                sides = stated_sides(p)
                if self.mode == "ignore" or not sides or p == f["text"].lower():
                    m = own
                else:
                    m = self._side_entities(c, sides.pop(), f.get("category"))
            outs.append(to_reader(m.astype(np.uint8), c["aff"]))
        return np.stack(outs)
