import json
import os
import subprocess
import sys
import tempfile

import nibabel as nib
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from common import load_packed, pack_mask, reader_to_original, save_packed, unpack_mask  # noqa: E402
from swap import eligibility, stated_sides, strip_side, swap_side  # noqa: E402


# ------------------------------------------------------------------ text transforms

@pytest.mark.parametrize("text,side,swap,strip", [
    ("Nodule in the right upper lobe.", "right", "Nodule in the left upper lobe.", "Nodule in the upper lobe."),
    ("Left pleural effusion.", "left", "Right pleural effusion.", "pleural effusion."),
    ("Consolidation in the LUL.", "left", "Consolidation in the RUL.", "Consolidation in the upper lobe."),
    ("Nodule in the right middle lobe.", "right", "Nodule in the lingula.", None),
    ("Lingular nodule.", "left", "right middle lobe nodule.", None),
    ("Opacity in the middle lobe", "right", "Opacity in the lingula", None),
    ("RLL atelectasis", "right", "LLL atelectasis", "lower lobe atelectasis"),
])
def test_swap(text, side, swap, strip):
    assert eligibility(text) == (side, "ok")
    assert swap_side(text) == swap
    assert stated_sides(swap_side(text)) == ({"left", "right"} - {side})
    assert strip_side(text) == strip


@pytest.mark.parametrize("text,reason", [
    ("Bilateral pleural effusion.", "bilateral"), ("Mild cardiomegaly.", "no_side"),
    ("Nodules in the right and left lower lobes.", "both_sides"), ("Bibasilar atelectasis", "bilateral"),
])
def test_ineligible(text, reason):
    assert eligibility(text) == (None, reason)


# ------------------------------------------------------------------ orientation round trip vs nnU-Net writer

AFFS = [np.diag([-0.7, -0.7, 1.5, 1]), np.diag([0.7, -0.7, -1.5, 1]),
        np.array([[0, 0, 1.5, 0], [-0.8, 0, 0, 0], [0, 0.8, 0, 0], [0, 0, 0, 1.0]]),
        np.array([[0, -0.9, 0, 0], [0, 0, -2.0, 0], [0.9, 0, 0, 0], [0, 0, 0, 1.0]])]


@pytest.mark.parametrize("aff", AFFS)
def test_reader_to_original_matches_nnunet_writer(aff, tmp_path):
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    rng = np.random.default_rng(1)
    vol = rng.integers(0, 3, (11, 7, 5)).astype(np.int16)
    p = tmp_path / "img.nii.gz"
    nib.save(nib.Nifti1Image(vol, aff), p)
    io = NibabelIOWithReorient()
    data, props = io.read_images([str(p)])
    seg = (data[0] > 0).astype(np.uint8)            # a "prediction" in reader space
    io.write_seg(seg, str(tmp_path / "seg.nii.gz"), props)
    via_writer = np.asanyarray(nib.load(tmp_path / "seg.nii.gz").dataobj)
    assert np.array_equal(reader_to_original(seg, props), via_writer)
    assert np.array_equal(via_writer, (vol > 0).astype(np.uint8))   # and it really is the original grid


def test_pack_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    m = np.zeros((30, 20, 10), bool)
    m[3:9, 5:6, 2:8] = rng.random((6, 1, 6)) > 0.5
    assert np.array_equal(unpack_mask(pack_mask(m)), m)
    save_packed(tmp_path / "x.npz", {"a": m, "b": np.zeros_like(m)})
    back = load_packed(tmp_path / "x.npz")
    assert np.array_equal(back["a"], m) and not back["b"].any()


# ------------------------------------------------------------------ end to end with a mock VoxTell

def run(*args, env=None):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, **(env or {})})
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@pytest.fixture(scope="module")
def synthetic():
    import make_synthetic
    d = tempfile.mkdtemp()
    ct, rex, lobes = make_synthetic.build(d)
    return d, ct, rex, lobes


@pytest.mark.parametrize("mode,sfr", [("follow", 0.0), ("ignore", 1.0)])
def test_end_to_end(synthetic, mode, sfr):
    d, ct, rex, lobes = synthetic
    work = os.path.join(d, f"work_{mode}")
    run("prepare.py", "--rex-dir", rex, "--ctrate-dir", ct, "--work", work, "--source", "dataset.json:val")
    os.makedirs(os.path.join(work, "lungs"), exist_ok=True)
    for case, lab in lobes.items():   # stand-in for TotalSegmentator: true lobes in the CT grid
        aff = nib.load(os.path.join(ct, f"{case}.nii.gz")).affine
        nib.save(nib.Nifti1Image(lab, aff), os.path.join(work, "lungs", f"{case}.nii.gz"))
    run("infer_voxtell.py", "--work", work, "--device", "cpu", env={"LATKT_MOCK": mode, "PYTHONPATH": ROOT})
    run("analyze.py", "--work", work)
    s = json.load(open(os.path.join(work, "results", "summary.json")))
    assert s["gate_lung_control_pass_rate"] == 1.0
    assert s["gate_alignment_median_in_roi"] >= 0.85
    assert s["n_scored"] == 6 and s["n_hit"] == 6, s       # 6 single-side findings, all consistent with masks
    assert s["SFR"] == pytest.approx(sfr)
    assert s["verdict"].startswith("UNDERPOWERED")          # 5 hits < 40, by design of the toy set
    if mode == "follow":
        assert s["swap_suppressed_rate"] > 0                  # no contralateral lesion -> empty swap prediction
        assert s["swap_correct_side_mean_(non-empty)"] == pytest.approx(1.0)
