import json
import os
import subprocess
import sys
import tempfile

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("mode,verdict", [("use", "USES_TEXT"), ("ignore", "NEGLIGIBLE")])
def test_verdict(mode, verdict, tmp_path):
    ck = tmp_path / "m.pt"
    torch.save({"state_dict": {"w": torch.zeros(1)}}, ck)
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "shuffle_test.py"), "--data_dir", "x", "--ckpt", str(ck),
                        "--out", str(out)], capture_output=True, text=True,
                       env={**os.environ, "TBKT_MOCK": mode})
    assert r.returncode == 0, r.stdout + r.stderr
    s = json.load(open(out / "summary.json"))
    assert s["verdict"].startswith(verdict), s["verdict"]
    assert s["n"] == 8


def test_derangement():
    import numpy as np
    sys.path.insert(0, ROOT)
    from shuffle_test import derangement
    rng = np.random.default_rng(1)
    for n in (2, 5, 94):
        p = derangement(n, rng)
        assert sorted(p) == list(range(n)) and not (p == np.arange(n)).any()
