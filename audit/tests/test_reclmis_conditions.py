import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reclmis_shuffle import build_conditions, lr_swap  # noqa: E402


def test_conditions():
    df = pd.DataFrame({"Image": [f"mask_{i}.png" for i in range(6)],
                       "Description": ["Bilateral, all left lung and all right lung."] * 3 +
                                      ["Unilateral, lower right lung.", "Unilateral, lower left lung.",
                                       "Bilateral, lower left lung and lower right lung."]})
    c, lr_ok, common = build_conditions(df, 3)
    assert common.startswith("Bilateral, all")
    for s in range(3):
        for k, t in c[f"shuffled_s{s}"].items():
            assert t != c["matched"][k]
    assert lr_ok == {"mask_3.png", "mask_4.png"}
    assert lr_swap("Unilateral, lower Right lung.") == "Unilateral, lower Left lung."
