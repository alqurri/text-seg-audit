"""Figure 1: slot-wise counterfactual effects (Dice points lost, 95% paired bootstrap CI clustered by patient
(QaTa-COV19) or CT volume (MosMedData+); values from analysis/cluster_ci.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

slots = ["side\n(left$\\leftrightarrow$right)", "extent\n(all/upper/...)", "count\n(# areas)", "bi-/uni-\nlateral"]
series = [
    ("RecLMIS, QaTa (X-ray)", "#2b6cb0",
     [(16.18, 13.97, 18.60), (11.08, 10.20, 11.98), (0.15, 0.05, 0.25), (1.16, 0.86, 1.48)]),
    ("Ariadne, QaTa (X-ray)", "#6b9bd1",
     [(20.19, 17.64, 22.87), (7.76, 7.10, 8.43), (0.08, -0.02, 0.17), (0.15, 0.05, 0.25)]),
    ("RecLMIS, MosMed (CT)", "#c05621",
     [(-0.06, -0.25, 0.45), (0.13, -0.42, 0.39), (1.19, 0.60, 1.86), (1.41, 0.55, 1.98)]),
    ("Ariadne, MosMed (CT)", "#e8a06a",
     [(0.24, 0.15, 0.52), (0.04, -0.46, 0.25), (0.02, -0.08, 0.10), (0.29, -0.02, 0.88)]),
]
plt.rcParams.update({"font.family": "serif", "font.size": 7.5})
fig, ax = plt.subplots(figsize=(3.4, 1.8))
x = np.arange(len(slots))
w = 0.2
for i, (lab, col, data) in enumerate(series):
    m = np.array([d[0] for d in data])
    err = np.array([[d[0] - d[1] for d in data], [d[2] - d[0] for d in data]])
    ax.bar(x + (i - 1.5) * w, m, w, yerr=err, capsize=1.3, color=col, label=lab, error_kw={"lw": 0.6})
ax.axhline(0, color="k", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(slots)
ax.set_ylabel("Dice drop (points)")
ax.legend(frameon=False, loc="upper right", fontsize=6.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.2)
fig.savefig("fig_slots.pdf")
print("ok")
