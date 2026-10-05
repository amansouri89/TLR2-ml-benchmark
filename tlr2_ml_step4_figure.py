"""
TLR2 ML project - Step 4: make the figure.

Needs (made by step 3, in the same folder):
  tlr2_train_oof.csv, tlr2_heldout_predictions.csv, tlr2_ml_fruit_scores.csv

Install once:  pip install matplotlib

Run:  python -u tlr2_ml_step4_figure.py
Output: tlr2_ml_figure.png (300 dpi) and tlr2_ml_figure.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

oof = pd.read_csv("tlr2_train_oof.csv")
held = pd.read_csv("tlr2_heldout_predictions.csv")
fruit = pd.read_csv("tlr2_ml_fruit_scores.csv")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

# ---- A: leave-families-out ----
rho = spearmanr(oof["oof_pred"], oof["p"]).correlation
mae = np.mean(np.abs(oof["oof_pred"] - oof["p"]))
base = np.mean(np.abs(oof["p"] - oof["p"].median()))
a = ax[0]
a.scatter(oof["p"], oof["oof_pred"], s=22, alpha=0.7, color="#3b6ea5")
lo, hi = oof["p"].min() - 0.2, oof["p"].max() + 0.2
a.plot([lo, hi], [lo, hi], "--", color="grey", lw=1)
a.set_xlim(lo, hi)
a.set_ylim(lo, hi)
a.set_xlabel("Measured pIC50 (ChEMBL)")
a.set_ylabel("Predicted pIC50")
a.set_title(f"A. Unseen chemical families\n"
            f"n = {len(oof)}, {oof['family'].nunique()} families, Spearman \u03c1 = {rho:.2f}")
a.text(0.04, 0.95, f"model error {mae:.2f}\nguess-the-median {base:.2f}",
       transform=a.transAxes, va="top", fontsize=9)

# ---- B: the 8 known antagonists ----
b = ax[1]
b.scatter(held["reported_p"], held["pred_p"], s=40, color="#c0392b")
OFFSETS = {"T12": (-26, 8), "T15": (-30, -14), "CU-CPT22": (8, -12)}  # stop labels overlapping
for _, r in held.iterrows():
    b.annotate(r["name"], (r["reported_p"], r["pred_p"]), fontsize=8,
               xytext=OFFSETS.get(r["name"], (4, 3)), textcoords="offset points")
lo2 = min(held["reported_p"].min(), held["pred_p"].min()) - 0.4
hi2 = max(held["reported_p"].max(), held["pred_p"].max()) + 0.4
b.plot([lo2, hi2], [lo2, hi2], "--", color="grey", lw=1)
b.set_xlim(lo2, hi2)
b.set_ylim(lo2, hi2)
b.set_xlabel("Reported potency (p-value)")
b.set_ylabel("Predicted pIC50")
b.set_title("B. Eight known TLR2 antagonists\n(held out of training)")
b.text(0.04, 0.95, "C11, C13, CU-CPT22: K$_D$ values\n(rough comparison)",
       transform=b.transAxes, va="top", fontsize=8)

# ---- C: how far are the fruit compounds from the training data ----
c = ax[2]
sims = fruit["closest_train_sim"]
c.hist(sims, bins=np.arange(0, 1.05, 0.05), color="#2e8b57", edgecolor="white")
c.axvline(0.4, color="black", ls="--", lw=1)
n_ok = int((sims >= 0.4).sum())
c.set_xlabel("Similarity to closest training compound (Tanimoto)")
c.set_ylabel("Number of fruit compounds")
c.set_title(f"C. Fruit phytochemicals vs training data\n"
            f"{n_ok} of {len(fruit)} within range (\u2265 0.4)")
c.text(0.41, c.get_ylim()[1] * 0.9, "in range \u2192", fontsize=9)

plt.tight_layout()
plt.savefig("tlr2_ml_figure.png", dpi=300)
plt.savefig("tlr2_ml_figure.pdf")
print("Saved: tlr2_ml_figure.png, tlr2_ml_figure.pdf")
print(f"A: rho={rho:.2f}, model error {mae:.2f} vs median-guess {base:.2f}")
print(f"C: {n_ok}/{len(fruit)} fruit compounds with similarity >= 0.4")
