"""
TLR2 ML project - Step 3: train a potency model on ChEMBL data and test it.

Needs: tlr2_chembl_raw.csv and tlr2_heldout_8.csv (made by step 1),
and tlr2_ml_step1_data.py in the same folder.

Install once:  pip install scikit-learn numpy openpyxl

  python -u tlr2_ml_step3_train.py --train
      1. builds the training set (pIC50 = how strong; no inactives needed)
      2. tests the model by chemical family (leave one family out)
      3. tests the model on the 8 known antagonists (never used in training)
      4. saves the model -> tlr2_model.pkl

  python -u tlr2_ml_step3_train.py --score apple.xlsx blackberry.xlsx blueberry.xlsx
      scores each fruit compound -> tlr2_ml_fruit_scores.csv
      (use the names of your fruit Excel files)

pIC50 = 9 - log10(value in nM).  Higher = stronger.  5.0 = 10 uM, 4.7 = 20 uM.
"""

import argparse
import math
import pickle
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from tlr2_ml_step1_data import standardise

SEED = 42
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Reported potency of the 8 known antagonists, converted to p-values (9 - log10 nM).
# Jiao set = IC50 in uM.  C11, C13, CU-CPT22 = K_D in M (different assay: so the
# comparison across the two groups is only rough).
HELDOUT_P = {
    "T9": 9 - math.log10(13.32e3),
    "T2": 9 - math.log10(14.88e3),
    "T5": 9 - math.log10(15.71e3),
    "T15": 9 - math.log10(16.63e3),
    "T12": 9 - math.log10(18.23e3),
    "C11": 9 - math.log10(1.51e-3 * 1e9),
    "C13": 9 - math.log10(1.94e-4 * 1e9),
    "CU-CPT22": 9 - math.log10(1.28e-5 * 1e9),
}


# ---------------------------------------------------------------- features
def featurise(smiles_list):
    """Fingerprint bits + 6 simple descriptors. Returns (X, fingerprints)."""
    X, fps = [], []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        fp = GEN.GetFingerprint(m)
        arr = np.zeros((2048,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        desc = [
            Descriptors.MolWt(m), Crippen.MolLogP(m),
            rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumHBD(m),
            rdMolDescriptors.CalcNumAromaticRings(m),
            rdMolDescriptors.CalcNumRotatableBonds(m),
        ]
        X.append(np.concatenate([arr, desc]))
        fps.append(fp)
    return np.array(X, dtype=float), fps


def make_model():
    return RandomForestRegressor(n_estimators=500, min_samples_leaf=1,
                                 random_state=SEED, n_jobs=-1)


def predict_with_spread(model, X):
    """Mean prediction and the spread between trees (a rough 'how unsure')."""
    per_tree = np.array([t.predict(X) for t in model.estimators_])
    return per_tree.mean(axis=0), per_tree.std(axis=0)


def max_sim(fps_query, fps_train):
    return np.array([max(DataStructs.BulkTanimotoSimilarity(f, fps_train)) for f in fps_query])


# ---------------------------------------------------------------- training set
def build_training_set():
    raw = pd.read_csv("tlr2_chembl_raw.csv")
    raw["standard_value"] = pd.to_numeric(raw["standard_value"], errors="coerce")
    # exact values only ("=" ), positive, antagonist-type measurements
    raw = raw[(raw["standard_relation"] == "=") & (raw["standard_value"] > 0)].copy()
    std = raw["canonical_smiles"].apply(standardise)
    raw["smiles"] = [s[0] for s in std]
    raw["ikey"] = [s[1] for s in std]
    raw = raw.dropna(subset=["smiles"])
    raw["p"] = 9 - np.log10(raw["standard_value"])

    held = pd.read_csv("tlr2_heldout_8.csv")
    raw = raw[~raw["ikey"].isin(set(held["ikey"]))]

    # one row per compound = median of its measurements (this also settles the
    # "conflicting results" compounds: we use the middle value)
    g = raw.groupby("ikey").agg(smiles=("smiles", "first"), p=("p", "median"),
                                n=("p", "size"), spread=("p", lambda x: x.max() - x.min()))
    g = g.reset_index()
    print(f"Training compounds with exact values: {len(g)}")
    print(f"  pIC50 range {g['p'].min():.2f} to {g['p'].max():.2f}, median {g['p'].median():.2f}")
    print(f"  compounds whose repeated measurements differ by > 1 log unit: {(g['spread'] > 1).sum()}")
    return g


def families(fps, cutoff=0.6):
    n = len(fps)
    d = []
    for i in range(1, n):
        d.extend([1 - x for x in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])])
    cl = Butina.ClusterData(d, n, cutoff, isDistData=True)
    fam = np.zeros(n, dtype=int)
    for k, c in enumerate(cl):
        for i in c:
            fam[i] = k
    return fam


def train():
    tr = build_training_set()
    X, fps = featurise(tr["smiles"])
    y = tr["p"].values
    fam = families(fps)
    print(f"  chemical families: {len(set(fam))}\n")

    # ---- test 1: leave families out ----
    n_splits = min(5, len(set(fam)))
    oof = np.zeros(len(y))
    for tri, tei in GroupKFold(n_splits=n_splits).split(X, y, groups=fam):
        m = make_model().fit(X[tri], y[tri])
        oof[tei] = m.predict(X[tei])
    rho = spearmanr(oof, y).correlation
    mae = np.mean(np.abs(oof - y))
    base = np.mean(np.abs(y - np.median(y)))
    print("TEST 1: leave-families-out (the model never sees the family it is tested on)")
    print(f"   Spearman rho = {rho:.2f}   (0 = random, 1 = perfect)")
    print(f"   mean error = {mae:.2f} log units   vs {base:.2f} for always guessing the median")
    q90 = float(np.quantile(np.abs(oof - y), 0.90))
    print(f"   90% of predictions are within +/- {q90:.2f} log units\n")

    # ---- final model on everything ----
    model = make_model().fit(X, y)

    # ---- test 2: the 8 known antagonists ----
    held = pd.read_csv("tlr2_heldout_8.csv")
    Xh, fph = featurise(held["smiles"])
    ph, sdh = predict_with_spread(model, Xh)
    sims = max_sim(fph, fps)
    true = np.array([HELDOUT_P[n] for n in held["name"]])
    print("TEST 2: the 8 known antagonists (never used in training)")
    print(f"   {'name':<9}{'reported':>9}{'predicted':>10}{'closest':>9}")
    for n, t, p, s in zip(held["name"], true, ph, sims):
        print(f"   {n:<9}{t:>9.2f}{p:>10.2f}{s:>9.2f}")
    rho_h = spearmanr(ph, true).correlation
    print(f"   Spearman rho (all 8) = {rho_h:.2f}")
    jiao = [i for i, n in enumerate(held["name"]) if n.startswith("T")]
    print(f"   mean error = {np.mean(np.abs(ph - true)):.2f} log units")
    print("   Note: the five T compounds are all close to 4.8 (a narrow range), so ranking")
    print("   among them is not expected to work. Look at the size of the error instead.")
    print("   C11/C13/CU-CPT22 are K_D values, so they are only a rough comparison.\n")

    with open("tlr2_model.pkl", "wb") as f:
        pickle.dump({"model": model, "fps": fps, "q90": q90}, f)
    pd.DataFrame({"smiles": tr["smiles"], "p": y, "family": fam, "oof_pred": oof}
                 ).to_csv("tlr2_train_oof.csv", index=False)
    pd.DataFrame({"name": held["name"], "reported_p": true, "pred_p": ph,
                  "closest_train_sim": sims}).to_csv("tlr2_heldout_predictions.csv", index=False)
    print("Saved: tlr2_model.pkl, tlr2_train_oof.csv, tlr2_heldout_predictions.csv")


# ---------------------------------------------------------------- score fruit
def read_fruit_sheet(path):
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    for ws in wb.worksheets:
        for r in range(1, 25):
            vals = [str(c.value).strip() if c.value is not None else "" for c in ws[r]]
            low = [v.lower() for v in vals]
            cand = [i for i, v in enumerate(low) if "smiles" in v]
            if not cand:
                continue
            # prefer "actual", then anything that is not the "search" (Kekule) column
            pick = ([i for i in cand if "actual" in low[i]]
                    or [i for i in cand if "search" not in low[i]]
                    or cand)
            smi_col = pick[0]
            name_col = next((i for i, v in enumerate(low) if "name" in v or "compound" in v), 1)
            print(f"  {path}: sheet '{ws.title}', header row {r}, "
                  f"SMILES column = '{vals[smi_col]}', name column = '{vals[name_col]}'", flush=True)
            rows = []
            for row in ws.iter_rows(min_row=r + 1, values_only=True):
                if row is None or len(row) <= smi_col or not row[smi_col]:
                    continue
                rows.append({"name": row[name_col] if name_col < len(row) else "",
                             "smiles_raw": str(row[smi_col])})
            if rows:
                return rows
    raise SystemExit(f"{path}: no column with 'SMILES' in its header was found on any sheet")


def score(paths):
    with open("tlr2_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model, fps_tr, q90 = bundle["model"], bundle["fps"], bundle["q90"]
    out = []
    for path in paths:
        fruit = path.replace("\\", "/").split("/")[-1].split(".")[0]
        for row in read_fruit_sheet(path):
            clean, _ = standardise(row["smiles_raw"])
            if clean is None:
                continue
            out.append({"fruit": fruit, "name": row["name"], "smiles": clean})
    df = pd.DataFrame(out)
    X, fps = featurise(df["smiles"])
    p, sd = predict_with_spread(model, X)
    sim = max_sim(fps, fps_tr)
    df["pred_pIC50"] = p.round(2)
    df["pred_uM"] = (10 ** (6 - p)).round(1)  # pIC50 -> uM
    df["tree_spread"] = sd.round(2)
    df["closest_train_sim"] = sim.round(2)
    df["trust"] = np.where(sim >= 0.4, "ok", np.where(sim >= 0.25, "low", "outside"))
    df = df.sort_values("pred_pIC50", ascending=False)
    df.to_csv("tlr2_ml_fruit_scores.csv", index=False)
    print(f"Scored {len(df)} fruit compounds -> tlr2_ml_fruit_scores.csv")
    print(f"Typical error of the model: +/- {q90:.2f} log units (90% interval, from the family test)")
    print("\nHow close the fruit compounds are to the training data:")
    print(df["trust"].value_counts().to_string())
    print("  ok = similar compounds were in training; low = weak match; outside = the model")
    print("  has not seen anything like it, so its score is a guess.")
    print("\nTop 15:")
    print(df.head(15)[["fruit", "name", "pred_pIC50", "pred_uM", "trust"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--score", nargs="+", help="fruit Excel files")
    a = ap.parse_args()
    if a.train:
        train()
    elif a.score:
        score(a.score)
    else:
        print("Use --train, then --score file1.xlsx file2.xlsx ...")
        sys.exit(1)
