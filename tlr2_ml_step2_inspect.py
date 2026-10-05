"""
TLR2 ML project - Step 2: look at the ChEMBL data before we trust it.

Put this file in the SAME folder as tlr2_ml_step1_data.py and the files it made
(tlr2_chembl_raw.csv, tlr2_train_clean.csv, tlr2_heldout_8.csv).

  python tlr2_ml_step2_inspect.py --inspect
      (a) saves the compounds with conflicting results -> tlr2_conflicts.csv
      (b) shows which assays the data comes from
      (c) counts the chemical families among the actives
      (d) shows how close each of the 8 known antagonists is to the training data
      No internet needed.

  python tlr2_ml_step2_inspect.py --pubchem
      lists PubChem assays linked to TLR2 -> pubchem_tlr2_assays.csv
      Internet needed. We read the list together and choose which assays to use.
"""

import argparse
import sys

import pandas as pd

from tlr2_ml_step1_data import standardise, label_row, CUTOFF_NM


def inspect():
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina

    raw = pd.read_csv("tlr2_chembl_raw.csv")
    raw["standard_value"] = pd.to_numeric(raw["standard_value"], errors="coerce")
    raw["label"] = [label_row(r, v) for r, v in zip(raw["standard_relation"], raw["standard_value"])]
    raw = raw.dropna(subset=["label"]).copy()
    raw["label"] = raw["label"].astype(int)
    std = raw["canonical_smiles"].apply(standardise)
    raw["smiles_clean"] = [s[0] for s in std]
    raw["ikey"] = [s[1] for s in std]
    raw = raw.dropna(subset=["smiles_clean"])

    # ---------- (a) conflicts ----------
    mixed = raw.groupby("ikey")["label"].nunique()
    conflict_keys = mixed[mixed > 1].index
    conf = raw[raw["ikey"].isin(conflict_keys)].sort_values(["ikey", "standard_value"])
    cols = ["ikey", "smiles_clean", "standard_type", "standard_relation",
            "standard_value", "label", "assay_chembl_id", "assay_description",
            "document_chembl_id", "target_used"]
    cols = [c for c in cols if c in conf.columns]
    conf[cols].to_csv("tlr2_conflicts.csv", index=False)
    print(f"(a) {len(conflict_keys)} compounds have conflicting results -> tlr2_conflicts.csv")
    print("    Open it in Excel and look at assay_description for each pair.\n")

    # ---------- (b) assays ----------
    train = pd.read_csv("tlr2_train_clean.csv")
    used = raw[raw["ikey"].isin(train["ikey"])]
    print("(b) Top assays in the training data (rows, compounds):")
    top = (used.groupby(["assay_chembl_id", "assay_description"])
               .agg(rows=("ikey", "size"), compounds=("ikey", "nunique"))
               .sort_values("rows", ascending=False).head(10))
    for (aid, desc), r in top.iterrows():
        print(f"    {aid}  {r['rows']:>3} rows, {r['compounds']:>3} cpds  {str(desc)[:90]}")
    print("    Look for words like 'agonist' / 'EC50' / 'activation' - those are NOT antagonists.\n")

    # ---------- (c) chemical families ----------
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = [Chem.MolFromSmiles(s) for s in train["smiles"]]
    fps = [gen.GetFingerprint(m) for m in mols]
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    clusters = Butina.ClusterData(dists, n, 0.6, isDistData=True)  # similarity >= 0.4 = same family
    sizes = sorted([len(c) for c in clusters], reverse=True)
    singles = sum(1 for s in sizes if s == 1)
    print(f"(c) {n} training compounds fall into {len(sizes)} chemical families")
    print(f"    biggest families: {sizes[:8]}")
    print(f"    compounds that are alone in their family: {singles}")
    train = train.copy()
    fam = {}
    for k, c in enumerate(clusters):
        for idx in c:
            fam[idx] = k
    train["family"] = [fam[i] for i in range(n)]
    train.to_csv("tlr2_train_with_families.csv", index=False)
    print("    saved -> tlr2_train_with_families.csv\n")

    # ---------- (d) how close are the 8 known antagonists to the training data ----------
    held = pd.read_csv("tlr2_heldout_8.csv")
    print("(d) Closest training compound for each known antagonist (Tanimoto, 1.0 = identical):")
    for _, row in held.iterrows():
        m = Chem.MolFromSmiles(row["smiles"])
        sims = DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m), fps)
        print(f"    {row['name']:<9} {max(sims):.2f}")
    print("    Below ~0.4 means the model has not seen anything like it: a hard, fair test.")


def pubchem_list():
    import requests
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    print("Asking PubChem for assays linked to TLR2 (max 60 s per call)...", flush=True)
    r = requests.get(f"{base}/assay/target/genesymbol/TLR2/aids/JSON", timeout=60)
    r.raise_for_status()
    aids = r.json().get("IdentifierList", {}).get("AID", [])
    print(f"Found {len(aids)} assays. Reading their names...", flush=True)
    rows = []
    for k in range(0, len(aids), 20):
        chunk = aids[k:k + 20]
        ids = ",".join(str(a) for a in chunk)
        try:
            rr = requests.get(f"{base}/assay/aid/{ids}/summary/JSON", timeout=60)
            rr.raise_for_status()
            for s in rr.json().get("AssaySummaries", {}).get("AssaySummary", []):
                rows.append({
                    "AID": s.get("AID"),
                    "name": s.get("Name"),
                    "source": s.get("SourceName"),
                    "tested": s.get("SIDCountAll"),
                    "active": s.get("SIDCountActive"),
                    "inactive": s.get("SIDCountInactive"),
                })
        except Exception as e:
            print(f"  chunk {k}: {e}", flush=True)
        print(f"  {min(k + 20, len(aids))}/{len(aids)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("pubchem_tlr2_assays.csv", index=False)
    print("\nSaved -> pubchem_tlr2_assays.csv")
    if not df.empty:
        print(df.sort_values("tested", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--pubchem", action="store_true")
    a = ap.parse_args()
    if a.inspect:
        inspect()
    elif a.pubchem:
        pubchem_list()
    else:
        print("Use --inspect or --pubchem")
        sys.exit(1)
