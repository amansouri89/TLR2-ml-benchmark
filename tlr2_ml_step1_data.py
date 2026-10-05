"""
TLR2 ML project - Step 1: get TLR2 data from ChEMBL and clean it.

Run it in two parts:

  Part A (find the target):
      python tlr2_ml_step1_data.py --search
    This prints ChEMBL targets that match "Toll-like receptor 2".
    Pick the HUMAN single-protein TLR2 and copy its ID (CHEMBL...).

  Part B (download + clean):
      python tlr2_ml_step1_data.py --target CHEMBLxxxx

Install first:
      pip install requests rdkit pandas

Output:
  tlr2_chembl_raw.csv       all activities downloaded (for your records)
  tlr2_train_clean.csv      one row per compound, with label 1 (active) / 0 (inactive)
  tlr2_heldout_8.csv        the 8 known antagonists (NOT used for training)

Notes:
  - No BBB filter is used here (not needed for the TLR2 binding model).
  - Activity cut-off: 20 uM  (IC50/Ki/Kd <= 20 uM = active). Change CUTOFF_NM if needed.
  - We only keep IC50, Ki and Kd. EC50 is skipped on purpose because in
    ChEMBL it is mostly for AGONISTS (for example Pam3CSK4), and we want antagonists.
"""

import argparse
import sys

import pandas as pd
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

CUTOFF_NM = 20000  # 20 uM
KEEP_TYPES = ["IC50", "Ki", "Kd"]

# 8 known TLR2 antagonists (from the two papers + CU-CPT22). Kept out of training.
HELDOUT = {
    "C11": "O=C(NCc1n[nH]c2c1CCCCC2)CC1N(CCC(C)C)CCNC1=O",
    "C13": "COC(=O)c1c2CCCc2sc1NC(=O)NCCN1CCOCC1",
    "T9": "CC(=O)c1cccc(NC(=O)COC(=O)c2c3ccccc3nc3c2CCCC3)c1",
    "T2": "Oc1cccc(NC(=S)NC(=O)c2cc(-c3cccc(OCC(C)C)c3)nc4ccccc24)c1",
    "T5": "COc1ccc(CNC(=O)c2ccc(Nc3ccnc(-c4ccc[nH]4)n3)cc2)cc1OC",
    "T15": "COc1ccc(CCN(CCc2nc(-c3ccc(Cl)cc3)no2)C(=O)c2ccccc2)cc1OC",
    "T12": "O=C(Nc1cccc(C(F)(F)F)c1)C2Sc3nnc(CCC)n3NC2c1ccc(OC)cc1",
    "CU-CPT22": "O=C(C(C=C1O)=Cc2cc(OC)c(O)c(O)c2C1=O)OCCCCCC",
}


def standardise(smiles):
    """Return (clean_smiles, inchikey_first_block) or (None, None) if the SMILES is bad."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    try:
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)  # remove salts
        mol = rdMolStandardize.Uncharger().uncharge(mol)             # neutralise
        Chem.SanitizeMol(mol)
    except Exception:
        return None, None
    clean = Chem.MolToSmiles(mol)
    key = inchi.MolToInchiKey(mol)
    if not key:
        return None, None
    return clean, key.split("-")[0]  # first block = same skeleton, ignores stereo


BASE = "https://www.ebi.ac.uk/chembl/api/data"


def get_json(url, params=None):
    """Call the ChEMBL web service with a time limit, so it never hangs silently."""
    r = requests.get(url, params=params, timeout=60,
                     headers={"Accept": "application/json"})
    r.raise_for_status()
    return r.json()


def search_targets():
    print("Searching ChEMBL (max 60 s per call)...", flush=True)
    data = get_json(f"{BASE}/target/search.json",
                    {"q": "Toll-like receptor 2", "limit": 50})
    rows = []
    for t in data.get("targets", []):
        rows.append({
            "target_chembl_id": t.get("target_chembl_id"),
            "pref_name": t.get("pref_name"),
            "organism": t.get("organism"),
            "target_type": t.get("target_type"),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\nChoose the HUMAN, SINGLE PROTEIN target for TLR2, then run:")
    print("  python tlr2_ml_step1_data.py --target CHEMBLxxxx")


def download_activities(target_id):
    print(f"Downloading activities for {target_id} ...", flush=True)
    url = f"{BASE}/activity.json"
    params = {
        "target_chembl_id": target_id,
        "standard_type__in": ",".join(KEEP_TYPES),
        "standard_units": "nM",
        "limit": 1000,
    }
    rows = []
    page = 1
    while url:
        data = get_json(url, params)
        rows.extend(data.get("activities", []))
        print(f"  page {page}: {len(rows)} rows so far", flush=True)
        nxt = data.get("page_meta", {}).get("next")
        url = ("https://www.ebi.ac.uk" + nxt) if nxt else None
        params = None  # the "next" link already has the settings
        page += 1
    return pd.DataFrame(rows)


def label_row(rel, val):
    """Return 1 (active), 0 (inactive) or None (cannot decide)."""
    if pd.isna(val):
        return None
    if rel in ("=", "~", None) or pd.isna(rel):
        return 1 if val <= CUTOFF_NM else 0
    if rel in ("<", "<="):
        return 1 if val <= CUTOFF_NM else None
    if rel in (">", ">="):
        return 0 if val >= CUTOFF_NM else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true", help="list TLR2 targets in ChEMBL")
    ap.add_argument("--target", nargs="+", help="one or more ChEMBL target IDs (separate with spaces)")
    args = ap.parse_args()

    if args.search:
        search_targets()
        return
    if not args.target:
        print("Use --search first, then --target CHEMBLxxxx")
        sys.exit(1)

    # ---- download ----
    parts = []
    for tid in args.target:
        d = download_activities(tid)
        if not d.empty:
            d["target_used"] = tid
            parts.append(d)
        print(f"{tid}: {len(d)} rows", flush=True)
    raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if raw.empty:
        print("No activities found for this target. Check the ID.")
        sys.exit(1)
    raw.to_csv("tlr2_chembl_raw.csv", index=False)
    print(f"Downloaded {len(raw)} activity rows -> tlr2_chembl_raw.csv")

    # ---- label ----
    raw["standard_value"] = pd.to_numeric(raw["standard_value"], errors="coerce")
    raw["label"] = [label_row(r, v) for r, v in zip(raw["standard_relation"], raw["standard_value"])]
    raw = raw.dropna(subset=["label"]).copy()
    raw["label"] = raw["label"].astype(int)

    # ---- clean structures ----
    std = raw["canonical_smiles"].apply(standardise)
    raw["smiles_clean"] = [s[0] for s in std]
    raw["ikey"] = [s[1] for s in std]
    n_bad = raw["smiles_clean"].isna().sum()
    raw = raw.dropna(subset=["smiles_clean"])
    print(f"Removed {n_bad} rows with bad structures")

    # ---- one row per compound ----
    # A compound can have several measurements. If they disagree, we drop it
    # (we cannot trust it). Otherwise we keep one row.
    groups = raw.groupby("ikey")
    rows = []
    n_conflict = 0
    for key, g in groups:
        if g["label"].nunique() > 1:
            n_conflict += 1
            continue
        rows.append({
            "ikey": key,
            "smiles": g["smiles_clean"].iloc[0],
            "label": int(g["label"].iloc[0]),
            "n_measurements": len(g),
            "best_value_nM": g["standard_value"].min(),
            "molecule_chembl_id": g["molecule_chembl_id"].iloc[0],
        })
    train = pd.DataFrame(rows)
    print(f"Dropped {n_conflict} compounds with conflicting labels")

    # ---- remove the 8 known antagonists from training ----
    held_rows = []
    held_keys = set()
    for name, smi in HELDOUT.items():
        clean, key = standardise(smi)
        held_rows.append({"name": name, "smiles": clean, "ikey": key})
        held_keys.add(key)
    pd.DataFrame(held_rows).to_csv("tlr2_heldout_8.csv", index=False)
    before = len(train)
    train = train[~train["ikey"].isin(held_keys)]
    print(f"Removed {before - len(train)} held-out compounds that were in ChEMBL")

    train.to_csv("tlr2_train_clean.csv", index=False)

    # ---- summary ----
    print("\n=== SUMMARY ===")
    print(f"Compounds for training: {len(train)}")
    print(f"  active   (<= {CUTOFF_NM/1000:.0f} uM): {int((train['label'] == 1).sum())}")
    print(f"  inactive (>  {CUTOFF_NM/1000:.0f} uM): {int((train['label'] == 0).sum())}")
    print("Saved: tlr2_train_clean.csv, tlr2_heldout_8.csv")
    print("\nSend me these 3 numbers. If actives or inactives are below ~50,")
    print("we will need to add PubChem data or change the plan.")


if __name__ == "__main__":
    main()
