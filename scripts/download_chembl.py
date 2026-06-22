#!/usr/bin/env python3
"""
download_chembl.py — pull ChEMBL subsets into data/corpus/ for RAG ingestion.

Downloads via the ChEMBL REST API (no account needed):
  1. All approved small-molecule drugs (max_phase=4) with properties
  2. Bioactivity data (IC50/Ki/Kd) for 12 high-value drug targets
  3. Drug indication / mechanism-of-action table
  4. Approved drugs with ATC codes (therapeutic class)

Saves as CSVs that ingest_rag.py handles with SAR-table-aware chunking.
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.ebi.ac.uk/chembl/api/data"
OUT  = Path("data/corpus/chembl")
OUT.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers["Accept"] = "application/json"


def get(endpoint: str, params: dict, label: str) -> list[dict]:
    """Page through a ChEMBL endpoint and return all records."""
    params = {**params, "format": "json", "limit": 1000, "offset": 0}
    records = []
    while True:
        try:
            r = SESSION.get(f"{BASE}/{endpoint}", params=params, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"  [warn] {label}: {e}")
            break
        data = r.json()
        page = data.get(endpoint, data.get(list(data.keys())[0], []))
        records.extend(page)
        total = data.get("page_meta", {}).get("total_count", len(records))
        print(f"  {label}: {len(records)}/{total}", end="\r")
        if len(page) < 1000 or len(records) >= total:
            break
        params["offset"] += 1000
        time.sleep(0.2)
    print(f"  {label}: {len(records)} records          ")
    return records


# ---------------------------------------------------------------------------
# 1. Approved small-molecule drugs
# ---------------------------------------------------------------------------
print("\n[1/4] Approved small-molecule drugs ...")
drugs = get("molecule", {
    "max_phase": 4,
    "molecule_type": "Small molecule",
    "molecule_properties__mw_freebase__lte": 900,
}, "drugs")

rows = []
for d in drugs:
    cs   = d.get("molecule_structures") or {}
    prop = d.get("molecule_properties") or {}
    rows.append({
        "chembl_id":      d.get("molecule_chembl_id"),
        "name":           d.get("pref_name"),
        "smiles":         cs.get("canonical_smiles"),
        "inchi_key":      cs.get("standard_inchi_key"),
        "mw":             prop.get("mw_freebase"),
        "alogp":          prop.get("alogp"),
        "hbd":            prop.get("hbd"),
        "hba":            prop.get("hba"),
        "tpsa":           prop.get("psa"),
        "rotatable_bonds":prop.get("rtb"),
        "qed":            prop.get("qed_weighted"),
        "max_phase":      d.get("max_phase"),
        "oral":           d.get("oral"),
        "black_box_warning": d.get("black_box_warning"),
    })

df_drugs = pd.DataFrame(rows).dropna(subset=["smiles"])
df_drugs.to_csv(OUT / "approved_drugs.csv", index=False)
print(f"  Saved {len(df_drugs)} approved drugs → {OUT}/approved_drugs.csv")


# ---------------------------------------------------------------------------
# 2. Bioactivity data for key targets
# ---------------------------------------------------------------------------
TARGETS = {
    "CHEMBL203":  "EGFR kinase",
    "CHEMBL301":  "CDK2",
    "CHEMBL246":  "HIV-1 protease",
    "CHEMBL4523":  "SARS-CoV-2 main protease (3CLpro)",
    "CHEMBL279":  "VEGFR2 (KDR)",
    "CHEMBL1862": "Bcr-Abl kinase",
    "CHEMBL230":  "COX-2 (PTGS2)",
    "CHEMBL5145": "BRAF",
    "CHEMBL4523582": "ALK kinase",
    "CHEMBL2971":  "JAK1",
    "CHEMBL2835":  "JAK2",
    "CHEMBL240":  "hERG (cardiac safety)",
}

print(f"\n[2/4] Bioactivity data for {len(TARGETS)} targets ...")
all_bio = []
for chembl_id, name in TARGETS.items():
    print(f"  → {name} ({chembl_id})")
    bio = get("activity", {
        "target_chembl_id": chembl_id,
        "standard_type__in": "IC50,Ki,Kd,EC50",
        "standard_relation": "=",
        "assay_type": "B",
    }, name)
    for b in bio:
        b["target_name"] = name
    all_bio.extend(bio)
    time.sleep(0.3)

if all_bio:
    keep = ["molecule_chembl_id", "target_name", "target_chembl_id",
            "standard_type", "standard_value", "standard_units",
            "pchembl_value", "canonical_smiles", "assay_chembl_id",
            "document_chembl_id", "activity_comment"]
    df_bio = pd.DataFrame(all_bio)
    df_bio = df_bio[[c for c in keep if c in df_bio.columns]]
    df_bio = df_bio.dropna(subset=["canonical_smiles", "standard_value"])
    df_bio.to_csv(OUT / "bioactivity_key_targets.csv", index=False)
    print(f"  Saved {len(df_bio)} bioactivity rows → {OUT}/bioactivity_key_targets.csv")


# ---------------------------------------------------------------------------
# 3. Drug mechanisms of action
# ---------------------------------------------------------------------------
print("\n[3/4] Drug mechanisms of action ...")
mechs = get("mechanism", {}, "mechanisms")
if mechs:
    df_mech = pd.DataFrame(mechs)
    cols = ["molecule_chembl_id", "mechanism_of_action", "target_chembl_id",
            "action_type", "direct_interaction", "disease_efficacy"]
    df_mech = df_mech[[c for c in cols if c in df_mech.columns]]
    df_mech.to_csv(OUT / "drug_mechanisms.csv", index=False)
    print(f"  Saved {len(df_mech)} mechanism rows → {OUT}/drug_mechanisms.csv")


# ---------------------------------------------------------------------------
# 4. Drug indications
# ---------------------------------------------------------------------------
print("\n[4/4] Drug indications ...")
inds = get("drug_indication", {}, "indications")
if inds:
    df_ind = pd.DataFrame(inds)
    cols = ["molecule_chembl_id", "drugind_id", "indication_refs",
            "efo_id", "efo_term", "max_phase_for_ind", "mesh_id", "mesh_heading"]
    df_ind = df_ind[[c for c in cols if c in df_ind.columns]]
    df_ind.to_csv(OUT / "drug_indications.csv", index=False)
    print(f"  Saved {len(df_ind)} indication rows → {OUT}/drug_indications.csv")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\nDone. Files written to", OUT)
for f in sorted(OUT.iterdir()):
    size = f.stat().st_size / 1024
    print(f"  {f.name:45s}  {size:8.1f} KB")

print("\nNext: python scripts/ingest_rag.py --corpus data/corpus --store .chroma")
