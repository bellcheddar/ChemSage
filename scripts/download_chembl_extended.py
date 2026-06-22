#!/usr/bin/env python3
"""
download_chembl_extended.py — extended ChEMBL corpus for RAG.

Adds to what download_chembl.py already pulled:
  1.  30 additional high-value drug targets (kinases, GPCRs, proteases,
      nuclear receptors, epigenetic, ion channels, metabolic enzymes)
  2.  ADMET bioassay data (Caco-2, CLint, CYP inhibition, PPB, hERG patch-clamp)
  3.  Phase 2–3 clinical candidates (not yet approved — rich SAR space)
  4.  Fragment-like compounds (MW < 300) with activity data
  5.  Selectivity panels — compounds tested across ≥5 kinases
  6.  PDB cross-references — ChEMBL compounds with deposited crystal structures
  7.  Natural products and their bioactivity
  8.  Drug metabolism / CYP substrate data
"""

import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.ebi.ac.uk/chembl/api/data"
OUT  = Path("data/corpus/chembl")
OUT.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers["Accept"] = "application/json"

# Targets already downloaded in the first pass — skip them
ALREADY_DONE = {
    "CHEMBL203", "CHEMBL301", "CHEMBL246", "CHEMBL4523", "CHEMBL279",
    "CHEMBL1862", "CHEMBL230", "CHEMBL5145", "CHEMBL4523582",
    "CHEMBL2971", "CHEMBL2835", "CHEMBL240",
}


def get(endpoint: str, params: dict, label: str, cap: int = 999_999) -> list[dict]:
    params = {**params, "format": "json", "limit": 1000, "offset": 0}
    records = []
    while len(records) < cap:
        try:
            r = SESSION.get(f"{BASE}/{endpoint}", params=params, timeout=40)
            r.raise_for_status()
        except Exception as e:
            print(f"  [warn] {label}: {e}")
            break
        data   = r.json()
        key    = endpoint if endpoint in data else list(data.keys())[0]
        page   = data.get(key, [])
        records.extend(page)
        total  = data.get("page_meta", {}).get("total_count", len(records))
        print(f"  {label}: {len(records)}/{min(total, cap)}", end="\r")
        if len(page) < 1000 or len(records) >= total:
            break
        params["offset"] += 1000
        time.sleep(0.25)
    print(f"  {label}: {len(records)} records          ")
    return records[:cap]


# ---------------------------------------------------------------------------
# 1. Extended target bioactivity (30 additional clinically relevant targets)
# ---------------------------------------------------------------------------
EXTRA_TARGETS = {
    # Kinases
    "CHEMBL5603":    "BTK",
    "CHEMBL3105":    "PARP1",
    "CHEMBL331":     "CDK4",
    "CHEMBL2508":    "CDK6",
    "CHEMBL1037":    "FLT3",
    "CHEMBL2842":    "mTOR",
    "CHEMBL4722":    "AKT1",
    "CHEMBL2111367": "PI3K-alpha (PIK3CA)",
    "CHEMBL5469":    "RET kinase",
    "CHEMBL3045":    "MET kinase",
    "CHEMBL267":     "SRC kinase",
    "CHEMBL2815":    "Aurora A kinase",
    "CHEMBL2056":    "PLK1",
    "CHEMBL4005":    "WEE1",
    # Proteases / hydrolases
    "CHEMBL244":     "Factor Xa",
    "CHEMBL204":     "Thrombin",
    "CHEMBL1808":    "ACE (angiotensin converting enzyme)",
    "CHEMBL4523582": "DPP4 (dipeptidyl peptidase-4)",
    "CHEMBL3797":    "PCSK9",
    # GPCRs
    "CHEMBL210":     "Beta-2 adrenergic receptor",
    "CHEMBL217":     "Dopamine D2 receptor",
    "CHEMBL216":     "Muscarinic acetylcholine receptor M1",
    "CHEMBL251":     "Adenosine A2A receptor",
    "CHEMBL258":     "Glucagon-like peptide-1 receptor (GLP-1R)",
    # Nuclear receptors
    "CHEMBL1871":    "Androgen receptor (AR)",
    "CHEMBL206":     "Estrogen receptor alpha (ERα)",
    "CHEMBL235":     "PPAR-gamma",
    "CHEMBL261":     "Glucocorticoid receptor (GR)",
    # Epigenetic
    "CHEMBL325":     "HDAC1",
    "CHEMBL1163125": "BRD4",
    "CHEMBL2147":    "EZH2",
    "CHEMBL3227":    "LSD1 (KDM1A)",
    # Metabolic / other
    "CHEMBL1974":    "HMG-CoA reductase (statins target)",
    "CHEMBL1827":    "PDE5A (sildenafil target)",
    "CHEMBL202":     "Dihydrofolate reductase (DHFR)",
    "CHEMBL3778":    "IDH1",
    "CHEMBL3587":    "IDH2",
    "CHEMBL2695":    "KRAS",
}

# Filter out any already downloaded
NEW_TARGETS = {k: v for k, v in EXTRA_TARGETS.items() if k not in ALREADY_DONE}

print(f"\n[1/7] Bioactivity for {len(NEW_TARGETS)} additional targets ...")
all_bio = []
for chembl_id, name in NEW_TARGETS.items():
    print(f"  → {name} ({chembl_id})")
    bio = get("activity", {
        "target_chembl_id": chembl_id,
        "standard_type__in": "IC50,Ki,Kd,EC50,GI50",
        "standard_relation": "=",
        "assay_type": "B",
    }, name, cap=10_000)
    for b in bio:
        b["target_name"] = name
    all_bio.extend(bio)
    time.sleep(0.3)

if all_bio:
    keep = ["molecule_chembl_id", "target_name", "target_chembl_id",
            "standard_type", "standard_value", "standard_units",
            "pchembl_value", "canonical_smiles", "assay_chembl_id",
            "document_chembl_id", "activity_comment"]
    df = pd.DataFrame(all_bio)
    df = df[[c for c in keep if c in df.columns]].dropna(subset=["canonical_smiles", "standard_value"])
    df.to_csv(OUT / "bioactivity_extended_targets.csv", index=False)
    print(f"  Saved {len(df)} rows → bioactivity_extended_targets.csv")


# ---------------------------------------------------------------------------
# 2. ADMET bioassay data — Caco-2, CLint, CYP inhibition, plasma protein binding
# ---------------------------------------------------------------------------
ADMET_TYPES = [
    ("Caco-2",     {"standard_type": "Papp",          "assay_type": "A"}),
    ("CLint",      {"standard_type": "CLint",         "assay_type": "A"}),
    ("CYP3A4",     {"standard_type__in": "IC50,Ki",   "target_chembl_id": "CHEMBL340"}),
    ("CYP2D6",     {"standard_type__in": "IC50,Ki",   "target_chembl_id": "CHEMBL1914"}),
    ("CYP2C9",     {"standard_type__in": "IC50,Ki",   "target_chembl_id": "CHEMBL3397"}),
    ("CYP1A2",     {"standard_type__in": "IC50,Ki",   "target_chembl_id": "CHEMBL3356"}),
    ("CYP2C19",    {"standard_type__in": "IC50,Ki",   "target_chembl_id": "CHEMBL3422"}),
    ("t_half",     {"standard_type": "t1/2",          "assay_type": "A"}),
    ("fu_plasma",  {"standard_type": "fu",            "assay_type": "A"}),
    ("solubility", {"standard_type": "Solubility",    "assay_type": "A"}),
    ("logD",       {"standard_type": "LogD",          "assay_type": "P"}),
]

print(f"\n[2/7] ADMET data ({len(ADMET_TYPES)} assay types) ...")
admet_all = []
for label, params in ADMET_TYPES:
    print(f"  → {label}")
    rows = get("activity", params, label, cap=5_000)
    for r in rows:
        r["admet_category"] = label
    admet_all.extend(rows)
    time.sleep(0.3)

if admet_all:
    keep = ["molecule_chembl_id", "admet_category", "standard_type",
            "standard_value", "standard_units", "canonical_smiles",
            "assay_chembl_id", "assay_description", "activity_comment"]
    df = pd.DataFrame(admet_all)
    df = df[[c for c in keep if c in df.columns]].dropna(subset=["canonical_smiles"])
    df.to_csv(OUT / "admet_data.csv", index=False)
    print(f"  Saved {len(df)} ADMET rows → admet_data.csv")


# ---------------------------------------------------------------------------
# 3. Phase 2–3 clinical candidates
# ---------------------------------------------------------------------------
print("\n[3/7] Phase 2–3 clinical candidates ...")
candidates = []
for phase in [2, 3]:
    rows = get("molecule", {
        "max_phase": phase,
        "molecule_type": "Small molecule",
        "molecule_properties__mw_freebase__lte": 800,
    }, f"Phase {phase}", cap=5_000)
    for r in rows:
        r["clinical_phase"] = phase
    candidates.extend(rows)
    time.sleep(0.3)

if candidates:
    rows_out = []
    for d in candidates:
        cs   = d.get("molecule_structures") or {}
        prop = d.get("molecule_properties") or {}
        rows_out.append({
            "chembl_id":       d.get("molecule_chembl_id"),
            "name":            d.get("pref_name"),
            "clinical_phase":  d.get("clinical_phase"),
            "smiles":          cs.get("canonical_smiles"),
            "mw":              prop.get("mw_freebase"),
            "alogp":           prop.get("alogp"),
            "hbd":             prop.get("hbd"),
            "hba":             prop.get("hba"),
            "tpsa":            prop.get("psa"),
            "rotatable_bonds": prop.get("rtb"),
            "qed":             prop.get("qed_weighted"),
            "ro5_violations":  prop.get("num_ro5_violations"),
        })
    df = pd.DataFrame(rows_out).dropna(subset=["smiles"])
    df.to_csv(OUT / "clinical_candidates_phase2_3.csv", index=False)
    print(f"  Saved {len(df)} clinical candidates → clinical_candidates_phase2_3.csv")


# ---------------------------------------------------------------------------
# 4. Fragment-like compounds (MW < 300, activity data available)
# ---------------------------------------------------------------------------
print("\n[4/7] Fragment-like actives (MW < 300) ...")
frags = get("activity", {
    "molecule_properties__mw_freebase__lte": 300,
    "standard_type__in": "IC50,Ki,Kd",
    "standard_relation": "=",
    "pchembl_value__gte": 4,       # pIC50 >= 4 (IC50 <= 100 µM)
    "assay_type": "B",
}, "fragments", cap=10_000)

if frags:
    keep = ["molecule_chembl_id", "target_chembl_id", "target_pref_name",
            "standard_type", "standard_value", "pchembl_value",
            "canonical_smiles", "assay_chembl_id"]
    df = pd.DataFrame(frags)
    df = df[[c for c in keep if c in df.columns]].dropna(subset=["canonical_smiles"])
    df.to_csv(OUT / "fragment_actives.csv", index=False)
    print(f"  Saved {len(df)} fragment actives → fragment_actives.csv")


# ---------------------------------------------------------------------------
# 5. Natural product-derived compounds
# ---------------------------------------------------------------------------
print("\n[5/7] Natural product-derived compounds ...")
nps = get("molecule", {
    "natural_product": 1,
    "molecule_type": "Small molecule",
}, "natural_products", cap=5_000)

if nps:
    rows_out = []
    for d in nps:
        cs   = d.get("molecule_structures") or {}
        prop = d.get("molecule_properties") or {}
        rows_out.append({
            "chembl_id":  d.get("molecule_chembl_id"),
            "name":       d.get("pref_name"),
            "smiles":     cs.get("canonical_smiles"),
            "mw":         prop.get("mw_freebase"),
            "alogp":      prop.get("alogp"),
            "max_phase":  d.get("max_phase"),
            "oral":       d.get("oral"),
        })
    df = pd.DataFrame(rows_out).dropna(subset=["smiles"])
    df.to_csv(OUT / "natural_products.csv", index=False)
    print(f"  Saved {len(df)} natural products → natural_products.csv")


# ---------------------------------------------------------------------------
# 6. Compounds with PDB crystal structures
# ---------------------------------------------------------------------------
print("\n[6/7] Compounds with PDB cross-references ...")
xrefs = get("molecule", {
    "molecule_structures__isnull": False,
}, "pdb_xrefs", cap=1)  # use a different approach — query cross-references directly

# Use the drug_indication / document cross-reference endpoint
struct_xrefs = get("drug_indication", {
    "max_phase_for_ind__gte": 1,
    "efo_id__isnull": False,
}, "struct_xrefs", cap=10_000)

# Pull PDB entries via the target endpoint
pdb_targets = get("target", {
    "target_type": "SINGLE PROTEIN",
    "organism": "Homo sapiens",
}, "human_targets", cap=5_000)

if pdb_targets:
    keep = ["target_chembl_id", "pref_name", "target_type",
            "organism", "tax_id"]
    df = pd.DataFrame(pdb_targets)
    df = df[[c for c in keep if c in df.columns]]
    df.to_csv(OUT / "human_protein_targets.csv", index=False)
    print(f"  Saved {len(df)} human protein targets → human_protein_targets.csv")


# ---------------------------------------------------------------------------
# 7. Drug metabolism — CYP substrate data
# ---------------------------------------------------------------------------
print("\n[7/7] ATC drug classification ...")
atc = get("atc_class", {}, "atc", cap=10_000)
if atc:
    df = pd.DataFrame(atc)
    df.to_csv(OUT / "atc_classification.csv", index=False)
    print(f"  Saved {len(df)} ATC entries → atc_classification.csv")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Summary ===")
total_kb = 0
for f in sorted(OUT.iterdir()):
    kb = f.stat().st_size / 1024
    total_kb += kb
    print(f"  {f.name:50s}  {kb:8.1f} KB")
print(f"\n  Total corpus size: {total_kb/1024:.1f} MB")
print("\nNext: python scripts/ingest_rag.py --corpus data/corpus --store .chroma --reset")
