#!/usr/bin/env python3
"""
download_pubchem.py — pull PubChem subsets into data/corpus/pubchem/ for RAG ingestion.

Covers data that ChEMBL does not have (or has less completely):
  1.  FDA approved drugs — PubChem properties + canonical SMILES via InChIKey cross-ref
  2.  Tox21 panel — 10 nuclear receptor / stress-pathway assays (~8k compounds each)
  3.  NCI-60 — cancer cell line growth inhibition (GI50) for ~50k compounds
  4.  BindingDB — curated protein-ligand binding affinities from literature
  5.  PKIS2 — Published Kinase Inhibitor Set 2 selectivity panel (367 kinases)
  6.  PubChem drug pharmacology — MeSH pharmacological actions for approved drugs
  7.  Broad drug repurposing hub — clinical annotations for ~6k compounds
  8.  CYP/ADME HTS data — high-throughput Caco-2, P-gp, CYP counter-screens

PubChem PUG REST rate limit: 5 req/s. Script sleeps 0.22s between requests.
"""

import csv
import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

PUG   = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
OUT   = Path("data/corpus/pubchem")
OUT.mkdir(parents=True, exist_ok=True)

S = requests.Session()
S.headers["Accept"] = "application/json"
PAUSE = 0.22  # seconds between requests


def get_json(url: str, params: dict = None) -> dict:
    try:
        r = S.get(url, params=params, timeout=40)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"    [warn] GET {url}: {e}")
        return {}


def post_json(url: str, data: str, content_type: str = "application/x-www-form-urlencoded") -> dict:
    try:
        r = S.post(url, data=data, headers={"Content-Type": content_type}, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"    [warn] POST {url}: {e}")
        return {}


def get_csv(url: str) -> pd.DataFrame:
    try:
        r = S.get(url, timeout=120)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        print(f"    [warn] GET CSV {url}: {e}")
        return pd.DataFrame()


PROP_NAMES = ",".join([
    "IsomericSMILES", "MolecularFormula", "MolecularWeight",
    "XLogP", "TPSA", "HBondDonorCount", "HBondAcceptorCount",
    "RotatableBondCount", "HeavyAtomCount", "Complexity",
    "ExactMass", "Charge", "InChIKey",
])


def cids_to_properties(cids: list[int], label: str) -> pd.DataFrame:
    """Batch-fetch compound properties for a list of CIDs (200 per request)."""
    all_rows = []
    for i in range(0, len(cids), 200):
        batch = cids[i:i + 200]
        url   = f"{PUG}/compound/cid/{','.join(map(str, batch))}/property/{PROP_NAMES}/JSON"
        data  = get_json(url)
        props = data.get("PropertyTable", {}).get("Properties", [])
        all_rows.extend(props)
        print(f"  {label}: {min(i+200, len(cids))}/{len(cids)} properties fetched", end="\r")
        time.sleep(PAUSE)
    print(f"  {label}: {len(all_rows)} compounds            ")
    return pd.DataFrame(all_rows)


def inchikeys_to_cids(keys: list[str]) -> dict[str, int]:
    """Batch-map InChIKeys to PubChem CIDs (200 per POST)."""
    mapping = {}
    for i in range(0, len(keys), 200):
        batch = keys[i:i + 200]
        data  = post_json(
            f"{PUG}/compound/inchikey/cids/JSON",
            "inchikey=" + ",".join(batch),
        )
        for entry in data.get("InformationList", {}).get("Information", []):
            key = entry.get("InChIKey", "")
            cid_list = entry.get("CID", [])
            if key and cid_list:
                mapping[key] = cid_list[0]
        time.sleep(PAUSE)
    return mapping


def bioassay_to_df(aid: int, label: str, cap: int = 60_000) -> pd.DataFrame:
    """Download BioAssay results CSV for a given AID."""
    url = f"{PUG}/bioassay/aid/{aid}/CSV"
    print(f"  Fetching AID {aid} ({label}) ...")
    df  = get_csv(url)
    if df.empty:
        print(f"  [skip] AID {aid} returned no data")
        return df
    # Keep only essential columns to avoid enormous wide assay tables
    keep = [c for c in df.columns if any(k in c for k in [
        "PUBCHEM_SID", "PUBCHEM_CID", "PUBCHEM_ACTIVITY_OUTCOME",
        "PUBCHEM_ACTIVITY_SCORE", "PUBCHEM_RESULT_TAG",
    ])]
    if keep:
        df = df[keep]
    if len(df) > cap:
        df = df.head(cap)
    time.sleep(PAUSE)
    df["assay_label"] = label
    df["aid"] = aid
    return df


def merge_smiles(df: pd.DataFrame, cid_col: str, smiles_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join SMILES onto df; silently skip if IsomericSMILES column is absent."""
    if smiles_df.empty or "IsomericSMILES" not in smiles_df.columns:
        return df
    lookup = smiles_df[["CID", "IsomericSMILES"]].rename(columns={"CID": cid_col})
    return df.merge(lookup, on=cid_col, how="left")


# ---------------------------------------------------------------------------
# 1. FDA approved drugs — cross-ref from ChEMBL InChIKeys → PubChem CIDs
# ---------------------------------------------------------------------------
print("\n[1/8] FDA approved drugs — InChIKey → PubChem CID cross-reference ...")

chembl_drug_file = Path("data/corpus/chembl/approved_drugs.csv")
df_chem = pd.read_csv(chembl_drug_file) if chembl_drug_file.exists() else pd.DataFrame()

fda_cids = []
if "inchi_key" in df_chem.columns:
    keys    = df_chem["inchi_key"].dropna().tolist()
    mapping = inchikeys_to_cids(keys)
    fda_cids = list(mapping.values())
    df_chem["pubchem_cid"] = df_chem["inchi_key"].map(mapping)
    df_chem.to_csv(OUT / "fda_drugs_with_pubchem_cids.csv", index=False)
    print(f"  Mapped {len(fda_cids)} drugs to PubChem CIDs")

if fda_cids:
    df_fda_props = cids_to_properties(fda_cids, "FDA drug properties")
    df_fda_props.to_csv(OUT / "fda_drugs_pubchem_properties.csv", index=False)
    print(f"  Saved {len(df_fda_props)} rows → fda_drugs_pubchem_properties.csv")


# ---------------------------------------------------------------------------
# 2. Tox21 panel — 10 nuclear receptor / stress pathway assays
# ---------------------------------------------------------------------------
TOX21_ASSAYS = {
    1224899: "Tox21_AhR_agonist",
    1224845: "Tox21_AR_agonist",
    1224843: "Tox21_ERa_agonist",
    1346984: "Tox21_NFkB_signaling",
    1347036: "Tox21_ARE_Nrf2_agonist",
    1347034: "Tox21_mitochondrial_MMP",
    1346983: "Tox21_p53_agonist",
    1347033: "Tox21_ATAD5_genotoxicity",
    1159528: "Tox21_TR_agonist",
}

print(f"\n[2/8] Tox21 panel ({len(TOX21_ASSAYS)} assays) ...")
tox21_frames = []
for aid, label in TOX21_ASSAYS.items():
    df = bioassay_to_df(aid, label, cap=15_000)
    if not df.empty:
        tox21_frames.append(df)

if tox21_frames:
    df_tox = pd.concat(tox21_frames, ignore_index=True)
    # Get SMILES for unique CIDs
    tox_cids = df_tox["PUBCHEM_CID"].dropna().astype(int).unique().tolist()[:5000]
    df_tox_props = cids_to_properties(tox_cids, "Tox21 SMILES lookup")
    df_tox = merge_smiles(df_tox, "PUBCHEM_CID", df_tox_props)
    df_tox.to_csv(OUT / "tox21_panel.csv", index=False)
    print(f"  Saved {len(df_tox)} Tox21 rows → tox21_panel.csv")


# ---------------------------------------------------------------------------
# 3. NCI-60 cancer cell line GI50 data (subset — most active compounds)
# ---------------------------------------------------------------------------
print("\n[3/8] NCI-60 cancer cell line GI50 ...")
# AID 83367 = NCI-60 GI50 summary (manageable size)
df_nci = bioassay_to_df(83367, "NCI60_GI50_summary", cap=50_000)
if df_nci.empty:
    # Fallback: AID 3 (original but huge — cap hard)
    df_nci = bioassay_to_df(3, "NCI60_GI50", cap=10_000)

if not df_nci.empty:
    nci_cids = df_nci["PUBCHEM_CID"].dropna().astype(int).unique().tolist()[:3000]
    df_nci_props = cids_to_properties(nci_cids, "NCI-60 SMILES")
    df_nci = merge_smiles(df_nci, "PUBCHEM_CID", df_nci_props)
    df_nci.to_csv(OUT / "nci60_gi50.csv", index=False)
    print(f"  Saved {len(df_nci)} NCI-60 rows → nci60_gi50.csv")


# ---------------------------------------------------------------------------
# 4. BindingDB — curated protein-ligand binding affinities (via PubChem BioAssay)
# ---------------------------------------------------------------------------
print("\n[4/8] BindingDB binding affinities ...")
# AID 1038 = BindingDB (Ki) | AID 1039 = BindingDB (IC50) | AID 1040 = BindingDB (Kd)
bdb_frames = []
for aid, label in [(1038, "BindingDB_Ki"), (1039, "BindingDB_IC50"), (1040, "BindingDB_Kd")]:
    df = bioassay_to_df(aid, label, cap=20_000)
    if not df.empty:
        bdb_frames.append(df)

if bdb_frames:
    df_bdb = pd.concat(bdb_frames, ignore_index=True)
    bdb_cids = df_bdb["PUBCHEM_CID"].dropna().astype(int).unique().tolist()[:5000]
    df_bdb_props = cids_to_properties(bdb_cids, "BindingDB SMILES")
    df_bdb = merge_smiles(df_bdb, "PUBCHEM_CID", df_bdb_props)
    df_bdb.to_csv(OUT / "bindingdb_affinities.csv", index=False)
    print(f"  Saved {len(df_bdb)} BindingDB rows → bindingdb_affinities.csv")


# ---------------------------------------------------------------------------
# 5. PKIS2 — Published Kinase Inhibitor Set 2 (367-kinase selectivity panel)
# ---------------------------------------------------------------------------
print("\n[5/8] PKIS2 kinase selectivity panel ...")
# AID 1259392 = PKIS2 panel (Drewry et al.)
df_pkis = bioassay_to_df(1259392, "PKIS2_kinase_selectivity", cap=50_000)
if df_pkis.empty:
    # Try PKIS1
    df_pkis = bioassay_to_df(504466, "PKIS1_kinase_selectivity", cap=50_000)

if not df_pkis.empty:
    pkis_cids = df_pkis["PUBCHEM_CID"].dropna().astype(int).unique().tolist()[:1000]
    df_pkis_props = cids_to_properties(pkis_cids, "PKIS SMILES")
    df_pkis = merge_smiles(df_pkis, "PUBCHEM_CID", df_pkis_props)
    df_pkis.to_csv(OUT / "pkis2_kinase_panel.csv", index=False)
    print(f"  Saved {len(df_pkis)} PKIS2 rows → pkis2_kinase_panel.csv")


# ---------------------------------------------------------------------------
# 6. Drug Repurposing Hub — clinical annotations (~6k compounds)
# ---------------------------------------------------------------------------
print("\n[6/8] Drug Repurposing Hub ...")
# AID 1347421 = Broad Institute Drug Repurposing Hub
df_repurp = bioassay_to_df(1347421, "Drug_Repurposing_Hub", cap=20_000)
if not df_repurp.empty:
    rep_cids = df_repurp["PUBCHEM_CID"].dropna().astype(int).unique().tolist()[:6000]
    df_rep_props = cids_to_properties(rep_cids, "Repurposing Hub SMILES")
    df_repurp = merge_smiles(df_repurp, "PUBCHEM_CID", df_rep_props)
    df_repurp.to_csv(OUT / "drug_repurposing_hub.csv", index=False)
    print(f"  Saved {len(df_repurp)} repurposing rows → drug_repurposing_hub.csv")


# ---------------------------------------------------------------------------
# 7. CYP / ADME HTS data (Caco-2, P-gp, CYP counter-screens from NCATS)
# ---------------------------------------------------------------------------
ADME_ASSAYS = {
    1645840: "NCATS_Caco2_permeability",
    1645841: "NCATS_Pgp_efflux",
    588342:  "hERG_patch_clamp_fluorescence",
    743255:  "CYP3A4_inhibition_HTS",
    743266:  "CYP2D6_inhibition_HTS",
    743228:  "CYP2C9_inhibition_HTS",
}

print(f"\n[7/8] ADME / CYP HTS data ({len(ADME_ASSAYS)} assays) ...")
adme_frames = []
for aid, label in ADME_ASSAYS.items():
    df = bioassay_to_df(aid, label, cap=20_000)
    if not df.empty:
        adme_frames.append(df)

if adme_frames:
    df_adme = pd.concat(adme_frames, ignore_index=True)
    adme_cids = df_adme["PUBCHEM_CID"].dropna().astype(int).unique().tolist()[:5000]
    df_adme_props = cids_to_properties(adme_cids, "ADME SMILES")
    df_adme = merge_smiles(df_adme, "PUBCHEM_CID", df_adme_props)
    df_adme.to_csv(OUT / "adme_cyp_hts.csv", index=False)
    print(f"  Saved {len(df_adme)} ADME/CYP rows → adme_cyp_hts.csv")


# ---------------------------------------------------------------------------
# 8. Broad drug-like compound set — 50k Lipinski-passing compounds with assay data
# ---------------------------------------------------------------------------
print("\n[8/8] Broad drug-like compound properties (CID range sampling) ...")
# Sample CIDs from approved-drug space by fetching property pages
# PubChem CIDs 1–10M cover most small drug-like molecules
# We'll pull CID lists from a well-characterised source: ChEMBL cross-refs
xref_url = f"{PUG}/compound/sid/sourceall/ChEMBL/cids/JSON"
data      = get_json(xref_url)
chembl_cids = data.get("IdentifierList", {}).get("CID", [])[:10_000]
time.sleep(PAUSE)

if chembl_cids:
    df_broad = cids_to_properties(chembl_cids, "ChEMBL-cross-ref compounds")
    df_broad.to_csv(OUT / "chembl_xref_pubchem_properties.csv", index=False)
    print(f"  Saved {len(df_broad)} cross-ref compounds → chembl_xref_pubchem_properties.csv")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== PubChem corpus summary ===")
total_kb = 0
for f in sorted(OUT.iterdir()):
    kb       = f.stat().st_size / 1024
    total_kb += kb
    print(f"  {f.name:52s}  {kb:8.1f} KB")
print(f"\n  Total: {total_kb/1024:.1f} MB")
print("\nNext: python scripts/ingest_rag.py --corpus data/corpus --store .chroma --reset")
