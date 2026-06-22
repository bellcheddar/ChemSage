#!/usr/bin/env python3
"""
download_pdb_supplement.py — fill gaps left by download_pdb.py.

Fixes:
  A. CCD SMILES — reads comp_ids from the already-downloaded ligand-structure
     pairs table and batch-fetches SMILES via RCSB GraphQL. Only fetches
     compounds that actually appear in PDB structures (better than the full
     40k CCD anyway).
  B. Drug-like pairs — rebuild pdb_druglike_structure_pairs.csv now that
     SMILES are available.
  C. SIFTS CATH — probe for the correct URL and download.
"""

import gzip
import io
import time
from pathlib import Path

import pandas as pd
import requests

OUT   = Path("data/corpus/pdb")
RCSB_GQL  = "https://data.rcsb.org/graphql"
SIFTS_BASE = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv"

S = requests.Session()
S.headers["User-Agent"] = "ChemSage/1.0 (marc@marcdeller.com)"
PAUSE = 0.15

CHEMCOMP_GQL = """
query GetChemComps($comp_ids: [String!]!) {
  chem_comps(comp_ids: $comp_ids) {
    chem_comp {
      id
      name
      formula
      formula_weight
      type
      pdbx_release_status
    }
    rcsb_chem_comp_descriptor {
      SMILES
      SMILES_stereo
      InChI
      InChIKey
    }
    rcsb_chem_comp_related {
      resource_name
      resource_accession_code
    }
  }
}
"""


def graphql(query: str, variables: dict = None) -> dict:
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    try:
        r = S.post(RCSB_GQL, json=payload,
                   headers={"Content-Type": "application/json"}, timeout=60)
        r.raise_for_status()
        return r.json().get("data", {})
    except Exception as e:
        print(f"    [warn] GraphQL: {e}")
        return {}


def download_gz_csv(url: str, label: str) -> pd.DataFrame:
    print(f"  Trying {url} ...")
    try:
        r = S.get(url, timeout=600, stream=True)
        r.raise_for_status()
        buf = io.BytesIO(r.content)
        with gzip.open(buf) as gz:
            df = pd.read_csv(gz, comment="#", low_memory=False)
        print(f"  {label}: {len(df):,} rows")
        return df
    except Exception as e:
        print(f"    [warn] {label}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# A. CCD SMILES — batch-fetch from RCSB GraphQL using comp_ids from pairs
# ---------------------------------------------------------------------------
print("\n[A] CCD SMILES via RCSB GraphQL ...")

pairs_file = OUT / "pdb_ligand_structure_pairs.csv"
if not pairs_file.exists():
    print("  [error] pdb_ligand_structure_pairs.csv not found — run download_pdb.py first")
else:
    df_pairs = pd.read_csv(pairs_file, low_memory=False)
    comp_ids = sorted(df_pairs["comp_id"].dropna().unique().tolist())
    print(f"  {len(comp_ids):,} unique comp_ids in the pairs table")

    BATCH = 100
    cc_rows = []

    for i in range(0, len(comp_ids), BATCH):
        batch = comp_ids[i:i + BATCH]
        data  = graphql(CHEMCOMP_GQL, {"comp_ids": batch})
        comps = data.get("chem_comps") or []
        for comp in comps:
            if not comp:
                continue
            cc   = comp.get("chem_comp") or {}
            descs = comp.get("rcsb_chem_comp_descriptor") or []
            rels  = comp.get("rcsb_chem_comp_related") or []

            # descs is a single dict (not a list) in the updated RCSB schema
            desc_obj = descs if isinstance(descs, dict) else {}
            smiles   = desc_obj.get("SMILES_stereo") or desc_obj.get("SMILES") or ""
            inchi    = desc_obj.get("InChI") or ""
            inchikey = desc_obj.get("InChIKey") or ""

            xrefs = {r.get("resource_name", ""): r.get("resource_accession_code", "")
                     for r in rels if r}

            cc_rows.append({
                "comp_id":        cc.get("id", ""),
                "name":           cc.get("name", ""),
                "formula":        cc.get("formula", ""),
                "formula_weight": cc.get("formula_weight"),
                "type":           cc.get("type", ""),
                "release_status": cc.get("pdbx_release_status", ""),
                "smiles":         smiles,
                "inchi":          inchi,
                "inchikey":       inchikey,
                "chembl_id":      xrefs.get("ChEMBL", ""),
                "drugbank_id":    xrefs.get("DrugBank", ""),
                "pubchem_cid":    xrefs.get("PubChem CID", ""),
            })

        done = min(i + BATCH, len(comp_ids))
        if done % 3000 < BATCH:
            print(f"  CCD SMILES: {done:,}/{len(comp_ids):,} fetched", end="\r")
        time.sleep(PAUSE)

    print(f"  CCD SMILES: {len(cc_rows):,} records fetched          ")
    df_cc = pd.DataFrame(cc_rows)
    df_cc.to_csv(OUT / "pdb_ligands_all.csv", index=False)
    print(f"  Saved {len(df_cc):,} rows → pdb_ligands_all.csv")

    # Drug-like subset
    if not df_cc.empty and "smiles" in df_cc.columns:
        df_drug = df_cc[
            df_cc["smiles"].notna() & (df_cc["smiles"] != "") &
            df_cc["formula_weight"].notna() &
            (df_cc["formula_weight"].astype(float) >= 100) &
            (df_cc["formula_weight"].astype(float) <= 800) &
            ~df_cc["type"].str.lower().str.contains("polymer|rna|dna|saccharide", na=True)
        ].copy()
        df_drug.to_csv(OUT / "pdb_ligands_druglike.csv", index=False)
        print(f"  {len(df_drug):,} drug-like ligands → pdb_ligands_druglike.csv")
        druglike_ids = set(df_drug["comp_id"].dropna().tolist())
        smiles_map   = df_cc.set_index("comp_id")["smiles"].to_dict()

        # Rebuild drug-like pairs
        df_pairs["smiles"] = df_pairs["comp_id"].map(smiles_map)
        df_pairs.to_csv(OUT / "pdb_ligand_structure_pairs.csv", index=False)

        df_drug_pairs = df_pairs[df_pairs["comp_id"].isin(druglike_ids)]
        df_drug_pairs.to_csv(OUT / "pdb_druglike_structure_pairs.csv", index=False)
        print(f"  {len(df_drug_pairs):,} drug-like pairs → pdb_druglike_structure_pairs.csv")


# ---------------------------------------------------------------------------
# B. SIFTS CATH — try several candidate URLs
# ---------------------------------------------------------------------------
print("\n[B] SIFTS CATH classification ...")

CATH_CANDIDATES = [
    f"{SIFTS_BASE}/pdb_chain_cath.csv.gz",
    f"{SIFTS_BASE}/pdb_chain_cath_scop.csv.gz",
    f"{SIFTS_BASE}/pdb_chain_cath_b.csv.gz",
    f"{SIFTS_BASE}/pdb_chain_scop2b_sf.csv.gz",
    f"{SIFTS_BASE}/pdb_chain_scop2.csv.gz",
    f"{SIFTS_BASE}/pdb_chain_scop.csv.gz",
]

df_cath = pd.DataFrame()
for url in CATH_CANDIDATES:
    df_cath = download_gz_csv(url, "SIFTS CATH/SCOP")
    if not df_cath.empty:
        cath_cols = [c for c in df_cath.columns
                     if any(k in c.upper() for k in
                            ["PDB", "CHAIN", "CATH", "SCOP", "CLASS",
                             "ARCH", "TOPOL", "SUPER", "SF"])]
        if cath_cols:
            df_cath = df_cath[cath_cols].drop_duplicates(
                subset=[c for c in ["PDB", "CHAIN"] if c in cath_cols]
            )
        df_cath.to_csv(OUT / "sifts_pdb_cath.csv", index=False)
        print(f"  Saved {len(df_cath):,} rows → sifts_pdb_cath.csv")
        break

if df_cath.empty:
    print("  [warn] Could not find SIFTS CATH file at any candidate URL")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Supplemental PDB corpus summary ===")
total_kb = 0
for f in sorted(OUT.iterdir()):
    if f.is_file():
        kb = f.stat().st_size / 1024
        total_kb += kb
        try:
            n = sum(1 for _ in open(f)) - 1
            row_hint = f"  ({n:,} rows)"
        except Exception:
            row_hint = ""
        print(f"  {f.name:55s}  {kb:8.1f} KB{row_hint}")
print(f"\n  Total PDB corpus: {total_kb / 1024:.1f} MB")
