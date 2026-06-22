#!/usr/bin/env python3
"""
download_pdb.py — pull RCSB PDB data into data/corpus/pdb/ for RAG ingestion.

Downloads:
  1.  All PDB entry metadata (~220k structures: method, resolution, title, date)
  2.  All Chemical Component Dictionary (CCD) entries with SMILES (~40k unique ligands)
  3.  Drug-like CCD subset (MW 100–800 Da, non-polymer, organic)
  4.  Drug-like ligand–structure pairs (which proteins bind which drug-like ligands)
  5.  SIFTS PDB → UniProt chain mapping (from EBI)
  6.  SIFTS PDB → Pfam domain mapping (from EBI)
  7.  SIFTS PDB → CATH classification (from EBI)
  8.  Structures with measured binding affinities (Kd / Ki / IC50 — PDBbind subset)

APIs used:
  - RCSB PDB Data API   https://data.rcsb.org/rest/v1/
  - RCSB PDB GraphQL    https://data.rcsb.org/graphql
  - RCSB Search API     https://search.rcsb.org/rcsbsearch/v2/query
  - wwPDB derived data  https://files.wwpdb.org/pub/pdb/derived_data/
  - EBI SIFTS           https://ftp.ebi.ac.uk/pub/databases/msd/sifts/

RCSB rate limit: ~10 requests/s. Script sleeps 0.15 s between requests.
SIFTS downloads are large (10–80 MB compressed); the script deduplicates to
chain-level rows so the resulting CSVs stay within RAG-friendly size.
"""

import gzip
import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

OUT         = Path("data/corpus/pdb")
OUT.mkdir(parents=True, exist_ok=True)

RCSB_DATA   = "https://data.rcsb.org"
RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_GQL    = f"{RCSB_DATA}/graphql"
WWPDB       = "https://files.wwpdb.org/pub/pdb/derived_data"
SIFTS_BASE  = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv"

S = requests.Session()
S.headers["User-Agent"] = "ChemSage/1.0 (drug-discovery-rag; marc@marcdeller.com)"
PAUSE = 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_json(url: str, params: dict = None, timeout: int = 60) -> dict:
    try:
        r = S.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"    [warn] GET {url}: {e}")
        return {}


def post_search(payload: dict, label: str) -> list[dict]:
    """POST to RCSB search v2 and return result_set list."""
    try:
        r = S.post(RCSB_SEARCH, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("result_set", [])
    except Exception as e:
        print(f"    [warn] RCSB search {label}: {e}")
        return []


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


def download_text(url: str, label: str, timeout: int = 300) -> str:
    print(f"  Downloading {label} ...")
    try:
        r = S.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"    [warn] {label}: {e}")
        return ""


def download_gz_csv(url: str, label: str, comment_char: str = None,
                    nrows: int = None) -> pd.DataFrame:
    """Download a gzip-compressed CSV from SIFTS/EBI."""
    print(f"  Downloading {label} ...")
    try:
        r = S.get(url, timeout=600, stream=True)
        r.raise_for_status()
        buf = io.BytesIO(r.content)
        with gzip.open(buf) as gz:
            df = pd.read_csv(gz, comment=comment_char,
                             low_memory=False, nrows=nrows)
        print(f"  {label}: {len(df):,} rows raw")
        return df
    except Exception as e:
        print(f"    [warn] {label}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# GraphQL fragments
# ---------------------------------------------------------------------------

ENTRY_GQL = """
query GetEntries($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    struct { title }
    exptl { method }
    struct_keywords { pdbx_keywords text }
    rcsb_entry_info {
      resolution_combined
      deposited_atom_count
      deposited_polymer_entity_instance_count
      deposited_nonpolymer_entity_instance_count
      structure_determination_methodology
    }
    pdbx_database_status { recvd_initial_deposition_date status_code }
    rcsb_entry_container_identifiers { assembly_ids }
  }
}
"""

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

AFFINITY_GQL = """
query GetAffinity($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    rcsb_binding_affinity {
      comp_id
      type
      value
      unit
      provenance_code
      reference_sequence_identity
      symbol
    }
    exptl { method }
    rcsb_entry_info { resolution_combined }
  }
}
"""


# ---------------------------------------------------------------------------
# 1. All PDB entry metadata (from wwPDB derived data index files)
# ---------------------------------------------------------------------------
print("\n[1/8] All PDB entry metadata (derived data index files) ...")

# entries.idx: IDCODE  HEADER  COMPOUND  SOURCE  AUTHOR  DEPOSITION  RELEASE  REVTYPE
entries_txt = download_text(f"{WWPDB}/index/entries.idx", "entries.idx")
resolu_txt  = download_text(f"{WWPDB}/index/resolu.idx",  "resolu.idx")

entries_rows = []
for line in entries_txt.splitlines():
    if not line.strip() or line.startswith("IDCODE") or line.startswith("---"):
        continue
    parts = line.split("\t") if "\t" in line else line.split()
    if len(parts) >= 1:
        pdb_id = parts[0].strip().upper()
        if len(pdb_id) == 4:
            entries_rows.append({
                "pdb_id": pdb_id,
                "header": parts[1].strip() if len(parts) > 1 else "",
                "compound": parts[2].strip() if len(parts) > 2 else "",
                "source": parts[3].strip() if len(parts) > 3 else "",
                "deposition_date": parts[5].strip() if len(parts) > 5 else "",
                "release_date": parts[6].strip() if len(parts) > 6 else "",
            })

df_entries = pd.DataFrame(entries_rows)
print(f"  Parsed {len(df_entries):,} entries from entries.idx")

# Parse resolution index
resolu_map = {}
for line in resolu_txt.splitlines():
    if not line.strip() or line.startswith("IDCODE") or line.startswith("--"):
        continue
    parts = line.split()
    if len(parts) >= 2:
        try:
            resolu_map[parts[0].upper()] = float(parts[-1])
        except ValueError:
            pass

if df_entries.shape[0] > 0:
    df_entries["resolution_A"] = df_entries["pdb_id"].map(resolu_map)
    df_entries.to_csv(OUT / "pdb_all_entries.csv", index=False)
    print(f"  Saved {len(df_entries):,} entries → pdb_all_entries.csv")
    all_pdb_ids = df_entries["pdb_id"].tolist()
else:
    # Fallback: use RCSB holdings endpoint
    print("  Falling back to RCSB holdings endpoint ...")
    holdings = get_json(f"{RCSB_DATA}/rest/v1/holdings/current/entry_ids", timeout=120)
    all_pdb_ids = holdings if isinstance(holdings, list) else []
    print(f"  Got {len(all_pdb_ids):,} PDB IDs from holdings")
    df_entries = pd.DataFrame({"pdb_id": all_pdb_ids})
    df_entries.to_csv(OUT / "pdb_all_entries.csv", index=False)


# ---------------------------------------------------------------------------
# 2. Chemical Component Dictionary — all unique PDB ligands with SMILES
# ---------------------------------------------------------------------------
print("\n[2/8] Chemical Component Dictionary — all ligands with SMILES ...")

# Get all chem comp IDs from the wwPDB usage counts file
cc_counts_txt = download_text(f"{WWPDB}/cc-counts.tdd", "cc-counts.tdd")
all_cc_ids = []
for line in cc_counts_txt.splitlines():
    parts = line.strip().split()
    if parts and parts[0] and not parts[0].startswith("#"):
        all_cc_ids.append(parts[0])
print(f"  {len(all_cc_ids):,} CCD IDs from cc-counts.tdd")

# Fallback: RCSB Search API if the file was empty
if len(all_cc_ids) < 100:
    print("  Falling back to RCSB Search API for chem comp IDs ...")
    search_payload = {
        "query": {
            "type": "terminal",
            "service": "text_chem",
            "parameters": {
                "attribute": "chem_comp.pdbx_release_status",
                "operator": "in",
                "value": ["REL", "OBS"],
            },
        },
        "return_type": "chem_comp",
        "request_options": {"return_all_hits": True},
    }
    cc_results = post_search(search_payload, "chem_comp IDs")
    all_cc_ids = [r["identifier"] for r in cc_results]
    print(f"  {len(all_cc_ids):,} CCD IDs from RCSB Search")

print(f"  {len(all_cc_ids):,} CCD entries to fetch ...")

BATCH = 100  # GraphQL batch size for chem comps
cc_rows = []

for i in range(0, len(all_cc_ids), BATCH):
    batch = all_cc_ids[i:i + BATCH]
    data  = graphql(CHEMCOMP_GQL, {"comp_ids": batch})
    comps = data.get("chem_comps") or []
    for comp in comps:
        if not comp:
            continue
        cc   = comp.get("chem_comp") or {}
        descs = comp.get("rcsb_chem_comp_descriptor") or []
        rels  = comp.get("rcsb_chem_comp_related") or []

        # descs is a single dict in the current RCSB GraphQL schema
        desc_obj = descs if isinstance(descs, dict) else {}
        smiles   = desc_obj.get("SMILES_stereo") or desc_obj.get("SMILES") or ""
        inchi    = desc_obj.get("InChI") or ""
        inchikey = desc_obj.get("InChIKey") or ""

        # External cross-references (ChEMBL, DrugBank, etc.)
        xrefs = {r.get("resource_name", ""): r.get("resource_accession_code", "")
                 for r in rels if r}

        cc_rows.append({
            "comp_id":       cc.get("id", ""),
            "name":          cc.get("name", ""),
            "formula":       cc.get("formula", ""),
            "formula_weight": cc.get("formula_weight"),
            "type":          cc.get("type", ""),
            "release_status": cc.get("pdbx_release_status", ""),
            "smiles":        smiles,
            "inchi":         inchi,
            "inchikey":      inchikey,
            "chembl_id":     xrefs.get("ChEMBL", ""),
            "drugbank_id":   xrefs.get("DrugBank", ""),
            "pubchem_cid":   xrefs.get("PubChem CID", ""),
        })

    done = min(i + BATCH, len(all_cc_ids))
    if done % 5000 < BATCH:
        print(f"  CCD: {done:,}/{len(all_cc_ids):,} fetched ({len(cc_rows):,} records)", end="\r")
    time.sleep(PAUSE)

print(f"  CCD: {len(cc_rows):,} records fetched          ")
df_cc = pd.DataFrame(cc_rows)
df_cc.to_csv(OUT / "pdb_ligands_all.csv", index=False)
print(f"  Saved {len(df_cc):,} CCD entries → pdb_ligands_all.csv")


# ---------------------------------------------------------------------------
# 3. Drug-like CCD subset (MW 100–800 Da, non-polymer, with SMILES)
# ---------------------------------------------------------------------------
print("\n[3/8] Drug-like CCD subset ...")

smiles_map: dict = {}
druglike_ids: set = set()

if not df_cc.empty and "smiles" in df_cc.columns:
    df_drug = df_cc[
        df_cc["smiles"].notna() & (df_cc["smiles"] != "") &
        df_cc["formula_weight"].notna() &
        (df_cc["formula_weight"].astype(float) >= 100) &
        (df_cc["formula_weight"].astype(float) <= 800) &
        ~df_cc["type"].str.lower().str.contains("polymer|rna|dna|saccharide", na=True)
    ].copy()
    df_drug.to_csv(OUT / "pdb_ligands_druglike.csv", index=False)
    print(f"  {len(df_drug):,} drug-like CCD entries → pdb_ligands_druglike.csv")
    druglike_ids = set(df_drug["comp_id"].dropna().tolist())
    smiles_map   = df_cc.set_index("comp_id")["smiles"].to_dict()
else:
    print("  [skip] CCD table empty — no drug-like subset to write")


# ---------------------------------------------------------------------------
# 4. Drug-like ligand–structure pairs (RCSB search for entries with ligands)
# ---------------------------------------------------------------------------
print("\n[4/8] Drug-like ligand–structure pairs (RCSB Search API) ...")

# Search for entries that contain non-polymer entities with MW in drug-like range
LIGAND_SEARCH = {
    "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.deposited_nonpolymer_entity_instance_count",
                    "operator": "greater",
                    "value": 0,
                    "negation": False,
                },
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "exptl.method",
                    "operator": "in",
                    "value": ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY",
                              "NEUTRON DIFFRACTION"],
                    "negation": False,
                },
            },
        ],
    },
    "return_type": "entry",
    "request_options": {
        "return_all_hits": True,
        "results_content_type": ["experimental"],
    },
}

print("  Querying RCSB for all structures with bound ligands ...")
results = post_search(LIGAND_SEARCH, "ligand-bound structures")
entry_ids_with_ligands = [r["identifier"] for r in results]
print(f"  {len(entry_ids_with_ligands):,} structures with bound non-polymer ligands")
time.sleep(PAUSE)

# For each entry, get the ligand inventory via the nonpolymer entity endpoint
# We'll batch-fetch the data API for nonpolymer entity details
# Use a GraphQL query for entry + nonpolymer info
NONPOLYMER_GQL = """
query GetNonPolymers($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    exptl { method }
    rcsb_entry_info { resolution_combined }
    nonpolymer_entities {
      nonpolymer_comp {
        chem_comp { id name formula_weight type }
      }
      rcsb_nonpolymer_entity {
        formula_weight
        pdbx_description
      }
    }
  }
}
"""

pair_rows = []
BATCH_E = 50
total_e  = min(len(entry_ids_with_ligands), 50_000)  # cap at 50k for speed

for i in range(0, total_e, BATCH_E):
    batch = entry_ids_with_ligands[i:i + BATCH_E]
    data  = graphql(NONPOLYMER_GQL, {"ids": batch})
    entries = data.get("entries") or []
    for entry in entries:
        if not entry:
            continue
        pdb_id  = entry.get("rcsb_id", "")
        method  = ((entry.get("exptl") or [{}])[0]).get("method", "")
        reso    = (entry.get("rcsb_entry_info") or {}).get("resolution_combined")
        for npe in (entry.get("nonpolymer_entities") or []):
            if not npe:
                continue
            cc_info = ((npe.get("nonpolymer_comp") or {}).get("chem_comp") or {})
            comp_id = cc_info.get("id", "")
            pair_rows.append({
                "pdb_id":      pdb_id,
                "comp_id":     comp_id,
                "comp_name":   cc_info.get("name", ""),
                "comp_mw":     cc_info.get("formula_weight"),
                "comp_type":   cc_info.get("type", ""),
                "method":      method,
                "resolution_A": reso,
                "is_druglike": comp_id in druglike_ids,
            })
    if i % 5000 < BATCH_E:
        print(f"  Ligand pairs: {min(i + BATCH_E, total_e):,}/{total_e:,}", end="\r")
    time.sleep(PAUSE)

print(f"  Ligand pairs: {len(pair_rows):,} fetched          ")
df_pairs = pd.DataFrame(pair_rows)
if not df_pairs.empty:
    df_pairs["smiles"] = df_pairs["comp_id"].map(smiles_map)
    df_pairs.to_csv(OUT / "pdb_ligand_structure_pairs.csv", index=False)
    df_drug_pairs = df_pairs[df_pairs["is_druglike"] == True]
    df_drug_pairs.to_csv(OUT / "pdb_druglike_structure_pairs.csv", index=False)
    print(f"  Saved {len(df_pairs):,} total pairs → pdb_ligand_structure_pairs.csv")
    print(f"  Saved {len(df_drug_pairs):,} drug-like pairs → pdb_druglike_structure_pairs.csv")


# ---------------------------------------------------------------------------
# 5. SIFTS PDB → UniProt chain mapping
# ---------------------------------------------------------------------------
print("\n[5/8] SIFTS PDB → UniProt chain mapping ...")

df_sifts_up = download_gz_csv(
    f"{SIFTS_BASE}/pdb_chain_uniprot.csv.gz",
    "SIFTS PDB-UniProt",
    comment_char="#",
)

if not df_sifts_up.empty:
    # Deduplicate to chain-level (drop per-residue rows if present)
    chain_cols = [c for c in ["PDB", "CHAIN", "SP_PRIMARY", "SP_BEG", "SP_END",
                               "PDB_BEG", "PDB_END", "RES_BEG", "RES_END"]
                  if c in df_sifts_up.columns]
    df_sifts_up = df_sifts_up[chain_cols].drop_duplicates(
        subset=[c for c in ["PDB", "CHAIN", "SP_PRIMARY"] if c in chain_cols]
    )
    df_sifts_up.to_csv(OUT / "sifts_pdb_uniprot.csv", index=False)
    print(f"  Saved {len(df_sifts_up):,} chain-level rows → sifts_pdb_uniprot.csv")


# ---------------------------------------------------------------------------
# 6. SIFTS PDB → Pfam domain mapping
# ---------------------------------------------------------------------------
print("\n[6/8] SIFTS PDB → Pfam domain mapping ...")

df_sifts_pfam = download_gz_csv(
    f"{SIFTS_BASE}/pdb_chain_pfam.csv.gz",
    "SIFTS PDB-Pfam",
    comment_char="#",
)

if not df_sifts_pfam.empty:
    pfam_cols = [c for c in ["PDB", "CHAIN", "PFAM_ID", "PFAM_NAME",
                              "AUTH_CHAIN_ID", "PDB_BEG", "PDB_END",
                              "PFAM_ACC", "CLAN_ID", "CLAN_ACC"]
                 if c in df_sifts_pfam.columns]
    df_sifts_pfam = df_sifts_pfam[pfam_cols].drop_duplicates()
    df_sifts_pfam.to_csv(OUT / "sifts_pdb_pfam.csv", index=False)
    print(f"  Saved {len(df_sifts_pfam):,} rows → sifts_pdb_pfam.csv")


# ---------------------------------------------------------------------------
# 7. SIFTS PDB → CATH/SCOP classification
# ---------------------------------------------------------------------------
print("\n[7/8] SIFTS PDB → CATH classification ...")

df_sifts_cath = download_gz_csv(
    f"{SIFTS_BASE}/pdb_chain_cath_scop.csv.gz",
    "SIFTS PDB-CATH-SCOP",
    comment_char="#",
)

if not df_sifts_cath.empty:
    cath_cols = [c for c in ["PDB", "CHAIN", "CATH_ID", "SCOP_ID",
                              "AUTH_CHAIN_ID", "PDB_BEG", "PDB_END",
                              "CATHCODE", "CLASS", "ARCHITECTURE",
                              "TOPOLOGY", "SUPERFAMILY"]
                 if c in df_sifts_cath.columns]
    df_sifts_cath = df_sifts_cath[cath_cols].drop_duplicates(
        subset=[c for c in ["PDB", "CHAIN"] if c in cath_cols]
    )
    df_sifts_cath.to_csv(OUT / "sifts_pdb_cath.csv", index=False)
    print(f"  Saved {len(df_sifts_cath):,} rows → sifts_pdb_cath.csv")


# ---------------------------------------------------------------------------
# 8. Structures with measured binding affinities (PDBbind subset via RCSB)
# ---------------------------------------------------------------------------
print("\n[8/8] Structures with measured binding affinities ...")

AFFINITY_SEARCH = {
    "query": {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": "rcsb_binding_affinity.type",
            "operator": "exists",
            "negation": False,
        },
    },
    "return_type": "entry",
    "request_options": {"return_all_hits": True},
}

aff_results = post_search(AFFINITY_SEARCH, "binding affinity")
aff_ids = [r["identifier"] for r in aff_results]
print(f"  {len(aff_ids):,} structures with measured binding affinities")
time.sleep(PAUSE)

aff_rows = []
BATCH_A = 50
for i in range(0, len(aff_ids), BATCH_A):
    batch = aff_ids[i:i + BATCH_A]
    data  = graphql(AFFINITY_GQL, {"ids": batch})
    entries = data.get("entries") or []
    for entry in entries:
        if not entry:
            continue
        pdb_id  = entry.get("rcsb_id", "")
        method  = ((entry.get("exptl") or [{}])[0]).get("method", "")
        reso    = (entry.get("rcsb_entry_info") or {}).get("resolution_combined")
        for aff in (entry.get("rcsb_binding_affinity") or []):
            if not aff:
                continue
            comp_id = aff.get("comp_id", "")
            aff_rows.append({
                "pdb_id":        pdb_id,
                "comp_id":       comp_id,
                "affinity_type": aff.get("type", ""),
                "affinity_value": aff.get("value"),
                "affinity_unit": aff.get("unit", ""),
                "provenance":    aff.get("provenance_code", ""),
                "seq_identity":  aff.get("reference_sequence_identity"),
                "symbol":        aff.get("symbol", ""),
                "method":        method,
                "resolution_A":  reso,
                "smiles":        smiles_map.get(comp_id, ""),
            })
    if i % 2000 < BATCH_A:
        print(f"  Affinity: {min(i + BATCH_A, len(aff_ids)):,}/{len(aff_ids):,}", end="\r")
    time.sleep(PAUSE)

print(f"  Affinity: {len(aff_rows):,} measurements fetched          ")
if aff_rows:
    df_aff = pd.DataFrame(aff_rows)
    df_aff.to_csv(OUT / "pdb_binding_affinities.csv", index=False)
    print(f"  Saved {len(df_aff):,} affinity rows → pdb_binding_affinities.csv")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== PDB corpus summary ===")
total_kb = 0
for f in sorted(OUT.iterdir()):
    if f.is_file():
        kb        = f.stat().st_size / 1024
        total_kb += kb
        row_hint  = ""
        try:
            n = sum(1 for _ in open(f)) - 1
            row_hint = f"  ({n:,} rows)"
        except Exception:
            pass
        print(f"  {f.name:55s}  {kb:8.1f} KB{row_hint}")
print(f"\n  Total: {total_kb / 1024:.1f} MB")
print("\nNext: python scripts/ingest_rag.py --corpus data/corpus --store .chroma --reset")
