#!/usr/bin/env python3
"""
corpus_lookup.py — fast, grounded keyword search across every ChemSage corpus CSV.

Bypasses the LLM for enumeration / factual-recall queries.  All data returned
is read directly from the corpus files — zero hallucination risk.

Usage:
    from rag.corpus_lookup import lookup
    result = lookup("EGFR structures")
    if result:
        print(result)          # str ready to print; None → fall through to LLM
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

_MAX_LOAD = 500   # hard cap on rows loaded per file; all are available for paging/export

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass
class CorpusFile:
    name:              str
    path:              Path
    search_cols:       list[str]               # columns searched for the keyword
    display_cols:      list[str]               # columns shown in output
    col_aliases:       dict      = field(default_factory=dict)   # rename for display
    bracket_clean:     list[str] = field(default_factory=list)   # strip "[]" from these cols
    pdb_cross_ref:     bool      = False       # join via PDB IDs from pdb_all_entries


@dataclass
class LookupResult:
    """One section of corpus search results, carrying the raw DataFrame for rendering."""
    title:   str
    df:      pd.DataFrame   # cleaned display columns, up to _MAX_LOAD rows
    total:   int            # true match count in file (may exceed len(df))
    keyword: str


# pdb_all_entries.csv note: columns are shifted vs. their names.
# Actual content: compound=deposition_date, source=compound_name, release_date=resolution_A.
_PDB_ENTRIES_PATH = Path("data/corpus/pdb/pdb_all_entries.csv")

REGISTRY: list[CorpusFile] = [
    # ---- ChEMBL ----------------------------------------------------------------
    CorpusFile(
        name="Approved drugs",
        path=Path("data/corpus/chembl/approved_drugs.csv"),
        search_cols=["name", "chembl_id"],
        display_cols=["name", "chembl_id", "smiles", "mw", "alogp", "qed", "max_phase"],
    ),
    CorpusFile(
        name="Clinical candidates (Phase 2/3)",
        path=Path("data/corpus/chembl/clinical_candidates_phase2_3.csv"),
        search_cols=["name", "chembl_id"],
        display_cols=["name", "chembl_id", "clinical_phase", "smiles", "mw", "qed"],
    ),
    CorpusFile(
        name="Natural products",
        path=Path("data/corpus/chembl/natural_products.csv"),
        search_cols=["name", "chembl_id"],
        display_cols=["name", "chembl_id", "smiles", "mw", "max_phase"],
    ),
    CorpusFile(
        name="Bioactivity — key targets (ChEMBL)",
        path=Path("data/corpus/chembl/bioactivity_key_targets.csv"),
        search_cols=["target_name", "molecule_chembl_id"],
        display_cols=["molecule_chembl_id", "target_name", "standard_type",
                      "standard_value", "standard_units", "pchembl_value"],

    ),
    CorpusFile(
        name="Bioactivity — extended targets (ChEMBL)",
        path=Path("data/corpus/chembl/bioactivity_extended_targets.csv"),
        search_cols=["target_name", "molecule_chembl_id"],
        display_cols=["molecule_chembl_id", "target_name", "standard_type",
                      "standard_value", "standard_units", "pchembl_value"],

    ),
    CorpusFile(
        name="ADMET data (ChEMBL)",
        path=Path("data/corpus/chembl/admet_data.csv"),
        search_cols=["molecule_chembl_id", "standard_type", "assay_description"],
        display_cols=["molecule_chembl_id", "admet_category", "standard_type",
                      "standard_value", "standard_units"],

    ),
    CorpusFile(
        name="Drug mechanisms (ChEMBL)",
        path=Path("data/corpus/chembl/drug_mechanisms.csv"),
        search_cols=["mechanism_of_action", "action_type"],
        display_cols=["molecule_chembl_id", "mechanism_of_action", "action_type"],

    ),
    CorpusFile(
        name="Drug indications (ChEMBL)",
        path=Path("data/corpus/chembl/drug_indications.csv"),
        search_cols=["efo_term", "mesh_heading"],
        display_cols=["molecule_chembl_id", "efo_term", "mesh_heading", "max_phase_for_ind"],

    ),
    CorpusFile(
        name="ATC classification",
        path=Path("data/corpus/chembl/atc_classification.csv"),
        search_cols=["who_name", "level1_description", "level2_description",
                     "level3_description", "level4_description"],
        display_cols=["level5", "who_name", "level1_description", "level2_description"],

    ),
    CorpusFile(
        name="Fragment actives (ChEMBL)",
        path=Path("data/corpus/chembl/fragment_actives.csv"),
        search_cols=["target_pref_name", "molecule_chembl_id"],
        display_cols=["molecule_chembl_id", "target_pref_name", "standard_type",
                      "standard_value", "pchembl_value"],

    ),
    # ---- PDB -------------------------------------------------------------------
    CorpusFile(
        name="PDB structures",
        path=_PDB_ENTRIES_PATH,
        # 'source' col actually contains compound descriptions; 'release_date' = resolution
        search_cols=["header", "source"],
        display_cols=["pdb_id", "source", "release_date", "header"],
        col_aliases={"source": "compound", "release_date": "resolution_A"},

    ),
    CorpusFile(
        name="PDB binding affinities",
        path=Path("data/corpus/pdb/pdb_binding_affinities.csv"),
        search_cols=["comp_id", "smiles"],
        display_cols=["pdb_id", "comp_id", "affinity_type", "affinity_value",
                      "affinity_unit", "resolution_A", "provenance"],
        bracket_clean=["affinity_value", "resolution_A"],
        pdb_cross_ref=True,   # also match by target via pdb_all_entries

    ),
    CorpusFile(
        name="PDB ligands (CCD)",
        path=Path("data/corpus/pdb/pdb_ligands_all.csv"),
        search_cols=["comp_id", "name", "chembl_id", "drugbank_id", "inchikey"],
        display_cols=["comp_id", "name", "formula", "formula_weight", "smiles",
                      "chembl_id", "drugbank_id"],

    ),
    CorpusFile(
        name="PDB ligand–structure pairs",
        path=Path("data/corpus/pdb/pdb_ligand_structure_pairs.csv"),
        search_cols=["comp_id", "comp_name", "smiles"],
        display_cols=["pdb_id", "comp_id", "comp_name", "smiles", "method", "resolution_A", "is_druglike"],
        bracket_clean=["resolution_A"],
        pdb_cross_ref=True,

    ),
    CorpusFile(
        name="PDB drug-like ligand–structure pairs",
        path=Path("data/corpus/pdb/pdb_druglike_structure_pairs.csv"),
        search_cols=["comp_id", "comp_name", "smiles"],
        display_cols=["pdb_id", "comp_id", "comp_name", "smiles", "method", "resolution_A"],
        bracket_clean=["resolution_A"],
        pdb_cross_ref=True,

    ),
    CorpusFile(
        name="SIFTS UniProt mappings",
        path=Path("data/corpus/pdb/sifts_pdb_uniprot.csv"),
        search_cols=["SP_PRIMARY"],
        display_cols=["PDB", "CHAIN", "SP_PRIMARY", "SP_BEG", "SP_END"],

    ),
    CorpusFile(
        name="SIFTS Pfam mappings",
        path=Path("data/corpus/pdb/sifts_pdb_pfam.csv"),
        search_cols=["PFAM_ID"],
        display_cols=["PDB", "CHAIN", "PFAM_ID"],

    ),
    # ---- PubChem ---------------------------------------------------------------
    CorpusFile(
        name="FDA drugs with PubChem CIDs",
        path=Path("data/corpus/pubchem/fda_drugs_with_pubchem_cids.csv"),
        search_cols=["name", "chembl_id"],
        display_cols=["name", "chembl_id", "pubchem_cid", "smiles", "mw", "qed"],

    ),
]

# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset([
    # query verbs / articles
    "list", "show", "give", "find", "search", "get", "what", "which", "are", "all",
    "every", "the", "some", "any", "me", "please", "have", "does", "this",
    "for", "in", "of", "with", "from", "about", "related", "against", "across",
    # output-format words (describe what kind of data, not the subject)
    "smiles", "smile", "string", "strings", "inchi", "inchikey", "sequence",
    "small", "bound", "binding",
    # corpus concept nouns — these describe the data type, not the subject
    "pdb", "structure", "structures", "entry", "entries", "crystal", "complex",
    "complexes", "binding", "affinity", "affinities", "data", "ic50", "kd",
    "potency", "values", "value", "cross", "reference", "drug", "drugs",
    "compound", "compounds", "molecule", "molecules", "target", "targets",
    "inhibitor", "inhibitors", "ligand", "ligands", "protein", "proteins",
    "kinase", "kinases", "receptor", "receptors", "enzyme", "enzymes",
    "indication", "indications", "mechanism", "mechanisms", "action",
    "activity", "activities", "assay", "assays", "measurement", "measurements",
    "result", "results", "record", "records", "annotation", "annotations",
    "available", "corpus", "database", "table", "file", "source",
    # common drug-discovery adjectives
    "active", "approved", "clinical", "experimental", "known", "measured",
    "targeting", "selective", "potent", "oral", "human",
    # document-type words (describe what to retrieve, not the subject)
    "reference", "code", "example", "examples", "snippet", "snippets",
    "command", "commands", "function", "functions", "method", "methods",
    "recipe", "recipes", "syntax", "tutorial", "guide", "docs", "documentation",
    "complete", "extended", "overview", "intro", "introduction",
])


# ALL-CAPS words that describe output format, not a biological/chemical subject.
# Excluded from the caps-priority path so the real target keyword wins.
_CAPS_STOPWORDS = frozenset(["SMILES", "INCHI", "INCHIKEY", "SMARTS", "SDF", "MOL"])


def _extract_keyword(query: str) -> str | None:
    """Pull the most informative keyword from a lookup query.

    Priority: longest all-caps acronym → longest title-case word → longest
    non-stopword, else None.
    """
    # Prefer ALL-CAPS acronyms (e.g. EGFR, BCR-ABL, CDK2); skip format words
    caps = [c for c in re.findall(r"\b[A-Z][A-Z0-9\-]{1,}\b", query)
            if c not in _CAPS_STOPWORDS]
    if caps:
        return max(caps, key=len)
    # Fall back to any word not in the stopword list
    words = [w.strip(".,?!") for w in query.split()
             if len(w) > 3 and w.lower().strip(".,?!") not in _STOPWORDS]
    return max(words, key=len) if words else None


# ---------------------------------------------------------------------------
# Per-file search
# ---------------------------------------------------------------------------

def _search_file(cf: CorpusFile, keyword: str,
                 pdb_target_ids: set[str] | None = None) -> LookupResult | None:
    """Search one registered file; return LookupResult with raw DataFrame or None."""
    if not cf.path.exists():
        return None

    pattern = rf"\b{re.escape(keyword)}\b"

    try:
        df = pd.read_csv(cf.path, dtype=str, low_memory=False).fillna("")
    except Exception:
        return None

    search_cols  = [c for c in cf.search_cols  if c in df.columns]
    display_cols = [c for c in cf.display_cols if c in df.columns]
    if not search_cols:
        return None

    mask = df[search_cols].apply(
        lambda col: col.str.contains(pattern, case=False, na=False, regex=True)
    ).any(axis=1)

    if cf.pdb_cross_ref and pdb_target_ids and "pdb_id" in df.columns:
        mask = mask | df["pdb_id"].isin(pdb_target_ids)

    hits = df[mask]
    if len(hits) == 0:
        return None

    # Deduplicate by comp_id (same ligand appears in many structures).
    # Sort by comp_mw descending first so the highest-MW — most drug-like — entry survives.
    if "comp_id" in hits.columns:
        if "comp_mw" in hits.columns:
            hits = hits.copy()
            hits["_mw"] = pd.to_numeric(hits["comp_mw"], errors="coerce")
            hits = hits.sort_values("_mw", ascending=False, na_position="last").drop(columns="_mw")
        hits = hits.drop_duplicates(subset="comp_id", keep="first")

    total = len(hits)
    show = hits.head(_MAX_LOAD).copy()

    # Clean bracket notation (e.g. "[2.05]" → "2.05")
    for col in cf.bracket_clean:
        if col in show.columns:
            show[col] = show[col].str.strip("[]").str.split(",").str[0]

    show = show[display_cols].rename(columns=cf.col_aliases)
    return LookupResult(title=cf.name, df=show, total=total, keyword=keyword)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Trigger regex — NARROW on purpose: only bulk enumeration where the LLM cannot
# return a complete set (hallucination risk on IDs/counts).  Anything more
# nuanced ("get the SMILES of …", "what is the IC50 for …") should reach the LLM.
_LOOKUP_RE = re.compile(
    r"\b(list|show|display|fetch|get)\b"
    r"|\b(all|every)\b.{0,40}\b(structure|drug|compound|affinity|entry|entries|ligand|mechanism|indication|kinase|molecule|molecules|smiles?|sequence)\b",
    re.IGNORECASE,
)

# Columns that signal what kind of data the user asked for.
# Used to drop irrelevant sources from the menu.
_INTENT_COLS: dict[str, re.Pattern] = {
    "smiles":   re.compile(r"\bsmiles?\b", re.IGNORECASE),
    "affinity": re.compile(r"\b(affinity|ic50|ki|kd|potency|pchembl)\b", re.IGNORECASE),
    "sequence": re.compile(r"\b(sequence|fasta|uniprot)\b", re.IGNORECASE),
}


def lookup(query: str) -> list[LookupResult] | None:
    """Bulk-enumeration fast path — only fires for explicit list/show/all queries.

    Returns a list of LookupResult or None.  None means the caller (chat loop)
    should pass the query to the LLM instead.
    """
    if not _LOOKUP_RE.search(query):
        return None

    keyword = _extract_keyword(query)
    if not keyword:
        return None

    # Pre-build PDB target IDs via pdb_all_entries so cross-ref files can reuse them.
    pdb_target_ids: set[str] = set()
    if _PDB_ENTRIES_PATH.exists():
        try:
            entries = pd.read_csv(_PDB_ENTRIES_PATH, dtype=str, low_memory=False).fillna("")
            struct_pattern = rf"\b{re.escape(keyword)}\b"
            scols = [c for c in ("header", "source") if c in entries.columns]
            if scols:
                m = entries[scols].apply(
                    lambda col: col.str.contains(struct_pattern, case=False, na=False, regex=True)
                ).any(axis=1)
                pdb_target_ids = set(entries.loc[m, "pdb_id"])
        except Exception:
            pass

    results: list[LookupResult] = []
    for cf in REGISTRY:
        r = _search_file(cf, keyword, pdb_target_ids)
        if r:
            results.append(r)

    if not results:
        return None

    # Intent filtering: if the query asks for a specific column type (e.g. SMILES),
    # drop sources that don't have that column populated — don't clutter the menu
    # with irrelevant tables.
    for col, pattern in _INTENT_COLS.items():
        if pattern.search(query):
            filtered = [r for r in results
                        if col in r.df.columns and r.df[col].str.len().gt(0).any()]
            if filtered:
                results = filtered
            break   # only apply the first matching intent

    return results if results else None
