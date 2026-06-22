#!/usr/bin/env python3
"""
build_corpus_dataset.py — generate SFT training examples from the ChemSage corpus.

Replaces build_dataset.py. Instead of 25 hardcoded SEED_SMILES, this script reads
the downloaded corpus CSVs and generates diverse Q&A examples that teach the model
to reason about real experimental chemistry data.

Six example classes:
  1. smiles_properties     — drug-likeness and property reasoning (approved drugs corpus)
  2. bioactivity           — potency interpretation (ChEMBL bioactivity tables)
  3. admet                 — ADMET property discussion (ChEMBL ADMET data)
  4. target_drug_mapping   — which drugs hit which targets (drug_mechanisms.csv)
  5. pdb_structure         — crystal structure and binding affinity reasoning (PDB corpus)
  6. scaffold_sar          — Murcko scaffold and matched molecular pair reasoning

Output: data/sft/train.jsonl, valid.jsonl, test.jsonl (80/10/10 split)

Usage:
    python scripts/build_corpus_dataset.py --n 3000 --seed 42
    python scripts/build_corpus_dataset.py --n 5000 --seed 42 --out data/sft
"""

import argparse
import json
import random
import textwrap
from pathlib import Path

import pandas as pd

# RDKit is imported lazily inside validators so missing RDKit gives a clear error
CORPUS = Path("data/corpus")
OUT    = Path("data/sft")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rdkit_props(smiles: str) -> dict | None:
    """Return a dict of computed properties or None if SMILES is invalid."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED, rdMolDescriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            "mw":    round(Descriptors.ExactMolWt(mol), 2),
            "logp":  round(Descriptors.MolLogP(mol), 2),
            "hbd":   rdMolDescriptors.CalcNumHBD(mol),
            "hba":   rdMolDescriptors.CalcNumHBA(mol),
            "tpsa":  round(Descriptors.TPSA(mol), 1),
            "rotb":  rdMolDescriptors.CalcNumRotatableBonds(mol),
            "rings": rdMolDescriptors.CalcNumRings(mol),
            "qed":   round(QED.qed(mol), 3),
            "ro5":   sum([
                Descriptors.ExactMolWt(mol) > 500,
                Descriptors.MolLogP(mol) > 5,
                rdMolDescriptors.CalcNumHBD(mol) > 5,
                rdMolDescriptors.CalcNumHBA(mol) > 10,
            ]),
        }
    except Exception:
        return None


def murcko(smiles: str) -> str | None:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(sc) if sc else None
    except Exception:
        return None


def msg(user: str, assistant: str) -> dict:
    return {"messages": [
        {"role": "user",      "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ]}


def activity_label(val_nm: float) -> str:
    if val_nm < 10:    return "highly potent (<10 nM)"
    if val_nm < 100:   return "potent (10–100 nM)"
    if val_nm < 1000:  return "moderate (100 nM–1 µM)"
    if val_nm < 10000: return "weak (1–10 µM)"
    return "inactive (>10 µM)"


def ro5_verdict(props: dict) -> str:
    if props["ro5"] == 0:
        return "passes Lipinski's Rule of Five with no violations"
    if props["ro5"] == 1:
        return "passes Lipinski's Rule of Five (1 violation — acceptable for clinical drugs)"
    return f"violates Lipinski's Rule of Five ({props['ro5']} violations)"


# ---------------------------------------------------------------------------
# Generator 1 — SMILES properties (from approved drugs corpus)
# ---------------------------------------------------------------------------

def gen_smiles_properties(rng: random.Random, n: int) -> list[dict]:
    # approved_drugs.csv columns: chembl_id, name, smiles, mw, alogp, hbd, hba, tpsa, qed ...
    path = CORPUS / "chembl/approved_drugs.csv"
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return []
    df = pd.read_csv(path).dropna(subset=["smiles"]).drop_duplicates("smiles")
    rows = df.sample(min(n * 3, len(df)), random_state=rng.randint(0, 9999)).to_dict("records")

    examples = []
    for r in rows:
        if len(examples) >= n:
            break
        smi  = r["smiles"]
        name = r.get("name") or "this compound"
        cid  = r.get("chembl_id", "")
        p    = rdkit_props(smi)
        if p is None:
            continue
        if rng.random() < 0.5:
            examples.append(msg(
                f"Is the compound with SMILES {smi!r} drug-like? Show your working.",
                f"Computing physicochemical properties for {name} (ChEMBL: {cid}):\n\n"
                f"- Molecular weight: {p['mw']} Da (limit ≤500)\n"
                f"- LogP: {p['logp']} (limit ≤5)\n"
                f"- HBD: {p['hbd']} (limit ≤5)\n"
                f"- HBA: {p['hba']} (limit ≤10)\n"
                f"- TPSA: {p['tpsa']} Å² (oral absorption typically ≤140 Å²)\n"
                f"- QED: {p['qed']} (0–1; >0.5 considered drug-like)\n\n"
                f"Verdict: this compound {ro5_verdict(p)}. "
                f"QED score of {p['qed']} indicates "
                f"{'good' if p['qed'] > 0.5 else 'moderate'} drug-likeness."
            ))
        else:
            examples.append(msg(
                f"What are the key physicochemical properties of {name}?",
                f"{name} (SMILES: {smi}) has:\n"
                f"MW {p['mw']} Da · LogP {p['logp']} · TPSA {p['tpsa']} Å² · "
                f"HBD {p['hbd']} · HBA {p['hba']} · {p['rotb']} rotatable bonds · "
                f"QED {p['qed']}.\n"
                f"It {ro5_verdict(p)}."
            ))

    print(f"  smiles_properties: {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# Generator 2 — Bioactivity interpretation (ChEMBL IC50/Ki/Kd)
# ---------------------------------------------------------------------------

def gen_bioactivity(rng: random.Random, n: int) -> list[dict]:
    examples = []
    for fname in ["bioactivity_key_targets.csv", "bioactivity_extended_targets.csv"]:
        path = CORPUS / "chembl" / fname
        if not path.exists():
            continue
        df = (pd.read_csv(path)
              .dropna(subset=["canonical_smiles", "standard_value", "target_name"])
              .query("standard_type in ['IC50','Ki','Kd','EC50']")
              .query("standard_value > 0"))
        if df.empty:
            continue
        rows = df.sample(min(n * 2, len(df)), random_state=rng.randint(0, 9999)).to_dict("records")

        for r in rows:
            if len(examples) >= n:
                break
            try:
                val   = float(r["standard_value"])
                unit  = str(r.get("standard_units", "nM"))
                val_nm = val if "nM" in unit else val * 1000 if "uM" in unit or "µM" in unit else val
                label = activity_label(val_nm)
                pchembl = r.get("pchembl_value", "")
                pchembl_str = f" (pIC50 = {float(pchembl):.2f})" if pchembl and str(pchembl) != "nan" else ""
                smiles = r["canonical_smiles"]
                target = r["target_name"]
                stype  = r["standard_type"]
                chembl = r.get("molecule_chembl_id", "")

                examples.append(msg(
                    f"How potent is the compound {chembl} (SMILES: {smiles}) against {target}?",
                    f"{chembl} has a {stype} of {val:.1f} {unit}{pchembl_str} against {target}.\n\n"
                    f"This is {label}. "
                    f"{'Compounds with IC50 < 100 nM are generally considered good starting points for drug development.' if val_nm < 100 else 'Optimisation of potency would be required before this series could advance.' if val_nm < 1000 else 'Significant potency optimisation would be needed.'}"
                ))
            except Exception:
                continue

    print(f"  bioactivity: {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# Generator 3 — ADMET property discussion
# ---------------------------------------------------------------------------

def gen_admet(rng: random.Random, n: int) -> list[dict]:
    path = CORPUS / "chembl/admet_data.csv"
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return []

    df = pd.read_csv(path).dropna(subset=["canonical_smiles", "standard_value"])
    # Pivot: get one ADMET record per SMILES/category
    cats = df.groupby(["canonical_smiles", "admet_category"]).first().reset_index()
    # Find SMILES that have at least 2 ADMET categories
    smi_counts = cats.groupby("canonical_smiles").size()
    rich_smi   = smi_counts[smi_counts >= 2].index.tolist()
    rng.shuffle(rich_smi)

    cat_desc = {
        "Caco-2":    ("Caco-2 permeability (Papp)",    "cm/s", "intestinal absorption"),
        "CLint":     ("intrinsic clearance (CLint)",   "µL/min/mg protein", "metabolic stability"),
        "CYP3A4":    ("CYP3A4 inhibition (IC50)",      "µM", "drug-drug interaction risk via CYP3A4"),
        "CYP2D6":    ("CYP2D6 inhibition (IC50)",      "µM", "drug-drug interaction risk via CYP2D6"),
        "CYP2C9":    ("CYP2C9 inhibition (IC50)",      "µM", "drug-drug interaction risk via CYP2C9"),
        "t_half":    ("half-life (t½)",                 "h",   "duration of action"),
        "fu_plasma": ("fraction unbound in plasma",    "",    "free drug fraction available for target engagement"),
        "solubility":("aqueous solubility",            "µg/mL","formulation and absorption"),
        "logD":      ("logD",                          "",    "lipophilicity at physiological pH"),
    }

    examples = []
    for smi in rich_smi:
        if len(examples) >= n:
            break
        group = cats[cats["canonical_smiles"] == smi]
        lines = []
        for _, row in group.iterrows():
            cat  = row["admet_category"]
            val  = row["standard_value"]
            if cat not in cat_desc:
                continue
            name, unit, meaning = cat_desc[cat]
            lines.append(f"- {name}: {val:.3g} {unit} ({meaning})")
        if not lines:
            continue
        examples.append(msg(
            f"Summarise the ADMET profile of the compound with SMILES: {smi}",
            f"Experimental ADMET data for this compound:\n" + "\n".join(lines) + "\n\n"
            "These values are measured in vitro / in vivo assays from ChEMBL and indicate "
            "the compound's pharmacokinetic behaviour. Always verify with RDKit-computed "
            "descriptors for a quick estimate before experimental data are available."
        ))

    print(f"  admet: {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# Generator 4 — Target-to-drug mapping (drug_mechanisms.csv)
# ---------------------------------------------------------------------------

def gen_target_drug_mapping(rng: random.Random, n: int) -> list[dict]:
    # drug_mechanisms.csv: molecule_chembl_id, mechanism_of_action, target_chembl_id, action_type
    mech_path = CORPUS / "chembl/drug_mechanisms.csv"
    drug_path = CORPUS / "chembl/approved_drugs.csv"
    if not mech_path.exists():
        print(f"  [skip] {mech_path.name} not found")
        return []

    df_mech = pd.read_csv(mech_path).dropna(subset=["mechanism_of_action", "action_type"])
    # Build a chembl_id → name lookup from approved_drugs
    name_map: dict = {}
    if drug_path.exists():
        df_d = pd.read_csv(drug_path).dropna(subset=["chembl_id", "name"])
        name_map = dict(zip(df_d["chembl_id"], df_d["name"]))

    targets = df_mech.groupby("mechanism_of_action")
    examples = []
    for moa, group in targets:
        if len(examples) >= n:
            break
        cids = group["molecule_chembl_id"].dropna().unique().tolist()
        if len(cids) < 2:
            continue
        names     = [name_map.get(c, c) for c in cids]
        actions   = group["action_type"].dropna().unique().tolist()
        action_str = "/".join(actions[:3]) if actions else "modulator"
        drug_list  = ", ".join(names[:8])
        if len(names) > 8:
            drug_list += f" and {len(names)-8} others"

        examples.append(msg(
            f"Which approved drugs work via {moa}?",
            f"{len(names)} approved drug(s) act via {moa} ({action_str}):\n"
            f"{drug_list}.\n\n"
            f"Source: ChEMBL drug mechanism annotations."
        ))

    print(f"  target_drug_mapping: {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# Generator 5 — PDB structure and binding affinity reasoning
# ---------------------------------------------------------------------------

def gen_pdb_structure(rng: random.Random, n: int) -> list[dict]:
    aff_path  = CORPUS / "pdb/pdb_binding_affinities.csv"
    lig_path  = CORPUS / "pdb/pdb_ligands_all.csv"
    if not aff_path.exists() or not lig_path.exists():
        print("  [skip] pdb affinity/ligand files not found")
        return []

    df_aff = pd.read_csv(aff_path).dropna(subset=["pdb_id", "comp_id", "affinity_value"])
    df_lig = pd.read_csv(lig_path).dropna(subset=["comp_id"]).drop_duplicates("comp_id").set_index("comp_id")

    rows = df_aff.sample(min(n * 3, len(df_aff)), random_state=rng.randint(0, 9999)).to_dict("records")

    examples = []
    for r in rows:
        if len(examples) >= n:
            break
        try:
            comp_id = str(r["comp_id"])
            if comp_id in df_lig.index:
                lig_row    = df_lig.loc[comp_id]
                lig_name   = str(lig_row["name"]) if "name" in df_lig.columns else comp_id
                lig_smiles = str(lig_row["smiles"]) if "smiles" in df_lig.columns else ""
            else:
                lig_name, lig_smiles = comp_id, ""

            atype = r["affinity_type"]
            raw   = str(r["affinity_value"]).strip("[]").split(",")[0]
            aval  = float(raw)
            aunit = r.get("affinity_unit", "")
            pdb   = r["pdb_id"]
            method   = r.get("method", "")
            reso_raw = str(r.get("resolution_A", "")).strip("[]").split(",")[0]
            try:
                reso_str = f", resolution {float(reso_raw):.1f} Å"
            except (ValueError, TypeError):
                reso_str = ""
            smiles_str = f" SMILES: {lig_smiles}." if lig_smiles else ""

            examples.append(msg(
                f"What is the measured binding affinity of {lig_name} in PDB structure {pdb}?",
                f"PDB entry {pdb} ({method}{reso_str}) contains {lig_name} (CCD code: {comp_id}){smiles_str}\n\n"
                f"Measured {atype}: {aval} {aunit}.\n\n"
                f"Source: PDBbind binding affinity annotation ingested by RCSB PDB."
            ))
        except Exception:
            continue

    print(f"  pdb_structure: {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# Generator 6 — Scaffold and SAR analysis
# ---------------------------------------------------------------------------

def gen_scaffold_sar(rng: random.Random, n: int) -> list[dict]:
    path = CORPUS / "chembl/approved_drugs.csv"
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return []

    df   = pd.read_csv(path).dropna(subset=["smiles", "name"])
    rows = df.sample(min(n * 3, len(df)), random_state=rng.randint(0, 9999)).to_dict("records")

    examples = []
    for r in rows:
        if len(examples) >= n:
            break
        smi  = r["smiles"]
        name = r.get("name", "the compound")
        sc   = murcko(smi)
        p    = rdkit_props(smi)
        if sc is None or p is None:
            continue

        sc_p = rdkit_props(sc)
        if sc_p is None:
            continue

        complexity = "complex" if p["rings"] > 3 else "moderate" if p["rings"] > 1 else "simple"
        examples.append(msg(
            f"What is the Murcko scaffold of {name} (SMILES: {smi}), and what does it tell us about the SAR?",
            f"The Murcko scaffold of {name} is: {sc}\n\n"
            f"The scaffold has MW {sc_p['mw']} Da, {sc_p['rings']} ring(s), and logP {sc_p['logp']}. "
            f"The parent molecule has a {complexity} ring system with {p['rotb']} rotatable bonds "
            f"and TPSA {p['tpsa']} Å².\n\n"
            f"In SAR analysis, the Murcko scaffold defines the minimal pharmacophoric framework. "
            f"Side-chain modifications (R-groups) on this scaffold are where potency, selectivity, "
            f"and ADMET properties are typically optimised."
        ))

    print(f"  scaffold_sar: {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GENERATORS = [
    ("smiles_properties",    gen_smiles_properties),
    ("bioactivity",          gen_bioactivity),
    ("admet",                gen_admet),
    ("target_drug_mapping",  gen_target_drug_mapping),
    ("pdb_structure",        gen_pdb_structure),
    ("scaffold_sar",         gen_scaffold_sar),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",    type=int, default=3000, help="Total examples to generate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",  type=Path, default=OUT)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    per_gen = args.n // len(GENERATORS)
    print(f"Generating {args.n} total examples ({per_gen} per generator) ...")

    all_examples = []
    for name, fn in GENERATORS:
        print(f"\n[{name}]")
        examples = fn(rng, per_gen)
        all_examples.extend(examples)

    rng.shuffle(all_examples)
    total = len(all_examples)
    n_train = int(total * 0.80)
    n_valid = int(total * 0.10)

    train = all_examples[:n_train]
    valid = all_examples[n_train:n_train + n_valid]
    test  = all_examples[n_train + n_valid:]

    args.out.mkdir(parents=True, exist_ok=True)
    for split_name, split in [("train", train), ("valid", valid), ("test", test)]:
        out_path = args.out / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for ex in split:
                f.write(json.dumps(ex) + "\n")
        print(f"\n{split_name}: {len(split)} → {out_path}")

    print(f"\nTotal: {total} examples  (train {len(train)} / valid {len(valid)} / test {len(test)})")
    print("\nNext: mlx_lm.lora --config config/train_config.yaml --train")


if __name__ == "__main__":
    main()
