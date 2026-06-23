#!/usr/bin/env python3
"""
build_dataset.py — generate and validate chemistry SFT pairs (Phase 3).

The key trick: generate ground truth by RUNNING RDKit, then template the question and
worked solution around the real values. Never hand-author a numeric property.

Writes MLX-LM's expected layout: train.jsonl + valid.jsonl + test.jsonl (80/10/10),
each line a {"messages": [...]} chat record.

Usage:
    python scripts/build_dataset.py --out data/sft --n 1500
"""

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import (
        Descriptors, Lipinski, rdMolDescriptors, AllChem, rdFingerprintGenerator
    )
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    raise SystemExit("RDKit is required: pip install rdkit")

SYSTEM = (
    "You are ChemSage, a chemically literate research assistant for drug discovery. "
    "You compute chemical properties with RDKit rather than estimating them, and you "
    "emit correct PyMOL scripts for structural work."
)

SEED_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",                           # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",                    # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                       # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",                                # paracetamol
    "Cn1cnc2c1c(=O)n(c(=O)n2C)C",                       # theophylline
    "OC(=O)c1cccnc1",                                    # nicotinic acid
    "Cc1ccc(cc1)S(=O)(=O)N",                            # p-toluenesulfonamide
    "OC(=O)c1ccc(N)cc1",                                 # 4-aminobenzoic acid
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",              # testosterone
    "CC(N)Cc1ccc(O)cc1",                                 # tyramine
    "COc1ccc(CCN)cc1O",                                  # dopamine methyl ether
    "c1ccc(cc1)C(=O)O",                                  # benzoic acid
    "OC(Cc1ccccc1)C(=O)O",                              # phenyllactic acid
    "OC(=O)c1ccc(Cl)cc1",                               # 4-chlorobenzoic acid
    "CC(=O)Nc1ccc(Cl)cc1",                              # chloroacetanilide
    "CCOc1ccc(NC(=O)c2cc(C(F)(F)F)ccc2Cl)cc1",         # sorafenib-like
    "c1ccc(Nc2ncnc3ccsc23)cc1",                         # erlotinib-like core
    "CC1=CC(=O)c2ccccc2C1=O",                           # menadione
    "O=C(O)c1cccc(c1)c1ccc(cc1)C#N",                   # fenoprofen analogue
    "Cn1cc(C(=O)Nc2ccc(F)cc2)c2ccccc21",               # indole amide
    "COc1cc(C=O)ccc1O",                                  # vanillin
    "O=C1NC(=O)c2ccccc21",                              # isatoic anhydride core
    "Cc1nc2ccccc2c(=O)n1Cc1ccccc1",                    # benzodiazepine-like
    "CC(C)(C)OC(=O)Nc1ccc(cc1)C(=O)O",                # Boc-4-aminobenzoic acid
    "O=C(O)c1ccc2ccccc2c1",                             # 2-naphthoic acid
]

PYMOL_CONTEXTS = [
    {"pdb": "1HSG", "ligand": "MK1", "target": "HIV protease"},
    {"pdb": "2GS6", "ligand": "STI", "target": "Abl kinase (imatinib complex)"},
    {"pdb": "3HTB", "ligand": "N3",  "target": "SARS-CoV-2 main protease"},
    {"pdb": "1OYT", "ligand": "IMA", "target": "Bcr-Abl (imatinib)"},
    {"pdb": "4DKL", "ligand": "LIG", "target": "CDK2 kinase"},
]

SUBSTRUCTURE_QUERIES = [
    ("[CX3](=O)[OX2H1]",      "carboxylic acid"),
    ("[NX3;H2]",               "primary amine"),
    ("[OX2H]",                 "hydroxyl"),
    ("c1ccccc1",               "benzene ring"),
    ("C(=O)N",                 "amide bond"),
    ("[F,Cl,Br,I]",            "halogen"),
    ("[SX4](=O)(=O)",          "sulphone"),
    ("[#6]1~[#6]~[#6]~[#6]~[#6]~1", "five-membered carbocycle"),
]

CODE_BLOCK = re.compile(r"```python\s*(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------------
# Behaviour class 1: SMILES literacy — canonicalise and identify a functional group
# ---------------------------------------------------------------------------

def smiles_canonicalise_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canon = Chem.MolToSmiles(mol)
    fg    = _identify_fg(mol)
    answer = (
        "I will canonicalise the SMILES with RDKit to produce a unique representation.\n\n"
        "```python\n"
        "from rdkit import Chem\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "print(Chem.MolToSmiles(mol))\n"
        "```\n\n"
        f"Canonical SMILES: `{canon}`. {fg}"
    )
    return _record(
        f"Canonicalise this SMILES and identify any obvious functional groups: {smiles}",
        answer,
    )


def _identify_fg(mol) -> str:
    smarts_map = [
        ("[CX3](=O)[OX2H1]",  "carboxylic acid"),
        ("[NX3;H2][CX4]",     "primary amine"),
        ("[OX2H]",            "hydroxyl group"),
        ("[CX3](=O)[OX2][CX4]", "ester"),
        ("[CX3](=O)[NX3]",   "amide"),
        ("[nX2r6]",           "pyridine-type ring nitrogen"),
        ("[F,Cl,Br,I]",       "halogen substituent"),
        ("[SX4](=O)(=O)",     "sulphone"),
    ]
    for smarts, name in smarts_map:
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            return f"The molecule contains a {name}."
    return "No simple functional group matched from the checked set."


# ---------------------------------------------------------------------------
# Behaviour class 2: Property reasoning via tools — Lipinski + Veber
# ---------------------------------------------------------------------------

def druglikeness_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canon = Chem.MolToSmiles(mol)
    mw   = round(Descriptors.MolWt(mol), 1)
    logp = round(Descriptors.MolLogP(mol), 2)
    hbd  = Lipinski.NumHDonors(mol)
    hba  = Lipinski.NumHAcceptors(mol)
    tpsa = round(rdMolDescriptors.CalcTPSA(mol), 1)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    lip  = mw < 500 and logp < 5 and hbd <= 5 and hba <= 10
    veb  = tpsa <= 140 and rotb <= 10
    verdict = (
        "passes both Lipinski and Veber" if (lip and veb) else
        "passes Lipinski but fails Veber" if lip else
        ("fails Lipinski and Veber" if not veb else "fails Lipinski")
    )
    answer = (
        "I will compute all descriptors with RDKit.\n\n"
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "print(Chem.MolToSmiles(mol))\n"
        "print(\n"
        "    round(Descriptors.MolWt(mol), 1),\n"
        "    round(Descriptors.MolLogP(mol), 2),\n"
        "    Lipinski.NumHDonors(mol),\n"
        "    Lipinski.NumHAcceptors(mol),\n"
        "    round(rdMolDescriptors.CalcTPSA(mol), 1),\n"
        "    rdMolDescriptors.CalcNumRotatableBonds(mol),\n"
        ")\n"
        "```\n\n"
        f"Canonical: `{canon}`. MW {mw}, logP {logp}, HBD {hbd}, HBA {hba}, "
        f"TPSA {tpsa}, RotBonds {rotb}. "
        f"The compound {verdict} "
        "(Lipinski: MW < 500, logP < 5, HBD ≤ 5, HBA ≤ 10; Veber: TPSA ≤ 140, RotBonds ≤ 10)."
    )
    return _record(
        f"Assess the oral drug-likeness of {smiles} using Lipinski and Veber rules.",
        answer,
    )


# ---------------------------------------------------------------------------
# Behaviour class 3a: RDKit tool invocation — Murcko scaffold
# ---------------------------------------------------------------------------

def murcko_scaffold_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return None
    smi_scaffold = Chem.MolToSmiles(scaffold) if scaffold else None
    if not smi_scaffold:
        return None
    canon = Chem.MolToSmiles(mol)
    answer = (
        "I will extract the Murcko scaffold with RDKit.\n\n"
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem.Scaffolds import MurckoScaffold\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "scaffold = MurckoScaffold.GetScaffoldForMol(mol)\n"
        "print(Chem.MolToSmiles(scaffold))\n"
        "```\n\n"
        f"Canonical: `{canon}`. Murcko scaffold: `{smi_scaffold}`. "
        "The scaffold retains ring systems and linkers, stripping all terminal substituents. "
        "Use it to cluster an SAR series by core or to align analogues."
    )
    return _record(
        f"Extract the Murcko scaffold from {smiles}.",
        answer,
    )


# ---------------------------------------------------------------------------
# Behaviour class 3b: RDKit tool invocation — Tanimoto similarity
# ---------------------------------------------------------------------------

def tanimoto_example(smi_a: str, smi_b: str):
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None:
        return None
    gen  = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp_a = gen.GetFingerprint(mol_a)
    fp_b = gen.GetFingerprint(mol_b)
    sim  = round(DataStructs.TanimotoSimilarity(fp_a, fp_b), 3)
    interp = (
        "The compounds are closely related (Tc ≥ 0.7) — likely analogues in the same series."
        if sim >= 0.7 else
        "The compounds share scaffold features (0.4 ≤ Tc < 0.7) but differ in substituents."
        if sim >= 0.4 else
        "The compounds are structurally dissimilar (Tc < 0.4) by Morgan fingerprint."
    )
    answer = (
        "I will compute Tanimoto similarity on Morgan (r=2, 2048-bit) fingerprints.\n\n"
        "```python\n"
        "from rdkit import Chem, DataStructs\n"
        "from rdkit.Chem import rdFingerprintGenerator\n"
        f"mol_a = Chem.MolFromSmiles('{smi_a}')\n"
        f"mol_b = Chem.MolFromSmiles('{smi_b}')\n"
        "gen  = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)\n"
        "fp_a = gen.GetFingerprint(mol_a)\n"
        "fp_b = gen.GetFingerprint(mol_b)\n"
        "print(round(DataStructs.TanimotoSimilarity(fp_a, fp_b), 3))\n"
        "```\n\n"
        f"Tanimoto (Morgan r=2): {sim}. {interp}"
    )
    return _record(
        f"How structurally similar are {smi_a} and {smi_b}? Use Morgan fingerprints.",
        answer,
    )


# ---------------------------------------------------------------------------
# Behaviour class 3c: RDKit tool invocation — substructure search
# ---------------------------------------------------------------------------

def substructure_example(query_smarts: str, query_name: str, target_smi: str):
    q_mol  = Chem.MolFromSmarts(query_smarts)
    t_mol  = Chem.MolFromSmiles(target_smi)
    if q_mol is None or t_mol is None:
        return None
    matches = t_mol.GetSubstructMatches(q_mol)
    present = bool(matches)
    canon   = Chem.MolToSmiles(t_mol)
    result_str = f"present ({len(matches)} match(es))" if present else "absent"
    answer = (
        f"I will run a SMARTS substructure search with RDKit.\n\n"
        "```python\n"
        "from rdkit import Chem\n"
        f"query  = Chem.MolFromSmarts('{query_smarts}')  # {query_name}\n"
        f"mol    = Chem.MolFromSmiles('{target_smi}')\n"
        "matches = mol.GetSubstructMatches(query)\n"
        "print(bool(matches), len(matches))\n"
        "```\n\n"
        f"Canonical target: `{canon}`. "
        f"The {query_name} pattern (`{query_smarts}`) is {result_str} in this molecule."
    )
    return _record(
        f"Does the molecule {target_smi} contain a {query_name} ({query_smarts})?",
        answer,
    )


# ---------------------------------------------------------------------------
# Behaviour class 4: PyMOL tool invocation
# ---------------------------------------------------------------------------

def pymol_example():
    ctx    = random.choice(PYMOL_CONTEXTS)
    pdb    = ctx["pdb"]
    ligand = ctx["ligand"]
    target = ctx["target"]
    style  = random.choice(["bfactor", "raytrace", "distance"])

    if style == "bfactor":
        question = "show the binding site within 5 Å of the ligand, coloured by B-factor"
        script   = (
            f"fetch {pdb}, async=0\n"
            f"remove solvent\n"
            f"select ligand, resn {ligand}\n"
            f"select bsite, byres (polymer within 5 of ligand)\n"
            f"hide everything\n"
            f"show cartoon, polymer\n"
            f"show sticks, ligand or bsite\n"
            f"spectrum b, blue_white_red, bsite\n"
            f"center ligand\n"
            f"zoom ligand, 5\n"
        )
    elif style == "raytrace":
        question = "generate a ray-traced figure of the ligand in sticks against a cartoon protein"
        script   = (
            f"fetch {pdb}, async=0\n"
            f"remove solvent\n"
            f"select ligand, resn {ligand}\n"
            f"show cartoon, polymer\n"
            f"show sticks, ligand\n"
            f"set ray_shadows, 0\n"
            f"set antialias, 2\n"
            f"center ligand\n"
            f"zoom ligand, 5\n"
            f"ray 1200, 900\n"
            f"png {pdb}_binding_site.png, dpi=300\n"
        )
    else:
        question = "measure the distance from the ligand to residue 25 of chain A"
        script   = (
            f"fetch {pdb}, async=0\n"
            f"remove solvent\n"
            f"select ligand, resn {ligand}\n"
            f"select cat_res, resi 25 and chain A\n"
            f"distance d1, ligand, cat_res\n"
            f"show sticks, ligand or cat_res\n"
            f"center ligand\n"
            f"zoom ligand, 6\n"
        )

    answer = (
        f"I will generate a PyMOL command script for {pdb} ({target}).\n\n"
        "```python\n"
        f"# Save as script.pml and run: pymol -c script.pml  (or File > Run Script in the GUI)\n"
        f"{script}"
        "```\n\n"
        f"The script fetches {pdb} from the PDB, removes solvent, selects the ligand "
        f"({ligand}), and {question}. Adjust residue numbers and chain IDs to match your system."
    )
    return _record(
        f"Write a PyMOL script for {pdb} ({target}) to {question}.",
        answer,
    )


# ---------------------------------------------------------------------------
# Behaviour class 5: Structural / docking interpretation
# ---------------------------------------------------------------------------

def plip_interpretation_example():
    ctx   = random.choice(PYMOL_CONTEXTS)
    n_hb  = random.randint(1, 3)
    n_hyd = random.randint(2, 5)
    n_sb  = random.randint(0, 1)

    hb_text  = f"Hydrogen bonds ({n_hb}): ASP25:OD1 - ligand:N-H (2.8 Å)"
    hb_text += (", GLY27:O - ligand:O-H (3.0 Å)" if n_hb > 1 else "")
    hb_text += (", THR26:N-H - ligand:C=O (2.9 Å)" if n_hb > 2 else "")
    hyd_text = f"Hydrophobic contacts ({n_hyd}): VAL82, ILE50, LEU23, PRO81" + (", ALA28" if n_hyd > 4 else "")
    sb_text  = f"Salt bridge (1): ARG8 - ligand:COO- (3.2 Å)." if n_sb else ""
    summary  = f"{hb_text}. {hyd_text}. {sb_text}"

    answer = (
        f"The PLIP output for {ctx['pdb']} ({ctx['target']}) shows:\n\n"
        f"- **{n_hb} hydrogen bond(s):** The H-bond(s) to ASP25 are likely essential for potency. "
        "Loss of the donor (e.g. N-methylation) typically costs 1 to 2 kcal/mol per bond. "
        "These interactions should be preserved in any analogue series.\n"
        f"- **{n_hyd} hydrophobic contacts (VAL82, ILE50, LEU23):** The lipophilic pocket is "
        "well-packed. Adding bulk here risks steric clash; stripping substituents loses van der "
        "Waals contacts. Fluorination or chlorination at the periphery can maintain volume with "
        "lower lipophilicity.\n"
        + (f"- **Salt bridge to ARG8:** A strong, geometry-sensitive anchor. The complementary "
           "carboxylate (or bioisostere: tetrazole, acylsulfonamide) is likely essential and "
           "should be preserved.\n" if n_sb else "")
        + "\n**SAR implication:** maintain the H-bond donor(s) and the hydrophobic core in "
        "analogues. Introduce polarity only at solvent-exposed positions. "
        "Use PLIP on each new docked pose to confirm the interaction pattern is conserved."
    )
    return _record(
        (
            f"Here is a PLIP interaction summary for a ligand docked into {ctx['pdb']} ({ctx['target']}):\n"
            f"{summary}\n"
            "Interpret these interactions and give SAR implications."
        ),
        answer,
    )


# ---------------------------------------------------------------------------
# Behaviour class 6: Medchem reasoning — matched molecular pair / logP SAR
# ---------------------------------------------------------------------------

def medchem_mmp_example(smi_a: str, smi_b: str):
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None:
        return None
    canon_a = Chem.MolToSmiles(mol_a)
    canon_b = Chem.MolToSmiles(mol_b)
    logp_a  = round(Descriptors.MolLogP(mol_a), 2)
    logp_b  = round(Descriptors.MolLogP(mol_b), 2)
    delta   = round(logp_b - logp_a, 2)
    mw_a    = round(Descriptors.MolWt(mol_a), 1)
    mw_b    = round(Descriptors.MolWt(mol_b), 1)
    direction = "increases" if delta > 0 else "decreases"
    answer = (
        "I will compute logP and MW for both compounds with RDKit to quantify the substituent effect.\n\n"
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors\n"
        f"mol_a = Chem.MolFromSmiles('{smi_a}')\n"
        f"mol_b = Chem.MolFromSmiles('{smi_b}')\n"
        "for mol in (mol_a, mol_b):\n"
        "    print(Chem.MolToSmiles(mol),\n"
        "          round(Descriptors.MolWt(mol), 1),\n"
        "          round(Descriptors.MolLogP(mol), 2))\n"
        "```\n\n"
        f"A: `{canon_a}` — MW {mw_a}, logP {logp_a}.\n"
        f"B: `{canon_b}` — MW {mw_b}, logP {logp_b}.\n\n"
        f"The structural change {direction} logP by {abs(delta):.2f} units (ΔlogP = {delta:+.2f}). "
        "In a matched molecular pair analysis, this delta captures the substituent contribution. "
        "If cell permeability or plasma protein binding tracks logP in your series, this quantifies "
        "the trade-off between the two modifications. "
        "Prioritise the analogue whose logP keeps the series within the Lipinski window (< 5) "
        "while maintaining target engagement."
    )
    return _record(
        (
            f"Compare these two analogues as a matched molecular pair and reason about "
            f"the SAR implication of the structural change:\nA: {smi_a}\nB: {smi_b}"
        ),
        answer,
    )


# ---------------------------------------------------------------------------
# Shared record builder
# ---------------------------------------------------------------------------

def _record(user_content: str, assistant_content: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ---------------------------------------------------------------------------
# Validation: SMILES parsing + code block execution
# ---------------------------------------------------------------------------

def validate(example: dict) -> bool:
    """Reject if any in-code SMILES fails to parse, or any RDKit code block errors."""
    for msg in example["messages"]:
        if msg["role"] != "assistant":
            continue
        for block in CODE_BLOCK.findall(msg["content"]):
            for smi in re.findall(r"MolFromSmiles\(['\"]([^'\"]+)['\"]\)", block):
                if Chem.MolFromSmiles(smi) is None:
                    return False
            if "rdkit" not in block.lower():
                continue
            if _exec_block(block).startswith("[error]"):
                return False
    return True


def _exec_block(code: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(code)
        path = fh.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=15,
        )
        return proc.stdout if proc.returncode == 0 else f"[error]\n{proc.stderr}"
    except subprocess.TimeoutExpired:
        return "[error] timeout"


# ---------------------------------------------------------------------------
# Generator registry + main
# ---------------------------------------------------------------------------

def _make_generators():
    def _subst():
        q, name = random.choice(SUBSTRUCTURE_QUERIES)
        t = random.choice(SEED_SMILES)
        return substructure_example(q, name, t)

    return [
        ("smiles_literacy",      lambda: smiles_canonicalise_example(random.choice(SEED_SMILES))),
        ("property_reasoning",   lambda: druglikeness_example(random.choice(SEED_SMILES))),
        ("murcko_scaffold",      lambda: murcko_scaffold_example(random.choice(SEED_SMILES))),
        ("tanimoto",             lambda: tanimoto_example(*random.sample(SEED_SMILES, 2))),
        ("substructure",         _subst),
        ("pymol_script",         pymol_example),
        ("plip_interpretation",  plip_interpretation_example),
        ("medchem_mmp",          lambda: medchem_mmp_example(*random.sample(SEED_SMILES, 2))),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",  type=Path, default=Path("data/sft"))
    ap.add_argument("--n",    type=int,  default=1500)
    ap.add_argument("--seed", type=int,  default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    generators = _make_generators()
    examples, skipped = [], 0
    idx = 0
    max_attempts = args.n * 12

    print(f"Generating {args.n} validated SFT examples across {len(generators)} behaviour classes ...")
    while len(examples) < args.n and idx < max_attempts:
        name, gen = generators[idx % len(generators)]
        idx += 1
        ex = gen()
        if ex is None:
            skipped += 1
            continue
        if validate(ex):
            examples.append(ex)
            if len(examples) % 100 == 0:
                print(f"  {len(examples)}/{args.n} validated")
        else:
            skipped += 1

    if len(examples) < args.n:
        print(f"Warning: only {len(examples)}/{args.n} examples validated ({skipped} rejected).")

    random.shuffle(examples)
    n      = len(examples)
    train  = examples[:int(.8 * n)]
    valid  = examples[int(.8 * n):int(.9 * n)]
    test   = examples[int(.9 * n):]

    for split_name, split in [("train", train), ("valid", valid), ("test", test)]:
        out_path = args.out / f"{split_name}.jsonl"
        with open(out_path, "w") as fh:
            for ex in split:
                fh.write(json.dumps(ex) + "\n")

    print(f"\nWrote {len(train)} train / {len(valid)} valid / {len(test)} test → {args.out}")
    print(f"({skipped} examples rejected during validation)")
    print("Next: python scripts/train_qlora.py --config config/train_config.yaml")


if __name__ == "__main__":
    main()
