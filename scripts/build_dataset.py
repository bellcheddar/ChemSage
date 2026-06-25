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
    # Approved drugs from corpus (diverse scaffolds)
    "CN1CCC[C@H]1c1cccnc1",                            # nicotine
    "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",     # ciprofloxacin
    "CC(C)NCC(O)COc1cccc2ccccc12",                      # propranolol
    "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",             # diazepam
    "COCCc1ccc(OCC(O)CNC(C)C)cc1",                     # metoprolol
    "CC(C)NCC(O)c1ccc(O)c(O)c1",                       # isoproterenol
    "O=C(O)c1ccccc1O",                                  # salicylic acid
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                      # ibuprofen (dup removed in dedup)
    "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1", # indomethacin
    "COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC",   # prazosin
    "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1",               # phenytoin
    "CC(=O)Nc1nnc(S(N)(=O)=O)s1",                      # acetazolamide
    "Nc1ccc(S(N)(=O)=O)cc1",                            # sulfanilamide
    "COc1ccc(Cc2cnc(N)nc2N)cc(OC)c1OC",                # trimethoprim
    "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",                  # atenolol
    "CC(C)NCC(O)COc1ccc(CCOCC2CC2)cc1",                # betaxolol
    "O=c1ccc2ccccc2o1",                                  # coumarin
    "COc1cc(C=O)ccc1O",                                  # vanillin
    "CC(N)Cc1ccccc1",                                    # amphetamine
    "CC(=O)Oc1ccc(O)cc1",                               # hydroquinone monoacetate
    "OC(=O)/C=C/c1ccccc1",                              # cinnamic acid
    "NC(=O)c1ccc(O)cc1",                                # 4-hydroxybenzamide
    "CC(=O)c1ccc(O)cc1",                                # 4-hydroxyacetophenone
    "OC(=O)CCc1ccccc1",                                  # hydrocinnamic acid
    "Oc1ccc(Cl)cc1",                                     # 4-chlorophenol
    "CC(O)c1ccccc1",                                     # 1-phenylethanol
]

INVALID_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)X",     # invalid atom X
    "CC((=O)O",                    # unbalanced parens
    "c1ccccc",                     # unclosed ring
    "C=C=C=C=C=C=C=C=C=C=C",      # unusual but maybe valid
    "not_a_smiles_at_all",         # garbage
    "[Cu+3]INVALID",               # partial garbage
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
# Behaviour class 7: Single property computation (code-first)
# ---------------------------------------------------------------------------

_PROPERTY_TEMPLATES = [
    {
        "prop": "molecular weight",
        "code": "round(Descriptors.MolWt(mol), 2)",
        "imports": "from rdkit import Chem\nfrom rdkit.Chem import Descriptors",
        "fmt": lambda v: f"{v} Da",
        "compute": lambda mol: round(Descriptors.MolWt(mol), 2),
    },
    {
        "prop": "logP",
        "code": "round(Descriptors.MolLogP(mol), 2)",
        "imports": "from rdkit import Chem\nfrom rdkit.Chem import Descriptors",
        "fmt": lambda v: str(v),
        "compute": lambda mol: round(Descriptors.MolLogP(mol), 2),
    },
    {
        "prop": "TPSA",
        "code": "round(rdMolDescriptors.CalcTPSA(mol), 1)",
        "imports": "from rdkit import Chem\nfrom rdkit.Chem import rdMolDescriptors",
        "fmt": lambda v: f"{v} A^2",
        "compute": lambda mol: round(rdMolDescriptors.CalcTPSA(mol), 1),
    },
    {
        "prop": "number of hydrogen bond donors",
        "code": "Lipinski.NumHDonors(mol)",
        "imports": "from rdkit import Chem\nfrom rdkit.Chem import Lipinski",
        "fmt": lambda v: str(v),
        "compute": lambda mol: Lipinski.NumHDonors(mol),
    },
    {
        "prop": "number of hydrogen bond acceptors",
        "code": "Lipinski.NumHAcceptors(mol)",
        "imports": "from rdkit import Chem\nfrom rdkit.Chem import Lipinski",
        "fmt": lambda v: str(v),
        "compute": lambda mol: Lipinski.NumHAcceptors(mol),
    },
    {
        "prop": "number of rotatable bonds",
        "code": "rdMolDescriptors.CalcNumRotatableBonds(mol)",
        "imports": "from rdkit import Chem\nfrom rdkit.Chem import rdMolDescriptors",
        "fmt": lambda v: str(v),
        "compute": lambda mol: rdMolDescriptors.CalcNumRotatableBonds(mol),
    },
]


def single_property_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    tmpl = random.choice(_PROPERTY_TEMPLATES)
    value = tmpl["compute"](mol)
    answer = (
        f"```python\n"
        f"{tmpl['imports']}\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        f"print({tmpl['code']})\n"
        f"```\n\n"
        f"The {tmpl['prop']} is {tmpl['fmt'](value)}."
    )
    questions = [
        f"What is the {tmpl['prop']} of {smiles}?",
        f"Calculate the {tmpl['prop']} for this molecule: {smiles}",
        f"Compute the {tmpl['prop']} of {smiles} using RDKit.",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# Behaviour class 8: QED drug-likeness score
# ---------------------------------------------------------------------------

def qed_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        from rdkit.Chem.QED import qed
        score = round(qed(mol), 3)
    except Exception:
        return None
    interp = (
        "highly drug-like (QED > 0.67)" if score > 0.67 else
        "moderately drug-like (0.33 < QED < 0.67)" if score > 0.33 else
        "poorly drug-like (QED < 0.33)"
    )
    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem.QED import qed\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "print(round(qed(mol), 3))\n"
        "```\n\n"
        f"QED = {score}. The compound is {interp}. "
        "QED (Quantitative Estimate of Drug-likeness) integrates MW, logP, HBD, HBA, PSA, "
        "rotatable bonds, aromatic rings, and alerts into a single 0-1 score."
    )
    return _record(
        f"What is the QED score of {smiles}?",
        answer,
    )


# ---------------------------------------------------------------------------
# Behaviour class 9: Molecular formula and atom counts
# ---------------------------------------------------------------------------

def molecular_formula_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    formula = rdMolDescriptors.CalcMolFormula(mol)
    n_heavy = mol.GetNumHeavyAtoms()
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_arom  = rdMolDescriptors.CalcNumAromaticRings(mol)
    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import rdMolDescriptors\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "print(rdMolDescriptors.CalcMolFormula(mol))\n"
        "print('Heavy atoms:', mol.GetNumHeavyAtoms())\n"
        "print('Rings:', rdMolDescriptors.CalcNumRings(mol))\n"
        "print('Aromatic rings:', rdMolDescriptors.CalcNumAromaticRings(mol))\n"
        "```\n\n"
        f"Molecular formula: {formula}. "
        f"{n_heavy} heavy atoms, {n_rings} ring(s) ({n_arom} aromatic)."
    )
    questions = [
        f"What is the molecular formula of {smiles}?",
        f"How many rings does {smiles} have?",
        f"Give me the atom counts for {smiles}.",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# Behaviour class 10: SMILES validation (valid and invalid)
# ---------------------------------------------------------------------------

def smiles_validation_example():
    if random.random() < 0.5:
        smi = random.choice(INVALID_SMILES)
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return None
        answer = (
            "```python\n"
            "from rdkit import Chem\n"
            f"mol = Chem.MolFromSmiles('{smi}')\n"
            "print('Valid' if mol is not None else 'Invalid')\n"
            "```\n\n"
            f"The SMILES `{smi}` is **invalid**. RDKit cannot parse it. "
            "Check for unbalanced parentheses, unclosed rings, or invalid atom symbols."
        )
    else:
        smi = random.choice(SEED_SMILES)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        canon = Chem.MolToSmiles(mol)
        answer = (
            "```python\n"
            "from rdkit import Chem\n"
            f"mol = Chem.MolFromSmiles('{smi}')\n"
            "print('Valid' if mol is not None else 'Invalid')\n"
            "if mol:\n"
            "    print(Chem.MolToSmiles(mol))\n"
            "```\n\n"
            f"The SMILES `{smi}` is **valid**. Canonical form: `{canon}`."
        )
    return _record(f"Is this a valid SMILES string? {smi}", answer)


# ---------------------------------------------------------------------------
# Behaviour class 11: Full property profile (code-heavy)
# ---------------------------------------------------------------------------

def full_profile_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canon = Chem.MolToSmiles(mol)
    mw   = round(Descriptors.MolWt(mol), 2)
    logp = round(Descriptors.MolLogP(mol), 2)
    hbd  = Lipinski.NumHDonors(mol)
    hba  = Lipinski.NumHAcceptors(mol)
    tpsa = round(rdMolDescriptors.CalcTPSA(mol), 1)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    try:
        from rdkit.Chem.QED import qed
        qed_val = round(qed(mol), 3)
    except Exception:
        qed_val = None
    qed_str = f", QED {qed_val}" if qed_val else ""
    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors\n"
        "from rdkit.Chem.QED import qed\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "print('SMILES:', Chem.MolToSmiles(mol))\n"
        "print('Formula:', rdMolDescriptors.CalcMolFormula(mol))\n"
        "print('MW:', round(Descriptors.MolWt(mol), 2))\n"
        "print('logP:', round(Descriptors.MolLogP(mol), 2))\n"
        "print('HBD:', Lipinski.NumHDonors(mol))\n"
        "print('HBA:', Lipinski.NumHAcceptors(mol))\n"
        "print('TPSA:', round(rdMolDescriptors.CalcTPSA(mol), 1))\n"
        "print('RotBonds:', rdMolDescriptors.CalcNumRotatableBonds(mol))\n"
        "print('Rings:', rdMolDescriptors.CalcNumRings(mol))\n"
        "print('QED:', round(qed(mol), 3))\n"
        "```\n\n"
        f"**{canon}**\n"
        f"Formula: {formula}, MW: {mw}, logP: {logp}, "
        f"HBD: {hbd}, HBA: {hba}, TPSA: {tpsa}, "
        f"RotBonds: {rotb}, Rings: {n_rings}{qed_str}."
    )
    questions = [
        f"Give me the full physicochemical profile of {smiles}.",
        f"Compute all drug-likeness properties for {smiles}.",
        f"Run a complete property calculation on {smiles}.",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# Behaviour class 12: Multi-property comparison (two molecules)
# ---------------------------------------------------------------------------

def property_comparison_example(smi_a: str, smi_b: str):
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None:
        return None
    canon_a, canon_b = Chem.MolToSmiles(mol_a), Chem.MolToSmiles(mol_b)
    mw_a, mw_b   = round(Descriptors.MolWt(mol_a), 1), round(Descriptors.MolWt(mol_b), 1)
    logp_a, logp_b = round(Descriptors.MolLogP(mol_a), 2), round(Descriptors.MolLogP(mol_b), 2)
    tpsa_a, tpsa_b = round(rdMolDescriptors.CalcTPSA(mol_a), 1), round(rdMolDescriptors.CalcTPSA(mol_b), 1)
    hbd_a, hbd_b = Lipinski.NumHDonors(mol_a), Lipinski.NumHDonors(mol_b)
    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors\n"
        f"smi_a = '{smi_a}'\n"
        f"smi_b = '{smi_b}'\n"
        "for smi in (smi_a, smi_b):\n"
        "    mol = Chem.MolFromSmiles(smi)\n"
        "    print(Chem.MolToSmiles(mol),\n"
        "          'MW:', round(Descriptors.MolWt(mol), 1),\n"
        "          'logP:', round(Descriptors.MolLogP(mol), 2),\n"
        "          'TPSA:', round(rdMolDescriptors.CalcTPSA(mol), 1),\n"
        "          'HBD:', Lipinski.NumHDonors(mol))\n"
        "```\n\n"
        f"| Property | A | B |\n"
        f"|----------|---|---|\n"
        f"| SMILES | `{canon_a}` | `{canon_b}` |\n"
        f"| MW | {mw_a} | {mw_b} |\n"
        f"| logP | {logp_a} | {logp_b} |\n"
        f"| TPSA | {tpsa_a} | {tpsa_b} |\n"
        f"| HBD | {hbd_a} | {hbd_b} |\n\n"
        f"Compound {'A' if logp_a < logp_b else 'B'} is more polar (lower logP), "
        f"which generally favours aqueous solubility and renal clearance. "
        f"Compound {'A' if tpsa_a < tpsa_b else 'B'} has lower TPSA, "
        f"suggesting better passive permeability."
    )
    return _record(
        f"Compare the physicochemical properties of these two compounds:\nA: {smi_a}\nB: {smi_b}",
        answer,
    )


# ---------------------------------------------------------------------------
# Behaviour class 13: Corpus-aware drug lookup
# ---------------------------------------------------------------------------

_CORPUS_DRUGS = None

def _load_corpus_drugs():
    global _CORPUS_DRUGS
    if _CORPUS_DRUGS is not None:
        return _CORPUS_DRUGS
    import pandas as pd
    path = Path("data/corpus/chembl/approved_drugs.csv")
    if not path.exists():
        _CORPUS_DRUGS = []
        return _CORPUS_DRUGS
    df = pd.read_csv(path, dtype=str, low_memory=False).fillna("")
    drugs = []
    for _, row in df.iterrows():
        smi = row.get("smiles", "")
        name = row.get("name", "")
        if smi and name and 10 < len(smi) < 120:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                drugs.append({"smiles": smi, "name": name})
    _CORPUS_DRUGS = drugs
    return drugs


def corpus_drug_example():
    drugs = _load_corpus_drugs()
    if not drugs:
        return None
    drug = random.choice(drugs)
    smi, name = drug["smiles"], drug["name"]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    canon = Chem.MolToSmiles(mol)
    mw   = round(Descriptors.MolWt(mol), 1)
    logp = round(Descriptors.MolLogP(mol), 2)
    hbd  = Lipinski.NumHDonors(mol)
    hba  = Lipinski.NumHAcceptors(mol)
    tpsa = round(rdMolDescriptors.CalcTPSA(mol), 1)
    lip  = mw < 500 and logp < 5 and hbd <= 5 and hba <= 10
    verdict = "passes" if lip else "fails"
    answer = (
        f"{name} has SMILES: `{canon}`.\n\n"
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors\n"
        f"mol = Chem.MolFromSmiles('{smi}')\n"
        "print('MW:', round(Descriptors.MolWt(mol), 1))\n"
        "print('logP:', round(Descriptors.MolLogP(mol), 2))\n"
        "print('HBD:', Lipinski.NumHDonors(mol))\n"
        "print('HBA:', Lipinski.NumHAcceptors(mol))\n"
        "print('TPSA:', round(rdMolDescriptors.CalcTPSA(mol), 1))\n"
        "```\n\n"
        f"MW: {mw}, logP: {logp}, HBD: {hbd}, HBA: {hba}, TPSA: {tpsa}. "
        f"The compound {verdict} Lipinski's Rule of Five."
    )
    questions = [
        f"What are the properties of {name}?",
        f"Compute the physicochemical properties of {name} (SMILES: {smi}).",
        f"Is {name} drug-like? Calculate its Lipinski properties.",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# Behaviour class 14: Murcko scaffold with SAR context
# ---------------------------------------------------------------------------

def murcko_with_sar_example(smi_a: str, smi_b: str):
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None:
        return None
    try:
        scaf_a = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol_a))
        scaf_b = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol_b))
    except Exception:
        return None
    if not scaf_a or not scaf_b:
        return None
    same = scaf_a == scaf_b
    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem.Scaffolds import MurckoScaffold\n"
        f"mol_a = Chem.MolFromSmiles('{smi_a}')\n"
        f"mol_b = Chem.MolFromSmiles('{smi_b}')\n"
        "scaf_a = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol_a))\n"
        "scaf_b = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol_b))\n"
        "print('A scaffold:', scaf_a)\n"
        "print('B scaffold:', scaf_b)\n"
        "print('Same scaffold:', scaf_a == scaf_b)\n"
        "```\n\n"
        f"A scaffold: `{scaf_a}`\n"
        f"B scaffold: `{scaf_b}`\n\n"
        + (f"Both compounds share the same Murcko scaffold, making them part of the same "
           "structural series. Differences are in peripheral substituents only, which is "
           "ideal for matched molecular pair analysis."
           if same else
           f"The compounds have different Murcko scaffolds, indicating distinct chemical "
           "series. A scaffold hop between them could broaden IP space or improve selectivity.")
    )
    return _record(
        f"Do these two compounds share the same scaffold?\nA: {smi_a}\nB: {smi_b}",
        answer,
    )


# ---------------------------------------------------------------------------
# Behaviour class 15: PyMOL selection algebra
# ---------------------------------------------------------------------------

_PYMOL_SELECTION_EXAMPLES = [
    {
        "intent": "select all residues within 4 angstroms of the ligand",
        "selection": "select near_lig, byres (all within 4 of organic)",
        "explanation": "`byres` expands atom-level hits to full residues. `organic` selects non-polymer ligands. Adjust the distance cutoff for your pocket.",
    },
    {
        "intent": "select chain A only",
        "selection": "select chainA, chain A",
        "explanation": "Selects all atoms in chain A. Use `chain A+B` for multiple chains.",
    },
    {
        "intent": "select all aromatic residues (Phe, Tyr, Trp, His) in the binding site",
        "selection": "select aromatic_bs, (resn PHE+TYR+TRP+HIS) and (byres (all within 5 of organic))",
        "explanation": "Combines residue-name selection with proximity to the ligand.",
    },
    {
        "intent": "select protein backbone atoms only",
        "selection": "select bb, name CA+C+N+O and polymer.protein",
        "explanation": "`name` selects by atom name, `polymer.protein` restricts to protein chains.",
    },
    {
        "intent": "select water molecules near the binding site",
        "selection": "select waters_near, resn HOH and (all within 3.5 of organic)",
        "explanation": "Finds crystallographic waters that may mediate hydrogen bonds between protein and ligand.",
    },
    {
        "intent": "select all beta-sheet residues",
        "selection": "select sheets, ss s",
        "explanation": "`ss s` selects by secondary structure (s=sheet, h=helix, l=loop). Requires DSSP assignment (`dss` command).",
    },
    {
        "intent": "select all atoms in residues 50-100 of chain B",
        "selection": "select region, chain B and resi 50-100",
        "explanation": "`resi` selects by residue number. Use `resi 50+75+100` for non-contiguous residues.",
    },
]


def pymol_selection_example():
    ex = random.choice(_PYMOL_SELECTION_EXAMPLES)
    ctx = random.choice(PYMOL_CONTEXTS)
    answer = (
        f"```python\n"
        f"# PyMOL selection for {ctx['pdb']} ({ctx['target']})\n"
        f"fetch {ctx['pdb']}, async=0\n"
        f"remove solvent\n"
        f"{ex['selection']}\n"
        f"show sticks, {ex['selection'].split(',')[0].split()[-1]}\n"
        "```\n\n"
        f"{ex['explanation']}"
    )
    return _record(
        f"Write a PyMOL selection to {ex['intent']} in {ctx['pdb']} ({ctx['target']}).",
        answer,
    )


# ---------------------------------------------------------------------------
# Behaviour class 16: PyMOL visualization recipes
# ---------------------------------------------------------------------------

_PYMOL_VIZ_RECIPES = [
    {
        "intent": "show the electrostatic surface of the binding pocket",
        "script": (
            "fetch {pdb}, async=0\n"
            "remove solvent\n"
            "select ligand, resn {ligand}\n"
            "select pocket, byres (polymer within 5 of ligand)\n"
            "hide everything\n"
            "show cartoon, polymer\n"
            "show surface, pocket\n"
            "set surface_color, white, pocket\n"
            "util.cbaw pocket\n"
            "show sticks, ligand\n"
            "util.cbag ligand\n"
            "center ligand\n"
            "zoom ligand, 8\n"
        ),
        "notes": "The surface shows shape complementarity between the ligand and pocket. White surface with colored sticks highlights buried vs exposed regions.",
    },
    {
        "intent": "align two structures and show RMSD",
        "script": (
            "fetch {pdb}, async=0\n"
            "fetch 1HSG, async=0\n"
            "align {pdb}, 1HSG\n"
            "show cartoon\n"
            "color cyan, {pdb}\n"
            "color salmon, 1HSG\n"
            "center\n"
            "zoom vis, 5\n"
        ),
        "notes": "The `align` command performs sequence-dependent structural alignment and reports the RMSD. Use `super` for sequence-independent alignment (useful for comparing homologues).",
    },
    {
        "intent": "show symmetry mates in the crystal packing",
        "script": (
            "fetch {pdb}, async=0\n"
            "remove solvent\n"
            "symexp sym, {pdb}, all, 5\n"
            "show cartoon\n"
            "color grey80, sym*\n"
            "color cyan, {pdb}\n"
            "center\n"
        ),
        "notes": "`symexp` generates symmetry-related copies within the specified radius (5 A). Useful for checking crystal contacts that might distort the binding site.",
    },
    {
        "intent": "colour the protein by B-factor to show flexibility",
        "script": (
            "fetch {pdb}, async=0\n"
            "remove solvent\n"
            "show cartoon\n"
            "spectrum b, blue_white_red\n"
            "set cartoon_putty_scale_min, auto\n"
            "set cartoon_putty_scale_max, auto\n"
            "cartoon putty\n"
            "center\n"
        ),
        "notes": "B-factors (temperature factors) indicate atomic displacement. High B-factors (red) suggest flexible or disordered regions. Putty representation scales the cartoon tube width by B-factor.",
    },
    {
        "intent": "create a publication-quality figure with transparent surface",
        "script": (
            "fetch {pdb}, async=0\n"
            "remove solvent\n"
            "select ligand, resn {ligand}\n"
            "bg_color white\n"
            "hide everything\n"
            "show cartoon, polymer\n"
            "color grey90, polymer\n"
            "show sticks, ligand\n"
            "util.cbag ligand\n"
            "show surface, polymer\n"
            "set transparency, 0.7\n"
            "set surface_color, white\n"
            "center ligand\n"
            "zoom ligand, 6\n"
            "set ray_opaque_background, 1\n"
            "ray 2400, 1800\n"
            "png {pdb}_figure.png, dpi=300\n"
        ),
        "notes": "Transparent surface reveals the ligand inside the pocket. `ray` renders at high resolution for publication.",
    },
]


def pymol_viz_example():
    recipe = random.choice(_PYMOL_VIZ_RECIPES)
    ctx = random.choice(PYMOL_CONTEXTS)
    script = recipe["script"].replace("{pdb}", ctx["pdb"]).replace("{ligand}", ctx["ligand"])
    answer = (
        f"```python\n"
        f"# PyMOL script for {ctx['pdb']} ({ctx['target']})\n"
        f"{script}"
        "```\n\n"
        f"{recipe['notes']}"
    )
    return _record(
        f"Write a PyMOL script to {recipe['intent']} for {ctx['pdb']} ({ctx['target']}).",
        answer,
    )


# ---------------------------------------------------------------------------
# Behaviour class 17: PLIP interaction analysis
# ---------------------------------------------------------------------------

_PLIP_INTERACTION_TYPES = [
    {"type": "hydrogen bond", "residues": "ASP25, GLY27, THR26", "distances": "2.8, 3.0, 2.9", "angles": "165, 155, 170", "note": "H-bonds to catalytic residues are typically essential for potency. Loss of a donor (N-methylation) costs 1-2 kcal/mol."},
    {"type": "hydrophobic contact", "residues": "VAL82, ILE50, LEU23, PRO81, ALA28", "distances": "3.6, 3.8, 3.5, 3.9, 4.0", "angles": "", "note": "The lipophilic pocket is well-packed. Adding bulk risks steric clash; fluorination can maintain volume with lower lipophilicity."},
    {"type": "pi-stacking", "residues": "PHE53, TRP84", "distances": "3.8, 4.1", "angles": "15, 8", "note": "Parallel pi-stacking with offset. The angle < 30 degrees indicates parallel (not T-shaped). Replacing the aromatic with cyclohexyl would lose this interaction."},
    {"type": "salt bridge", "residues": "ARG8, LYS103", "distances": "3.2, 3.5", "angles": "", "note": "Strong, geometry-sensitive anchors. The complementary carboxylate (or bioisostere: tetrazole, acylsulfonamide) is likely essential."},
    {"type": "water bridge", "residues": "ASN155 via HOH301", "distances": "2.7 (lig-water), 2.9 (water-protein)", "angles": "", "note": "Water-mediated H-bond. Displacing this water with a well-placed functional group could improve potency (entropy gain from water release)."},
    {"type": "halogen bond", "residues": "GLU166 backbone O", "distances": "3.1", "angles": "165", "note": "C-Cl...O halogen bond. The angle ~165 degrees is near-ideal (sigma-hole geometry). Replacing Cl with Br strengthens this interaction."},
]


def plip_analysis_example():
    ctx = random.choice(PYMOL_CONTEXTS)
    interactions = random.sample(_PLIP_INTERACTION_TYPES, k=min(3, len(_PLIP_INTERACTION_TYPES)))
    plip_output = ""
    analysis = ""
    for ix in interactions:
        plip_output += f"- {ix['type']}: {ix['residues']} ({ix['distances']} A)\n"
        analysis += f"- **{ix['type'].title()} ({ix['residues']}):** {ix['note']}\n"

    answer = (
        f"Running PLIP on {ctx['pdb']}:\n\n"
        "```bash\n"
        f"plip -f {ctx['pdb']}.pdb -o plip_output/ -x\n"
        "```\n\n"
        f"PLIP detected the following interactions with {ctx['ligand']}:\n"
        f"{plip_output}\n"
        f"**Analysis:**\n{analysis}\n"
        f"**SAR recommendation:** Preserve the key H-bond donors/acceptors and the hydrophobic core. "
        "Introduce polarity only at solvent-exposed positions."
    )
    return _record(
        f"Run PLIP on {ctx['pdb']} ({ctx['target']}) and interpret the interactions with {ctx['ligand']}.",
        answer,
    )


# ---------------------------------------------------------------------------
# Behaviour class 18: PDB file format knowledge
# ---------------------------------------------------------------------------

_PDB_FORMAT_QUESTIONS = [
    {
        "q": "What do the columns in a PDB ATOM record mean?",
        "a": (
            "A PDB ATOM record has fixed-width fields:\n\n"
            "```\n"
            "ATOM      1  N   ALA A   1      27.340  24.430   2.614  1.00  9.67           N\n"
            "│         │  │   │   │   │      │       │       │      │     │              │\n"
            "record  serial name res ch resi    x       y       z    occ   B-factor     element\n"
            "```\n\n"
            "- **Columns 1-6:** Record type (ATOM or HETATM)\n"
            "- **Columns 7-11:** Atom serial number\n"
            "- **Columns 13-16:** Atom name (CA, N, O, CB, etc.)\n"
            "- **Columns 17:** Alternate location indicator (blank or A/B for alt conformers)\n"
            "- **Columns 18-20:** Residue name (ALA, GLY, etc.)\n"
            "- **Column 22:** Chain ID (A, B, etc.)\n"
            "- **Columns 23-26:** Residue sequence number\n"
            "- **Columns 31-54:** X, Y, Z coordinates (8.3 format)\n"
            "- **Columns 55-60:** Occupancy (1.00 = fully occupied)\n"
            "- **Columns 61-66:** B-factor (temperature factor)\n"
            "- **Columns 77-78:** Element symbol\n\n"
            "HETATM records use the same format but denote non-polymer atoms (ligands, waters, ions)."
        ),
    },
    {
        "q": "What is the difference between ATOM and HETATM records?",
        "a": (
            "**ATOM** records are for standard polymer residues (amino acids in proteins, nucleotides in DNA/RNA). "
            "**HETATM** records are for everything else: ligands, cofactors, metal ions, modified residues, and water molecules (HOH).\n\n"
            "In PyMOL: `select polymer.protein` gets ATOM records; `select organic` gets non-water HETATM; `select solvent` gets waters.\n\n"
            "In Biopython: `residue.id[0]` returns `' '` for ATOM and `'H_XXX'` for HETATM (where XXX is the residue name)."
        ),
    },
    {
        "q": "How do I handle alternate conformations in a PDB file?",
        "a": (
            "Alternate conformations (altlocs) appear when an atom has multiple positions in the electron density. "
            "Column 17 of the ATOM record contains the altloc indicator (A, B, or blank).\n\n"
            "```python\n"
            "# pdb-tools: keep only conformer A\n"
            "# pdb_selaltloc -A input.pdb > output.pdb\n"
            "\n"
            "# Biopython: filter altlocs\n"
            "from Bio.PDB import PDBParser, PDBIO, Select\n"
            "class AltlocSelect(Select):\n"
            "    def accept_atom(self, atom):\n"
            "        return atom.get_altloc() in (' ', 'A')\n"
            "parser = PDBParser(QUIET=True)\n"
            "structure = parser.get_structure('prot', 'input.pdb')\n"
            "io = PDBIO()\n"
            "io.set_structure(structure)\n"
            "io.save('output.pdb', AltlocSelect())\n"
            "```\n\n"
            "Always resolve altlocs before docking or MD — most tools expect a single conformer."
        ),
    },
    {
        "q": "What does the B-factor tell us about a protein structure?",
        "a": (
            "The B-factor (temperature factor, Debye-Waller factor) quantifies atomic displacement from "
            "the mean position. Units are A^2.\n\n"
            "- **Low B-factor (< 20):** Well-ordered, rigid. The atom position is reliable.\n"
            "- **Medium B-factor (20-40):** Moderate flexibility. Common for surface loops.\n"
            "- **High B-factor (> 40):** Disordered or flexible. Coordinates are less reliable.\n\n"
            "In PyMOL, visualise with: `spectrum b, blue_white_red` (blue=low, red=high).\n\n"
            "For drug design: high B-factors in the binding site suggest an induced-fit mechanism. "
            "Rigid binding sites (low B-factors) are better targets for structure-based virtual screening."
        ),
    },
]


def pdb_format_example():
    ex = random.choice(_PDB_FORMAT_QUESTIONS)
    return _record(ex["q"], ex["a"])


# ---------------------------------------------------------------------------
# Behaviour class 19: pdb-tools commands
# ---------------------------------------------------------------------------

_PDBTOOLS_RECIPES = [
    {"intent": "extract chain A from a PDB file", "cmd": "pdb_selchain -A input.pdb > chainA.pdb", "note": "Use `-A,B` for multiple chains."},
    {"intent": "select only residues 50-150", "cmd": "pdb_selres -50:150 input.pdb > region.pdb", "note": "Useful for extracting a domain or binding site region."},
    {"intent": "remove all water molecules", "cmd": "pdb_delhetatm input.pdb | pdb_keepcoord > no_water.pdb", "note": "Or specifically: `pdb_selres -HOH input.pdb` to select (not remove) waters. Pipe through `pdb_delresname` to remove."},
    {"intent": "renumber residues starting from 1", "cmd": "pdb_reres -1 input.pdb > renumbered.pdb", "note": "Useful when residue numbering has gaps or doesn't start at 1."},
    {"intent": "tidy up a PDB file (fix formatting)", "cmd": "pdb_tidy input.pdb > clean.pdb", "note": "Fixes column alignment, adds TER records between chains, and ensures proper formatting."},
    {"intent": "convert a PDB to a clean format for docking", "cmd": "pdb_selchain -A input.pdb | pdb_delhetatm | pdb_tidy > receptor.pdb", "note": "Pipeline: select chain, remove ligands/waters/ions, tidy formatting. Add hydrogens separately with `reduce` or PDBFixer."},
    {"intent": "split a multi-model NMR structure into individual models", "cmd": "pdb_splitmodel input.pdb", "note": "Creates input_1.pdb, input_2.pdb, etc. Use model 1 for most applications."},
    {"intent": "remove alternate conformations keeping only conformer A", "cmd": "pdb_selaltloc -A input.pdb > single_conf.pdb", "note": "Essential before docking or MD. Most tools cannot handle multiple conformers."},
]


def pdbtools_example():
    recipe = random.choice(_PDBTOOLS_RECIPES)
    ctx = random.choice(PYMOL_CONTEXTS)
    answer = (
        f"```bash\n"
        f"# {recipe['intent']} for {ctx['pdb']}\n"
        f"{recipe['cmd'].replace('input.pdb', ctx['pdb'] + '.pdb')}\n"
        "```\n\n"
        f"{recipe['note']}\n\n"
        "pdb-tools is a collection of command-line utilities that each do one thing. "
        "They can be piped together for complex workflows."
    )
    return _record(
        f"How do I {recipe['intent']}? I have {ctx['pdb']}.pdb.",
        answer,
    )


# ---------------------------------------------------------------------------
# Behaviour class 20: Biopython Bio.PDB patterns
# ---------------------------------------------------------------------------

_BIOPYTHON_RECIPES = [
    {
        "intent": "parse a PDB file and iterate over residues",
        "code": (
            "from Bio.PDB import PDBParser\n"
            "parser = PDBParser(QUIET=True)\n"
            "structure = parser.get_structure('prot', '{pdb}.pdb')\n"
            "model = structure[0]\n"
            "for chain in model:\n"
            "    for residue in chain:\n"
            "        if residue.id[0] == ' ':  # skip HETATM\n"
            "            print(chain.id, residue.get_resname(), residue.id[1])\n"
        ),
        "note": "The SMCRA hierarchy: Structure > Model > Chain > Residue > Atom. `residue.id` is a tuple (hetflag, resseq, icode).",
    },
    {
        "intent": "find all residues within 5 angstroms of a ligand",
        "code": (
            "from Bio.PDB import PDBParser, NeighborSearch\n"
            "parser = PDBParser(QUIET=True)\n"
            "structure = parser.get_structure('prot', '{pdb}.pdb')\n"
            "model = structure[0]\n"
            "atoms = [a for a in model.get_atoms()]\n"
            "ns = NeighborSearch(atoms)\n"
            "# Find ligand center\n"
            "ligand_atoms = [a for r in model.get_residues()\n"
            "                if r.id[0].startswith('H_') for a in r]\n"
            "for la in ligand_atoms:\n"
            "    nearby = ns.search(la.get_vector().get_array(), 5.0, 'R')\n"
            "    for res in nearby:\n"
            "        if res.id[0] == ' ':\n"
            "            print(res.get_resname(), res.id[1])\n"
        ),
        "note": "NeighborSearch uses a KD-tree for fast spatial queries. 'R' returns residue-level results.",
    },
    {
        "intent": "calculate the RMSD between two structures",
        "code": (
            "from Bio.PDB import PDBParser, Superimposer\n"
            "parser = PDBParser(QUIET=True)\n"
            "s1 = parser.get_structure('s1', 'structure1.pdb')\n"
            "s2 = parser.get_structure('s2', 'structure2.pdb')\n"
            "atoms1 = [a for a in s1[0]['A'].get_atoms() if a.name == 'CA']\n"
            "atoms2 = [a for a in s2[0]['A'].get_atoms() if a.name == 'CA']\n"
            "sup = Superimposer()\n"
            "sup.set_atoms(atoms1[:min(len(atoms1), len(atoms2))],\n"
            "              atoms2[:min(len(atoms1), len(atoms2))])\n"
            "print(f'RMSD: {sup.rms:.2f} A')\n"
        ),
        "note": "Superimposer aligns on CA atoms and reports RMSD. For whole-structure alignment, use all backbone atoms (CA, C, N, O).",
    },
]


def biopython_example():
    recipe = random.choice(_BIOPYTHON_RECIPES)
    ctx = random.choice(PYMOL_CONTEXTS)
    code = recipe["code"].replace("{pdb}", ctx["pdb"])
    answer = (
        f"```python\n{code}```\n\n"
        f"{recipe['note']}"
    )
    return _record(
        f"How do I {recipe['intent']} using Biopython? I'm working with {ctx['pdb']} ({ctx['target']}).",
        answer,
    )


# ---------------------------------------------------------------------------
# Behaviour class 21: ChimeraX commands
# ---------------------------------------------------------------------------

_CHIMERAX_RECIPES = [
    {
        "intent": "open a structure and show the binding site",
        "script": "open {pdb}\ndelete solvent\nselect ::{ligand}\nshow sel atoms\nshow sel nearAtoms < 5 atoms\ncolor byhetero\nview sel",
        "note": "ChimeraX atom specification uses `::` for residue name. `nearAtoms < 5` selects atoms within 5 A.",
    },
    {
        "intent": "calculate hydrogen bonds",
        "script": "open {pdb}\ndelete solvent\nhbonds ::{ligand} restrict cross reveal true log true",
        "note": "`hbonds` with `restrict cross` shows only inter-molecular H-bonds (ligand-protein). `log true` writes distances and angles to the log.",
    },
    {
        "intent": "find clashes between the ligand and protein",
        "script": "open {pdb}\ndelete solvent\nclashes ::{ligand} restrict cross reveal true\nlog true",
        "note": "Clashes indicate steric overlap (van der Waals radii). Useful for checking docked poses or mutations.",
    },
    {
        "intent": "align two structures using matchmaker",
        "script": "open {pdb}\nopen 1HSG\nmatchmaker #2 to #1\ncolor #1 cornflower blue\ncolor #2 salmon",
        "note": "`matchmaker` performs sequence-dependent structural alignment. Reports RMSD and number of aligned residues.",
    },
    {
        "intent": "colour by conservation or sequence variability",
        "script": "open {pdb}\ndelete solvent\ncolor byattr bfactor palette blue:white:red\ncartoon style width 2\nview",
        "note": "Uses B-factor as a proxy for flexibility. For true conservation colouring, load a sequence alignment first.",
    },
]


def chimerax_example():
    recipe = random.choice(_CHIMERAX_RECIPES)
    ctx = random.choice(PYMOL_CONTEXTS)
    script = recipe["script"].replace("{pdb}", ctx["pdb"]).replace("{ligand}", ctx["ligand"])
    answer = (
        f"```\n"
        f"# ChimeraX commands for {ctx['pdb']} ({ctx['target']})\n"
        f"{script}\n"
        "```\n\n"
        f"{recipe['note']}"
    )
    return _record(
        f"Write ChimeraX commands to {recipe['intent']} for {ctx['pdb']} ({ctx['target']}).",
        answer,
    )


# ---------------------------------------------------------------------------
# Behaviour class 22: Gemmi patterns
# ---------------------------------------------------------------------------

_GEMMI_RECIPES = [
    {
        "intent": "parse an mmCIF file and list all chains",
        "code": (
            "import gemmi\n"
            "doc = gemmi.cif.read('{pdb}.cif')\n"
            "st = gemmi.make_structure_from_block(doc[0])\n"
            "for model in st:\n"
            "    for chain in model:\n"
            "        print(f'Chain {chain.name}: {len(chain)} residues')\n"
        ),
        "note": "gemmi is the fastest Python mmCIF/PDB parser. It handles symmetry, unit cells, and electron density maps natively.",
    },
    {
        "intent": "find all ligand residues in a structure",
        "code": (
            "import gemmi\n"
            "st = gemmi.read_structure('{pdb}.pdb')\n"
            "for model in st:\n"
            "    for chain in model:\n"
            "        for res in chain:\n"
            "            if res.het_flag == 'H' and res.name != 'HOH':\n"
            "                print(f'{chain.name} {res.name} {res.seqid}')\n"
        ),
        "note": "`het_flag == 'H'` identifies HETATM residues. Filter out HOH for non-water ligands.",
    },
    {
        "intent": "find all contacts within 4 angstroms between protein and ligand",
        "code": (
            "import gemmi\n"
            "st = gemmi.read_structure('{pdb}.pdb')\n"
            "ns = gemmi.NeighborSearch(st[0], st.cell, 5.0)\n"
            "ns.populate()\n"
            "# Find contacts from ligand atoms\n"
            "for chain in st[0]:\n"
            "    for res in chain:\n"
            "        if res.het_flag == 'H' and res.name != 'HOH':\n"
            "            for atom in res:\n"
            "                for nb in ns.find_atoms(atom.pos, '\\0', radius=4.0):\n"
            "                    mark = nb.to_cra(st[0])\n"
            "                    if mark.residue.het_flag != 'H':\n"
            "                        print(f'{atom.name}-{mark.chain.name}/{mark.residue.name}{mark.residue.seqid}/{mark.atom.name}: {nb.dist():.2f} A')\n"
        ),
        "note": "gemmi's NeighborSearch is symmetry-aware and handles crystal contacts correctly.",
    },
]


def gemmi_example():
    recipe = random.choice(_GEMMI_RECIPES)
    ctx = random.choice(PYMOL_CONTEXTS)
    code = recipe["code"].replace("{pdb}", ctx["pdb"])
    answer = (
        f"```python\n{code}```\n\n"
        f"{recipe['note']}"
    )
    return _record(
        f"How do I {recipe['intent']} with gemmi? I'm working with {ctx['pdb']} ({ctx['target']}).",
        answer,
    )


# ---------------------------------------------------------------------------
# Behaviour class 23: Structure preparation for docking/MD
# ---------------------------------------------------------------------------

_PREP_RECIPES = [
    {
        "intent": "prepare a receptor for docking",
        "code": (
            "from pdbfixer import PDBFixer\n"
            "from openmm.app import PDBFile\n"
            "fixer = PDBFixer(filename='{pdb}.pdb')\n"
            "fixer.findMissingResidues()\n"
            "fixer.findMissingAtoms()\n"
            "fixer.addMissingAtoms()\n"
            "fixer.addMissingHydrogens(7.4)  # pH 7.4\n"
            "fixer.removeHeterogens(keepWater=False)\n"
            "PDBFile.writeFile(fixer.topology, fixer.positions,\n"
            "                  open('{pdb}_prepped.pdb', 'w'))\n"
        ),
        "note": "PDBFixer adds missing atoms, residues, and hydrogens at physiological pH. Always remove heterogens (ligands, ions) for receptor-only preparation.",
    },
    {
        "intent": "add hydrogens to a ligand for docking",
        "code": (
            "# Using Open Babel\n"
            "# obabel ligand.sdf -O ligand_h.sdf -h --gen3d\n"
            "\n"
            "# Or using RDKit\n"
            "from rdkit import Chem\n"
            "from rdkit.Chem import AllChem\n"
            "mol = Chem.MolFromMolFile('ligand.sdf')\n"
            "mol = Chem.AddHs(mol)\n"
            "AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())\n"
            "AllChem.MMFFOptimizeMolecule(mol)\n"
            "Chem.MolToMolFile(mol, 'ligand_h.sdf')\n"
        ),
        "note": "Hydrogens and 3D coordinates are required for docking. ETKDGv3 generates good initial conformers; MMFF optimisation refines them.",
    },
]


def struct_prep_example():
    recipe = random.choice(_PREP_RECIPES)
    ctx = random.choice(PYMOL_CONTEXTS)
    code = recipe["code"].replace("{pdb}", ctx["pdb"])
    answer = (
        f"```python\n{code}```\n\n"
        f"{recipe['note']}"
    )
    return _record(
        f"How do I {recipe['intent']}? Working with {ctx['pdb']} ({ctx['target']}).",
        answer,
    )


# ---------------------------------------------------------------------------
# Behaviour class 24: Docking interpretation
# ---------------------------------------------------------------------------

def docking_interpretation_example():
    ctx = random.choice(PYMOL_CONTEXTS)
    score = round(random.uniform(-10.5, -5.0), 1)
    rmsd_lb = round(random.uniform(0.0, 2.5), 1)
    n_poses = random.randint(5, 20)
    top_cluster = random.randint(3, 8)

    potency = (
        "excellent predicted binding (< -9 kcal/mol)" if score < -9 else
        "good predicted binding (-7 to -9 kcal/mol)" if score < -7 else
        "moderate predicted binding (-5 to -7 kcal/mol)"
    )

    answer = (
        f"Docking results for {ctx['ligand']} into {ctx['pdb']} ({ctx['target']}):\n\n"
        f"- **Best score:** {score} kcal/mol ({potency})\n"
        f"- **RMSD lower bound:** {rmsd_lb} A (vs crystallographic pose)\n"
        f"- **Poses generated:** {n_poses}, top cluster: {top_cluster} poses\n\n"
        "```bash\n"
        f"# AutoDock Vina command\n"
        f"vina --receptor {ctx['pdb']}_receptor.pdbqt \\\n"
        f"     --ligand {ctx['ligand']}.pdbqt \\\n"
        f"     --center_x 10.0 --center_y 20.0 --center_z 15.0 \\\n"
        f"     --size_x 20 --size_y 20 --size_z 20 \\\n"
        f"     --exhaustiveness 32 \\\n"
        f"     --out {ctx['ligand']}_docked.pdbqt\n"
        "```\n\n"
        f"An RMSD < 2.0 A from the crystal pose indicates the docking protocol can reproduce "
        f"the experimental binding mode. A score of {score} kcal/mol corresponds to a predicted "
        f"Kd of ~{10**(score/1.364)*1e6:.0f} uM (using deltaG = RT ln Kd at 298K).\n\n"
        "Always validate docking results with PLIP interaction analysis and visual inspection in PyMOL."
    )
    return _record(
        f"I docked {ctx['ligand']} into {ctx['pdb']} ({ctx['target']}) with Vina. "
        f"The best score is {score} kcal/mol with RMSD {rmsd_lb} A. Interpret the results.",
        answer,
    )


# ---------------------------------------------------------------------------
# Behaviour class 25: Plotly visualization (Python)
# ---------------------------------------------------------------------------

_PLOTLY_RECIPES = [
    {
        "intent": "create a scatter plot of MW vs logP for a compound set",
        "code": (
            "import plotly.express as px\n"
            "import pandas as pd\n"
            "from rdkit import Chem\n"
            "from rdkit.Chem import Descriptors\n"
            "\n"
            "smiles_list = {smiles_list}\n"
            "data = []\n"
            "for smi in smiles_list:\n"
            "    mol = Chem.MolFromSmiles(smi)\n"
            "    if mol:\n"
            "        data.append({\n"
            "            'SMILES': Chem.MolToSmiles(mol),\n"
            "            'MW': round(Descriptors.MolWt(mol), 1),\n"
            "            'logP': round(Descriptors.MolLogP(mol), 2),\n"
            "        })\n"
            "df = pd.DataFrame(data)\n"
            "fig = px.scatter(df, x='MW', y='logP', hover_data=['SMILES'],\n"
            "                 title='MW vs logP')\n"
            "fig.add_hrect(y0=-0.5, y1=5, fillcolor='green', opacity=0.1,\n"
            "              annotation_text='Lipinski logP zone')\n"
            "fig.add_vrect(x0=0, x1=500, fillcolor='blue', opacity=0.1,\n"
            "              annotation_text='Lipinski MW zone')\n"
            "fig.write_html('mw_logp.html')\n"
        ),
        "note": "The shaded regions show the Lipinski-compliant space. Compounds outside both zones are likely to have poor oral bioavailability.",
    },
    {
        "intent": "create a radar chart of drug-likeness properties",
        "code": (
            "import plotly.graph_objects as go\n"
            "from rdkit import Chem\n"
            "from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors\n"
            "\n"
            "smi = '{smi}'\n"
            "mol = Chem.MolFromSmiles(smi)\n"
            "props = {\n"
            "    'MW (/ 100)': Descriptors.MolWt(mol) / 100,\n"
            "    'logP': Descriptors.MolLogP(mol),\n"
            "    'HBD': Lipinski.NumHDonors(mol),\n"
            "    'HBA (/ 2)': Lipinski.NumHAcceptors(mol) / 2,\n"
            "    'TPSA (/ 30)': rdMolDescriptors.CalcTPSA(mol) / 30,\n"
            "    'RotBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),\n"
            "}\n"
            "fig = go.Figure(go.Scatterpolar(\n"
            "    r=list(props.values()),\n"
            "    theta=list(props.keys()),\n"
            "    fill='toself', name=Chem.MolToSmiles(mol)\n"
            "))\n"
            "fig.update_layout(title='Drug-likeness Radar')\n"
            "fig.write_html('radar.html')\n"
        ),
        "note": "Properties are scaled so the Lipinski limits fall at roughly the same radius. Ideal compounds form a compact polygon within the outer ring.",
    },
]


def plotly_example():
    recipe = random.choice(_PLOTLY_RECIPES)
    smiles_sample = random.sample(SEED_SMILES, min(5, len(SEED_SMILES)))
    smi = random.choice(SEED_SMILES)
    code = recipe["code"].replace("{smiles_list}", repr(smiles_sample)).replace("{smi}", smi)
    answer = (
        f"```python\n{code}```\n\n"
        f"{recipe['note']}"
    )
    return _record(
        f"Create a Plotly visualization to {recipe['intent']}.",
        answer,
    )


# ---------------------------------------------------------------------------
# Behaviour class 26: R / ggplot2 / bio3d recipes
# ---------------------------------------------------------------------------

_R_RECIPES = [
    {
        "intent": "plot a SAR scatter of pIC50 vs logP coloured by series",
        "code": (
            'library(tidyverse)\n'
            '\n'
            'df <- read_csv("sar_data.csv")\n'
            'ggplot(df, aes(x = logP, y = pIC50, colour = series)) +\n'
            '  geom_point(size = 3, alpha = 0.7) +\n'
            '  geom_smooth(method = "lm", se = FALSE, linetype = "dashed") +\n'
            '  labs(title = "SAR: pIC50 vs logP by series",\n'
            '       x = "clogP", y = "pIC50") +\n'
            '  theme_minimal() +\n'
            '  scale_colour_brewer(palette = "Set1")\n'
            'ggsave("sar_plot.png", width = 8, height = 6, dpi = 300)\n'
        ),
        "note": "The linear trend line per series reveals whether potency tracks lipophilicity (a common SAR trap). Flat or negative slopes suggest you can reduce logP without losing potency.",
    },
    {
        "intent": "perform RMSD analysis on an MD trajectory using bio3d",
        "code": (
            'library(bio3d)\n'
            '\n'
            'pdb <- read.pdb("{pdb}.pdb")\n'
            'traj <- read.dcd("trajectory.dcd")\n'
            'ca.inds <- atom.select(pdb, "calpha")\n'
            '\n'
            '# Superimpose trajectory frames on first frame\n'
            'xyz <- fit.xyz(fixed = pdb$xyz, mobile = traj,\n'
            '               fixed.inds = ca.inds$xyz, mobile.inds = ca.inds$xyz)\n'
            '\n'
            'rd <- rmsd(xyz[1,ca.inds$xyz], xyz[,ca.inds$xyz])\n'
            'plot(rd, type = "l", ylab = "RMSD (A)", xlab = "Frame",\n'
            '     main = "CA RMSD over trajectory")\n'
        ),
        "note": "bio3d provides the standard R toolkit for structural bioinformatics. The RMSD plot shows whether the simulation has equilibrated (flat plateau) or is still drifting.",
    },
    {
        "intent": "create an interactive Plotly chart of compound properties in R",
        "code": (
            'library(plotly)\n'
            'library(readr)\n'
            '\n'
            'df <- read_csv("compounds.csv")\n'
            'p <- plot_ly(df, x = ~MW, y = ~logP, color = ~QED,\n'
            '             text = ~paste("Name:", name, "<br>SMILES:", smiles),\n'
            '             type = "scatter", mode = "markers",\n'
            '             marker = list(size = 10, opacity = 0.7)) %>%\n'
            '  layout(title = "Compound Property Space",\n'
            '         xaxis = list(title = "Molecular Weight"),\n'
            '         yaxis = list(title = "logP"),\n'
            '         shapes = list(\n'
            '           list(type = "rect", x0 = 0, x1 = 500, y0 = -0.5, y1 = 5,\n'
            '                fillcolor = "green", opacity = 0.05, line = list(width = 0))\n'
            '         ))\n'
            'htmlwidgets::saveWidget(p, "property_space.html")\n'
        ),
        "note": "Plotly in R creates interactive HTML widgets. Hover shows compound details. The green rectangle marks the Lipinski-compliant zone.",
    },
    {
        "intent": "calculate RMSF per residue from an MD simulation",
        "code": (
            'library(bio3d)\n'
            '\n'
            'pdb <- read.pdb("{pdb}.pdb")\n'
            'traj <- read.dcd("trajectory.dcd")\n'
            'ca.inds <- atom.select(pdb, "calpha")\n'
            '\n'
            'xyz <- fit.xyz(fixed = pdb$xyz, mobile = traj,\n'
            '               fixed.inds = ca.inds$xyz, mobile.inds = ca.inds$xyz)\n'
            '\n'
            'rf <- rmsf(xyz[,ca.inds$xyz])\n'
            'plot.bio3d(rf, sse = pdb, ylab = "RMSF (A)", xlab = "Residue",\n'
            '           main = "Per-residue flexibility")\n'
            '# Highlight binding site residues\n'
            'abline(v = c(25, 50, 82), col = "red", lty = 2)\n'
        ),
        "note": "RMSF (root mean square fluctuation) per residue shows which parts of the protein are flexible. Peaks often correspond to loops and termini. Low RMSF in the binding site suggests a rigid pocket suitable for structure-based design.",
    },
]


def r_recipe_example():
    recipe = random.choice(_R_RECIPES)
    ctx = random.choice(PYMOL_CONTEXTS)
    code = recipe["code"].replace("{pdb}", ctx["pdb"])
    answer = (
        f"```r\n{code}```\n\n"
        f"{recipe['note']}"
    )
    return _record(
        f"Write an R script to {recipe['intent']}.",
        answer,
    )


# ---------------------------------------------------------------------------
# Behaviour class 27: Refusal / out-of-scope examples
# ---------------------------------------------------------------------------

_REFUSAL_QUESTIONS = [
    {
        "q": "What is the IC50 of compound XYZ-12345 against BRAF?",
        "a": "I don't have data for compound XYZ-12345 in my corpus. To find its IC50 against BRAF, check ChEMBL (search by compound name or structure) or the primary literature. I can help you query the ChEMBL API if you have a ChEMBL ID or SMILES string.",
    },
    {
        "q": "Synthesise a molecule that cures cancer.",
        "a": "I can't design a cure for cancer — drug discovery requires extensive experimental validation that goes far beyond computational prediction. What I can do is help you analyse existing SAR data, compute physicochemical properties, search for bioactive compounds against a specific target, or interpret structural data. What target or compound class are you working on?",
    },
    {
        "q": "What is the crystal structure of PDB entry 99ZZ?",
        "a": "PDB entry 99ZZ does not exist in my corpus (which contains 255,239 structures). PDB IDs are 4-character alphanumeric codes (e.g. 1HSG, 4DKL). Please check the ID and try again, or search the RCSB PDB at rcsb.org.",
    },
    {
        "q": "What is the binding affinity of water?",
        "a": "Water is not a drug-like ligand and does not have a meaningful binding affinity in the drug discovery sense. However, crystallographic water molecules in binding sites can be important for ligand design — they can mediate hydrogen bonds between the protein and ligand. If you're interested in water-mediated interactions, I can run PLIP analysis on a specific PDB structure.",
    },
]


def refusal_example():
    ex = random.choice(_REFUSAL_QUESTIONS)
    return _record(ex["q"], ex["a"])


# ---------------------------------------------------------------------------
# Behaviour class 28: Protein family structures (corpus-grounded)
# ---------------------------------------------------------------------------

_PROTEIN_FAMILIES = {
    "kinase": {
        "search": ["kinase"],
        "context": "Kinases are key drug targets in oncology. The ATP binding site (hinge region, P-loop, activation loop, DFG motif) is the primary druggable pocket. Type I inhibitors bind the active (DFG-in) conformation; Type II inhibitors stabilise DFG-out.",
        "examples": ["imatinib", "erlotinib", "vemurafenib", "crizotinib", "lapatinib"],
    },
    "dark kinase": {
        "search": ["kinase"],
        "context": "Dark kinases are understudied members of the human kinome with limited annotation. They represent untapped therapeutic potential. The Illuminating the Druggable Genome (IDG) project prioritises these targets. Structural data can reveal druggable pockets even when biology is poorly understood.",
        "examples": [],
    },
    "antibody": {
        "search": ["antibody", "Fab", "immunoglobulin"],
        "context": "Antibody structures typically show Fab (fragment antigen-binding) or scFv (single-chain variable fragment) regions. The CDR loops (complementarity-determining regions: H1, H2, H3, L1, L2, L3) form the antigen-binding surface. H3 is the most variable and often dominates binding specificity.",
        "examples": ["trastuzumab", "rituximab", "pembrolizumab", "nivolumab"],
    },
    "GPCR": {
        "search": ["GPCR", "G-protein coupled", "rhodopsin", "adrenergic", "muscarinic", "serotonin receptor", "dopamine receptor", "opioid receptor"],
        "context": "GPCRs are the largest family of membrane receptors and targets of ~34% of approved drugs. The orthosteric binding site sits within the 7-transmembrane helix bundle. Allosteric sites on the extracellular or intracellular face offer selectivity advantages. Active-state structures (with G-protein or arrestin) vs inactive-state structures inform agonist vs antagonist design.",
        "examples": ["adrenaline", "salmeterol", "olanzapine", "morphine", "sumatriptan"],
    },
    "GTPase": {
        "search": ["GTPase", "RAS", "KRAS", "HRAS", "NRAS", "RAB", "RHO", "CDC42", "RAC"],
        "context": "Small GTPases (RAS superfamily) act as molecular switches cycling between GTP-bound (active) and GDP-bound (inactive) states. KRAS G12C was considered undruggable until covalent inhibitors (sotorasib, adagrasib) exploited a cryptic pocket (switch II pocket) adjacent to the mutant cysteine. Other RAS mutations remain challenging.",
        "examples": ["sotorasib", "adagrasib"],
    },
    "cytokine": {
        "search": ["cytokine", "interleukin", "interferon", "TNF", "chemokine"],
        "context": "Cytokines are signalling proteins in immunity and inflammation. Structural data shows receptor-cytokine interfaces that inform biologics design. Small-molecule inhibitors of cytokine-receptor interactions are rare but emerging (e.g. JAK inhibitors downstream of cytokine receptors).",
        "examples": ["tofacitinib", "ruxolitinib", "baricitinib"],
    },
    "receptor": {
        "search": ["receptor"],
        "context": "Receptor structures reveal ligand-binding pockets, activation mechanisms, and allosteric sites. Nuclear receptors (ER, AR, PPAR) have well-defined ligand-binding domains. Receptor tyrosine kinases combine extracellular ligand binding with intracellular kinase activity.",
        "examples": ["tamoxifen", "enzalutamide", "rosiglitazone"],
    },
}

_FAMILY_PDB_CACHE = {}

def _load_family_pdbs():
    if _FAMILY_PDB_CACHE:
        return _FAMILY_PDB_CACHE
    import pandas as pd
    entries = pd.read_csv("data/corpus/pdb/pdb_all_entries.csv", dtype=str, low_memory=False).fillna("")
    pairs = pd.read_csv("data/corpus/pdb/pdb_ligand_structure_pairs.csv", dtype=str, low_memory=False).fillna("")
    affinities = pd.read_csv("data/corpus/pdb/pdb_binding_affinities.csv", dtype=str, low_memory=False).fillna("")

    for family, info in _PROTEIN_FAMILIES.items():
        mask = pd.Series(False, index=entries.index)
        for term in info["search"]:
            mask = mask | entries["header"].str.contains(term, case=False) | entries["source"].str.contains(term, case=False)
        pdb_ids = set(entries.loc[mask, "pdb_id"])
        family_pairs = pairs[pairs["pdb_id"].isin(pdb_ids) & (pairs["smiles"].str.len() > 5)]
        family_aff = affinities[affinities["pdb_id"].isin(pdb_ids)]

        structures = []
        for _, row in entries.loc[mask].head(200).iterrows():
            structures.append({
                "pdb_id": row["pdb_id"],
                "header": row.get("header", ""),
                "source": row.get("source", ""),
                "resolution": row.get("release_date", ""),
            })

        ligands = []
        for _, row in family_pairs.drop_duplicates("comp_id").head(50).iterrows():
            ligands.append({
                "pdb_id": row["pdb_id"],
                "comp_id": row["comp_id"],
                "comp_name": row.get("comp_name", ""),
                "smiles": row["smiles"],
            })

        _FAMILY_PDB_CACHE[family] = {
            "structures": structures,
            "ligands": ligands,
            "n_structures": len(pdb_ids),
            "n_affinities": len(family_aff),
        }
    return _FAMILY_PDB_CACHE


def protein_family_example():
    cache = _load_family_pdbs()
    family = random.choice(list(_PROTEIN_FAMILIES.keys()))
    info = _PROTEIN_FAMILIES[family]
    data = cache.get(family, {})

    if not data.get("structures"):
        return None

    struct = random.choice(data["structures"])
    pdb_id = struct["pdb_id"]

    style = random.choice(["overview", "ligand", "compare"])

    if style == "overview":
        answer = (
            f"PDB entry **{pdb_id}** is a {family} structure"
            + (f" ({struct['header']})" if struct["header"] else "") + ".\n\n"
            f"{info['context']}\n\n"
            f"The corpus contains **{data['n_structures']}** {family} structures"
            f" with **{data['n_affinities']}** measured binding affinities.\n\n"
            "```python\n"
            f"# PyMOL: visualise {pdb_id}\n"
            f"fetch {pdb_id}, async=0\n"
            "remove solvent\n"
            "show cartoon, polymer\n"
            "show sticks, organic\n"
            "util.cbag organic\n"
            "center organic\n"
            "zoom organic, 8\n"
            "```"
        )
        return _record(
            f"What do we know about the {family} structure {pdb_id}?",
            answer,
        )

    elif style == "ligand" and data.get("ligands"):
        lig = random.choice(data["ligands"])
        mol = Chem.MolFromSmiles(lig["smiles"])
        if mol is None:
            return None
        mw = round(Descriptors.MolWt(mol), 1)
        logp = round(Descriptors.MolLogP(mol), 2)
        answer = (
            f"The ligand **{lig['comp_id']}**"
            + (f" ({lig['comp_name']})" if lig["comp_name"] else "")
            + f" is bound in {family} structure {lig['pdb_id']}.\n\n"
            "```python\n"
            "from rdkit import Chem\n"
            "from rdkit.Chem import Descriptors\n"
            f"mol = Chem.MolFromSmiles('{lig['smiles']}')\n"
            "print('MW:', round(Descriptors.MolWt(mol), 1))\n"
            "print('logP:', round(Descriptors.MolLogP(mol), 2))\n"
            "```\n\n"
            f"MW: {mw}, logP: {logp}.\n\n"
            f"{info['context']}"
        )
        return _record(
            f"What ligand is bound in {family} structure {lig['pdb_id']}? Compute its properties.",
            answer,
        )

    else:
        if len(data["structures"]) < 2:
            return None
        s1, s2 = random.sample(data["structures"], 2)
        answer = (
            f"Comparing {family} structures {s1['pdb_id']} and {s2['pdb_id']}:\n\n"
            f"- **{s1['pdb_id']}:** {s1['header']}\n"
            f"- **{s2['pdb_id']}:** {s2['header']}\n\n"
            "```python\n"
            f"# PyMOL: align and compare\n"
            f"fetch {s1['pdb_id']}, async=0\n"
            f"fetch {s2['pdb_id']}, async=0\n"
            f"align {s2['pdb_id']}, {s1['pdb_id']}\n"
            "show cartoon\n"
            f"color cyan, {s1['pdb_id']}\n"
            f"color salmon, {s2['pdb_id']}\n"
            "show sticks, organic\n"
            "center organic\n"
            "zoom vis, 5\n"
            "```\n\n"
            f"The `align` command reports the RMSD between the two structures. "
            f"Compare the binding sites to identify conserved interactions and structural differences "
            f"that could inform selectivity.\n\n{info['context']}"
        )
        return _record(
            f"Compare these two {family} structures: {s1['pdb_id']} and {s2['pdb_id']}.",
            answer,
        )


# ---------------------------------------------------------------------------
# Behaviour class 29: Code-then-quote (numerical fidelity improvement)
# ---------------------------------------------------------------------------

def code_then_quote_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canon = Chem.MolToSmiles(mol)
    mw   = round(Descriptors.MolWt(mol), 1)
    logp = round(Descriptors.MolLogP(mol), 2)
    hbd  = Lipinski.NumHDonors(mol)
    hba  = Lipinski.NumHAcceptors(mol)
    tpsa = round(rdMolDescriptors.CalcTPSA(mol), 1)

    prop = random.choice([
        ("molecular weight", f"round(Descriptors.MolWt(mol), 1)", f"{mw} Da"),
        ("logP", f"round(Descriptors.MolLogP(mol), 2)", str(logp)),
        ("hydrogen bond donors", f"Lipinski.NumHDonors(mol)", str(hbd)),
        ("hydrogen bond acceptors", f"Lipinski.NumHAcceptors(mol)", str(hba)),
        ("topological polar surface area", f"round(rdMolDescriptors.CalcTPSA(mol), 1)", f"{tpsa} A^2"),
    ])

    answer = (
        "I will compute this with RDKit rather than estimating.\n\n"
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        f"result = {prop[1]}\n"
        f"print(result)\n"
        "```\n\n"
        f"The RDKit output shows: **{prop[0]} = {prop[2]}**.\n\n"
        "This value is computed directly by RDKit, not estimated from memory."
    )
    questions = [
        f"What is the exact {prop[0]} of {smiles}?",
        f"Compute the {prop[0]} of {canon}.",
        f"Use RDKit to calculate the {prop[0]} for {smiles}.",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# Behaviour class 30: Precise rounding examples (fidelity drill)
# ---------------------------------------------------------------------------

def rounding_precision_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canon = Chem.MolToSmiles(mol)
    mw_full = Descriptors.MolWt(mol)
    mw_1 = round(mw_full, 1)
    mw_2 = round(mw_full, 2)
    logp_full = Descriptors.MolLogP(mol)
    logp_2 = round(logp_full, 2)
    tpsa_full = rdMolDescriptors.CalcTPSA(mol)
    tpsa_1 = round(tpsa_full, 1)

    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors, rdMolDescriptors\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        f"mw = Descriptors.MolWt(mol)\n"
        f"logp = Descriptors.MolLogP(mol)\n"
        f"tpsa = rdMolDescriptors.CalcTPSA(mol)\n"
        f"print(f'MW:   {{mw:.4f}} -> {{round(mw, 1)}}')\n"
        f"print(f'logP: {{logp:.4f}} -> {{round(logp, 2)}}')\n"
        f"print(f'TPSA: {{tpsa:.4f}} -> {{round(tpsa, 1)}}')\n"
        "```\n\n"
        f"**{canon}:**\n"
        f"- MW: {mw_full:.4f} -> rounded to 1 dp: **{mw_1}** Da\n"
        f"- logP: {logp_full:.4f} -> rounded to 2 dp: **{logp_2}**\n"
        f"- TPSA: {tpsa_full:.4f} -> rounded to 1 dp: **{tpsa_1}** A^2\n\n"
        "Always report values at the precision shown in the code output. "
        "MW to 1 decimal place, logP to 2, TPSA to 1."
    )
    return _record(
        f"Calculate the MW, logP, and TPSA of {smiles} with full precision and proper rounding.",
        answer,
    )


# ---------------------------------------------------------------------------
# Behaviour class 31: Personality — druglikeness with humor and emoji
# ---------------------------------------------------------------------------

_MW_JOKES = [
    (500, 800, "Getting into peptide territory — your formulation scientist just felt a chill 😄"),
    (800, 2000, "This isn't a small molecule any more, it's a small protein 😄"),
    (0, 150, "This is basically a fragment — you'll need to grow it before it does anything useful 😄"),
]

_LOGP_JOKES = [
    (5, 7, "This molecule is greasier than a chip shop. Consider adding some polarity 😄"),
    (7, 15, "logP over 7 — you could use this as suntan oil 😄"),
    (-3, -1, "Extremely polar — this molecule would rather be in the ocean than cross a membrane 😄"),
]

_TPSA_JOKES = [
    (0, 20, "Flatter than Kansas. Good luck getting this past the BBB without a carrier 😄"),
    (140, 300, "More polar surface than the Arctic — oral bioavailability is going to be a challenge 😄"),
]

_ROTBOND_JOKES = [
    (10, 15, "This molecule has more flexibility than a yoga instructor 😄"),
    (15, 50, "With this many rotatable bonds, your docking software is going to need therapy 😄"),
]

_QED_JOKES = [
    (0, 0.2, "QED {qed} — Lipinski would like a word 😄"),
    (0.2, 0.4, "QED {qed} — not winning any beauty contests in drug space 😄"),
    (0.7, 1.0, "QED {qed} — a textbook drug-like molecule. The medchem gods are smiling 😄"),
]

_PASS_COMMENTS = [
    "💊 **PASSES** — comfortably drug-like. Ship it to the assay queue.",
    "💊 **PASSES** — all clear on the Lipinski front. This one's a keeper.",
    "💊 **PASSES** — well within the drug-like sweet spot. Nice work.",
    "💊 **PASSES** — Lipinski gives the thumbs up 👍",
]

_FAIL_COMMENTS = [
    "⚠️ **FAILS** — back to the drawing board. Consider trimming the fat.",
    "⚠️ **FAILS** — this compound needs a medchem makeover.",
    "⚠️ **FAILS** — the Lipinski police have flagged this one 🚨 😄",
]


def personality_druglikeness_example(smiles: str):
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
    try:
        from rdkit.Chem.QED import qed
        qed_val = round(qed(mol), 2)
    except Exception:
        qed_val = None

    lip = mw < 500 and logp < 5 and hbd <= 5 and hba <= 10

    jokes = []
    for lo, hi, joke in _MW_JOKES:
        if lo <= mw < hi:
            jokes.append(joke)
    for lo, hi, joke in _LOGP_JOKES:
        if lo <= logp < hi:
            jokes.append(joke)
    for lo, hi, joke in _TPSA_JOKES:
        if lo <= tpsa < hi:
            jokes.append(joke)
    for lo, hi, joke in _ROTBOND_JOKES:
        if lo <= rotb < hi:
            jokes.append(joke)
    if qed_val is not None:
        for lo, hi, joke in _QED_JOKES:
            if lo <= qed_val < hi:
                jokes.append(joke.replace("{qed}", str(qed_val)))

    verdict = random.choice(_PASS_COMMENTS) if lip else random.choice(_FAIL_COMMENTS)
    joke_line = f"\n\n{random.choice(jokes)}" if jokes else ""
    qed_line = f"  QED  {'🌟' if qed_val and qed_val > 0.5 else '  '} {qed_val}\n" if qed_val else ""

    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "print('MW:', round(Descriptors.MolWt(mol), 1))\n"
        "print('logP:', round(Descriptors.MolLogP(mol), 2))\n"
        "print('HBD:', Lipinski.NumHDonors(mol))\n"
        "print('HBA:', Lipinski.NumHAcceptors(mol))\n"
        "print('TPSA:', round(rdMolDescriptors.CalcTPSA(mol), 1))\n"
        "print('RotBonds:', rdMolDescriptors.CalcNumRotatableBonds(mol))\n"
        "```\n\n"
        f"```\n"
        f"═══ Drug-likeness Verdict for {canon} ═══\n"
        f"  MW   {'✅' if mw < 500 else '❌'}  {mw} Da {'(< 500)' if mw < 500 else '(≥ 500)'}\n"
        f"  logP {'✅' if logp < 5 else '❌'}  {logp} {'(< 5)' if logp < 5 else '(≥ 5)'}\n"
        f"  HBD  {'✅' if hbd <= 5 else '❌'}  {hbd} {'(≤ 5)' if hbd <= 5 else '(> 5)'}\n"
        f"  HBA  {'✅' if hba <= 10 else '❌'}  {hba} {'(≤ 10)' if hba <= 10 else '(> 10)'}\n"
        f"  TPSA    {tpsa} A^2\n"
        f"  RotB    {rotb}\n"
        f"{qed_line}"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  {verdict}\n"
        f"```"
        f"{joke_line}"
    )
    questions = [
        f"Is {smiles} drug-like?",
        f"Assess the drug-likeness of {smiles}.",
        f"Run a Lipinski check on {smiles}.",
        f"How does {canon} look from a drug-likeness perspective?",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# Behaviour class 32: Personality — SMILES validation with flair
# ---------------------------------------------------------------------------

_VALID_REACTIONS = [
    "RDKit says: this SMILES is legit ✅",
    "Valid SMILES confirmed — RDKit parsed it without breaking a sweat 💪",
    "✅ SMILES is valid. Canonical form shown above.",
]

_INVALID_REACTIONS = [
    "RDKit tried its best, but this SMILES is broken 💔 😄",
    "❌ Invalid SMILES — RDKit choked on this one. Check for typos.",
    "This SMILES has issues. RDKit threw it back like a bad fish 🐟 😄",
    "❌ Nope — this SMILES won't parse. Common culprits: unbalanced parentheses, unclosed rings, or made-up atoms.",
]


def personality_validation_example():
    if random.random() < 0.4:
        smi = random.choice(INVALID_SMILES)
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return None
        answer = (
            "```python\n"
            "from rdkit import Chem\n"
            f"mol = Chem.MolFromSmiles('{smi}')\n"
            "print('Valid ✅' if mol is not None else 'Invalid ❌')\n"
            "```\n\n"
            f"{random.choice(_INVALID_REACTIONS)}\n\n"
            f"`{smi}` — check for unbalanced parentheses, unclosed ring numbers, or invalid atom symbols."
        )
    else:
        smi = random.choice(SEED_SMILES)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        canon = Chem.MolToSmiles(mol)
        mw = round(Descriptors.MolWt(mol), 1)
        answer = (
            "```python\n"
            "from rdkit import Chem\n"
            "from rdkit.Chem import Descriptors\n"
            f"mol = Chem.MolFromSmiles('{smi}')\n"
            "print('Valid ✅' if mol is not None else 'Invalid ❌')\n"
            "if mol:\n"
            "    print('Canonical:', Chem.MolToSmiles(mol))\n"
            "    print('MW:', round(Descriptors.MolWt(mol), 1))\n"
            "```\n\n"
            f"{random.choice(_VALID_REACTIONS)}\n\n"
            f"Canonical: `{canon}` (MW {mw} Da)"
        )
    return _record(f"Is this a valid SMILES? {smi}", answer)


# ---------------------------------------------------------------------------
# Behaviour class 33: Personality — structural biology with character
# ---------------------------------------------------------------------------

_STRUCT_OBSERVATIONS = [
    "B-factor over 60 in the binding site — this region is doing the Macarena 💃 😄",
    "Resolution 1.2 Å — crisp enough to see individual water molecules. Beautiful data 🔬",
    "Resolution 3.5 Å — squint hard enough and you can almost see the side chains 😄",
    "15 hydrogen bonds to the ligand — this compound is clinging on for dear life 🤝",
    "RMSD 0.3 Å between the two chains — almost identical. Mother Nature's copy-paste 🧬",
    "The ligand is completely buried — good luck getting it out without denaturing the protein 😄",
    "3 salt bridges — this binding mode is anchored like a battleship ⚓",
    "No electron density for residues 45-60 — the loop has left the building 🏃 😄",
]

_BINDING_ASSESSMENTS = [
    {"range": (-12, -10), "emoji": "🏆", "comment": "Extraordinary predicted binding — this is a clinical candidate-level score"},
    {"range": (-10, -8), "emoji": "🎯", "comment": "Excellent predicted binding — worth progressing to biochemical assay"},
    {"range": (-8, -6), "emoji": "👍", "comment": "Good predicted binding — typical for a hit compound"},
    {"range": (-6, -4), "emoji": "🤔", "comment": "Weak predicted binding — needs optimisation"},
    {"range": (-4, 0), "emoji": "😬", "comment": "Poor predicted binding — back to the virtual library 😄"},
]


def personality_structure_example():
    ctx = random.choice(PYMOL_CONTEXTS)
    style = random.choice(["observation", "binding", "bfactor"])

    if style == "observation":
        obs = random.choice(_STRUCT_OBSERVATIONS)
        answer = (
            f"Looking at {ctx['pdb']} ({ctx['target']}):\n\n"
            "```python\n"
            f"# PyMOL quick look\n"
            f"fetch {ctx['pdb']}, async=0\n"
            "remove solvent\n"
            "show cartoon\n"
            "show sticks, organic\n"
            "spectrum b, blue_white_red\n"
            "center organic\n"
            "zoom organic, 8\n"
            "```\n\n"
            f"🔬 **Observation:** {obs}"
        )
        return _record(
            f"Take a look at {ctx['pdb']} ({ctx['target']}) and tell me what you see.",
            answer,
        )

    elif style == "binding":
        score = round(random.uniform(-11.0, -3.0), 1)
        assessment = None
        for a in _BINDING_ASSESSMENTS:
            if a["range"][0] <= score < a["range"][1]:
                assessment = a
                break
        if not assessment:
            return None
        answer = (
            f"Docking score for {ctx['ligand']} in {ctx['pdb']}: **{score} kcal/mol**\n\n"
            f"{assessment['emoji']} {assessment['comment']}.\n\n"
            f"Estimated Kd: ~{10**(score/1.364)*1e6:.0f} µM "
            f"(using ΔG = RT ln Kd at 298K)\n\n"
            "Remember: docking scores are estimates, not measurements. "
            "Always validate with a biochemical assay before celebrating 🧪"
        )
        return _record(
            f"I docked {ctx['ligand']} into {ctx['pdb']} ({ctx['target']}) and got a score of {score} kcal/mol. Is that good?",
            answer,
        )

    else:
        avg_b = round(random.uniform(15, 65), 1)
        flexibility = (
            "well-ordered and rigid — high confidence in atomic positions 🎯" if avg_b < 25 else
            "moderately flexible — typical for a surface-exposed region" if avg_b < 40 else
            "quite flexible — coordinates should be interpreted with caution ⚠️" if avg_b < 55 else
            "highly disordered — this region is doing the Macarena 💃 😄"
        )
        answer = (
            f"B-factor analysis for {ctx['pdb']} ({ctx['target']}):\n\n"
            "```python\n"
            f"# PyMOL: colour by B-factor\n"
            f"fetch {ctx['pdb']}, async=0\n"
            "remove solvent\n"
            "show cartoon\n"
            "spectrum b, blue_white_red\n"
            "cartoon putty\n"
            "center\n"
            "```\n\n"
            f"Average B-factor in the binding site: **{avg_b} Å²**\n\n"
            f"This region is {flexibility}.\n\n"
            "🔵 Blue = rigid (low B) → 🔴 Red = flexible (high B)\n"
            "Putty representation scales the cartoon tube width by B-factor — thick = floppy."
        )
        return _record(
            f"How flexible is the binding site in {ctx['pdb']} ({ctx['target']})? Show me the B-factors.",
            answer,
        )


# ---------------------------------------------------------------------------
# Behaviour class 34: RAFT-style "quote from execution output" records
# ---------------------------------------------------------------------------

def raft_single_property_example(smiles: str):
    """Two-turn RAFT: user provides RDKit stdout, assistant quotes from it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canon = Chem.MolToSmiles(mol)

    code = (
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "print('MW:', round(Descriptors.MolWt(mol), 1))\n"
        "print('logP:', round(Descriptors.MolLogP(mol), 2))\n"
        "print('HBD:', Lipinski.NumHDonors(mol))\n"
        "print('HBA:', Lipinski.NumHAcceptors(mol))\n"
        "print('TPSA:', round(rdMolDescriptors.CalcTPSA(mol), 1))\n"
    )
    stdout = _exec_block(code).strip()
    if stdout.startswith("[error]"):
        return None

    user_msg = (
        f"I ran this RDKit code on {canon}:\n\n"
        f"```python\n{code}```\n\n"
        f"Here is the output:\n```\n{stdout}\n```\n\n"
        "State the property values from the output above."
    )
    answer = (
        f"From the RDKit output for `{canon}`:\n\n"
        f"- {stdout.replace(chr(10), chr(10) + '- ')}\n\n"
        "These values are computed directly by RDKit — always trust the tool output over estimation."
    )
    return _record(user_msg, answer)


def raft_druglikeness_example(smiles: str):
    """RAFT: user provides RDKit stdout, assistant interprets and quotes exactly."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canon = Chem.MolToSmiles(mol)

    code = (
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "mw = round(Descriptors.MolWt(mol), 1)\n"
        "logp = round(Descriptors.MolLogP(mol), 2)\n"
        "hbd = Lipinski.NumHDonors(mol)\n"
        "hba = Lipinski.NumHAcceptors(mol)\n"
        "tpsa = round(rdMolDescriptors.CalcTPSA(mol), 1)\n"
        "rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)\n"
        "lip = mw < 500 and logp < 5 and hbd <= 5 and hba <= 10\n"
        "print(f'MW: {mw}')\n"
        "print(f'logP: {logp}')\n"
        "print(f'HBD: {hbd}')\n"
        "print(f'HBA: {hba}')\n"
        "print(f'TPSA: {tpsa}')\n"
        "print(f'RotBonds: {rotb}')\n"
        "print(f'Lipinski: {\"PASS\" if lip else \"FAIL\"}')\n"
    )
    stdout = _exec_block(code).strip()
    if stdout.startswith("[error]"):
        return None

    passes = "PASS" in stdout
    verdict = "passes" if passes else "fails"

    user_msg = (
        f"Assess the drug-likeness of {canon}. Here is the RDKit output:\n\n"
        f"```\n{stdout}\n```\n\n"
        "Interpret these results."
    )
    answer = (
        f"Based on the RDKit output, `{canon}` **{verdict}** Lipinski's Rule of Five:\n\n"
    )
    for line in stdout.strip().split("\n"):
        if ":" in line:
            prop, val = line.split(":", 1)
            prop, val = prop.strip(), val.strip()
            if prop in ("MW", "logP", "HBD", "HBA"):
                limits = {"MW": "< 500", "logP": "< 5", "HBD": "≤ 5", "HBA": "≤ 10"}
                answer += f"- **{prop}: {val}** ({limits.get(prop, '')})\n"
            else:
                answer += f"- **{prop}: {val}**\n"
    answer += (
        "\nAll values are from the RDKit computation above, not estimated."
    )
    return _record(user_msg, answer)


def raft_with_distractor_example(smiles: str):
    """RAFT with distractor: user provides output + a wrong value, assistant picks the correct one."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canon = Chem.MolToSmiles(mol)
    mw = round(Descriptors.MolWt(mol), 1)
    logp = round(Descriptors.MolLogP(mol), 2)

    wrong_mw = round(mw + random.uniform(15, 50) * random.choice([-1, 1]), 1)
    wrong_logp = round(logp + random.uniform(1, 3) * random.choice([-1, 1]), 2)

    code = (
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "print('MW:', round(Descriptors.MolWt(mol), 1))\n"
        "print('logP:', round(Descriptors.MolLogP(mol), 2))\n"
    )
    stdout = _exec_block(code).strip()
    if stdout.startswith("[error]"):
        return None

    user_msg = (
        f"I found conflicting values for {canon}.\n\n"
        f"A database entry says MW = {wrong_mw} Da and logP = {wrong_logp}.\n\n"
        f"But RDKit gives:\n```\n{stdout}\n```\n\n"
        "Which values should I trust?"
    )
    answer = (
        f"Trust the **RDKit computation**. The database values (MW {wrong_mw}, logP {wrong_logp}) "
        f"may be from a different salt form, tautomer, or simply an error.\n\n"
        f"The correct values from RDKit are:\n"
        f"- **MW: {mw} Da**\n"
        f"- **logP: {logp}**\n\n"
        "When in doubt, always recompute with RDKit from the canonical SMILES."
    )
    return _record(user_msg, answer)


# ---------------------------------------------------------------------------
# Generator registry + main
# ---------------------------------------------------------------------------

def _make_generators():
    def _subst():
        q, name = random.choice(SUBSTRUCTURE_QUERIES)
        t = random.choice(SEED_SMILES)
        return substructure_example(q, name, t)

    return [
        # --- SMILES / RDKit (original + Round 3) ---
        ("smiles_literacy",      lambda: smiles_canonicalise_example(random.choice(SEED_SMILES))),
        ("property_reasoning",   lambda: druglikeness_example(random.choice(SEED_SMILES))),
        ("murcko_scaffold",      lambda: murcko_scaffold_example(random.choice(SEED_SMILES))),
        ("tanimoto",             lambda: tanimoto_example(*random.sample(SEED_SMILES, 2))),
        ("substructure",         _subst),
        ("medchem_mmp",          lambda: medchem_mmp_example(*random.sample(SEED_SMILES, 2))),
        ("single_property",      lambda: single_property_example(random.choice(SEED_SMILES))),
        ("single_property",      lambda: single_property_example(random.choice(SEED_SMILES))),
        ("qed_score",            lambda: qed_example(random.choice(SEED_SMILES))),
        ("molecular_formula",    lambda: molecular_formula_example(random.choice(SEED_SMILES))),
        ("smiles_validation",    smiles_validation_example),
        ("full_profile",         lambda: full_profile_example(random.choice(SEED_SMILES))),
        ("full_profile",         lambda: full_profile_example(random.choice(SEED_SMILES))),
        ("property_comparison",  lambda: property_comparison_example(*random.sample(SEED_SMILES, 2))),
        ("corpus_drug",          corpus_drug_example),
        ("corpus_drug",          corpus_drug_example),
        ("murcko_sar",           lambda: murcko_with_sar_example(*random.sample(SEED_SMILES, 2))),
        # --- Structural biology (Round 4) ---
        ("pymol_selection",      pymol_selection_example),
        ("pymol_selection",      pymol_selection_example),
        ("pymol_viz",            pymol_viz_example),
        ("pymol_viz",            pymol_viz_example),
        ("pymol_script",         pymol_example),
        ("plip_analysis",        plip_analysis_example),
        ("plip_analysis",        plip_analysis_example),
        ("plip_interpretation",  plip_interpretation_example),
        ("pdb_format",           pdb_format_example),
        ("pdb_format",           pdb_format_example),
        ("pdbtools",             pdbtools_example),
        ("pdbtools",             pdbtools_example),
        ("biopython",            biopython_example),
        ("biopython",            biopython_example),
        ("chimerax",             chimerax_example),
        ("chimerax",             chimerax_example),
        ("gemmi",                gemmi_example),
        ("struct_prep",          struct_prep_example),
        ("docking_interp",       docking_interpretation_example),
        ("docking_interp",       docking_interpretation_example),
        # --- Visualization / R (Round 4) ---
        ("plotly",               plotly_example),
        ("plotly",               plotly_example),
        ("r_recipe",             r_recipe_example),
        ("r_recipe",             r_recipe_example),
        # --- Protein families (Round 4+) ---
        ("protein_family",       protein_family_example),
        ("protein_family",       protein_family_example),
        ("protein_family",       protein_family_example),
        # --- Numerical fidelity improvement ---
        ("code_then_quote",      lambda: code_then_quote_example(random.choice(SEED_SMILES))),
        ("code_then_quote",      lambda: code_then_quote_example(random.choice(SEED_SMILES))),
        ("code_then_quote",      lambda: code_then_quote_example(random.choice(SEED_SMILES))),
        ("rounding_precision",   lambda: rounding_precision_example(random.choice(SEED_SMILES))),
        ("rounding_precision",   lambda: rounding_precision_example(random.choice(SEED_SMILES))),
        # --- RAFT (quote from execution output) ---
        ("raft_single",          lambda: raft_single_property_example(random.choice(SEED_SMILES))),
        ("raft_single",          lambda: raft_single_property_example(random.choice(SEED_SMILES))),
        ("raft_single",          lambda: raft_single_property_example(random.choice(SEED_SMILES))),
        ("raft_druglike",        lambda: raft_druglikeness_example(random.choice(SEED_SMILES))),
        ("raft_druglike",        lambda: raft_druglikeness_example(random.choice(SEED_SMILES))),
        ("raft_druglike",        lambda: raft_druglikeness_example(random.choice(SEED_SMILES))),
        ("raft_distractor",      lambda: raft_with_distractor_example(random.choice(SEED_SMILES))),
        ("raft_distractor",      lambda: raft_with_distractor_example(random.choice(SEED_SMILES))),
        # --- Personality ---
        ("personality_drug",     lambda: personality_druglikeness_example(random.choice(SEED_SMILES))),
        ("personality_drug",     lambda: personality_druglikeness_example(random.choice(SEED_SMILES))),
        ("personality_drug",     lambda: personality_druglikeness_example(random.choice(SEED_SMILES))),
        ("personality_valid",    personality_validation_example),
        ("personality_valid",    personality_validation_example),
        ("personality_struct",   personality_structure_example),
        ("personality_struct",   personality_structure_example),
        ("personality_struct",   personality_structure_example),
        # --- Robustness ---
        ("refusal",              refusal_example),
        ("refusal",              refusal_example),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",  type=Path, default=Path("data/sft"))
    ap.add_argument("--n",    type=int,  default=7500)
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
