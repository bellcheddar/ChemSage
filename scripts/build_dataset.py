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
    from rdkit.Chem.QED import qed as _qed
    from rdkit.Chem.MolStandardize import rdMolStandardize as _rdMolStd
    from rdkit.Chem import inchi as _rdkitInchi
    from rdkit.Chem import FilterCatalog as _FilterCatalog
    from rdkit.Chem import rdFMCS as _rdFMCS
    from rdkit.Chem.Recap import RecapDecompose as _RecapDecompose
    from rdkit.Chem.BRICS import BRICSDecompose as _BRICSDecompose
    from rdkit.Chem import rdChemReactions as _rdChemReactions
except ImportError:
    raise SystemExit("RDKit is required: pip install rdkit")

# SA score from RDKit contrib
import os as _os
from rdkit.Chem import RDConfig as _RDConfig
sys.path.append(_os.path.join(_RDConfig.RDContribDir, "SA_Score"))
try:
    import sascorer as _sascorer
    _HAS_SASCORER = True
except ImportError:
    _HAS_SASCORER = False

# Build PAINS + Brenk filter catalog once at import time
def _build_filter_catalog():
    params = _FilterCatalog.FilterCatalogParams()
    params.AddCatalog(_FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    params.AddCatalog(_FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return _FilterCatalog.FilterCatalog(params)

_FILTER_CATALOG = _build_filter_catalog()

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
    "CC(=O)Nc1ccc(Cl)cc1",                              # 4-chloroacetanilide
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
    # Round 5: broader scaffold diversity
    "CN(C)CCCN1c2ccccc2CCc2ccccc21",                    # imipramine
    "OC(=O)c1ccc(Nc2ncc3cc(Cl)ccc3n2)cc1",             # erlotinib analogue
    "CC(C)(C)c1cc(CN2CCN(Cc3cccc(C(F)(F)F)c3)CC2)cc(C(C)(C)C)c1O",  # tetrahydropyrimidine
    "Clc1ccc(C2=NCC(=O)Nc3ccccc23)cc1",                # oxazepam scaffold
    "COCCOC1CN(C(=O)c2cc3cc(F)ccc3[nH]2)CC1",         # fluoroindole amide
    "CC1=C(C(=O)Nc2ccc(F)cc2F)C(c2ccc(Cl)cc2)NC(=O)N1",  # urea compound
    "O=C(c1ccc(F)cc1)c1ccc(NC2=NC(=O)CS2)cc1",        # thiazolinone
    "CC(=O)Nc1ccc2c(c1)C(C)(C)CC2=O",                 # naphthalenone acetamide
    "CN1CCN(c2nc(-c3ccccc3)c3ccccc3n2)CC1",           # quinoxaline piperazine
    "Fc1ccc(-c2cn(-c3ccccc3)nn2)cc1",                  # triazole fluorobenzene
    "O=C(O)c1ccc(NC(=O)c2ccc(Cl)s2)cc1",              # thiophene sulfonamide
    "CC(C)c1nc(N2CCOCC2)nc2ccccc12",                   # morpholine quinazoline
    "Nc1nc(Cl)nc2nccc12",                               # adenine analogue (Cl)
    "CC1(C)CCC(=C2CCCCC2)CC1",                         # bicyclic olefin
    "O=S(=O)(Nc1ccc(Br)cc1)c1ccccc1",                 # bromophenyl sulfonamide
    "O=C(O)c1cccc(NC(=O)Nc2ccc(Cl)cc2)c1",            # urea benzoic acid
    "CCOC(=O)c1ccc(NC(=O)c2ccccc2Cl)cc1",             # ester benzamide
    "CN(Cc1ccc(F)cc1)S(=O)(=O)c1ccc(C)cc1",           # N-methyl sulfonamide
    "O=C(O)CCc1ccc(Br)cc1",                             # bromophenylpropionic acid
    "CCc1cc(C)c(C)cc1NC(=O)c1ccc(Cl)cc1",             # alkylbenzamide
    "c1ccc(-c2nnco2)cc1",                               # phenyl oxadiazole
    "CC(=O)Nc1ccc(-c2nc3ccccc3s2)cc1",                # benzothiazole acetamide
    "O=C(Nc1ccc(Cl)cc1)c1cccs1",                       # thiophene amide
    "C(#N)c1ccc(C(=O)Nc2cccc(F)c2)cc1",               # cyano fluorobenzamide
    "CC(=O)c1ccc(NC(=O)c2ccc(F)cc2)cc1",              # fluorobenzamide
    "O=C(O)c1ccc(F)cc1NC(=O)c1ccncc1",                # pyridine fluorobenzoic acid
    "CCc1ccc(C(=O)Nc2ccccc2F)cc1",                     # fluoroaniline ethylbenzamide
    "O=C(Nc1cccc(Cl)c1)c1ccc(F)cc1",                  # fluorobenzamide chloroaniline
    "Cn1cc(-c2ccc(Cl)cc2)nn1",                          # chlorophenyl methyltriazole
    "CC1=NN(c2ccc(Cl)cc2)C(=O)C1",                    # pyrazolone chloro
    "CC(=O)Nc1ccc(OCC(=O)O)cc1",                      # phenoxy acetic acid acetamide
    "Brc1ccc(NC(=O)Cc2cccs2)cc1",                      # thienyl bromoanilide
    "O=C(Nc1cccc(C(F)(F)F)c1)c1ccc(Cl)cc1",          # trifluoromethyl amide
    "CCN(CC)C(=O)c1cccc(NC(=O)c2ccc(F)cc2)c1",       # fluorobenzamide diethylamine
    "O=C(O)c1ccc(F)cc1",                                # fluorobenzoic acid
    "Cc1ccc(C(=O)Nc2ccccc2OC)cc1",                    # methoxybenzamide
    "O=c1[nH]c2ccccc2n1-c1ccccc1",                    # benzimidazolone
    "Cc1nn(C)cc1C(=O)Nc1ccc(F)cc1",                   # methylpyrazole amide
    "CCOC(=O)c1cc2ccccc2[nH]1",                        # ethyl indole-2-carboxylate
    "O=C(Nc1ccccc1)c1ccncc1",                           # isonicotinamide
    # Macrolide / lactone scaffolds
    "O=C1CCCCCCCCO1",                                   # oxacyclodecan-2-one (macrolactone)
    "CC1OC(=O)C(C)C(=O)CCCC(C)CC1=O",                  # simplified macrolide
    # Natural product-like
    "OC(/C=C/c1ccc(O)cc1)c1ccc(O)cc1",                 # resveratrol analogue
    "COc1ccc([C@@H]2OCC3=CC(=O)c4c(OC)cccc4[C@@H]23)cc1",  # dehydropodophyllotoxin-like
    "O=C1OC[C@@H]2Cc3c(O)cc(O)cc3[C@H]12",             # flavanone scaffold
    "O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",            # apigenin-like
    # PROTAC-like / bivalent
    "O=C(CCCOCCOCCO)Nc1ccc(NC(=O)c2ccccc2)cc1",        # PEG linker bivalent
    "O=C(CCC(=O)Nc1ccc(F)cc1)Nc1cccc(F)c1",            # glutaramide bis-aniline
    # Kinase inhibitors (EGFR/VEGFR-like)
    "Nc1ccc(NC(=O)c2ccc(Cl)c(NC3=NCCN3)c2)cc1",        # EGFR-type anilinopyrimidine
    "CC(C)(C)c1cc(NC(=O)Nc2ccc(OC3CCNCC3)cc2)no1",     # isoxazole urea
    "Cc1ccc(-c2nn(C)cc2C(N)=O)cc1NC(=O)c1ccc(F)cc1",   # pyrazole amide kinase
    # Fragment-sized (MW 150–250)
    "c1ccc2[nH]ccc2c1",                                  # indole
    "c1cnc2[nH]ccc2c1",                                 # 7-azaindole
    "O=c1[nH]c2ccccc2n1C",                              # 1-methylbenzimidazolinone
    "Cc1ccc(CN2CCOCC2)cc1",                             # 4-methylbenzyl morpholine
    "OC(c1ccncc1)c1ccncc1",                             # di(4-pyridyl)carbinol
    "N#Cc1ccc(NC(=O)c2cccs2)cc1",                       # thienyl nitrile amide
    # Peptide mimetics / beta-lactam
    "O=C1CN(Cc2ccccc2)C1=O",                            # 1-benzylazetidine-2,3-dione
    "O=C1CCN1Cc1ccccc1",                                 # N-benzyl-beta-lactam
    "CC(NC(=O)OC(C)(C)C)C(=O)NC1CCCCC1",               # Boc-Ala-cyclohexylamide
    # Heterocycle diversity
    "c1cc2cccc3cccc1c23",                                # pyrene
    "n1ccnc2ccccc12",                                    # benzimidazole
    "O=c1cccc[nH]1",                                     # 2-pyridinone
    "Cc1cc(=O)oc2ccccc12",                               # 4-methylcoumarin
    "c1cnc2ccccn12",                                     # 1,5-naphthyridine
    "O=c1ccc2ccccc2[nH]1",                               # 2-quinolinone
    # High HBD (for HBD tests)
    "OCC(O)C(O)C(O)CO",                                  # xylitol
    "NC(CO)(CO)CO",                                      # tris(hydroxymethyl)aminomethane
    # Charged / zwitterionic
    "OC(=O)c1ccc(S(=O)(=O)[O-])cc1",                    # sulfosalicylic acid anion
    "CC(=[NH2+])NCCC[C@@H](N)C(=O)[O-]",               # arginine zwitterion-like
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
        score = round(_qed(mol), 3)
    except Exception:
        return None
    interp = (
        "highly drug-like (QED > 0.67)" if score > 0.67 else
        "moderately drug-like (0.33 < QED ≤ 0.67)" if score > 0.33 else
        "poorly drug-like (QED ≤ 0.33)"
    )
    # Alternate import style so model sees both forms
    import_style = random.choice([
        "from rdkit.Chem.QED import qed\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "print(round(qed(mol), 3))\n",
        "from rdkit.Chem import QED\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "print(round(QED.qed(mol), 3))\n",
    ])
    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        f"{import_style}"
        "```\n\n"
        f"QED = {score}. The compound is {interp}. "
        "QED (Quantitative Estimate of Drug-likeness) integrates MW, logP, HBD, HBA, PSA, "
        "rotatable bonds, aromatic rings, and structural alerts into a single 0–1 score."
    )
    questions = [
        f"What is the QED score of {smiles}?",
        f"Calculate the QED drug-likeness score for {smiles}.",
        f"Compute the QED of {smiles} using RDKit.",
        f"What is the quantitative estimate of drug-likeness for {smiles}?",
        f"How drug-like is {smiles}? Compute the QED score.",
        f"Give me the QED score for this molecule: {smiles}",
        f"Use RDKit to compute the QED drug-likeness of {smiles}.",
    ]
    return _record(random.choice(questions), answer)


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
# R5 NEW: Standalone TPSA — long-form and acronym phrasings
# ---------------------------------------------------------------------------

def tpsa_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    tpsa = round(rdMolDescriptors.CalcTPSA(mol), 1)
    veber = "meets" if tpsa <= 140 else "exceeds"
    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import rdMolDescriptors\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "print(round(rdMolDescriptors.CalcTPSA(mol), 1))\n"
        "```\n\n"
        f"TPSA = {tpsa} Å². "
        f"This compound {veber} the Veber oral absorption threshold (TPSA ≤ 140 Å²). "
        "TPSA is the sum of surface contributions of polar atoms (N, O, S and attached H)."
    )
    questions = [
        f"What is the TPSA of {smiles}?",
        f"Calculate the TPSA for {smiles}.",
        f"Compute the topological polar surface area of {smiles}.",
        f"What is the topological polar surface area of {smiles}?",
        f"Use RDKit to calculate the topological polar surface area for {smiles}.",
        f"What is the polar surface area of {smiles}?",
        f"Compute the TPSA of {smiles} using RDKit.",
        f"Calculate the polar surface area (TPSA) of {smiles}.",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# R5 NEW: Standalone scaffold equality (explicit Boolean, not Tanimoto)
# ---------------------------------------------------------------------------

def scaffold_equality_example(smi_a: str, smi_b: str):
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
    same = (scaf_a == scaf_b)
    answer = (
        "I will compare Murcko scaffolds to determine scaffold identity.\n\n"
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem.Scaffolds import MurckoScaffold\n"
        f"mol_a = Chem.MolFromSmiles('{smi_a}')\n"
        f"mol_b = Chem.MolFromSmiles('{smi_b}')\n"
        "scaf_a = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol_a))\n"
        "scaf_b = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol_b))\n"
        "print('A:', scaf_a)\n"
        "print('B:', scaf_b)\n"
        "print('Same scaffold:', scaf_a == scaf_b)\n"
        "```\n\n"
        f"Scaffold A: `{scaf_a}`  \n"
        f"Scaffold B: `{scaf_b}`  \n\n"
        + ("**Same scaffold: Yes.** Both compounds share the same Murcko core. "
           "Differences lie in peripheral substituents only."
           if same else
           "**Same scaffold: No.** The compounds have distinct Murcko scaffolds, "
           "indicating they belong to different structural series.")
    )
    questions = [
        f"Do these two compounds share the same scaffold?\nA: {smi_a}\nB: {smi_b}",
        f"Are these compounds in the same scaffold class?\nA: {smi_a}\nB: {smi_b}",
        f"Do A and B share the same Murcko scaffold?\nA: {smi_a}\nB: {smi_b}",
        f"Check if these two molecules have the same core scaffold:\nA: {smi_a}\nB: {smi_b}",
        f"Are {smi_a} and {smi_b} matched molecular pairs on the same scaffold?",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# R5 NEW: Standalone HBD / HBA / RotBonds (targeted single-property)
# ---------------------------------------------------------------------------

_HBX_TEMPLATES = [
    {
        "prop": "hydrogen bond donors",
        "code": "Lipinski.NumHDonors(mol)",
        "imports": "from rdkit import Chem\nfrom rdkit.Chem import Lipinski",
        "compute": lambda m: Lipinski.NumHDonors(m),
        "limit": "≤ 5 (Lipinski)",
        "questions": [
            "How many hydrogen bond donors does {s} have?",
            "Calculate the number of HBD for {s}.",
            "What is the HBD count of {s}?",
            "Compute the number of H-bond donors in {s} using RDKit.",
        ],
    },
    {
        "prop": "hydrogen bond acceptors",
        "code": "Lipinski.NumHAcceptors(mol)",
        "imports": "from rdkit import Chem\nfrom rdkit.Chem import Lipinski",
        "compute": lambda m: Lipinski.NumHAcceptors(m),
        "limit": "≤ 10 (Lipinski)",
        "questions": [
            "How many hydrogen bond acceptors does {s} have?",
            "Calculate the HBA count for {s}.",
            "What is the number of H-bond acceptors in {s}?",
            "Compute the number of hydrogen bond acceptors of {s} using RDKit.",
        ],
    },
    {
        "prop": "rotatable bonds",
        "code": "rdMolDescriptors.CalcNumRotatableBonds(mol)",
        "imports": "from rdkit import Chem\nfrom rdkit.Chem import rdMolDescriptors",
        "compute": lambda m: rdMolDescriptors.CalcNumRotatableBonds(m),
        "limit": "≤ 10 (Veber)",
        "questions": [
            "How many rotatable bonds does {s} have?",
            "Calculate the number of rotatable bonds in {s}.",
            "What is the rotatable bond count of {s}?",
            "Compute rotatable bonds for {s} using RDKit.",
        ],
    },
    {
        "prop": "aromatic rings",
        "code": "rdMolDescriptors.CalcNumAromaticRings(mol)",
        "imports": "from rdkit import Chem\nfrom rdkit.Chem import rdMolDescriptors",
        "compute": lambda m: rdMolDescriptors.CalcNumAromaticRings(m),
        "limit": None,
        "questions": [
            "How many aromatic rings are in {s}?",
            "Count the aromatic rings in {s}.",
            "What is the aromatic ring count of {s}?",
            "Compute the number of aromatic rings in {s} using RDKit.",
        ],
    },
]


def hbd_hba_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    tmpl = random.choice(_HBX_TEMPLATES)
    value = tmpl["compute"](mol)
    limit_note = f" Lipinski/Veber threshold: {tmpl['limit']}." if tmpl["limit"] else ""
    answer = (
        f"```python\n"
        f"{tmpl['imports']}\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        f"print({tmpl['code']})\n"
        "```\n\n"
        f"Number of {tmpl['prop']}: **{value}**.{limit_note}"
    )
    q_template = random.choice(tmpl["questions"])
    return _record(q_template.format(s=smiles), answer)


# ---------------------------------------------------------------------------
# R5 NEW: Largest fragment / salt stripping
# ---------------------------------------------------------------------------

def salt_strip_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        chooser = _rdMolStd.LargestFragmentChooser()
        parent = chooser.choose(mol)
        parent_smi = Chem.MolToSmiles(parent)
        orig_smi = Chem.MolToSmiles(mol)
    except Exception:
        return None
    changed = (parent_smi != orig_smi)
    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem.MolStandardize import rdMolStandardize\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "chooser = rdMolStandardize.LargestFragmentChooser()\n"
        "parent = chooser.choose(mol)\n"
        "print(Chem.MolToSmiles(parent))\n"
        "```\n\n"
        + (f"Parent (largest fragment): `{parent_smi}`. "
           "Counter-ions and co-crystals were stripped." if changed else
           f"The molecule `{parent_smi}` contains no salt or co-crystal; it is already a single fragment.")
    )
    questions = [
        f"Strip the salt from {smiles} and return the parent molecule.",
        f"What is the largest organic fragment of {smiles}?",
        f"Remove the counter-ion from {smiles}.",
        f"Get the parent compound from this salt form: {smiles}",
        f"Standardise {smiles} by choosing the largest fragment.",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# R5 NEW: InChI and InChIKey
# ---------------------------------------------------------------------------

def inchikey_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        inchi_str = _rdkitInchi.MolToInchi(mol)
        if not inchi_str:
            return None
        inchi_key = _rdkitInchi.InchiToInchiKey(inchi_str)
        if not inchi_key:
            return None
    except Exception:
        return None
    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import inchi\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "inchi_str = inchi.MolToInchi(mol)\n"
        "inchi_key = inchi.InchiToInchiKey(inchi_str)\n"
        "print('InChI:', inchi_str)\n"
        "print('InChIKey:', inchi_key)\n"
        "```\n\n"
        f"InChI: `{inchi_str}`  \n"
        f"InChIKey: `{inchi_key}`  \n\n"
        "The InChIKey is a fixed-length hash (27 chars) suitable as a database identifier. "
        "The first 14 chars encode connectivity; the last 9 encode stereo and charge."
    )
    questions = [
        f"Compute the InChIKey for {smiles}.",
        f"What is the InChIKey of {smiles}?",
        f"Generate the InChI and InChIKey for {smiles}.",
        f"Convert this SMILES to InChIKey: {smiles}",
        f"What is the standard InChI identifier for {smiles}?",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# R5 NEW: Stereo / chirality perception
# ---------------------------------------------------------------------------

def stereo_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    except Exception:
        return None
    n_chiral = len(chiral)
    stereo_desc = ", ".join(f"atom {idx} ({cip})" for idx, cip in chiral) if chiral else "none"
    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "Chem.AssignStereochemistry(mol, cleanIt=True, force=True)\n"
        "chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)\n"
        "print('Chiral centers:', chiral_centers)\n"
        "print('Count:', len(chiral_centers))\n"
        "```\n\n"
        + (f"**{n_chiral} chiral center(s):** {stereo_desc}. "
           f"Maximum {2**n_chiral} stereoisomers possible." if n_chiral > 0 else
           "**No chiral centers detected.** The molecule is achiral.")
    )
    questions = [
        f"How many chiral centers does {smiles} have?",
        f"Identify the stereocenters in {smiles}.",
        f"What is the stereochemistry of {smiles}?",
        f"Find all chiral centers in {smiles} using RDKit.",
        f"Does {smiles} have any stereocenters?",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# R5 NEW: Tautomer enumeration
# ---------------------------------------------------------------------------

def tautomer_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        enumerator = _rdMolStd.TautomerEnumerator()
        tauts = enumerator.Enumerate(mol)
        taut_smiles = sorted({Chem.MolToSmiles(t) for t in tauts})
        if len(taut_smiles) < 2:
            return None  # skip molecules with no meaningful tautomers
        canonical_taut = enumerator.Canonicalize(mol)
        canon_smi = Chem.MolToSmiles(canonical_taut)
    except Exception:
        return None
    n = len(taut_smiles)
    shown = taut_smiles[:4]
    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem.MolStandardize import rdMolStandardize\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "enumerator = rdMolStandardize.TautomerEnumerator()\n"
        "tauts = enumerator.Enumerate(mol)\n"
        "taut_smiles = sorted({Chem.MolToSmiles(t) for t in tauts})\n"
        "canonical = enumerator.Canonicalize(mol)\n"
        "print('Tautomers:', len(taut_smiles))\n"
        "print('Canonical tautomer:', Chem.MolToSmiles(canonical))\n"
        "```\n\n"
        f"**{n} tautomer(s) enumerated.** Canonical tautomer: `{canon_smi}`.\n\n"
        "Representative forms:\n"
        + "\n".join(f"- `{s}`" for s in shown)
        + ("\n- _(and more…)_" if n > 4 else "")
    )
    questions = [
        f"Enumerate the tautomers of {smiles}.",
        f"How many tautomers does {smiles} have?",
        f"What are the tautomeric forms of {smiles}?",
        f"Generate all tautomers of {smiles} using RDKit.",
        f"What is the canonical tautomer of {smiles}?",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# R5 NEW: Multi-molecule QED ranking (3-4 compounds)
# ---------------------------------------------------------------------------

def multi_mol_rank_example(smiles_list: list):
    mols, valid = [], []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append((s, m))
    if len(mols) < 3:
        return None
    try:
        results = [(s, round(_qed(m), 3)) for s, m in mols]
    except Exception:
        return None
    ranked = sorted(results, key=lambda x: x[1], reverse=True)
    code_lines = "\n".join(
        f"    ('{s}', round(qed(Chem.MolFromSmiles('{s}')), 3)),"
        for s, _ in mols
    )
    ranking_text = "\n".join(
        f"{i+1}. `{s}` — QED {score}" for i, (s, score) in enumerate(ranked)
    )
    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem.QED import qed\n"
        "compounds = [\n"
        f"{code_lines}\n"
        "]\n"
        "ranked = sorted(compounds, key=lambda x: x[1], reverse=True)\n"
        "for i, (smi, score) in enumerate(ranked, 1):\n"
        "    print(f'{i}. {smi}: QED = {score}')\n"
        "```\n\n"
        f"QED ranking (highest to lowest drug-likeness):\n\n{ranking_text}\n\n"
        f"`{ranked[0][0]}` has the highest QED ({ranked[0][1]}) and is the most drug-like. "
        f"`{ranked[-1][0]}` has the lowest QED ({ranked[-1][1]})."
    )
    letters = "ABCDEFG"
    compounds_block = "\n".join(f"{letters[i]}: {s}" for i, (s, _) in enumerate(mols))
    questions = [
        f"Rank these compounds by QED drug-likeness score:\n{compounds_block}",
        f"Which of these molecules is most drug-like? Compute QED for each:\n{compounds_block}",
        f"Sort these compounds from most to least drug-like using QED:\n{compounds_block}",
        f"Compare the QED scores of these compounds and rank them:\n{compounds_block}",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# R5 NEW: QED code-then-quote (QED value stated verbatim from code output)
# ---------------------------------------------------------------------------

def qed_code_then_quote_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        score = round(_qed(mol), 3)
    except Exception:
        return None
    mw   = round(Descriptors.MolWt(mol), 1)
    logp = round(Descriptors.MolLogP(mol), 2)
    interp = (
        "highly drug-like" if score > 0.67 else
        "moderately drug-like" if score > 0.33 else
        "poorly drug-like"
    )
    answer = (
        "I will compute the QED score and all major descriptors.\n\n"
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski\n"
        "from rdkit.Chem.QED import qed\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "print('MW:', round(Descriptors.MolWt(mol), 1))\n"
        "print('logP:', round(Descriptors.MolLogP(mol), 2))\n"
        "print('QED:', round(qed(mol), 3))\n"
        "```\n\n"
        f"MW: {mw}, logP: {logp}, QED: {score}. "
        f"The compound is {interp} (QED {score})."
    )
    return _record(
        f"Compute the QED score and key properties of {smiles}. "
        "Quote the exact values from the RDKit output.",
        answer,
    )


# ---------------------------------------------------------------------------
# R5 NEW: Fingerprint similarity (explicit, no scaffold confusion)
# ---------------------------------------------------------------------------

def fingerprint_similarity_example(smi_a: str, smi_b: str):
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None:
        return None
    gen  = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp_a = gen.GetFingerprint(mol_a)
    fp_b = gen.GetFingerprint(mol_b)
    sim  = round(DataStructs.TanimotoSimilarity(fp_a, fp_b), 3)
    interp = (
        "closely related (Tc ≥ 0.7)" if sim >= 0.7 else
        "moderately similar (0.4 ≤ Tc < 0.7)" if sim >= 0.4 else
        "structurally dissimilar (Tc < 0.4)"
    )
    answer = (
        "I will compute Tanimoto similarity using Morgan (ECFP4) fingerprints.\n\n"
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
        f"Tanimoto similarity (Morgan r=2): **{sim}**. The compounds are {interp}."
    )
    questions = [
        f"What is the Tanimoto similarity between {smi_a} and {smi_b}?",
        f"Compute the Morgan fingerprint similarity of {smi_a} and {smi_b}.",
        f"Calculate the structural similarity between {smi_a} and {smi_b} using ECFP4.",
        f"How similar are {smi_a} and {smi_b}? Use Tanimoto on Morgan fingerprints.",
        f"What is the ECFP4 Tanimoto coefficient between {smi_a} and {smi_b}?",
        f"Calculate the fingerprint similarity of these two compounds:\nA: {smi_a}\nB: {smi_b}",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# R5 NEW: Lipinski card with code-then-quote pattern
# ---------------------------------------------------------------------------

def lipinski_card_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw   = round(Descriptors.MolWt(mol), 1)
    logp = round(Descriptors.MolLogP(mol), 2)
    hbd  = Lipinski.NumHDonors(mol)
    hba  = Lipinski.NumHAcceptors(mol)
    tpsa = round(rdMolDescriptors.CalcTPSA(mol), 1)
    try:
        score = round(_qed(mol), 3)
    except Exception:
        score = None
    passes = mw < 500 and logp < 5 and hbd <= 5 and hba <= 10
    verdict = "✅ PASS" if passes else "❌ FAIL"
    checks = [
        ("MW", mw, "< 500", mw < 500),
        ("logP", logp, "< 5", logp < 5),
        ("HBD", hbd, "≤ 5", hbd <= 5),
        ("HBA", hba, "≤ 10", hba <= 10),
    ]
    card_lines = "\n".join(
        f"| {prop} | {val} | {limit} | {'✅' if ok else '❌'} |"
        for prop, val, limit, ok in checks
    )
    qed_line = f"\nQED drug-likeness score: {score}." if score else ""
    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors\n"
        "from rdkit.Chem.QED import qed\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "print('MW:',   round(Descriptors.MolWt(mol), 1))\n"
        "print('logP:', round(Descriptors.MolLogP(mol), 2))\n"
        "print('HBD:',  Lipinski.NumHDonors(mol))\n"
        "print('HBA:',  Lipinski.NumHAcceptors(mol))\n"
        "print('TPSA:', round(rdMolDescriptors.CalcTPSA(mol), 1))\n"
        "print('QED:',  round(qed(mol), 3))\n"
        "```\n\n"
        f"**Lipinski Rule of Five: {verdict}**\n\n"
        f"| Property | Value | Limit | Pass? |\n"
        f"|---|---|---|---|\n"
        f"{card_lines}\n"
        f"| TPSA | {tpsa} Å² | ≤ 140 (Veber) | {'✅' if tpsa <= 140 else '❌'} |"
        f"{qed_line}"
    )
    questions = [
        f"Generate a Lipinski card for {smiles}.",
        f"Assess the oral bioavailability of {smiles} using Lipinski and Veber rules.",
        f"Does {smiles} pass Lipinski's Rule of Five? Show a property table.",
        f"Compute MW, logP, HBD, HBA, TPSA and QED for {smiles}.",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# R5 NEW: PAINS / Brenk structural filter
# ---------------------------------------------------------------------------

_PAINS_CONTEXT_QUESTIONS = [
    "Screen {smiles} for PAINS and Brenk structural alerts.",
    "Does {smiles} contain any PAINS substructures?",
    "Check if {smiles} flags any pan-assay interference alerts.",
    "Run a structural alert screen on {smiles}.",
    "Is {smiles} a PAINS compound? Check with RDKit.",
]


def pains_filter_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    matches = list(_FILTER_CATALOG.GetMatches(mol))
    if matches:
        alerts = sorted({e.GetDescription() for e in matches})
        alert_text = ", ".join(f"`{a}`" for a in alerts)
        verdict = f"**PAINS/Brenk alert(s) detected:** {alert_text}."
        action = (
            "Avoid using this compound as a primary hit in biochemical assays without "
            "orthogonal confirmation (e.g., SPR, NanoBRET, or biophysical assay). "
            "Consider analogues with the alert removed."
        )
    else:
        verdict = "**Clean:** No PAINS or Brenk alerts detected."
        action = "This compound is suitable for primary screening."
    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import FilterCatalog\n"
        "params = FilterCatalog.FilterCatalogParams()\n"
        "params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)\n"
        "params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)\n"
        "catalog = FilterCatalog.FilterCatalog(params)\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "matches = list(catalog.GetMatches(mol))\n"
        "for entry in matches:\n"
        "    print(entry.GetDescription())\n"
        "print('Clean' if not matches else f'{len(matches)} alert(s)')\n"
        "```\n\n"
        f"{verdict} {action}"
    )
    return _record(
        random.choice(_PAINS_CONTEXT_QUESTIONS).format(smiles=smiles),
        answer,
    )


# ---------------------------------------------------------------------------
# R5 NEW: Structural alerts / reactive groups (hardcoded medchem knowledge)
# ---------------------------------------------------------------------------

_STRUCTURAL_ALERTS = [
    {
        "name": "Michael acceptor (α,β-unsaturated carbonyl)",
        "smarts": "[CX3](=[OX1])[CX3]=[CX3]",
        "risk": "Covalent warhead — reacts with Cys/Lys nucleophiles. Intentional in covalent drugs (e.g., ibrutinib); unacceptable as a non-covalent hit.",
        "example_smi": "O=C/C=C/c1ccccc1",  # cinnamaldehyde
        "example_name": "cinnamaldehyde",
    },
    {
        "name": "Aldehyde",
        "smarts": "[CX3H1](=O)[#6]",
        "risk": "Forms imines with Lys and Schiff bases — covalent, reversible. Flags as a Brenk alert. Can cause non-specific activity in HTS.",
        "example_smi": "O=Cc1ccccc1",  # benzaldehyde
        "example_name": "benzaldehyde",
    },
    {
        "name": "Epoxide",
        "smarts": "[OX2r3]1[#6r3][#6r3]1",
        "risk": "Reactive alkylating agent. Reacts non-specifically with DNA and proteins. Remove from non-covalent hit series.",
        "example_smi": "C1OC1",  # ethylene oxide
        "example_name": "epoxide",
    },
    {
        "name": "Nitroaromatic",
        "smarts": "[$([nX3](=O)),$([NX3](=O)=O),$([NX3+](=O)[O-])][a]",
        "risk": "Metabolically reduced to reactive nitroso/hydroxylamine intermediates — genotoxicity risk. Flagged in most drug safety panels.",
        "example_smi": "O=[N+]([O-])c1ccccc1",  # nitrobenzene
        "example_name": "nitrobenzene",
    },
    {
        "name": "Rhodanine",
        "smarts": "O=C1NC(=S)S1",
        "risk": "Classic PAINS scaffold — aggregates in solution and gives false positives in many biochemical assays. Promiscuous binder.",
        "example_smi": "O=C1NC(=S)S1",
        "example_name": "rhodanine",
    },
    {
        "name": "Acyl halide",
        "smarts": "[CX3](=[OX1])[F,Cl,Br,I]",
        "risk": "Highly reactive electrophile; hydrolyses rapidly in aqueous media. Not suitable as an HTS compound.",
        "example_smi": "ClC(=O)c1ccccc1",  # benzoyl chloride
        "example_name": "benzoyl chloride",
    },
]


def structural_alert_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Pick a random alert and check if this molecule triggers it OR use a known example
    alert = random.choice(_STRUCTURAL_ALERTS)
    patt = Chem.MolFromSmarts(alert["smarts"])
    if patt is None:
        return None
    # 60% of the time use the built-in example; 40% use seed SMILES (may be clean)
    if random.random() < 0.6:
        mol = Chem.MolFromSmiles(alert["example_smi"])
        smiles = alert["example_smi"]
        display_name = alert["example_name"]
    else:
        display_name = smiles
    hits = mol.HasSubstructMatch(patt) if mol else False
    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        f"patt = Chem.MolFromSmarts('{alert['smarts']}')\n"
        "print('Alert present:', mol.HasSubstructMatch(patt))\n"
        "```\n\n"
        f"**{alert['name']}** (`{alert['smarts']}`) — alert {'present' if hits else 'absent'} in `{display_name}`.\n\n"
        f"**Risk:** {alert['risk']}"
    )
    questions = [
        f"Does {display_name} ({smiles}) contain a {alert['name']}?",
        f"Check for reactive alerts in {smiles}: specifically {alert['name']}.",
        f"Is the substructure {alert['smarts']} present in {smiles}?",
        f"Assess the reactivity risk of {smiles} for {alert['name']}.",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# R5 NEW: Classic bioisostere pairs
# ---------------------------------------------------------------------------

_BIOISOSTERE_PAIRS = [
    {
        "original": "carboxylic acid (-COOH)",
        "bioisostere": "tetrazole (-CN₄H)",
        "original_smarts": "[CX3](=O)[OX2H1]",
        "iso_smarts": "c1nn[nH]n1",
        "rationale": (
            "Tetrazole is a classical bioisostere for carboxylic acid. "
            "Both are acidic (tetrazole pKa ~4.9 vs COOH ~4–5) and adopt similar geometry. "
            "Tetrazole is more metabolically stable (no glucuronidation) and often has better "
            "membrane permeability. Used in losartan (AT1 antagonist) in place of the COOH in valsartan."
        ),
    },
    {
        "original": "phenol (-OH on aromatic)",
        "bioisostere": "fluorine (-F)",
        "original_smarts": "[OX2H][c]",
        "iso_smarts": "[F][c]",
        "rationale": (
            "Fluorine is a weak bioisostere for phenol OH. Both have similar van der Waals radii "
            "(F=1.47 Å, O=1.52 Å) and can accept H-bonds, but F rarely donates. "
            "Replacing phenol with F removes a metabolic soft spot (glucuronidation, sulfation) "
            "and typically raises logP by ~0.1–0.5. Useful when the OH is not critical for binding."
        ),
    },
    {
        "original": "ester (-COOR)",
        "bioisostere": "amide (-CONH-)",
        "original_smarts": "[CX3](=O)[OX2H0][#6]",
        "iso_smarts": "[NX3][CX3](=[OX1])",
        "rationale": (
            "Amide is more metabolically stable than ester (less susceptible to esterase cleavage). "
            "H-bond donor capacity increases with the NH. The amide is more rigid (partial double bond "
            "character), which can improve selectivity but reduce flexibility. "
            "A prodrug strategy reverses this: use the ester for oral bioavailability, "
            "cleave to acid in vivo."
        ),
    },
    {
        "original": "halide (Cl, Br)",
        "bioisostere": "trifluoromethyl (-CF₃)",
        "original_smarts": "[Cl,Br][c]",
        "iso_smarts": "[CX4](F)(F)F",
        "rationale": (
            "-CF₃ is a metabolically stable bioisostere for aryl Cl/Br. It is electron-withdrawing "
            "(σp = +0.54), lipophilic (adds ~0.9 logP), and strongly resists oxidative metabolism. "
            "MW increases by ~51 Da (vs Cl=+16). Useful when the halogen is a metabolic hotspot."
        ),
    },
    {
        "original": "pyridine (N-heterocycle)",
        "bioisostere": "pyrimidine or pyridazine",
        "original_smarts": "n1ccccc1",
        "iso_smarts": "n1ccncc1",
        "rationale": (
            "Pyrimidine is a classical bioisostere for pyridine with an additional H-bond acceptor. "
            "Adding the second N reduces logP (~0.3–0.5 units) and increases aqueous solubility. "
            "Also reduces hERG liability (aromatic amines more basic). Used extensively in kinase "
            "inhibitors (imatinib, nilotinib, gefitinib)."
        ),
    },
]


def bioisostere_example(smiles: str):
    pair = random.choice(_BIOISOSTERE_PAIRS)
    answer = (
        f"**Bioisostere:** {pair['original']} → {pair['bioisostere']}\n\n"
        "```python\n"
        "from rdkit import Chem\n"
        "# Detect the original group\n"
        f"patt = Chem.MolFromSmarts('{pair['original_smarts']}')\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "has_group = mol.HasSubstructMatch(patt)\n"
        "print('Original group present:', has_group)\n"
        "```\n\n"
        f"**Rationale:** {pair['rationale']}"
    )
    questions = [
        f"What is a good bioisostere for the {pair['original']} group in {smiles}?",
        f"Suggest a bioisostere replacement for the {pair['original']} in {smiles}.",
        f"How can I replace the {pair['original']} in {smiles} with a bioisostere?",
        f"What is the bioisosteric equivalent of a {pair['original']}?",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# R5 NEW: Lead optimisation rules (CNS MPO, PFI, golden triangle)
# ---------------------------------------------------------------------------

def lead_opt_rules_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw   = round(Descriptors.MolWt(mol), 1)
    logp = round(Descriptors.MolLogP(mol), 2)
    hbd  = Lipinski.NumHDonors(mol)
    hba  = Lipinski.NumHAcceptors(mol)
    tpsa = round(rdMolDescriptors.CalcTPSA(mol), 1)
    rot  = rdMolDescriptors.CalcNumRotatableBonds(mol)
    arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    try:
        score = round(_qed(mol), 3)
    except Exception:
        score = None

    # PFI (Property Forecast Index) = logP + aromatic ring count
    pfi = round(logp + arom, 2)
    pfi_ok = pfi < 8
    # CNS MPO (simplified: 6 params, each 0-1)
    def _cns_mpo():
        s = 0.0
        s += 1.0 if mw <= 360 else (0.0 if mw >= 500 else (500 - mw) / 140)
        s += 1.0 if logp <= 3 else (0.0 if logp >= 5 else (5 - logp) / 2)
        s += 1.0 if tpsa >= 40 else (0.0 if tpsa == 0 else tpsa / 40)
        s += 0.0 if tpsa > 90 else (1.0 if tpsa <= 60 else (90 - tpsa) / 30)
        s += 1.0 if hbd <= 0 else (0.0 if hbd >= 3 else (3 - hbd) / 3)
        s += 1.0 if rot <= 8 else (0.0 if rot >= 10 else (10 - rot) / 2)
        return round(s, 2)
    cns_mpo = _cns_mpo()

    # Golden triangle: good CNS drugs have MW 200–450 and logP 1–3
    golden = (200 <= mw <= 450) and (1 <= logp <= 3)

    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "mw   = round(Descriptors.MolWt(mol), 1)\n"
        "logp = round(Descriptors.MolLogP(mol), 2)\n"
        "arom = rdMolDescriptors.CalcNumAromaticRings(mol)\n"
        "pfi  = round(logp + arom, 2)\n"
        "print('MW:', mw, 'logP:', logp, 'Arom rings:', arom, 'PFI:', pfi)\n"
        "```\n\n"
        f"**Lead-optimisation scorecard for `{smiles}`:**\n\n"
        f"| Rule | Value | Threshold | Result |\n"
        f"|---|---|---|---|\n"
        f"| PFI (logP + arom rings) | {pfi} | < 8 | {'✅' if pfi_ok else '❌'} |\n"
        f"| CNS MPO score | {cns_mpo}/6 | ≥ 4 = CNS-ready | {'✅' if cns_mpo >= 4 else '⚠️'} |\n"
        f"| Golden triangle (MW 200–450, logP 1–3) | MW {mw}, logP {logp} | — | {'✅' if golden else '❌'} |\n"
        f"| HBD | {hbd} | ≤ 3 (CNS) | {'✅' if hbd <= 3 else '❌'} |\n"
        f"| TPSA | {tpsa} Å² | < 90 Å² (BBB) | {'✅' if tpsa < 90 else '❌'} |\n\n"
        "**PFI** (Pfizer's Property Forecast Index) predicts solubility and non-specific binding: "
        "logP + number of aromatic rings < 8 is the target zone. "
        "**CNS MPO** integrates MW, logP, TPSA, HBD, and RotBonds into a 0–6 score; ≥ 4 predicts CNS penetration."
    )
    questions = [
        f"Evaluate {smiles} for CNS drug-likeness using PFI and CNS MPO.",
        f"Apply lead-optimisation rules (PFI, CNS MPO, golden triangle) to {smiles}.",
        f"Score {smiles} against Pfizer's PFI and the golden triangle rule.",
        f"Is {smiles} a suitable CNS drug candidate? Compute PFI and CNS MPO.",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# R5 NEW: ADME heuristics (BBB, solubility, P-gp)
# ---------------------------------------------------------------------------

_ADME_RULES = [
    {
        "name": "Blood-brain barrier (BBB) penetration",
        "criteria": "logP 1–3, PSA < 90 Å², MW < 450, HBD ≤ 3, rotatable bonds ≤ 8",
        "params": lambda mw, logp, hbd, tpsa, rot: {
            "BBB penetrant": (1 <= logp <= 3) and (tpsa < 90) and (mw < 450) and (hbd <= 3) and (rot <= 8),
            "logP 1–3": 1 <= logp <= 3,
            "PSA < 90 Å²": tpsa < 90,
            "MW < 450": mw < 450,
            "HBD ≤ 3": hbd <= 3,
        },
        "question": "Can {smiles} penetrate the blood-brain barrier?",
    },
    {
        "name": "Aqueous solubility (Lipinski/GSK rule)",
        "criteria": "logS (estimated) > −5; GSK: logP < 4 and MW < 400 for good solubility",
        "params": lambda mw, logp, hbd, tpsa, rot: {
            "Likely soluble (GSK)": logp < 4 and mw < 400,
            "logP < 4": logp < 4,
            "MW < 400": mw < 400,
        },
        "question": "Predict the aqueous solubility profile of {smiles}.",
    },
    {
        "name": "P-gp efflux liability",
        "criteria": "MW > 400, logP > 4, HBD > 2, PSA < 40: risk of P-gp substrate",
        "params": lambda mw, logp, hbd, tpsa, rot: {
            "P-gp risk": (mw > 400) and (logp > 4) and (hbd > 2),
            "MW > 400": mw > 400,
            "logP > 4": logp > 4,
            "HBD > 2": hbd > 2,
        },
        "question": "Assess P-gp efflux liability for {smiles}.",
    },
]


def adme_heuristics_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw   = round(Descriptors.MolWt(mol), 1)
    logp = round(Descriptors.MolLogP(mol), 2)
    hbd  = Lipinski.NumHDonors(mol)
    tpsa = round(rdMolDescriptors.CalcTPSA(mol), 1)
    rot  = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rule = random.choice(_ADME_RULES)
    checks = rule["params"](mw, logp, hbd, tpsa, rot)
    rows = "\n".join(
        f"| {k} | {'✅' if v else '❌'} |"
        for k, v in checks.items()
    )
    overall_key = list(checks.keys())[0]
    overall = checks[overall_key]
    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "print('MW:',   round(Descriptors.MolWt(mol), 1))\n"
        "print('logP:', round(Descriptors.MolLogP(mol), 2))\n"
        "print('HBD:',  Lipinski.NumHDonors(mol))\n"
        "print('TPSA:', round(rdMolDescriptors.CalcTPSA(mol), 1))\n"
        "print('RotB:', rdMolDescriptors.CalcNumRotatableBonds(mol))\n"
        "```\n\n"
        f"**{rule['name']}** ({rule['criteria']})\n\n"
        f"| Criterion | Met? |\n|---|---|\n{rows}\n\n"
        f"**Verdict:** {'Likely ' + overall_key if overall else 'May not meet: ' + overall_key}. "
        "These are heuristic rules — experimental PAMPA, Caco-2, or efflux assays are required for confirmation."
    )
    return _record(rule["question"].format(smiles=smiles), answer)


# ---------------------------------------------------------------------------
# R5 NEW: Synthetic accessibility score
# ---------------------------------------------------------------------------

def sa_score_example(smiles: str):
    if not _HAS_SASCORER:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        sa = round(_sascorer.calculateScore(mol), 2)
    except Exception:
        return None
    if sa <= 2:
        interp = "very easy to synthesise (SA ≤ 2)"
    elif sa <= 4:
        interp = "readily accessible (SA 2–4)"
    elif sa <= 6:
        interp = "moderately difficult to synthesise (SA 4–6)"
    else:
        interp = "challenging synthesis (SA > 6)"
    answer = (
        "```python\n"
        "import sys, os\n"
        "from rdkit.Chem import RDConfig\n"
        "sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))\n"
        "import sascorer\n"
        "from rdkit import Chem\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "print(round(sascorer.calculateScore(mol), 2))\n"
        "```\n\n"
        f"SA score: **{sa}** — {interp}. "
        "The SA score (Ertl & Schuffenhauer, 2009) runs from 1 (trivially simple) to 10 (extremely complex). "
        "It penalises rare ring systems, large molecular complexity, and unusual substitution patterns."
    )
    questions = [
        f"Calculate the synthetic accessibility score of {smiles}.",
        f"How easy is {smiles} to synthesise? Compute the SA score.",
        f"What is the SA score of {smiles}?",
        f"Assess the synthetic accessibility of {smiles} using the Ertl SA score.",
        f"Is {smiles} synthetically accessible? Use RDKit SA score.",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# R5 NEW: PDB structure quality interpretation
# ---------------------------------------------------------------------------

_PDB_QUALITY_QUESTIONS = [
    {
        "q": "How do I assess the quality of a PDB structure before using it for docking?",
        "a": (
            "Check these quality indicators before using any PDB entry:\n\n"
            "| Metric | Good | Acceptable | Caution |\n"
            "|---|---|---|---|\n"
            "| Resolution (Å) | < 2.0 | 2.0–2.5 | > 2.5 |\n"
            "| R-factor | < 0.20 | 0.20–0.25 | > 0.25 |\n"
            "| R-free | < 0.25 | 0.25–0.30 | > 0.30 |\n"
            "| R-free/R gap | < 0.05 | 0.05–0.07 | > 0.07 |\n"
            "| MolProbity score | < 1.0 | 1.0–2.0 | > 2.0 |\n"
            "| Ramachandran outliers (%) | < 0.5 | 0.5–2 | > 2 |\n\n"
            "```python\n"
            "# Fetch quality data from RCSB PDB validation API\n"
            "import requests\n"
            "pdb_id = '1HSG'\n"
            "url = f'https://data.rcsb.org/rest/v1/core/entry/{pdb_id}'\n"
            "data = requests.get(url).json()\n"
            "refine = data.get('refine', [{}])[0]\n"
            "print('Resolution:', data.get('rcsb_entry_info', {}).get('resolution_combined', 'N/A'))\n"
            "print('R-work:', refine.get('ls_r_factor_r_work'))\n"
            "print('R-free:', refine.get('ls_r_factor_r_free'))\n"
            "```\n\n"
            "**Also check:** crystal form (P212121 is common), space group, presence of "
            "buffer/co-solvent in the active site, and whether the ligand pose has clear electron density (RSCC > 0.8)."
        ),
    },
    {
        "q": "What does a high R-free/R-factor gap mean in a PDB structure?",
        "a": (
            "The R-factor measures how well the structural model explains the diffraction data; "
            "the R-free is computed on a held-out 5% test set.\n\n"
            "- **R-factor** reflects overall model fit. Target < 0.20 for high-quality structures.\n"
            "- **R-free** is the unbiased estimate. Target < 0.25.\n"
            "- **R-free/R gap > 0.07** suggests **model overfitting** (too many parameters chasing the data). "
            "The model may have been over-refined or contain model bias.\n\n"
            "For drug design, prioritise:\n"
            "1. Resolution < 2.0 Å (side-chain geometry reliable)\n"
            "2. R-free < 0.25 (not overfitted)\n"
            "3. Ligand RSCC > 0.8 (electron density supports the pose)\n"
            "4. No missing loops in the binding site\n\n"
            "Check electron density using the **EDS** server or validate locally with `phenix.molprobity`."
        ),
    },
    {
        "q": "How do I read the resolution and R-factors from a PDB file in Python?",
        "a": (
            "```python\n"
            "# Method 1: RCSB REST API (recommended, always up-to-date)\n"
            "import requests\n"
            "pdb_id = '1HSG'\n"
            "url = f'https://data.rcsb.org/rest/v1/core/entry/{pdb_id}'\n"
            "d = requests.get(url).json()\n"
            "info = d.get('rcsb_entry_info', {})\n"
            "refine = d.get('refine', [{}])[0]\n"
            "print(f\"Resolution: {info.get('resolution_combined')} Å\")\n"
            "print(f\"R-work: {refine.get('ls_r_factor_r_work')}\")\n"
            "print(f\"R-free: {refine.get('ls_r_factor_r_free')}\")\n"
            "\n"
            "# Method 2: parse REMARK 3 from PDB file header\n"
            "with open('1hsg.pdb') as f:\n"
            "    for line in f:\n"
            "        if line.startswith('REMARK   3') and 'R VALUE' in line:\n"
            "            print(line.strip())\n"
            "        if line.startswith('ATOM'):\n"
            "            break\n"
            "```\n\n"
            "REMARK 3 in PDB format contains the refinement statistics. "
            "The REST API is faster and machine-readable. "
            "For bulk queries, use `https://data.rcsb.org/graphql` with GraphQL queries."
        ),
    },
    {
        "q": "What is the RSCC and why does it matter for validating a bound ligand?",
        "a": (
            "**RSCC (Real-space correlation coefficient)** measures how well a ligand model "
            "fits the experimental electron density map at its position.\n\n"
            "- Range: 0 (no correlation) to 1 (perfect fit)\n"
            "- **RSCC > 0.8:** Ligand pose is supported by density — high confidence\n"
            "- **RSCC 0.6–0.8:** Moderate confidence — check the Fo-Fc map manually\n"
            "- **RSCC < 0.6:** Weak density — the pose may be unreliable; consider omit maps\n\n"
            "```python\n"
            "# RSCC data from RCSB PDB validation API\n"
            "import requests\n"
            "pdb_id = '1HSG'\n"
            "url = f'https://data.rcsb.org/rest/v1/core/chemcomp_instance/{pdb_id}/MK1/A/902'\n"
            "r = requests.get(url)\n"
            "# See pdbx_vrpt_summary_geometry for RSCC\n"
            "```\n\n"
            "Always validate the binding pose visually in Coot, PyMOL with the electron density "
            "map loaded, or via the PDB validation report at `https://www.rcsb.org/validate/{pdb_id}`."
        ),
    },
]


def pdb_quality_example():
    ex = random.choice(_PDB_QUALITY_QUESTIONS)
    return _record(ex["q"], ex["a"])


# ---------------------------------------------------------------------------
# R5 NEW: Cofactor and metal binding in protein structures
# ---------------------------------------------------------------------------

_COFACTOR_EXAMPLES = [
    {
        "cofactor": "zinc",
        "residues": ["Cys", "His", "Asp", "Glu"],
        "geometry": "tetrahedral (4-coordinate)",
        "pdb_examples": ["1SLN", "4HW3", "1A4E"],
        "comp_id": "ZN",
        "note": (
            "Zinc is the most common catalytic/structural metal in proteins. "
            "Catalytic zinc (e.g., carbonic anhydrase, thermolysin) is coordinated by "
            "3 protein ligands + 1 water. Structural zinc (zinc finger) uses 4 Cys/His ligands. "
            "SMARTS for Cys zinc binder: `[#16X2][Zn]`."
        ),
        "code": (
            "from Bio.PDB import PDBParser\n"
            "parser = PDBParser(QUIET=True)\n"
            "structure = parser.get_structure('prot', '{pdb}.pdb')\n"
            "for model in structure:\n"
            "    for chain in model:\n"
            "        for res in chain:\n"
            "            if res.get_resname() == 'ZN':\n"
            "                print('Zinc at:', chain.id, res.id)\n"
        ),
    },
    {
        "cofactor": "heme (iron porphyrin)",
        "residues": ["His (proximal)", "His or Cys (distal)"],
        "geometry": "square planar with axial ligands",
        "pdb_examples": ["1MBO", "1A6N", "3LGS"],
        "comp_id": "HEM",
        "note": (
            "Heme is the prosthetic group of haemoglobin, myoglobin, cytochromes, and "
            "CYP450 enzymes. The iron is coordinated by 4 porphyrin nitrogens + 1 axial His "
            "(proximal). In CYP450, the distal axial ligand is Cys (thiolate). "
            "The Fe can be Fe²⁺ (reduced, O₂-binding) or Fe³⁺ (oxidised). "
            "Heme compounds (comp_id=HEM) appear as HETATM records."
        ),
        "code": (
            "from Bio.PDB import PDBParser\n"
            "parser = PDBParser(QUIET=True)\n"
            "structure = parser.get_structure('mb', '{pdb}.pdb')\n"
            "for model in structure:\n"
            "    for chain in model:\n"
            "        for res in chain:\n"
            "            if res.get_resname() in ('HEM', 'HEC'):\n"
            "                print('Heme in chain', chain.id, 'at', res.id)\n"
        ),
    },
    {
        "cofactor": "NAD⁺/NADH (nicotinamide adenine dinucleotide)",
        "residues": ["Arg", "Ser", "Asp", "Thr"],
        "geometry": "extended conformation in Rossmann fold",
        "pdb_examples": ["1AXR", "3O3E", "5DHH"],
        "comp_id": "NAD",
        "note": (
            "NAD⁺ is a hydride acceptor; NADH is the reduced form. The Rossmann fold "
            "(βαβαβ) is the canonical NAD-binding domain — identified by the fingerprint "
            "GXGXXG. The nicotinamide ring sits at the re or si face. "
            "The ribose 2'-OH determines NAD vs NADP specificity: NADP has a 2'-phosphate. "
            "Comp IDs: NAD, NAI, NDP (NADP), NDN."
        ),
        "code": (
            "from Bio.PDB import PDBParser\n"
            "parser = PDBParser(QUIET=True)\n"
            "structure = parser.get_structure('dh', '{pdb}.pdb')\n"
            "for model in structure:\n"
            "    for chain in model:\n"
            "        for res in chain:\n"
            "            if res.get_resname() in ('NAD', 'NAI', 'NDP', 'NDN'):\n"
            "                print('Cofactor', res.get_resname(), 'in chain', chain.id)\n"
        ),
    },
    {
        "cofactor": "ATP (adenosine triphosphate)",
        "residues": ["Lys", "Asp", "Phe", "Gly-rich loop"],
        "geometry": "P-loop / glycine-rich loop (GXGXXG) cradles the phosphates",
        "pdb_examples": ["1ATP", "1FMK", "4GE1"],
        "comp_id": "ATP",
        "note": (
            "ATP is the phosphate donor in kinase reactions. The adenine sits in the "
            "hinge region; the phosphate tail contacts the DFG-Mg²⁺. "
            "Non-hydrolysable analogues (AMPPNP, ADP-BeF₃) are used in structural studies. "
            "The Mg²⁺ ion is essential and coordinates the β and γ phosphates. "
            "Comp IDs: ATP, ADP, ANP (AMPPNP), ACP (ACPCP)."
        ),
        "code": (
            "from Bio.PDB import PDBParser\n"
            "parser = PDBParser(QUIET=True)\n"
            "structure = parser.get_structure('kin', '{pdb}.pdb')\n"
            "for model in structure:\n"
            "    for chain in model:\n"
            "        for res in chain:\n"
            "            if res.get_resname() in ('ATP', 'ADP', 'ANP', 'ACP'):\n"
            "                print('Nucleotide', res.get_resname(), 'in', chain.id)\n"
        ),
    },
]


def cofactor_binding_example():
    ex = random.choice(_COFACTOR_EXAMPLES)
    pdb = random.choice(ex["pdb_examples"])
    code = ex["code"].replace("{pdb}", pdb.lower())
    answer = (
        f"**{ex['cofactor'].title()}** (comp_id: {ex['comp_id']})\n\n"
        f"- **Coordinating residues:** {', '.join(ex['residues'])}\n"
        f"- **Geometry:** {ex['geometry']}\n"
        f"- **Example PDB:** {pdb}\n\n"
        f"{ex['note']}\n\n"
        "```python\n"
        f"{code}"
        "```"
    )
    questions = [
        f"How is {ex['cofactor']} coordinated in protein structures? Use {pdb} as an example.",
        f"What residues coordinate {ex['cofactor']} in PDB structure {pdb}?",
        f"Describe the binding of {ex['cofactor']} in protein {pdb}.",
        f"How do I identify {ex['cofactor']} binding sites in a PDB file using Biopython?",
        f"What is the coordination geometry of {ex['cofactor']} in PDB {pdb}?",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# R5 NEW: RCSB PDB REST API for programmatic structure retrieval
# ---------------------------------------------------------------------------

_RCSB_API_RECIPES = [
    {
        "intent": "download a PDB structure as a PDB file",
        "code": (
            "import requests\n"
            "pdb_id = '{pdb}'\n"
            "url = f'https://files.rcsb.org/download/{pdb_id}.pdb'\n"
            "r = requests.get(url)\n"
            "with open(f'{pdb_id}.pdb', 'w') as f:\n"
            "    f.write(r.text)\n"
            "print(f'Downloaded {pdb_id}.pdb ({len(r.text)} bytes)')\n"
        ),
        "note": "The RCSB files endpoint serves PDB and mmCIF formats. Use `.cif` for the mmCIF format. Legacy `.pdb` files are capped at 99999 atoms; use mmCIF for large structures.",
    },
    {
        "intent": "search for all structures of a human kinase by UniProt ID",
        "code": (
            "import requests, json\n"
            "uniprot_id = 'P00533'  # EGFR\n"
            "query = {\n"
            "    'query': {\n"
            "        'type': 'terminal',\n"
            "        'service': 'text',\n"
            "        'parameters': {\n"
            "            'attribute': 'rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession',\n"
            "            'operator': 'exact_match',\n"
            "            'value': uniprot_id\n"
            "        }\n"
            "    },\n"
            "    'return_type': 'entry'\n"
            "}\n"
            "r = requests.post('https://search.rcsb.org/rcsbsearch/v2/query', json=query)\n"
            "results = r.json()\n"
            "pdb_ids = [hit['identifier'] for hit in results.get('result_set', [])]\n"
            "print(f'Found {len(pdb_ids)} structures for {uniprot_id}:', pdb_ids[:5])\n"
        ),
        "note": "The RCSB Search API v2 supports structured queries. `return_type: entry` returns PDB IDs; use `polymer_entity` for chain-level results.",
    },
    {
        "intent": "get metadata (resolution, R-factor, ligands) for a PDB entry",
        "code": (
            "import requests\n"
            "pdb_id = '{pdb}'\n"
            "url = f'https://data.rcsb.org/rest/v1/core/entry/{pdb_id}'\n"
            "d = requests.get(url).json()\n"
            "info = d.get('rcsb_entry_info', {})\n"
            "refine = d.get('refine', [{}])[0]\n"
            "print('Resolution:', info.get('resolution_combined'), 'Å')\n"
            "print('R-work:', refine.get('ls_r_factor_r_work'))\n"
            "print('R-free:', refine.get('ls_r_factor_r_free'))\n"
            "print('Ligands:', info.get('nonpolymer_bound_components', []))\n"
        ),
        "note": "The RCSB Data API returns structured JSON for every PDB entry. It's faster than parsing PDB headers and always reflects the latest deposited data.",
    },
    {
        "intent": "search for structures with a specific ligand (by comp_id)",
        "code": (
            "import requests\n"
            "comp_id = 'STI'  # imatinib\n"
            "query = {\n"
            "    'query': {\n"
            "        'type': 'terminal',\n"
            "        'service': 'text',\n"
            "        'parameters': {\n"
            "            'attribute': 'rcsb_nonpolymer_instance_feature_summary.comp_id',\n"
            "            'operator': 'exact_match',\n"
            "            'value': comp_id\n"
            "        }\n"
            "    },\n"
            "    'return_type': 'entry'\n"
            "}\n"
            "r = requests.post('https://search.rcsb.org/rcsbsearch/v2/query', json=query)\n"
            "pdb_ids = [h['identifier'] for h in r.json().get('result_set', [])]\n"
            "print(f'{comp_id} found in {len(pdb_ids)} structures:', pdb_ids[:5])\n"
        ),
        "note": "Searching by `comp_id` finds all co-crystal structures containing that ligand. Useful for finding alternative binding site geometries or selectivity data.",
    },
    {
        "intent": "find all PDB entries from a paper by DOI",
        "code": (
            "import requests\n"
            "doi = '10.2210/pdb1HSG/pdb'\n"
            "query = {\n"
            "    'query': {\n"
            "        'type': 'terminal',\n"
            "        'service': 'text',\n"
            "        'parameters': {\n"
            "            'attribute': 'rcsb_primary_citation.pdbx_database_id_doi',\n"
            "            'operator': 'exact_match',\n"
            "            'value': doi\n"
            "        }\n"
            "    },\n"
            "    'return_type': 'entry'\n"
            "}\n"
            "r = requests.post('https://search.rcsb.org/rcsbsearch/v2/query', json=query)\n"
            "print(r.json().get('result_set', []))\n"
        ),
        "note": "The RCSB assigns DOIs to each PDB entry (format: 10.2210/pdbXXXX/pdb). You can also search by PubMed ID or citation author.",
    },
]


def pdb_fetch_api_example():
    recipe = random.choice(_RCSB_API_RECIPES)
    ctx = random.choice(PYMOL_CONTEXTS)
    code = recipe["code"].replace("{pdb}", ctx["pdb"])
    answer = (
        f"```python\n{code}```\n\n"
        f"{recipe['note']}"
    )
    return _record(
        f"How do I {recipe['intent']} using the RCSB PDB API?",
        answer,
    )


# ---------------------------------------------------------------------------
# R5 NEW: H-bond network analysis in protein structures
# ---------------------------------------------------------------------------

def hbond_structure_example():
    ctx = random.choice(PYMOL_CONTEXTS)
    pdb = ctx["pdb"]
    ligand = ctx["ligand"]
    style = random.choice(["biopython", "chimerax", "pymol"])
    if style == "biopython":
        answer = (
            "```python\n"
            "from Bio.PDB import PDBParser, HSExposureCB\n"
            "from Bio.PDB.DSSP import DSSP\n"
            "# Use PLIP for the most detailed H-bond analysis\n"
            "# or use gemmi for fast contact search\n"
            "import gemmi\n"
            f"st = gemmi.read_structure('{pdb.lower()}.pdb')\n"
            "ns = gemmi.NeighborSearch(st[0], st.cell, 4.0)\n"
            "ns.populate(include_h=False)\n"
            "# Find contacts between ligand and protein\n"
            "hbond_candidates = []\n"
            "for chain in st[0]:\n"
            "    for res in chain:\n"
            f"        if res.name == '{ligand}':\n"
            "            for atom in res:\n"
            "                if atom.element.name in ('N', 'O', 'S'):\n"
            "                    for nb in ns.find_atoms(atom.pos, '\\0', radius=3.5):\n"
            "                        cra = nb.to_cra(st[0])\n"
            "                        if cra.atom.element.name in ('N', 'O') and cra.residue.name != res.name:\n"
            "                            hbond_candidates.append(\n"
            "                                (atom.name, cra.chain.name, cra.residue.name,\n"
            "                                 str(cra.residue.seqid), cra.atom.name, round(nb.dist(), 2))\n"
            "                            )\n"
            "for h in hbond_candidates:\n"
            "    print(f'Ligand {h[0]} — {h[2]}{h[3]} {h[4]} ({h[5]} Å)')\n"
            "```\n\n"
            f"This finds all N/O contacts within 3.5 Å between {ligand} and protein — "
            "strong H-bond candidates. For full H-bond analysis including angles, use PLIP: "
            f"`plip -f {pdb.lower()}.pdb -x -o plip_output/`"
        )
    elif style == "chimerax":
        answer = (
            "```\n"
            f"# ChimeraX: H-bond analysis for {pdb}\n"
            f"open {pdb}\n"
            "delete solvent\n"
            f"hbonds ::{ligand} restrict cross reveal true log true\n"
            "```\n\n"
            f"`hbonds` with `restrict cross` shows only ligand↔protein H-bonds. "
            "`reveal true` displays acceptor/donor atoms. "
            "`log true` writes distances and angles to the log."
        )
    else:
        answer = (
            "```python\n"
            f"# PyMOL: visualise H-bonds for {pdb}/{ligand}\n"
            f"fetch {pdb}, async=0\n"
            "remove solvent\n"
            f"select lig, resn {ligand}\n"
            "select prot, polymer\n"
            "distance hbonds, lig, prot, 3.5, mode=2\n"
            "show sticks, lig\n"
            "show sticks, byres (prot within 5 of lig)\n"
            "util.cbag lig\n"
            "center lig\n"
            "zoom lig, 8\n"
            "```\n\n"
            "`distance ... mode=2` shows all heavy-atom contacts within 3.5 Å — H-bond candidates. "
            "For angle-filtered H-bonds, use PLIP or ChimeraX `hbonds` command."
        )
    questions = [
        f"Analyse the H-bond network between {ligand} and {pdb} ({ctx.get('target','protein')}).",
        f"How do I find hydrogen bonds between the ligand and protein in {pdb}?",
        f"Show the H-bond interactions of {ligand} in PDB structure {pdb}.",
        f"Calculate H-bonds formed by {ligand} in {pdb} using Python.",
    ]
    return _record(random.choice(questions), answer)


# ===========================================================================
# METRIC-FIX GENERATORS
# Target: PyExec, Code→Quote, Numerical Fidelity, Rounding
# ===========================================================================

# ---------------------------------------------------------------------------
# PyExec fix: complete runnable RDKit scripts (no external files needed)
# ---------------------------------------------------------------------------

_PYEXEC_TASKS = [
    {
        "intent": "compute all Ro5 properties and print a table",
        "code_tpl": (
            "from rdkit import Chem\n"
            "from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors\n"
            "from rdkit.Chem.QED import qed\n"
            "mol = Chem.MolFromSmiles('{smi}')\n"
            "props = {{\n"
            "    'MW':   round(Descriptors.MolWt(mol), 1),\n"
            "    'logP': round(Descriptors.MolLogP(mol), 2),\n"
            "    'HBD':  Lipinski.NumHDonors(mol),\n"
            "    'HBA':  Lipinski.NumHAcceptors(mol),\n"
            "    'TPSA': round(rdMolDescriptors.CalcTPSA(mol), 1),\n"
            "    'QED':  round(qed(mol), 3),\n"
            "}}\n"
            "for k, v in props.items():\n"
            "    print(f'{{k}}: {{v}}')\n"
        ),
        "question": "Write a complete Python script to compute and print all Ro5 properties of {smi}.",
    },
    {
        "intent": "enumerate all substructure matches and print atom indices",
        "code_tpl": (
            "from rdkit import Chem\n"
            "mol = Chem.MolFromSmiles('{smi}')\n"
            "patt = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')\n"
            "matches = mol.GetSubstructMatches(patt)\n"
            "print(f'Found {{len(matches)}} carboxylic acid group(s)')\n"
            "for i, m in enumerate(matches):\n"
            "    print(f'  Match {{i+1}}: atoms {{m}}')\n"
        ),
        "question": "Write a complete RDKit script to find all carboxylic acid groups in {smi}.",
    },
    {
        "intent": "compute Morgan fingerprint and print the first 10 bit positions",
        "code_tpl": (
            "from rdkit import Chem\n"
            "from rdkit.Chem import rdFingerprintGenerator\n"
            "mol = Chem.MolFromSmiles('{smi}')\n"
            "gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)\n"
            "fp = gen.GetFingerprint(mol)\n"
            "bits = list(fp.GetOnBits())\n"
            "print(f'Non-zero bits: {{len(bits)}}')\n"
            "print(f'First 10 bit positions: {{bits[:10]}}')\n"
        ),
        "question": "Write a complete Python script to compute the Morgan fingerprint of {smi} and print the on-bits.",
    },
    {
        "intent": "generate a Murcko scaffold and compute its properties",
        "code_tpl": (
            "from rdkit import Chem\n"
            "from rdkit.Chem import Descriptors\n"
            "from rdkit.Chem.Scaffolds import MurckoScaffold\n"
            "mol = Chem.MolFromSmiles('{smi}')\n"
            "scaf = MurckoScaffold.GetScaffoldForMol(mol)\n"
            "scaf_smi = Chem.MolToSmiles(scaf)\n"
            "print(f'Scaffold: {{scaf_smi}}')\n"
            "print(f'Scaffold MW: {{round(Descriptors.MolWt(scaf), 1)}}')\n"
            "print(f'Scaffold HBD: {{Chem.rdMolDescriptors.CalcNumHBD(scaf)}}')\n"
        ),
        "question": "Write a runnable Python script to extract the Murcko scaffold of {smi} and compute its MW.",
    },
    {
        "intent": "check all PAINS alerts and print results",
        "code_tpl": (
            "from rdkit import Chem\n"
            "from rdkit.Chem import FilterCatalog\n"
            "params = FilterCatalog.FilterCatalogParams()\n"
            "params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)\n"
            "catalog = FilterCatalog.FilterCatalog(params)\n"
            "mol = Chem.MolFromSmiles('{smi}')\n"
            "matches = list(catalog.GetMatches(mol))\n"
            "if matches:\n"
            "    for e in matches:\n"
            "        print(f'ALERT: {{e.GetDescription()}}')\n"
            "else:\n"
            "    print('No PAINS alerts detected')\n"
        ),
        "question": "Write a complete Python script to check {smi} for PAINS alerts using RDKit FilterCatalog.",
    },
    {
        "intent": "compute SA score and QED and compare",
        "code_tpl": (
            "import sys, os\n"
            "from rdkit.Chem import RDConfig\n"
            "sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))\n"
            "import sascorer\n"
            "from rdkit import Chem\n"
            "from rdkit.Chem.QED import qed\n"
            "mol = Chem.MolFromSmiles('{smi}')\n"
            "sa = round(sascorer.calculateScore(mol), 2)\n"
            "qed_score = round(qed(mol), 3)\n"
            "print(f'SA score: {{sa}} (1=easy, 10=hard)')\n"
            "print(f'QED: {{qed_score}} (0=poor, 1=ideal drug-like)')\n"
        ),
        "question": "Write a complete Python script to compute and compare the SA score and QED of {smi}.",
    },
]


def pyexec_drill_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    task = random.choice(_PYEXEC_TASKS)
    code = task["code_tpl"].format(smi=smiles)
    # Verify the code actually runs (only for RDKIT tasks, skip SA score which needs contrib)
    if "sascorer" not in code:
        try:
            ns = {}
            exec(compile(code, "<string>", "exec"), ns)  # noqa: S102
        except Exception:
            return None  # reject if code doesn't run
    answer = f"```python\n{code}```\n\nThis script runs end-to-end without external files."
    return _record(task["question"].format(smi=smiles), answer)


# ---------------------------------------------------------------------------
# Code→Quote fix: explicit compute-then-quote with output shown verbatim
# ---------------------------------------------------------------------------

def code_then_quote_v2_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw   = round(Descriptors.MolWt(mol), 1)
    logp = round(Descriptors.MolLogP(mol), 2)
    hbd  = Lipinski.NumHDonors(mol)
    hba  = Lipinski.NumHAcceptors(mol)
    tpsa = round(rdMolDescriptors.CalcTPSA(mol), 1)
    try:
        score = round(_qed(mol), 3)
    except Exception:
        score = None
    # Format as explicit: "Running the code above gives: MW: X, ..."
    output_block = (
        f"MW: {mw}\n"
        f"logP: {logp}\n"
        f"HBD: {hbd}\n"
        f"HBA: {hba}\n"
        f"TPSA: {tpsa}\n"
        + (f"QED: {score}\n" if score else "")
    )
    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors\n"
        "from rdkit.Chem.QED import qed\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "print('MW:',   round(Descriptors.MolWt(mol), 1))\n"
        "print('logP:', round(Descriptors.MolLogP(mol), 2))\n"
        "print('HBD:',  Lipinski.NumHDonors(mol))\n"
        "print('HBA:',  Lipinski.NumHAcceptors(mol))\n"
        "print('TPSA:', round(rdMolDescriptors.CalcTPSA(mol), 1))\n"
        "print('QED:',  round(qed(mol), 3))\n"
        "```\n\n"
        f"Running this code gives:\n\n"
        f"```\n{output_block}```\n\n"
        f"Quoting directly from the output: MW = {mw}, logP = {logp}, "
        f"HBD = {hbd}, HBA = {hba}, TPSA = {tpsa} Å²"
        + (f", QED = {score}." if score else ".")
    )
    questions = [
        f"Compute and quote all physicochemical properties of {smiles}.",
        f"Run RDKit on {smiles} and quote the exact output values.",
        f"What are the exact MW, logP, HBD, HBA, TPSA and QED of {smiles}? Show your code and quote the output.",
        f"Calculate all key properties of {smiles} using RDKit and report the values from the output.",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# Rounding fix: explicit rounding conventions with all properties
# ---------------------------------------------------------------------------

def rounding_explicit_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw   = round(Descriptors.MolWt(mol), 1)
    logp = round(Descriptors.MolLogP(mol), 2)
    tpsa = round(rdMolDescriptors.CalcTPSA(mol), 1)
    try:
        score = round(_qed(mol), 3)
    except Exception:
        score = None
    rot  = rdMolDescriptors.CalcNumRotatableBonds(mol)
    # Deliberately show wrong and right rounding to drive the lesson home
    mw_wrong   = int(mw)       # 0 d.p. — wrong
    logp_wrong = round(logp, 0)  # 0 d.p. — wrong
    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors, rdMolDescriptors\n"
        "from rdkit.Chem.QED import qed\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "# Correct rounding conventions\n"
        "print('MW:',   round(Descriptors.MolWt(mol), 1))      # 1 decimal place\n"
        "print('logP:', round(Descriptors.MolLogP(mol), 2))    # 2 decimal places\n"
        "print('TPSA:', round(rdMolDescriptors.CalcTPSA(mol), 1))  # 1 decimal place\n"
        "print('QED:',  round(qed(mol), 3))                    # 3 decimal places\n"
        "```\n\n"
        f"**Correct values:** MW = {mw}, logP = {logp}, TPSA = {tpsa} Å²"
        + (f", QED = {score}." if score else ".")
        + "\n\n"
        "**Rounding conventions:**\n"
        "- MW: **1 decimal place** (e.g., 180.2, not 180 or 180.16)\n"
        "- logP: **2 decimal places** (e.g., 1.35, not 1 or 1.3)\n"
        "- TPSA: **1 decimal place** (e.g., 63.6, not 64)\n"
        "- QED: **3 decimal places** (e.g., 0.552, not 0.55 or 1)\n"
        "- HBD/HBA: **integers** (no decimal)\n\n"
        f"❌ MW = {mw_wrong} (wrong — no decimal)  \n"
        f"✅ MW = {mw} (correct — 1 d.p.)"
    )
    questions = [
        f"What are the correct rounded values of MW, logP, TPSA and QED for {smiles}?",
        f"Compute MW, logP, TPSA and QED for {smiles} with the correct number of decimal places.",
        f"What rounding should be applied to RDKit properties? Show for {smiles}.",
        f"Give me MW, logP, TPSA and QED of {smiles} — use the standard ChemSage rounding conventions.",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# Fidelity fix: multi-property compute with every value explicitly quoted
# ---------------------------------------------------------------------------

def fidelity_multistep_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw   = round(Descriptors.MolWt(mol), 1)
    logp = round(Descriptors.MolLogP(mol), 2)
    hbd  = Lipinski.NumHDonors(mol)
    hba  = Lipinski.NumHAcceptors(mol)
    tpsa = round(rdMolDescriptors.CalcTPSA(mol), 1)
    rot  = rdMolDescriptors.CalcNumRotatableBonds(mol)
    try:
        score = round(_qed(mol), 3)
    except Exception:
        score = None
    passes = (mw < 500) and (logp < 5) and (hbd <= 5) and (hba <= 10)
    verdict = "PASSES Lipinski Ro5" if passes else "FAILS Lipinski Ro5"
    answer = (
        "I will compute each property step by step from the SMILES, then state the exact values.\n\n"
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors\n"
        "from rdkit.Chem.QED import qed\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "mw   = round(Descriptors.MolWt(mol), 1)\n"
        "logp = round(Descriptors.MolLogP(mol), 2)\n"
        "hbd  = Lipinski.NumHDonors(mol)\n"
        "hba  = Lipinski.NumHAcceptors(mol)\n"
        "tpsa = round(rdMolDescriptors.CalcTPSA(mol), 1)\n"
        "rot  = rdMolDescriptors.CalcNumRotatableBonds(mol)\n"
        "qed_score = round(qed(mol), 3)\n"
        "print(mw, logp, hbd, hba, tpsa, rot, qed_score)\n"
        "```\n\n"
        f"Output: `{mw} {logp} {hbd} {hba} {tpsa} {rot}" + (f" {score}`" if score else "`") + "\n\n"
        f"Property summary:\n"
        f"- MW = **{mw}** Da\n"
        f"- logP = **{logp}**\n"
        f"- HBD = **{hbd}**, HBA = **{hba}**\n"
        f"- TPSA = **{tpsa}** Å²\n"
        f"- Rotatable bonds = **{rot}**\n"
        + (f"- QED = **{score}**\n" if score else "")
        + f"\n**{verdict}** ({'✅' if passes else '❌'})."
    )
    questions = [
        f"Step through the RDKit calculation of MW, logP, HBD, HBA, TPSA and QED for {smiles}.",
        f"Compute each physicochemical property of {smiles} separately and state the exact values.",
        f"What are MW, logP, HBD, HBA, TPSA, rotatable bonds and QED for {smiles}? Show each computation.",
        f"Break down the full property profile of {smiles} step by step.",
    ]
    return _record(random.choice(questions), answer)


# ===========================================================================
# CATEGORY 1: Medicinal Chemistry Reasoning
# ===========================================================================

# ---------------------------------------------------------------------------
# hERG liability assessment
# ---------------------------------------------------------------------------

def herg_liability_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw   = round(Descriptors.MolWt(mol), 1)
    logp = round(Descriptors.MolLogP(mol), 2)
    hbd  = Lipinski.NumHDonors(mol)
    arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    # Simple hERG risk heuristic: basic N + logP > 3 + aromatic rings
    has_basic_n = mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H0,H1;!$(NC=O)]"))
    risk_factors = []
    if logp > 3:
        risk_factors.append(f"logP {logp} > 3")
    if has_basic_n:
        risk_factors.append("basic nitrogen present")
    if arom >= 2:
        risk_factors.append(f"{arom} aromatic rings")
    if mw > 400:
        risk_factors.append(f"MW {mw} > 400")
    risk = "HIGH" if len(risk_factors) >= 3 else "MODERATE" if len(risk_factors) >= 2 else "LOW"
    answer = (
        "hERG (Kv11.1) is the major cardiac ion channel associated with drug-induced QT prolongation "
        "and Torsades de Pointes arrhythmia.\n\n"
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "logp = round(Descriptors.MolLogP(mol), 2)\n"
        "arom = rdMolDescriptors.CalcNumAromaticRings(mol)\n"
        "basic_n = mol.HasSubstructMatch(Chem.MolFromSmarts('[NX3;H0,H1;!$(NC=O)]'))\n"
        "mw = round(Descriptors.MolWt(mol), 1)\n"
        "print(f'logP: {logp}, Arom rings: {arom}, Basic N: {basic_n}, MW: {mw}')\n"
        "```\n\n"
        f"**hERG risk: {risk}**\n\n"
        "Risk factors flagged:\n"
        + ("\n".join(f"- {r}" for r in risk_factors) if risk_factors else "- None")
        + "\n\n"
        "**Key hERG liability rules:**\n"
        "- logP > 3 correlates strongly with hERG binding\n"
        "- Basic nitrogen (pKa > 7) at the correct distance from aromatic rings is a pharmacophore\n"
        "- ≥ 2 aromatic rings provide a flat lipophilic surface matching the hERG channel cavity\n"
        "- Mitigation: reduce logP, remove basic nitrogen, add polar groups, reduce ring count"
    )
    questions = [
        f"Assess the hERG liability of {smiles}.",
        f"Does {smiles} have hERG channel liability risk?",
        f"Evaluate {smiles} for cardiac QT prolongation risk (hERG).",
        f"Check {smiles} for hERG-related structural alerts.",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# Selectivity / kinome off-target discussion
# ---------------------------------------------------------------------------

_SELECTIVITY_EXAMPLES = [
    {
        "target": "EGFR (erbB1)",
        "family": "receptor tyrosine kinase",
        "pharmacophore": "anilinoquinazoline or anilinopyrimidine hinge binder",
        "selectivity_risks": ["erbB2 (HER2)", "erbB4", "VEGFR2 (KDR)", "Src"],
        "strategy": "exploit the Thr790 gatekeeper (→ Cys in erbB2) with bulky C-4 substituents; "
                    "add non-conserved hydrophobic contacts in the back pocket",
        "example_pdb": "1IEP",
    },
    {
        "target": "CDK2",
        "family": "serine/threonine kinase",
        "pharmacophore": "purine or aminopyrimidine hinge binder + hydrophobic DFG-out pocket",
        "selectivity_risks": ["CDK1", "CDK4", "CDK6", "CDK9"],
        "strategy": "exploit the large hydrophobic pocket adjacent to Phe80; "
                    "CDK4/6 have a larger gatekeeper (Phe vs Phe vs Met in CDK2)",
        "example_pdb": "1PXO",
    },
    {
        "target": "PI3Kα",
        "family": "PI3K lipid kinase",
        "pharmacophore": "morpholine O as hinge H-bond acceptor; central core fits ATP binding site",
        "selectivity_risks": ["PI3Kβ", "PI3Kδ", "PI3Kγ", "mTOR", "VPS34"],
        "strategy": "Ile800/Ile932 form a tighter pocket in α vs δ; "
                    "bulky substituents at C-2 of the morpholine confer α-selectivity",
        "example_pdb": "2RD0",
    },
]


def selectivity_profile_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    ex = random.choice(_SELECTIVITY_EXAMPLES)
    answer = (
        f"**Kinase selectivity strategy for {ex['target']} ({ex['family']})**\n\n"
        f"**Pharmacophore:** {ex['pharmacophore']}\n\n"
        f"**Key off-target risks:** {', '.join(ex['selectivity_risks'])}\n\n"
        f"**Selectivity strategy:** {ex['strategy']}\n\n"
        "```python\n"
        "# Screen structural analogues against the target pharmacophore\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import rdFingerprintGenerator, DataStructs\n"
        f"query = Chem.MolFromSmiles('{smiles}')\n"
        "gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)\n"
        "fp_q = gen.GetFingerprint(query)\n"
        "# Compare Tanimoto similarity to known selective inhibitors\n"
        "# (load from your compound collection or ChEMBL)\n"
        "```\n\n"
        f"**Reference structure:** PDB {ex['example_pdb']} shows the binding mode. "
        "Use PyMOL `align` to compare your compound's docked pose to the co-crystal ligand."
    )
    questions = [
        f"How do I design a selective inhibitor of {ex['target']}? Use {smiles} as the starting point.",
        f"What are the selectivity challenges for {ex['target']} inhibitors? Evaluate {smiles}.",
        f"Discuss the kinome selectivity profile for a {ex['target']} inhibitor like {smiles}.",
        f"What off-targets should I screen {smiles} against if targeting {ex['target']}?",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# Prodrug and BCS classification
# ---------------------------------------------------------------------------

_BCS_CLASSES = [
    {
        "class": "I",
        "solubility": "high",
        "permeability": "high",
        "absorption": "excellent oral bioavailability; absorption rate-limited by gastric emptying",
        "examples": "metoprolol, verapamil, propranolol",
        "strategy": "immediate release formulation; no special bioavailability concerns",
    },
    {
        "class": "II",
        "solubility": "low",
        "permeability": "high",
        "absorption": "solubility-limited absorption; highly variable bioavailability",
        "examples": "ibuprofen, carbamazepine, felodipine",
        "strategy": "salt formation, amorphous solid dispersions, nanosizing, lipid-based formulations",
    },
    {
        "class": "III",
        "solubility": "high",
        "permeability": "low",
        "absorption": "permeability-limited; often P-gp substrate",
        "examples": "cimetidine, acyclovir, ranitidine",
        "strategy": "prodrug strategies (lipophilic ester prodrug), P-gp inhibitors, permeation enhancers",
    },
    {
        "class": "IV",
        "solubility": "low",
        "permeability": "low",
        "absorption": "poor oral bioavailability; very challenging",
        "examples": "ritonavir (early form), cyclosporin A",
        "strategy": "lipid-based drug delivery (SMEDDS), cyclodextrin complexation, "
                    "or IV/parenteral route; consider fundamental redesign",
    },
]


def prodrug_bcs_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    logp = round(Descriptors.MolLogP(mol), 2)
    tpsa = round(rdMolDescriptors.CalcTPSA(mol), 1)
    # Simple BCS class prediction heuristic
    high_solub = logp < 2 or tpsa > 90  # rough proxy
    high_perm  = logp > 0 and tpsa < 130 and Lipinski.NumHDonors(mol) <= 3
    if high_solub and high_perm:
        cls = _BCS_CLASSES[0]
    elif not high_solub and high_perm:
        cls = _BCS_CLASSES[1]
    elif high_solub and not high_perm:
        cls = _BCS_CLASSES[2]
    else:
        cls = _BCS_CLASSES[3]
    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "logp = round(Descriptors.MolLogP(mol), 2)\n"
        "tpsa = round(rdMolDescriptors.CalcTPSA(mol), 1)\n"
        "hbd  = Lipinski.NumHDonors(mol)\n"
        "print(f'logP={logp}, TPSA={tpsa}, HBD={hbd}')\n"
        "# BCS class prediction heuristic:\n"
        "# High solubility proxy: logP < 2 or TPSA > 90\n"
        "# High permeability proxy: logP > 0 and TPSA < 130 and HBD <= 3\n"
        "```\n\n"
        f"Predicted **BCS Class {cls['class']}** (solubility: {cls['solubility']}, "
        f"permeability: {cls['permeability']}).\n\n"
        f"**Absorption:** {cls['absorption']}\n\n"
        f"**Representative drugs:** {cls['examples']}\n\n"
        f"**Formulation strategy:** {cls['strategy']}"
    )
    questions = [
        f"Classify {smiles} by BCS class and suggest a formulation strategy.",
        f"What BCS class is {smiles} likely to fall into? What are the absorption implications?",
        f"Discuss the oral bioavailability and BCS classification of {smiles}.",
        f"Would {smiles} benefit from a prodrug strategy? Assess BCS class first.",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# SAR delta table (matched molecular pair)
# ---------------------------------------------------------------------------

def sar_delta_example(smi_a: str, smi_b: str):
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None:
        return None
    mw_a   = round(Descriptors.MolWt(mol_a), 1)
    mw_b   = round(Descriptors.MolWt(mol_b), 1)
    logp_a = round(Descriptors.MolLogP(mol_a), 2)
    logp_b = round(Descriptors.MolLogP(mol_b), 2)
    tpsa_a = round(rdMolDescriptors.CalcTPSA(mol_a), 1)
    tpsa_b = round(rdMolDescriptors.CalcTPSA(mol_b), 1)
    try:
        qed_a  = round(_qed(mol_a), 3)
        qed_b  = round(_qed(mol_b), 3)
    except Exception:
        qed_a = qed_b = None
    dmw  = round(mw_b - mw_a, 1)
    dlogp= round(logp_b - logp_a, 2)
    dtpsa= round(tpsa_b - tpsa_a, 1)
    arrow = lambda d: ("↑" if d > 0 else "↓" if d < 0 else "→")
    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors, rdMolDescriptors\n"
        "from rdkit.Chem.QED import qed\n"
        f"mol_a = Chem.MolFromSmiles('{smi_a}')\n"
        f"mol_b = Chem.MolFromSmiles('{smi_b}')\n"
        "for label, mol in [('A', mol_a), ('B', mol_b)]:\n"
        "    mw   = round(Descriptors.MolWt(mol), 1)\n"
        "    logp = round(Descriptors.MolLogP(mol), 2)\n"
        "    tpsa = round(rdMolDescriptors.CalcTPSA(mol), 1)\n"
        "    print(f'{label}: MW={mw}, logP={logp}, TPSA={tpsa}')\n"
        "```\n\n"
        "**SAR delta table (A → B):**\n\n"
        "| Property | A | B | Δ | Direction |\n"
        "|---|---|---|---|---|\n"
        f"| MW | {mw_a} | {mw_b} | {dmw:+.1f} | {arrow(dmw)} |\n"
        f"| logP | {logp_a} | {logp_b} | {dlogp:+.2f} | {arrow(dlogp)} |\n"
        f"| TPSA | {tpsa_a} | {tpsa_b} | {dtpsa:+.1f} | {arrow(dtpsa)} |\n"
        + (f"| QED | {qed_a} | {qed_b} | {round(qed_b-qed_a,3):+.3f} | {arrow(qed_b-qed_a)} |\n" if qed_a and qed_b else "")
        + "\n"
        "**SAR interpretation:** The A → B transformation "
        + ("increases lipophilicity" if dlogp > 0.3 else "reduces lipophilicity" if dlogp < -0.3 else "has little effect on lipophilicity")
        + (f" and {'increases' if dtpsa > 5 else 'reduces' if dtpsa < -5 else 'barely changes'} polarity.")
    )
    questions = [
        f"Build an SAR delta table for compounds A and B:\nA: {smi_a}\nB: {smi_b}",
        f"Compare the physicochemical properties of {smi_a} and {smi_b} as an MMP delta table.",
        f"What is the effect of the structural change from {smi_a} to {smi_b} on key properties?",
        f"Compute the SAR delta (ΔlogP, ΔMW, ΔTPSA, ΔQED) between:\nA: {smi_a}\nB: {smi_b}",
    ]
    return _record(random.choice(questions), answer)


# ===========================================================================
# CATEGORY 2: Structural Biology Expansion
# ===========================================================================

# ---------------------------------------------------------------------------
# MDAnalysis trajectory analysis
# ---------------------------------------------------------------------------

_MDANALYSIS_RECIPES = [
    {
        "intent": "compute the Cα RMSD of a protein over an MD trajectory",
        "code": (
            "import MDAnalysis as mda\n"
            "from MDAnalysis.analysis import rms\n"
            "u = mda.Universe('topology.psf', 'trajectory.dcd')\n"
            "ref = mda.Universe('reference.pdb')\n"
            "ca = u.select_atoms('protein and name CA')\n"
            "R = rms.RMSD(ca, ref.select_atoms('protein and name CA'))\n"
            "R.run()\n"
            "import numpy as np\n"
            "print(f'Mean RMSD: {np.mean(R.rmsd[:,2]):.2f} Å')\n"
            "print(f'Max RMSD:  {np.max(R.rmsd[:,2]):.2f} Å')\n"
        ),
        "note": "RMSD[:,2] is the RMSD column; columns are [frame, time, RMSD]. "
                "Large RMSD (> 3 Å) sustained over the trajectory indicates conformational change.",
    },
    {
        "intent": "compute per-residue RMSF to identify flexible regions",
        "code": (
            "import MDAnalysis as mda\n"
            "from MDAnalysis.analysis import rms\n"
            "u = mda.Universe('topology.psf', 'trajectory.dcd')\n"
            "ca = u.select_atoms('protein and name CA')\n"
            "R = rms.RMSF(ca).run()\n"
            "# Print residues with RMSF > 2 Å (flexible)\n"
            "for atom, rmsf in zip(ca, R.rmsf):\n"
            "    if rmsf > 2.0:\n"
            "        print(f'Residue {atom.resname}{atom.resid} chain {atom.segid}: RMSF = {rmsf:.2f} Å')\n"
        ),
        "note": "RMSF measures per-residue flexibility. High RMSF (> 2 Å) indicates loops or "
                "termini. Binding-site residues should have low RMSF for reliable docking.",
    },
    {
        "intent": "calculate the distance between two residues over time",
        "code": (
            "import MDAnalysis as mda\n"
            "import numpy as np\n"
            "u = mda.Universe('topology.psf', 'trajectory.dcd')\n"
            "# Distance between Asp102 Cγ and His57 Nε2 (catalytic dyad)\n"
            "asp = u.select_atoms('resid 102 and name CG')\n"
            "his = u.select_atoms('resid 57 and name NE2')\n"
            "distances = []\n"
            "for ts in u.trajectory:\n"
            "    d = np.linalg.norm(asp.positions[0] - his.positions[0])\n"
            "    distances.append(d)\n"
            "print(f'Mean distance: {np.mean(distances):.2f} Å')\n"
            "print(f'H-bond occupancy (<3.5 Å): {np.mean(np.array(distances) < 3.5):.1%}')\n"
        ),
        "note": "Distance analysis tracks catalytic residue contacts and H-bond occupancies. "
                "H-bond criterion: distance < 3.5 Å (heavy atom) and angle > 150° ideally.",
    },
    {
        "intent": "measure ligand binding site contacts over an MD trajectory",
        "code": (
            "import MDAnalysis as mda\n"
            "import numpy as np\n"
            "u = mda.Universe('complex.prmtop', 'trajectory.nc')\n"
            "ligand = u.select_atoms('resname LIG')\n"
            "protein = u.select_atoms('protein')\n"
            "contact_counts = {{}}\n"
            "for ts in u.trajectory:\n"
            "    nearby = protein.select_atoms(\n"
            "        f'around 4.0 resname LIG', updating=True\n"
            "    )\n"
            "    for res in nearby.residues:\n"
            "        key = f'{res.resname}{res.resid}'\n"
            "        contact_counts[key] = contact_counts.get(key, 0) + 1\n"
            "n_frames = len(u.trajectory)\n"
            "for res, count in sorted(contact_counts.items(), key=lambda x: -x[1])[:10]:\n"
            "    print(f'{res}: {count/n_frames:.1%} occupancy')\n"
        ),
        "note": "Residue contact occupancy across the trajectory reveals which contacts are "
                "persistent (> 80%) vs transient. Focus SAR on persistent contacts.",
    },
]


def mdanalysis_example():
    recipe = random.choice(_MDANALYSIS_RECIPES)
    answer = (
        f"```python\n{recipe['code']}```\n\n"
        f"{recipe['note']}\n\n"
        "MDAnalysis supports GROMACS (.gro/.xtc), AMBER (.prmtop/.nc), CHARMM (.psf/.dcd), "
        "and NAMD formats natively."
    )
    return _record(
        f"How do I {recipe['intent']} using MDAnalysis?",
        answer,
    )


# ---------------------------------------------------------------------------
# DSSP secondary structure assignment
# ---------------------------------------------------------------------------

_DSSP_QUESTIONS = [
    {
        "q": "How do I assign secondary structure to residues in a PDB file?",
        "a": (
            "```python\n"
            "# Method 1: Biopython DSSP wrapper (requires DSSP binary)\n"
            "from Bio.PDB import PDBParser\n"
            "from Bio.PDB.DSSP import DSSP\n"
            "parser = PDBParser(QUIET=True)\n"
            "structure = parser.get_structure('prot', '1hsg.pdb')\n"
            "model = structure[0]\n"
            "dssp = DSSP(model, '1hsg.pdb', dssp='mkdssp')\n"
            "for key in dssp:\n"
            "    chain, (_, resnum, _) = key\n"
            "    ss = dssp[key][2]  # H=helix, E=strand, C=coil\n"
            "    res = dssp[key][1]\n"
            "    print(f'{chain} {res}{resnum}: {ss}')\n"
            "```\n\n"
            "```python\n"
            "# Method 2: mdtraj (no external binary needed)\n"
            "import mdtraj as md\n"
            "t = md.load('1hsg.pdb')\n"
            "ss = md.compute_dssp(t, simplified=False)\n"
            "# Returns array shape (n_frames, n_residues) with DSSP codes\n"
            "from collections import Counter\n"
            "counts = Counter(ss[0])\n"
            "print('Helix:', counts['H'], 'Strand:', counts['E'], 'Coil:', counts['C'])\n"
            "```\n\n"
            "DSSP codes: **H**=α-helix, **G**=3₁₀-helix, **I**=π-helix, **E**=β-strand, "
            "**B**=bridge, **T**=turn, **S**=bend, **C**=coil. "
            "Simplified DSSP collapses to H/E/C."
        ),
    },
    {
        "q": "What is the DSSP algorithm and what secondary structures does it detect?",
        "a": (
            "DSSP (Dictionary of Secondary Structure of Proteins, Kabsch & Sander 1983) assigns "
            "secondary structure by computing hydrogen bond energies using partial atomic charges.\n\n"
            "**Algorithm:** For each residue pair (i, j), DSSP computes:\n"
            "> E = 0.084 × (1/r_ON + 1/r_CH − 1/r_OH − 1/r_CN) × 332 kcal/mol\n"
            "If E < −0.5 kcal/mol, an H-bond is assigned.\n\n"
            "**Secondary structure patterns detected:**\n"
            "| Code | Structure | H-bond pattern |\n"
            "|---|---|---|\n"
            "| H | α-helix | i→i+4 |\n"
            "| G | 3₁₀-helix | i→i+3 |\n"
            "| I | π-helix | i→i+5 |\n"
            "| E | β-strand | between strands |\n"
            "| B | isolated β-bridge | single inter-strand |\n"
            "| T | turn | i→i+3 or i+4 |\n"
            "| S | bend | high curvature |\n"
            "| C | coil | none of above |\n\n"
            "Runs are simplified in most tools: H+G+I → helix, E+B → strand, rest → coil."
        ),
    },
]


def dssp_example():
    ex = random.choice(_DSSP_QUESTIONS)
    return _record(ex["q"], ex["a"])


# ---------------------------------------------------------------------------
# Protein-protein interface analysis
# ---------------------------------------------------------------------------

_PPI_EXAMPLES = [
    {
        "intent": "calculate the buried surface area at a protein-protein interface",
        "pdb": "3SGB",
        "partners": "chains A and B",
        "note": (
            "BSA = (SASA_A + SASA_B − SASA_complex) / 2. "
            "Typical protein-protein interfaces bury 1,500–3,000 Å². "
            "Hot-spot residues contribute > 50% of binding energy despite covering < 10% of BSA."
        ),
        "code": (
            "from Bio.PDB import PDBParser\n"
            "from Bio.PDB.SASA import ShrakeRupley\n"
            "parser = PDBParser(QUIET=True)\n"
            "structure = parser.get_structure('ppi', '3sgb.pdb')\n"
            "sr = ShrakeRupley()\n"
            "# SASA of the complex\n"
            "sr.compute(structure, level='S')\n"
            "sasa_complex = structure.sasa\n"
            "# SASA of chain A alone\n"
            "chainA = structure[0]['A']\n"
            "sr.compute(chainA, level='C')\n"
            "sasa_A = chainA.sasa\n"
            "# SASA of chain B alone\n"
            "chainB = structure[0]['B']\n"
            "sr.compute(chainB, level='C')\n"
            "sasa_B = chainB.sasa\n"
            "bsa = (sasa_A + sasa_B - sasa_complex) / 2\n"
            "print(f'BSA: {bsa:.0f} Å²')\n"
        ),
    },
    {
        "intent": "find hot-spot residues at a protein-protein interface",
        "pdb": "1BRS",
        "partners": "barnase (chain A) and barstar (chain D)",
        "note": (
            "Hot spots are residues at the interface where Ala-scanning shows ΔΔG > 2 kcal/mol. "
            "They cluster at the centre of the interface (O-ring theory). "
            "Arg, Trp, and Tyr appear disproportionately as hot spots."
        ),
        "code": (
            "import gemmi\n"
            "st = gemmi.read_structure('1brs.pdb')\n"
            "ns = gemmi.NeighborSearch(st[0], st.cell, 5.0)\n"
            "ns.populate()\n"
            "interface_res = set()\n"
            "for chain in st[0]:\n"
            "    if chain.name == 'A':  # barnase\n"
            "        for res in chain:\n"
            "            for atom in res:\n"
            "                for nb in ns.find_atoms(atom.pos, '\\0', radius=5.0):\n"
            "                    cra = nb.to_cra(st[0])\n"
            "                    if cra.chain.name == 'D':  # barstar\n"
            "                        interface_res.add((chain.name, res.name, str(res.seqid)))\n"
            "print('Interface residues (barnase side):')\n"
            "for r in sorted(interface_res):\n"
            "    print(f'  {r[0]}:{r[1]}{r[2]}')\n"
        ),
    },
]


def ppi_interface_example():
    ex = random.choice(_PPI_EXAMPLES)
    answer = (
        f"```python\n{ex['code']}```\n\n"
        f"{ex['note']}"
    )
    return _record(
        f"How do I {ex['intent']} in PDB {ex['pdb']} ({ex['partners']}) using Python?",
        answer,
    )


# ---------------------------------------------------------------------------
# UniProt REST API
# ---------------------------------------------------------------------------

_UNIPROT_RECIPES = [
    {
        "intent": "fetch protein sequence and function annotation by UniProt ID",
        "code": (
            "import requests\n"
            "uniprot_id = 'P00533'  # EGFR\n"
            "url = f'https://rest.uniprot.org/uniprotkb/{uniprot_id}.json'\n"
            "data = requests.get(url).json()\n"
            "print('Gene:',    data['genes'][0]['geneName']['value'])\n"
            "print('Organism:', data['organism']['scientificName'])\n"
            "print('Length:',  data['sequence']['length'])\n"
            "# Function comment\n"
            "for comment in data.get('comments', []):\n"
            "    if comment['commentType'] == 'FUNCTION':\n"
            "        print('Function:', comment['texts'][0]['value'][:200])\n"
            "        break\n"
        ),
        "note": "UniProt REST API v2. Use `.fasta` suffix for the FASTA sequence, "
                "`.txt` for flat-file format.",
    },
    {
        "intent": "search UniProt for all human kinases and get their PDB cross-references",
        "code": (
            "import requests\n"
            "query = 'organism_id:9606 AND keyword:KW-0418'  # Human + Kinase keyword\n"
            "url = 'https://rest.uniprot.org/uniprotkb/search'\n"
            "params = {'query': query, 'format': 'json', 'fields': 'accession,id,xref_pdb', 'size': 10}\n"
            "r = requests.get(url, params=params)\n"
            "for entry in r.json().get('results', []):\n"
            "    acc = entry['primaryAccession']\n"
            "    pdbs = [x['id'] for x in entry.get('uniProtKBCrossReferences', [])\n"
            "            if x['database'] == 'PDB'][:3]\n"
            "    print(f'{acc}: PDB structures = {pdbs}')\n"
        ),
        "note": "UniProt keyword KW-0418 = Kinase. KW-0597 = Phosphoprotein. "
                "Use `fields` to request only needed columns — much faster than full records.",
    },
    {
        "intent": "map a UniProt ID to its PDB structures and download them",
        "code": (
            "import requests\n"
            "uniprot_id = 'P00533'  # EGFR\n"
            "# Step 1: get PDB cross-references from UniProt\n"
            "url = f'https://rest.uniprot.org/uniprotkb/{uniprot_id}.json'\n"
            "data = requests.get(url).json()\n"
            "pdb_ids = [\n"
            "    x['id'] for x in data.get('uniProtKBCrossReferences', [])\n"
            "    if x['database'] == 'PDB'\n"
            "]\n"
            "print(f'PDB structures for {uniprot_id}:', pdb_ids[:5])\n"
            "# Step 2: download the first structure\n"
            "if pdb_ids:\n"
            "    pdb = pdb_ids[0]\n"
            "    pdb_url = f'https://files.rcsb.org/download/{pdb}.pdb'\n"
            "    with open(f'{pdb}.pdb', 'w') as f:\n"
            "        f.write(requests.get(pdb_url).text)\n"
            "    print(f'Downloaded {pdb}.pdb')\n"
        ),
        "note": "This workflow goes UniProt ID → PDB IDs → download structures. "
                "Use SIFTS (`data/corpus/pdb/sifts_pdb_uniprot.csv`) for a pre-built mapping.",
    },
]


def uniprot_api_example():
    recipe = random.choice(_UNIPROT_RECIPES)
    answer = (
        f"```python\n{recipe['code']}```\n\n"
        f"{recipe['note']}"
    )
    return _record(
        f"How do I {recipe['intent']} using the UniProt REST API?",
        answer,
    )


# ===========================================================================
# CATEGORY 3: RDKit Breadth
# ===========================================================================

# ---------------------------------------------------------------------------
# 3D conformer generation (ETKDG)
# ---------------------------------------------------------------------------

def conformer_3d_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw = round(Descriptors.MolWt(mol), 1)
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    # Rough number of conformers recommended
    n_confs = 50 if n_rot < 5 else 200 if n_rot < 10 else 500
    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import AllChem\n"
        f"mol = Chem.MolFromSmiles('{smiles}')\n"
        "mol_h = Chem.AddHs(mol)  # always add Hs before embedding\n"
        f"params = AllChem.ETKDGv3()\n"
        "params.randomSeed = 42\n"
        f"cids = AllChem.EmbedMultipleConfs(mol_h, numConfs={n_confs}, params=params)\n"
        "# Energy-minimise each conformer with MMFF94\n"
        "results = AllChem.MMFFOptimizeMoleculeConfs(mol_h, mmffVariant='MMFF94')\n"
        "energies = [(cid, e[1]) for cid, e in zip(cids, results) if e[0] == 0]\n"
        "energies.sort(key=lambda x: x[1])\n"
        "print(f'Generated {len(cids)} conformers, {len(energies)} converged')\n"
        "print(f'Lowest energy conformer: cid={energies[0][0]}, E={energies[0][1]:.2f} kcal/mol')\n"
        "# Write all conformers to SDF\n"
        "writer = Chem.SDWriter('conformers.sdf')\n"
        "for cid, _ in energies:\n"
        "    writer.write(mol_h, confId=cid)\n"
        "writer.close()\n"
        "```\n\n"
        f"**{n_confs} conformers** recommended for MW {mw}, {n_rot} rotatable bonds. "
        "ETKDGv3 (2022) uses torsion angle preferences from the Cambridge Structural Database. "
        "Always add Hs before embedding and remove after writing. "
        "MMFF94s (stereospecific) or MMFF94 are both suitable for drug-like molecules."
    )
    questions = [
        f"Generate 3D conformers for {smiles} using RDKit ETKDG.",
        f"How do I create an ensemble of 3D structures for {smiles}?",
        f"Compute energy-minimised conformers for {smiles} with RDKit.",
        f"Generate and rank conformers of {smiles} by MMFF94 energy.",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# Maximum Common Substructure (MCS)
# ---------------------------------------------------------------------------

def mcs_example(smi_a: str, smi_b: str):
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None:
        return None
    try:
        result = _rdFMCS.FindMCS(
            [mol_a, mol_b],
            timeout=1,
            bondCompare=_rdFMCS.BondCompare.CompareOrderExact,
            atomCompare=_rdFMCS.AtomCompare.CompareElements,
        )
        if not result.smartsString or result.numAtoms < 3:
            return None
        mcs_smarts = result.smartsString
        n_atoms = result.numAtoms
    except Exception:
        return None
    answer = (
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import rdFMCS\n"
        f"mol_a = Chem.MolFromSmiles('{smi_a}')\n"
        f"mol_b = Chem.MolFromSmiles('{smi_b}')\n"
        "result = rdFMCS.FindMCS(\n"
        "    [mol_a, mol_b],\n"
        "    timeout=10,\n"
        "    bondCompare=rdFMCS.BondCompare.CompareOrderExact,\n"
        "    atomCompare=rdFMCS.AtomCompare.CompareElements,\n"
        ")\n"
        "print('MCS SMARTS:', result.smartsString)\n"
        "print('MCS atoms:', result.numAtoms)\n"
        "print('MCS bonds:', result.numBonds)\n"
        "```\n\n"
        f"**MCS ({n_atoms} atoms):** `{mcs_smarts}`\n\n"
        "The MCS is the largest substructure shared by both molecules. "
        "Applications: scaffold alignment, RBFE perturbation network design, "
        "matched molecular pair identification, and activity cliff detection."
    )
    questions = [
        f"Find the maximum common substructure of:\nA: {smi_a}\nB: {smi_b}",
        f"What is the MCS between {smi_a} and {smi_b}?",
        f"Compute the largest shared scaffold between {smi_a} and {smi_b} using rdFMCS.",
        f"Identify the common core of {smi_a} and {smi_b}.",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# Reaction SMARTS
# ---------------------------------------------------------------------------

_REACTION_EXAMPLES = [
    {
        "name": "ester hydrolysis",
        "smarts": "[CX3:1](=[OX1])[OX2:2][#6:3]>>[CX3:1](=[OX1])O.[OX2H:2][#6:3]",
        "description": "Hydrolysis of an ester to carboxylic acid + alcohol",
        "example_smi": "CC(=O)OCC",
        "example_name": "ethyl acetate",
    },
    {
        "name": "amide bond formation",
        "smarts": "[CX3:1](=[OX1])[OX2H].[NX3:2]>>[CX3:1](=[OX1])[NX3:2]",
        "description": "Amide coupling (carboxylic acid + amine → amide + water)",
        "example_smi": "CC(=O)O.CN",
        "example_name": "acetic acid + methylamine",
    },
    {
        "name": "N-methylation",
        "smarts": "[NX3:1]>>[NX3:1][CH3]",
        "description": "N-methylation: adds methyl group to any nitrogen",
        "example_smi": "c1ccccc1N",
        "example_name": "aniline",
    },
    {
        "name": "Boc deprotection",
        "smarts": "[NX3:1]C(=O)OC(C)(C)C>>[NX3H:1]",
        "description": "Acid-mediated Boc deprotection to reveal free amine",
        "example_smi": "CC(C)(C)OC(=O)Nc1ccccc1",
        "example_name": "N-Boc aniline",
    },
    {
        "name": "bioisosteric COOH → tetrazole",
        "smarts": "[CX3:1](=[OX1])[OX2H]>>[#6:1]-c1nnn[nH]1",
        "description": "Replace carboxylic acid with tetrazole bioisostere",
        "example_smi": "CC(=O)O",
        "example_name": "acetic acid",
    },
]


def reaction_smarts_example(smiles: str):
    ex = random.choice(_REACTION_EXAMPLES)
    mol = Chem.MolFromSmiles(ex["example_smi"])
    if mol is None:
        return None
    try:
        rxn = _rdChemReactions.ReactionFromSmarts(ex["smarts"])
        products = rxn.RunReactants((mol,))
    except Exception:
        products = []
    prod_smiles = []
    for pset in products[:3]:
        try:
            psmi = ".".join(Chem.MolToSmiles(p) for p in pset if p is not None)
            if psmi:
                prod_smiles.append(psmi)
        except Exception:
            pass
    prod_str = ", ".join(f"`{s}`" for s in prod_smiles[:2]) if prod_smiles else "products depend on substrate"
    answer = (
        f"**Reaction: {ex['name'].title()}**\n\n"
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import rdChemReactions\n"
        f"rxn = rdChemReactions.ReactionFromSmarts('{ex['smarts']}')\n"
        f"mol = Chem.MolFromSmiles('{ex['example_smi']}')  # {ex['example_name']}\n"
        "products = rxn.RunReactants((mol,))\n"
        "for pset in products:\n"
        "    print([Chem.MolToSmiles(p) for p in pset])\n"
        "```\n\n"
        f"**SMARTS:** `{ex['smarts']}`\n\n"
        f"**Description:** {ex['description']}\n\n"
        f"**Applied to {ex['example_name']} (`{ex['example_smi']}`):** {prod_str}"
    )
    questions = [
        f"Write an RDKit reaction SMARTS for {ex['name']} and apply it to {ex['example_smi']}.",
        f"How do I perform {ex['name']} using RDKit reaction SMARTS?",
        f"Apply the {ex['name']} transformation to {ex['example_smi']} using rdChemReactions.",
        f"Encode {ex['name']} as a SMARTS reaction and run it on {ex['example_smi']}.",
    ]
    return _record(random.choice(questions), answer)


# ---------------------------------------------------------------------------
# RECAP / BRICS fragmentation
# ---------------------------------------------------------------------------

def recap_fragmentation_example(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    style = random.choice(["recap", "brics"])
    if style == "recap":
        try:
            tree = _RecapDecompose(mol)
            leaves = tree.GetLeaves()
            frags = sorted({Chem.MolToSmiles(Chem.MolFromSmiles(smi.replace("*", "[H]")))
                            for smi in leaves if smi and "*" in smi
                            and Chem.MolFromSmiles(smi.replace("*", "[H]")) is not None},
                           key=len)[:6]
        except Exception:
            return None
        if not frags:
            return None
        frag_list = "\n".join(f"- `{f}`" for f in frags)
        answer = (
            "```python\n"
            "from rdkit import Chem\n"
            "from rdkit.Chem.Recap import RecapDecompose\n"
            f"mol = Chem.MolFromSmiles('{smiles}')\n"
            "tree = RecapDecompose(mol)\n"
            "leaves = tree.GetLeaves()\n"
            "fragments = sorted(leaves.keys(), key=len)\n"
            "for frag in fragments[:6]:\n"
            "    print(frag)\n"
            "```\n\n"
            f"RECAP fragments of `{smiles}`:\n\n{frag_list}\n\n"
            "RECAP (Retrosynthetic Combinatorial Analysis Procedure) cleaves bonds at 11 "
            "chemical rules mimicking common medicinal chemistry transformations. "
            "Fragments retain the attachment point as `*`. "
            "Used in fragment-based drug discovery and virtual library enumeration."
        )
    else:
        try:
            pieces = _BRICSDecompose(mol)
            frags = sorted({Chem.MolToSmiles(Chem.MolFromSmiles(p))
                            for p in pieces
                            if p and Chem.MolFromSmiles(p) is not None},
                           key=len)[:6]
        except Exception:
            return None
        if not frags:
            return None
        frag_list = "\n".join(f"- `{f}`" for f in frags)
        answer = (
            "```python\n"
            "from rdkit import Chem\n"
            "from rdkit.Chem.BRICS import BRICSDecompose\n"
            f"mol = Chem.MolFromSmiles('{smiles}')\n"
            "fragments = BRICSDecompose(mol)\n"
            "for frag in sorted(fragments, key=len)[:6]:\n"
            "    print(frag)\n"
            "```\n\n"
            f"BRICS fragments of `{smiles}`:\n\n{frag_list}\n\n"
            "BRICS (Breaking of Retrosynthetically Interesting Chemical Substructures) "
            "uses 16 bond-breaking rules to generate synthetically accessible fragments. "
            "The attachment points are encoded as `[14CH2]` etc. "
            "Used for fragment-based design and library enumeration."
        )
    questions = [
        f"Fragment {smiles} using {'RECAP' if style == 'recap' else 'BRICS'} decomposition.",
        f"Apply {'RECAP' if style == 'recap' else 'BRICS'} fragmentation to {smiles} with RDKit.",
        f"What are the {'RECAP' if style == 'recap' else 'BRICS'} fragments of {smiles}?",
        f"Break {smiles} into fragments using RDKit {'Recap' if style == 'recap' else 'BRICS'}.",
    ]
    return _record(random.choice(questions), answer)


# ===========================================================================
# CATEGORY 4: PDB Depth
# ===========================================================================

# ---------------------------------------------------------------------------
# Electron density and structure validation
# ---------------------------------------------------------------------------

_ELECTRON_DENSITY_QA = [
    {
        "q": "How do I validate the electron density fit of a ligand pose?",
        "a": (
            "Three levels of electron density validation:\n\n"
            "**1. RSCC (Real-Space Correlation Coefficient)** — from the PDB validation report:\n"
            "```python\n"
            "import requests\n"
            "pdb_id = '1HSG'\n"
            "comp_id = 'MK1'\n"
            "# Fetch validation report from RCSB\n"
            "url = f'https://data.rcsb.org/rest/v1/core/nonpolymer_entity_instance/{pdb_id}/A/1403'\n"
            "data = requests.get(url).json()\n"
            "# Look for pdbx_vrpt_instance_results in the response\n"
            "print('RSCC:', data.get('pdbx_vrpt_instance_results', {}).get('RSCC'))\n"
            "```\n\n"
            "**2. Visual inspection in Coot or PyMOL:**\n"
            "```\n"
            "# PyMOL with electron density map\n"
            "fetch 1HSG\n"
            "fetch 1HSG, type=2fofc       # 2Fo-Fc map\n"
            "isomesh mesh1, 1HSG_2fofc, 1.5, resn MK1, carve=2\n"
            "show sticks, resn MK1\n"
            "center resn MK1\n"
            "```\n\n"
            "**3. Automated Mogul/CCDC check** for ligand geometry (bond lengths, angles, torsions).\n\n"
            "**Quality thresholds:**\n"
            "| RSCC | Confidence |\n"
            "|---|---|\n"
            "| > 0.90 | Excellent |\n"
            "| 0.80–0.90 | Good |\n"
            "| 0.60–0.80 | Marginal |\n"
            "| < 0.60 | Poor — query the pose |"
        ),
    },
    {
        "q": "What is the difference between Fo-Fc and 2Fo-Fc electron density maps?",
        "a": (
            "Both are difference Fourier maps computed during X-ray refinement:\n\n"
            "**2Fo-Fc (σA-weighted):** The main electron density map. "
            "Shows where atoms ARE. Contoured at 1.0–1.5σ (blue mesh in Coot). "
            "If the ligand is well-placed, it sits inside a blob of 2Fo-Fc density.\n\n"
            "**Fo-Fc (difference map):** Residuals after subtracting the model's calculated "
            "structure factors from the observed data. "
            "**Green (+3σ):** missing atoms or unmodelled density. "
            "**Red (−3σ):** atoms placed where there is no density (wrong position). "
            "The ideal model shows only noise (no significant Fo-Fc features).\n\n"
            "**Omit map:** Like Fo-Fc but the ligand is removed from the model before "
            "phasing — eliminates model bias. Used to confirm a ligand is genuinely present. "
            "If the ligand reappears as a positive Fo-Fc blob, binding is confirmed.\n\n"
            "```\n"
            "# PyMOL: view both maps for a ligand\n"
            "fetch 1HSG\n"
            "fetch 1HSG, type=2fofc\n"
            "fetch 1HSG, type=fofc\n"
            "isomesh mesh_2fofc, 1HSG_2fofc, 1.5, resn MK1, carve=2\n"
            "isomesh mesh_fofc, 1HSG_fofc, 3.0, resn MK1, carve=2\n"
            "color blue, mesh_2fofc\n"
            "color green, mesh_fofc\n"
            "show sticks, resn MK1\n"
            "```"
        ),
    },
    {
        "q": "How do I assess crystal contacts near the binding site?",
        "a": (
            "Crystal contacts are symmetry-related protein copies that pack against each other. "
            "They can distort the binding site (especially flexible loops) relative to the "
            "biological conformation.\n\n"
            "```python\n"
            "# Generate symmetry mates and check contacts in PyMOL\n"
            "fetch 1HSG, async=0\n"
            "remove solvent\n"
            "symexp sym, 1HSG, all, 5  # all symmetry mates within 5 Å\n"
            "show cartoon\n"
            "color grey80, sym*\n"
            "color cyan, 1HSG\n"
            "select binding_site, byres (1HSG within 8 of resn MK1)\n"
            "select crystal_contacts, byres (sym* within 5 of binding_site)\n"
            "show sticks, crystal_contacts\n"
            "color yellow, crystal_contacts\n"
            "```\n\n"
            "```python\n"
            "# Check with gemmi (symmetry-aware)\n"
            "import gemmi\n"
            "st = gemmi.read_structure('1hsg.pdb')\n"
            "ns = gemmi.NeighborSearch(st[0], st.cell, 5.0)\n"
            "ns.populate(include_h=False)\n"
            "print('Space group:', st.spacegroup_hm)\n"
            "print('Unit cell:', st.cell)\n"
            "```\n\n"
            "If a crystal contact residue is within 5 Å of the active site, "
            "the binding mode may be artefactual. Compare with other crystal forms or solution NMR."
        ),
    },
]


def electron_density_example():
    ex = random.choice(_ELECTRON_DENSITY_QA)
    return _record(ex["q"], ex["a"])


# ---------------------------------------------------------------------------
# Biological assembly vs asymmetric unit
# ---------------------------------------------------------------------------

_BIO_ASSEMBLY_QA = [
    {
        "q": "What is the difference between the asymmetric unit and the biological assembly in a PDB file?",
        "a": (
            "**Asymmetric unit (ASU):** the unique content in the crystal that, when combined with "
            "crystallographic symmetry, generates the full crystal lattice. "
            "It may contain a fraction or multiple copies of the biologically relevant molecule.\n\n"
            "**Biological assembly:** the functionally relevant oligomeric state. "
            "For example, HIV protease ASU contains a monomer, but the biological assembly is "
            "a homodimer. A virus capsid ASU may contain 1/60th of the icosahedral shell.\n\n"
            "```python\n"
            "import requests\n"
            "pdb_id = '1HSG'\n"
            "url = f'https://data.rcsb.org/rest/v1/core/entry/{pdb_id}'\n"
            "d = requests.get(url).json()\n"
            "assemblies = d.get('rcsb_entry_info', {}).get('assembly_count', 'unknown')\n"
            "stoich = d.get('rcsb_entry_info', {}).get('polymer_composition', 'unknown')\n"
            "print(f'Assemblies: {assemblies}')\n"
            "print(f'Composition: {stoich}')\n"
            "```\n\n"
            "```\n"
            "# PyMOL: load biological assembly (not ASU)\n"
            "fetch 1HSG, type=pdb1  # pdb1 = first biological assembly\n"
            "remove solvent\n"
            "show cartoon\n"
            "```\n\n"
            "**For drug design:** always use the biological assembly — modelling a monomer "
            "when the target is a dimer can misplace the binding site (e.g., HIV protease dimer interface)."
        ),
    },
    {
        "q": "How do I download and visualise the biological assembly for a PDB entry?",
        "a": (
            "```python\n"
            "import requests\n"
            "pdb_id = '1HSG'\n"
            "assembly_id = '1'  # first biological assembly\n"
            "# Download assembly (mmCIF format — recommended)\n"
            "url = f'https://files.rcsb.org/download/{pdb_id}-assembly{assembly_id}.cif'\n"
            "r = requests.get(url)\n"
            "with open(f'{pdb_id}_assembly{assembly_id}.cif', 'w') as f:\n"
            "    f.write(r.text)\n"
            "print(f'Downloaded {pdb_id} assembly {assembly_id}')\n"
            "```\n\n"
            "```\n"
            "# PyMOL method (easiest)\n"
            "fetch 1HSG, type=pdb1  # first biological assembly\n"
            "# For mmCIF assemblies:\n"
            "# fetch 1HSG, type=cif\n"
            "remove solvent\n"
            "show cartoon\n"
            "color cyan, chain A\n"
            "color salmon, chain B\n"
            "show sticks, organic\n"
            "center organic\n"
            "```\n\n"
            "RCSB provides assemblies as PDB, mmCIF, and PDBML. "
            "For large assemblies (virus capsids), use the `.bcif` (BinaryCIF) format for speed."
        ),
    },
]


def biological_assembly_example():
    ex = random.choice(_BIO_ASSEMBLY_QA)
    return _record(ex["q"], ex["a"])


# ===========================================================================
# CATEGORY 5: Corpus drug enrichment
# ===========================================================================

_DRUG_TARGET_FAMILIES = [
    {
        "family": "GPCR (G protein-coupled receptor)",
        "mechanism": "7-transmembrane receptor; agonists activate (increase cAMP/IP3/Ca²⁺), "
                     "antagonists block, inverse agonists suppress basal activity",
        "examples": [
            {"name": "salbutamol", "smi": "CC(CCc1ccc(O)c(O)c1)NCC(O)c1ccc(O)c(O)c1", "target": "β2-adrenoceptor agonist"},
            {"name": "propranolol", "smi": "CC(C)NCC(O)COc1cccc2ccccc12", "target": "β-adrenoceptor antagonist"},
            {"name": "clozapine", "smi": "CN1CCN(c2nc3ccccc3nc2Cl)CC1", "target": "D4/5-HT2A antagonist (antipsychotic)"},
        ],
        "pdb": "3SN6",
        "key_question": "What type of receptor does {name} target, and what is its mechanism of action?",
    },
    {
        "family": "Ion channel",
        "mechanism": "transmembrane pore; blockers bind within the channel (pore blockers) "
                     "or at extracellular/intracellular vestibules",
        "examples": [
            {"name": "lidocaine", "smi": "CCN(CC)CC(=O)Nc1c(C)cccc1C", "target": "Nav1 pore blocker (local anaesthetic)"},
            {"name": "nifedipine", "smi": "COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]", "target": "Cav1.2 blocker (DHP)"},
            {"name": "amiodarone", "smi": "CCCCc1oc2ccccc2c1CC(=O)Nc1ccc(I)c(OCC)c1", "target": "hERG/Nav/Cav blocker (antiarrhythmic)"},
        ],
        "pdb": "6J8E",
        "key_question": "What ion channel does {name} target and what is its binding mechanism?",
    },
    {
        "family": "Nuclear receptor",
        "mechanism": "ligand-gated transcription factor; agonists induce conformational change "
                     "that recruits co-activators; antagonists block co-activator binding",
        "examples": [
            {"name": "tamoxifen", "smi": "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1", "target": "ERα partial agonist/antagonist (SERM)"},
            {"name": "rosiglitazone", "smi": "CN(CCOc1ccc(CC2SC(=O)NC2=O)cc1)c1ccccc1C", "target": "PPARγ full agonist (insulin sensitiser)"},
            {"name": "dexamethasone", "smi": "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@H]3[C@@H](O)C[C@@]2(C)[C@]1(O)C(=O)CO", "target": "GR agonist (corticosteroid)"},
        ],
        "pdb": "3ERT",
        "key_question": "Describe the mechanism of action of {name} at its nuclear receptor target.",
    },
    {
        "family": "Protease",
        "mechanism": "enzyme active site binder; serine/cysteine/aspartyl/metalloprotease; "
                     "competitive inhibitors mimic the transition state",
        "examples": [
            {"name": "ritonavir", "smi": "CC(C)c1nc(CN(C)C(=O)N[C@@H](CCc2ccccc2)C(=O)N[C@@H](CC(C)C)C[C@H](O)[C@H](Cc2ccccc2)NC(=O)OCC2CCCO2)cs1", "target": "HIV protease aspartyl protease"},
            {"name": "saquinavir", "smi": "CC(C)(C)NC(=O)[C@@H]1C[C@@H]2CCCC[C@H]2CN1C[C@@H](O)[C@H](Cc1ccccc1)NC(=O)[C@@H](CC(N)=O)NC(=O)c1ccc2ccccc2n1", "target": "HIV protease"},
            {"name": "boceprevir", "smi": "CC(C)(C)NC(=O)[C@H]1[C@H](C(=O)NC(CC(=O)NC1)C(=O)c1nc2ccccc2s1)c1ccncc1", "target": "HCV NS3/4A serine protease"},
        ],
        "pdb": "1HSG",
        "key_question": "What type of protease does {name} inhibit and what is the catalytic mechanism?",
    },
]


def drug_target_family_example():
    family = random.choice(_DRUG_TARGET_FAMILIES)
    drug = random.choice(family["examples"])
    mol = Chem.MolFromSmiles(drug["smi"])
    if mol is None:
        return None
    mw   = round(Descriptors.MolWt(mol), 1)
    logp = round(Descriptors.MolLogP(mol), 2)
    answer = (
        f"**{drug['name'].title()}** is a **{drug['target']}**.\n\n"
        f"**Target class:** {family['family']}\n\n"
        f"**Mechanism:** {family['mechanism']}\n\n"
        "```python\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import Descriptors, rdMolDescriptors\n"
        "from rdkit.Chem.QED import qed\n"
        f"mol = Chem.MolFromSmiles('{drug['smi']}')\n"
        "print('MW:',   round(Descriptors.MolWt(mol), 1))\n"
        "print('logP:', round(Descriptors.MolLogP(mol), 2))\n"
        "print('QED:',  round(qed(mol), 3))\n"
        "```\n\n"
        f"MW = {mw}, logP = {logp}.\n\n"
        f"**See also:** PDB {family['pdb']} for a representative co-crystal structure.\n\n"
        f"```\n"
        f"fetch {family['pdb']}, async=0\n"
        f"remove solvent\n"
        f"show cartoon\n"
        f"show sticks, organic\n"
        f"util.cbag organic\n"
        f"center organic\n"
        f"```"
    )
    return _record(
        family["key_question"].format(name=drug["name"]),
        answer,
    )


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
        # R5: QED — 6 entries to fix 7:1 underrepresentation
        ("qed_score",            lambda: qed_example(random.choice(SEED_SMILES))),
        ("qed_score",            lambda: qed_example(random.choice(SEED_SMILES))),
        ("qed_score",            lambda: qed_example(random.choice(SEED_SMILES))),
        ("qed_score",            lambda: qed_example(random.choice(SEED_SMILES))),
        ("qed_score",            lambda: qed_example(random.choice(SEED_SMILES))),
        ("qed_score",            lambda: qed_example(random.choice(SEED_SMILES))),
        # R5: TPSA — 6 entries (long-form + acronym phrasings)
        ("tpsa_score",           lambda: tpsa_example(random.choice(SEED_SMILES))),
        ("tpsa_score",           lambda: tpsa_example(random.choice(SEED_SMILES))),
        ("tpsa_score",           lambda: tpsa_example(random.choice(SEED_SMILES))),
        ("tpsa_score",           lambda: tpsa_example(random.choice(SEED_SMILES))),
        ("tpsa_score",           lambda: tpsa_example(random.choice(SEED_SMILES))),
        ("tpsa_score",           lambda: tpsa_example(random.choice(SEED_SMILES))),
        # R5: Scaffold equality — explicit Boolean, disambiguated from Tanimoto
        ("scaffold_equality",    lambda: scaffold_equality_example(*random.sample(SEED_SMILES, 2))),
        ("scaffold_equality",    lambda: scaffold_equality_example(*random.sample(SEED_SMILES, 2))),
        ("scaffold_equality",    lambda: scaffold_equality_example(*random.sample(SEED_SMILES, 2))),
        ("scaffold_equality",    lambda: scaffold_equality_example(*random.sample(SEED_SMILES, 2))),
        ("scaffold_equality",    lambda: scaffold_equality_example(*random.sample(SEED_SMILES, 2))),
        # R5: Standalone HBD/HBA/RotBonds — fix Descriptors.HBDCount hallucination
        ("hbd_hba_count",        lambda: hbd_hba_example(random.choice(SEED_SMILES))),
        ("hbd_hba_count",        lambda: hbd_hba_example(random.choice(SEED_SMILES))),
        ("hbd_hba_count",        lambda: hbd_hba_example(random.choice(SEED_SMILES))),
        ("hbd_hba_count",        lambda: hbd_hba_example(random.choice(SEED_SMILES))),
        ("hbd_hba_count",        lambda: hbd_hba_example(random.choice(SEED_SMILES))),
        # R5: Salt stripping / largest fragment
        ("salt_strip",           lambda: salt_strip_example(random.choice(SEED_SMILES))),
        ("salt_strip",           lambda: salt_strip_example(random.choice(SEED_SMILES))),
        ("salt_strip",           lambda: salt_strip_example(random.choice(SEED_SMILES))),
        # R5: InChI and InChIKey
        ("inchikey",             lambda: inchikey_example(random.choice(SEED_SMILES))),
        ("inchikey",             lambda: inchikey_example(random.choice(SEED_SMILES))),
        ("inchikey",             lambda: inchikey_example(random.choice(SEED_SMILES))),
        # R5: Chirality / stereocenters
        ("stereo_centers",       lambda: stereo_example(random.choice(SEED_SMILES))),
        ("stereo_centers",       lambda: stereo_example(random.choice(SEED_SMILES))),
        ("stereo_centers",       lambda: stereo_example(random.choice(SEED_SMILES))),
        # R5: Tautomer enumeration
        ("tautomers",            lambda: tautomer_example(random.choice(SEED_SMILES))),
        ("tautomers",            lambda: tautomer_example(random.choice(SEED_SMILES))),
        ("tautomers",            lambda: tautomer_example(random.choice(SEED_SMILES))),
        # R5: Multi-molecule QED ranking
        ("multi_mol_rank",       lambda: multi_mol_rank_example(random.sample(SEED_SMILES, random.choice([3, 4])))),
        ("multi_mol_rank",       lambda: multi_mol_rank_example(random.sample(SEED_SMILES, random.choice([3, 4])))),
        ("multi_mol_rank",       lambda: multi_mol_rank_example(random.sample(SEED_SMILES, random.choice([3, 4])))),
        # R5: QED code-then-quote (explicit compute→quote pattern for QED)
        ("qed_code_then_quote",  lambda: qed_code_then_quote_example(random.choice(SEED_SMILES))),
        ("qed_code_then_quote",  lambda: qed_code_then_quote_example(random.choice(SEED_SMILES))),
        ("qed_code_then_quote",  lambda: qed_code_then_quote_example(random.choice(SEED_SMILES))),
        # R5: Fingerprint similarity (explicit, no scaffold confusion)
        ("fingerprint_sim",      lambda: fingerprint_similarity_example(*random.sample(SEED_SMILES, 2))),
        ("fingerprint_sim",      lambda: fingerprint_similarity_example(*random.sample(SEED_SMILES, 2))),
        ("fingerprint_sim",      lambda: fingerprint_similarity_example(*random.sample(SEED_SMILES, 2))),
        # R5: Lipinski card (full property table with QED)
        ("lipinski_card",        lambda: lipinski_card_example(random.choice(SEED_SMILES))),
        ("lipinski_card",        lambda: lipinski_card_example(random.choice(SEED_SMILES))),
        ("lipinski_card",        lambda: lipinski_card_example(random.choice(SEED_SMILES))),
        # Original generators (kept)
        ("molecular_formula",    lambda: molecular_formula_example(random.choice(SEED_SMILES))),
        ("smiles_validation",    smiles_validation_example),
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
        # --- R5: Medchem / structural filters ---
        ("pains_filter",         lambda: pains_filter_example(random.choice(SEED_SMILES))),
        ("pains_filter",         lambda: pains_filter_example(random.choice(SEED_SMILES))),
        ("pains_filter",         lambda: pains_filter_example(random.choice(SEED_SMILES))),
        ("structural_alert",     lambda: structural_alert_example(random.choice(SEED_SMILES))),
        ("structural_alert",     lambda: structural_alert_example(random.choice(SEED_SMILES))),
        ("structural_alert",     lambda: structural_alert_example(random.choice(SEED_SMILES))),
        ("bioisostere",          lambda: bioisostere_example(random.choice(SEED_SMILES))),
        ("bioisostere",          lambda: bioisostere_example(random.choice(SEED_SMILES))),
        ("bioisostere",          lambda: bioisostere_example(random.choice(SEED_SMILES))),
        ("lead_opt_rules",       lambda: lead_opt_rules_example(random.choice(SEED_SMILES))),
        ("lead_opt_rules",       lambda: lead_opt_rules_example(random.choice(SEED_SMILES))),
        ("lead_opt_rules",       lambda: lead_opt_rules_example(random.choice(SEED_SMILES))),
        ("adme_heuristics",      lambda: adme_heuristics_example(random.choice(SEED_SMILES))),
        ("adme_heuristics",      lambda: adme_heuristics_example(random.choice(SEED_SMILES))),
        ("adme_heuristics",      lambda: adme_heuristics_example(random.choice(SEED_SMILES))),
        ("sa_score",             lambda: sa_score_example(random.choice(SEED_SMILES))),
        ("sa_score",             lambda: sa_score_example(random.choice(SEED_SMILES))),
        ("sa_score",             lambda: sa_score_example(random.choice(SEED_SMILES))),
        # --- R5: PDB / structural biology expansion ---
        ("pdb_quality",          pdb_quality_example),
        ("pdb_quality",          pdb_quality_example),
        ("pdb_quality",          pdb_quality_example),
        ("cofactor_binding",     cofactor_binding_example),
        ("cofactor_binding",     cofactor_binding_example),
        ("cofactor_binding",     cofactor_binding_example),
        ("pdb_fetch_api",        pdb_fetch_api_example),
        ("pdb_fetch_api",        pdb_fetch_api_example),
        ("pdb_fetch_api",        pdb_fetch_api_example),
        ("hbond_structure",      hbond_structure_example),
        ("hbond_structure",      hbond_structure_example),
        ("hbond_structure",      hbond_structure_example),
        # --- R5: PyExec drills (×6 — directly target 75% PyExec gap) ---
        ("pyexec_drill",         lambda: pyexec_drill_example(random.choice(SEED_SMILES))),
        ("pyexec_drill",         lambda: pyexec_drill_example(random.choice(SEED_SMILES))),
        ("pyexec_drill",         lambda: pyexec_drill_example(random.choice(SEED_SMILES))),
        ("pyexec_drill",         lambda: pyexec_drill_example(random.choice(SEED_SMILES))),
        ("pyexec_drill",         lambda: pyexec_drill_example(random.choice(SEED_SMILES))),
        ("pyexec_drill",         lambda: pyexec_drill_example(random.choice(SEED_SMILES))),
        # --- R5: Code→Quote v2 (×5 — target 59% C→Q) ---
        ("code_then_quote_v2",   lambda: code_then_quote_v2_example(random.choice(SEED_SMILES))),
        ("code_then_quote_v2",   lambda: code_then_quote_v2_example(random.choice(SEED_SMILES))),
        ("code_then_quote_v2",   lambda: code_then_quote_v2_example(random.choice(SEED_SMILES))),
        ("code_then_quote_v2",   lambda: code_then_quote_v2_example(random.choice(SEED_SMILES))),
        ("code_then_quote_v2",   lambda: code_then_quote_v2_example(random.choice(SEED_SMILES))),
        # --- R5: Rounding drills (×5 — target 95% rounding) ---
        ("rounding_explicit",    lambda: rounding_explicit_example(random.choice(SEED_SMILES))),
        ("rounding_explicit",    lambda: rounding_explicit_example(random.choice(SEED_SMILES))),
        ("rounding_explicit",    lambda: rounding_explicit_example(random.choice(SEED_SMILES))),
        ("rounding_explicit",    lambda: rounding_explicit_example(random.choice(SEED_SMILES))),
        ("rounding_explicit",    lambda: rounding_explicit_example(random.choice(SEED_SMILES))),
        # --- R5: Multi-step fidelity (×5 — target 61% numerical fidelity) ---
        ("fidelity_multistep",   lambda: fidelity_multistep_example(random.choice(SEED_SMILES))),
        ("fidelity_multistep",   lambda: fidelity_multistep_example(random.choice(SEED_SMILES))),
        ("fidelity_multistep",   lambda: fidelity_multistep_example(random.choice(SEED_SMILES))),
        ("fidelity_multistep",   lambda: fidelity_multistep_example(random.choice(SEED_SMILES))),
        ("fidelity_multistep",   lambda: fidelity_multistep_example(random.choice(SEED_SMILES))),
        # --- R5 Cat 1: Medicinal chemistry ---
        ("herg_liability",       lambda: herg_liability_example(random.choice(SEED_SMILES))),
        ("herg_liability",       lambda: herg_liability_example(random.choice(SEED_SMILES))),
        ("herg_liability",       lambda: herg_liability_example(random.choice(SEED_SMILES))),
        ("selectivity_profile",  lambda: selectivity_profile_example(random.choice(SEED_SMILES))),
        ("selectivity_profile",  lambda: selectivity_profile_example(random.choice(SEED_SMILES))),
        ("selectivity_profile",  lambda: selectivity_profile_example(random.choice(SEED_SMILES))),
        ("prodrug_bcs",          lambda: prodrug_bcs_example(random.choice(SEED_SMILES))),
        ("prodrug_bcs",          lambda: prodrug_bcs_example(random.choice(SEED_SMILES))),
        ("prodrug_bcs",          lambda: prodrug_bcs_example(random.choice(SEED_SMILES))),
        ("sar_delta",            lambda: sar_delta_example(*random.sample(SEED_SMILES, 2))),
        ("sar_delta",            lambda: sar_delta_example(*random.sample(SEED_SMILES, 2))),
        ("sar_delta",            lambda: sar_delta_example(*random.sample(SEED_SMILES, 2))),
        # --- R5 Cat 2: Structural biology depth ---
        ("mdanalysis",           mdanalysis_example),
        ("mdanalysis",           mdanalysis_example),
        ("mdanalysis",           mdanalysis_example),
        ("dssp",                 dssp_example),
        ("dssp",                 dssp_example),
        ("dssp",                 dssp_example),
        ("ppi_interface",        ppi_interface_example),
        ("ppi_interface",        ppi_interface_example),
        ("ppi_interface",        ppi_interface_example),
        ("uniprot_api",          uniprot_api_example),
        ("uniprot_api",          uniprot_api_example),
        ("uniprot_api",          uniprot_api_example),
        # --- R5 Cat 3: RDKit breadth ---
        ("conformer_3d",         lambda: conformer_3d_example(random.choice(SEED_SMILES))),
        ("conformer_3d",         lambda: conformer_3d_example(random.choice(SEED_SMILES))),
        ("conformer_3d",         lambda: conformer_3d_example(random.choice(SEED_SMILES))),
        ("mcs_search",           lambda: mcs_example(*random.sample(SEED_SMILES, 2))),
        ("mcs_search",           lambda: mcs_example(*random.sample(SEED_SMILES, 2))),
        ("mcs_search",           lambda: mcs_example(*random.sample(SEED_SMILES, 2))),
        ("reaction_smarts",      lambda: reaction_smarts_example(random.choice(SEED_SMILES))),
        ("reaction_smarts",      lambda: reaction_smarts_example(random.choice(SEED_SMILES))),
        ("reaction_smarts",      lambda: reaction_smarts_example(random.choice(SEED_SMILES))),
        ("recap_fragmentation",  lambda: recap_fragmentation_example(random.choice(SEED_SMILES))),
        ("recap_fragmentation",  lambda: recap_fragmentation_example(random.choice(SEED_SMILES))),
        ("recap_fragmentation",  lambda: recap_fragmentation_example(random.choice(SEED_SMILES))),
        # --- R5 Cat 4: PDB depth ---
        ("electron_density",     electron_density_example),
        ("electron_density",     electron_density_example),
        ("electron_density",     electron_density_example),
        ("biological_assembly",  biological_assembly_example),
        ("biological_assembly",  biological_assembly_example),
        ("biological_assembly",  biological_assembly_example),
        # --- R5 Cat 5: Drug target families ---
        ("drug_target_family",   drug_target_family_example),
        ("drug_target_family",   drug_target_family_example),
        ("drug_target_family",   drug_target_family_example),
        ("drug_target_family",   drug_target_family_example),
        ("corpus_drug",          corpus_drug_example),
        ("corpus_drug",          corpus_drug_example),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",  type=Path, default=Path("data/sft"))
    ap.add_argument("--n",    type=int,  default=20000)
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
            ex["class"] = name
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
