from __future__ import print_function
import io
import jax.numpy as jnp
from prody import *
confProDy(verbosity="none")

restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
}

restype_str_to_int = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
    "X": 20,
}

restype_int_to_str = {
    0: "A",
    1: "C",
    2: "D",
    3: "E",
    4: "F",
    5: "G",
    6: "H",
    7: "I",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "P",
    13: "Q",
    14: "R",
    15: "S",
    16: "T",
    17: "V",
    18: "W",
    19: "Y",
    20: "X",
}

alphabet = list(restype_str_to_int)

element_list = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mb",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Uut",
    "Fl",
    "Uup",
    "Lv",
    "Uus",
    "Uuo",
]
element_list = [item.upper() for item in element_list]
element_dict_rev = dict(zip(range(1, len(element_list)), element_list))

def get_seq_rec(S: jnp.ndarray, S_pred: jnp.ndarray, mask: jnp.ndarray):
    """
    S : true sequence shape=[batch, length]
    S_pred : predicted sequence shape=[batch, length]
    mask : mask to compute average over the region shape=[batch, length]
    average : averaged sequence recovery shape=[batch]
    """
    match = S == S_pred
    average = jnp.sum(match * mask, axis=-1) / jnp.sum(mask, axis=-1)
    return average

def get_score(S: jnp.ndarray, log_probs: jnp.ndarray, mask: jnp.ndarray):
    """
    S : true sequence shape=[batch, length]
    log_probs : predicted sequence shape=[batch, length]
    mask : mask to compute average over the region shape=[batch, length]
    average_loss : averaged categorical cross entropy (CCE) [batch]
    loss_per_resdue : per position CCE [batch, length]
    """
    S_one_hot = jax.nn.one_hot(S, 21)
    loss_per_residue = -(S_one_hot * log_probs).sum(-1)  # [B, L]
    average_loss = jnp.sum(loss_per_residue * mask, axis=-1) / (
        jnp.sum(mask, axis=-1) + 1e-8
    )
    return average_loss, loss_per_residue

def write_full_PDB(
    save_path: str,
    X: jnp.ndarray,
    X_m: jnp.ndarray,
    b_factors: jnp.ndarray,
    R_idx: jnp.ndarray,
    chain_letters: jnp.ndarray,
    S: jnp.ndarray,
    other_atoms=None,
    icodes=None,
    force_hetatm=False,
):
    """
    save_path : path where the PDB will be written to
    X : protein atom xyz coordinates shape=[length, 14, 3]
    X_m : protein atom mask shape=[length, 14]
    b_factors: shape=[length, 14]
    R_idx: protein residue indices shape=[length]
    chain_letters: protein chain letters shape=[length]
    S : protein amino acid sequence shape=[length]
    other_atoms: other atoms parsed by prody
    icodes: a list of insertion codes for the PDB; e.g. antibody loops
    """
    restype_1to3 = {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "Q": "GLN",
        "E": "GLU",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
        "X": "UNK",
    }
    restype_INTtoSTR = {
        0: "A",
        1: "C",
        2: "D",
        3: "E",
        4: "F",
        5: "G",
        6: "H",
        7: "I",
        8: "K",
        9: "L",
        10: "M",
        11: "N",
        12: "P",
        13: "Q",
        14: "R",
        15: "S",
        16: "T",
        17: "V",
        18: "W",
        19: "Y",
        20: "X",
    }
    restype_name_to_atom14_names = {
        "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
        "ARG": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD",
            "NE",
            "CZ",
            "NH1",
            "NH2",
            "",
            "",
            "",
        ],
        "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
        "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
        "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
        "GLN": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD",
            "OE1",
            "NE2",
            "",
            "",
            "",
            "",
            "",
        ],
        "GLU": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD",
            "OE1",
            "OE2",
            "",
            "",
            "",
            "",
            "",
        ],
        "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
        "HIS": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "ND1",
            "CD2",
            "CE1",
            "NE2",
            "",
            "",
            "",
            "",
        ],
        "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
        "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
        "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
        "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
        "PHE": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "CE1",
            "CE2",
            "CZ",
            "",
            "",
            "",
        ],
        "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
        "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
        "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
        "TRP": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "CE2",
            "CE3",
            "NE1",
            "CZ2",
            "CZ3",
            "CH2",
        ],
        "TYR": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "CE1",
            "CE2",
            "CZ",
            "OH",
            "",
            "",
        ],
        "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
        "UNK": ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
    }
    S_str = [restype_1to3[AA] for AA in [restype_INTtoSTR[AA] for AA in S]]
    X_list = []
    b_factor_list = []
    atom_name_list = []
    element_name_list = []
    residue_name_list = []
    residue_number_list = []
    chain_id_list = []
    icodes_list = []
    for i, AA in enumerate(S_str):
        sel = X_m[i].astype(jnp.int32) == 1
        total = jnp.sum(sel)
        tmp = jnp.array(restype_name_to_atom14_names[AA])[sel]
        X_list.append(X[i][sel])
        b_factor_list.append(b_factors[i][sel])
        atom_name_list.append(tmp)
        element_name_list += [AA[:1] for AA in list(tmp)]
        residue_name_list += total * [AA]
        residue_number_list += total * [R_idx[i]]
        chain_id_list += total * [chain_letters[i]]
        icodes_list += total * [icodes[i]]
    X_stack = jnp.concatenate(X_list, 0)
    b_factor_stack = jnp.concatenate(b_factor_list, 0)
    atom_name_stack = jnp.concatenate(atom_name_list, 0)
    protein = prody.AtomGroup()
    protein.setCoords(X_stack)
    protein.setBetas(b_factor_stack)
    protein.setNames(atom_name_stack)
    protein.setResnames(residue_name_list)
    protein.setElements(element_name_list)
    protein.setOccupancies(jnp.ones([X_stack.shape[0]]))
    protein.setResnums(residue_number_list)
    protein.setChids(chain_id_list)
    protein.setIcodes(icodes_list)
    if other_atoms:
        other_atoms_g = prody.AtomGroup()
        other_atoms_g.setCoords(other_atoms.getCoords())
        other_atoms_g.setNames(other_atoms.getNames())
        other_atoms_g.setResnames(other_atoms.getResnames())
        other_atoms_g.setElements(other_atoms.getElements())
        other_atoms_g.setOccupancies(other_atoms.getOccupancies())
        other_atoms_g.setResnums(other_atoms.getResnums())
        other_atoms_g.setChids(other_atoms.getChids())
        if force_hetatm:
            other_atoms_g.setFlags("hetatm", other_atoms.getFlags("hetatm"))
        writePDB(save_path, protein + other_atoms_g)
    else:
        writePDB(save_path, protein)

def get_aligned_coordinates(protein_atoms, CA_dict: dict, atom_name: str):
    """
    protein_atoms: prody atom group
    CA_dict: mapping between chain_residue_idx_icodes and integers
    atom_name: atom to be parsed; e.g. CA
    """
    atom_atoms = protein_atoms.select(f"name {atom_name}")
    if atom_atoms != None:
        atom_coords = atom_atoms.getCoords()
        atom_resnums = atom_atoms.getResnums()
        atom_chain_ids = atom_atoms.getChids()
        atom_icodes = atom_atoms.getIcodes()
    atom_coords_ = jnp.zeros([len(CA_dict), 3], jnp.float32)
    atom_coords_m = jnp.zeros([len(CA_dict)], jnp.int32)
    if atom_atoms != None:
        for i in range(len(atom_resnums)):
            code = atom_chain_ids[i] + "_" + str(atom_resnums[i]) + "_" + atom_icodes[i]
            if code in list(CA_dict):
                atom_coords_[CA_dict[code], :] = atom_coords[i]
                atom_coords_m[CA_dict[code]] = 1
    return atom_coords_, atom_coords_m

def parse_PDB(
    input_path: str,
    chains: list = [],
    parse_all_atoms: bool = False,
    parse_atoms_with_zero_occupancy: bool = False
):
    """
    input_path : path for the input PDB
    chains: a list specifying which chains need to be parsed; e.g. ["A", "B"]
    parse_all_atoms: if False parse only N,CA,C,O otherwise all 37 atoms
    parse_atoms_with_zero_occupancy: if True atoms with zero occupancy will be parsed
    """
    element_list = [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mb",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Uut",
        "Fl",
        "Uup",
        "Lv",
        "Uus",
        "Uuo",
    ]
    element_list = [item.upper() for item in element_list]
    element_dict = dict(zip(element_list, range(1, len(element_list))))
    restype_3to1 = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
    restype_STRtoINT = {
        "A": 0,
        "C": 1,
        "D": 2,
        "E": 3,
        "F": 4,
        "G": 5,
        "H": 6,
        "I": 7,
        "K": 8,
        "L": 9,
        "M": 10,
        "N": 11,
        "P": 12,
        "Q": 13,
        "R": 14,
        "S": 15,
        "T": 16,
        "V": 17,
        "W": 18,
        "Y": 19,
        "X": 20,
    }
    atom_order = {
        "N": 0,
        "CA": 1,
        "C": 2,
        "CB": 3,
        "O": 4,
        "CG": 5,
        "CG1": 6,
        "CG2": 7,
        "OG": 8,
        "OG1": 9,
        "SG": 10,
        "CD": 11,
        "CD1": 12,
        "CD2": 13,
        "ND1": 14,
        "ND2": 15,
        "OD1": 16,
        "OD2": 17,
        "SD": 18,
        "CE": 19,
        "CE1": 20,
        "CE2": 21,
        "CE3": 22,
        "NE": 23,
        "NE1": 24,
        "NE2": 25,
        "OE1": 26,
        "OE2": 27,
        "CH2": 28,
        "NH1": 29,
        "NH2": 30,
        "OH": 31,
        "CZ": 32,
        "CZ2": 33,
        "CZ3": 34,
        "NZ": 35,
        "OXT": 36,
    }
    if not parse_all_atoms:
        atom_types = ["N", "CA", "C", "O"]
    else:
        atom_types = [
            "N",
            "CA",
            "C",
            "CB",
            "O",
            "CG",
            "CG1",
            "CG2",
            "OG",
            "OG1",
            "SG",
            "CD",
            "CD1",
            "CD2",
            "ND1",
            "ND2",
            "OD1",
            "OD2",
            "SD",
            "CE",
            "CE1",
            "CE2",
            "CE3",
            "NE",
            "NE1",
            "NE2",
            "OE1",
            "OE2",
            "CH2",
            "NH1",
            "NH2",
            "OH",
            "CZ",
            "CZ2",
            "CZ3",
            "NZ",
        ]
    if isinstance(input_path, str):
        atoms = parsePDB(input_path)
    elif isinstance(input_path, io.StringIO):
        atoms = parsePDBStream(input_path)
    if not parse_atoms_with_zero_occupancy:
        atoms = atoms.select("occupancy > 0")
    if chains:
        str_out = ""
        for item in chains:
            str_out += " chain " + item + " or"
        atoms = atoms.select(str_out[1:-3])
    protein_atoms = atoms.select("protein")
    backbone = protein_atoms.select("backbone")
    other_atoms = atoms.select("not protein and not water")
    water_atoms = atoms.select("water")
    CA_atoms = protein_atoms.select("name CA")
    CA_resnums = CA_atoms.getResnums()
    CA_chain_ids = CA_atoms.getChids()
    CA_icodes = CA_atoms.getIcodes()
    CA_dict = {}
    for i in range(len(CA_resnums)):
        code = CA_chain_ids[i] + "_" + str(CA_resnums[i]) + "_" + CA_icodes[i]
        CA_dict[code] = i
    xyz_37 = jnp.zeros([len(CA_dict), 37, 3], jnp.float32)
    xyz_37_m = jnp.zeros([len(CA_dict), 37], jnp.int32)
    for atom_name in atom_types:
        xyz, xyz_m = get_aligned_coordinates(protein_atoms, CA_dict, atom_name)
        xyz_37[:, atom_order[atom_name], :] = xyz
        xyz_37_m[:, atom_order[atom_name]] = xyz_m
    N = xyz_37[:, atom_order["N"], :]
    CA = xyz_37[:, atom_order["CA"], :]
    C = xyz_37[:, atom_order["C"], :]
    O = xyz_37[:, atom_order["O"], :]
    N_m = xyz_37_m[:, atom_order["N"]]
    CA_m = xyz_37_m[:, atom_order["CA"]]
    C_m = xyz_37_m[:, atom_order["C"]]
    O_m = xyz_37_m[:, atom_order["O"]]
    mask = N_m * CA_m * C_m * O_m  # must all 4 atoms exist
    b = CA - N
    c = C - CA
    a = jnp.cross(b, c, axis=-1)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
    chain_labels = jnp.array(CA_atoms.getChindices(), dtype=jnp.int32)
    R_idx = jnp.array(CA_resnums, dtype=jnp.int32)
    S = CA_atoms.getResnames()
    S = [restype_3to1[AA] if AA in list(restype_3to1) else "X" for AA in list(S)]
    S = jnp.array([restype_STRtoINT[AA] for AA in list(S)], jnp.int32)
    X = jnp.concatenate([N[:, None], CA[:, None], C[:, None], O[:, None]], 1)
    try:
        Y = jnp.array(other_atoms.getCoords(), dtype=jnp.float32)
        Y_t = list(other_atoms.getElements())
        Y_t = jnp.array(
            [
                element_dict[y_t.upper()] if y_t.upper() in element_list else 0
                for y_t in Y_t
            ],
            dtype=jnp.int32,
        )
        Y_m = (Y_t != 1) * (Y_t != 0)
        Y = Y[Y_m, :]
        Y_t = Y_t[Y_m]
        Y_m = Y_m[Y_m]
    except:
        Y = jnp.zeros([1, 3], jnp.float32)
        Y_t = jnp.zeros([1], jnp.int32)
        Y_m = jnp.zeros([1], jnp.int32)
    output_dict = {}
    output_dict["X"] = X
    output_dict["mask"] = mask
    output_dict["Y"] = Y
    output_dict["Y_t"] = Y_t
    output_dict["Y_m"] = Y_m
    output_dict["R_idx"] = R_idx
    output_dict["chain_labels"] = chain_labels
    output_dict["chain_letters"] = CA_chain_ids
    mask_c = []
    chain_list = list(set(output_dict["chain_letters"]))
    chain_list.sort()
    for chain in chain_list:
        mask_c.append(
            jnp.array(
                [chain == item for item in output_dict["chain_letters"]],
                dtype=bool,
            )
        )
    output_dict["mask_c"] = mask_c
    output_dict["chain_list"] = chain_list
    output_dict["S"] = S
    output_dict["xyz_37"] = xyz_37
    output_dict["xyz_37_m"] = xyz_37_m
    return output_dict, backbone, other_atoms, CA_icodes, water_atoms

def get_nearest_neighbours(CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms):
    mask_CBY = mask[:, None] * Y_m[None, :]  # [A,B]
    L2_AB = jnp.sum((CB[:, None, :] - Y[None, :, :]) ** 2, -1)
    L2_AB = L2_AB * mask_CBY + (1 - mask_CBY) * 1000.0
    nn_idx = jnp.argsort(L2_AB, -1)[:, :number_of_ligand_atoms]
    L2_AB_nn = jnp.take_along_axis(L2_AB, nn_idx, axis=1)
    D_AB_closest = jnp.sqrt(L2_AB_nn[:, 0])
    Y_r = jnp.repeat(Y[None, :, :], CB.shape[0], axis=0)
    Y_t_r = jnp.repeat(Y_t[None, :], CB.shape[0], axis=0)
    Y_m_r = jnp.repeat(Y_m[None, :], CB.shape[0], axis=0)
    Y_tmp = jnp.take_along_axis(Y_r, nn_idx[:, :, None], axis=1)
    Y_t_tmp = jnp.take_along_axis(Y_t_r, nn_idx, axis=1)
    Y_m_tmp = jnp.take_along_axis(Y_m_r, nn_idx, axis=1)
    Y = jnp.zeros([CB.shape[0], number_of_ligand_atoms, 3], dtype=jnp.float32)
    Y_t = jnp.zeros([CB.shape[0], number_of_ligand_atoms], dtype=jnp.int32)
    Y_m = jnp.zeros([CB.shape[0], number_of_ligand_atoms], dtype=jnp.int32)
    num_nn_update = Y_tmp.shape[1]
    Y = Y.at[:, :num_nn_update].set(Y_tmp)
    Y_t = Y_t.at[:, :num_nn_update].set(Y_t_tmp)
    Y_m = Y_m.at[:, :num_nn_update].set(Y_m_tmp)
    return Y, Y_t, Y_m, D_AB_closest

def featurize(
    input_dict,
    cutoff_for_score=8.0,
    use_atom_context=True,
    number_of_ligand_atoms=16,
    model_type="protein_mpnn",
):
    output_dict = {}
    if model_type == "ligand_mpnn":
        mask = input_dict["mask"]
        Y = input_dict["Y"]
        Y_t = input_dict["Y_t"]
        Y_m = input_dict["Y_m"]
        N = input_dict["X"][:, 0, :]
        CA = input_dict["X"][:, 1, :]
        C = input_dict["X"][:, 2, :]
        b = CA - N
        c = C - CA
        a = jnp.cross(b, c, axis=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        Y, Y_t, Y_m, D_XY = get_nearest_neighbours(
            CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms
        )
        mask_XY = (D_XY < cutoff_for_score) * mask * Y_m[:, 0]
        output_dict["mask_XY"] = mask_XY[None,]
        if "side_chain_mask" in list(input_dict):
            output_dict["side_chain_mask"] = input_dict["side_chain_mask"][None,]
        output_dict["Y"] = Y[None,]
        output_dict["Y_t"] = Y_t[None,]
        output_dict["Y_m"] = Y_m[None,]
        if not use_atom_context:
            output_dict["Y_m"] = 0.0 * output_dict["Y_m"]
    elif (
        model_type == "per_residue_label_membrane_mpnn"
        or model_type == "global_label_membrane_mpnn"
    ):
        output_dict["membrane_per_residue_labels"] = input_dict[
            "membrane_per_residue_labels"
        ][None,]
    R_idx_list = []
    count = 0
    R_idx_prev = -100000
    for R_idx in list(input_dict["R_idx"]):
        if R_idx_prev == R_idx:
            count += 1
        R_idx_list.append(R_idx + count)
        R_idx_prev = R_idx
    R_idx_renumbered = jnp.array(R_idx_list)
    output_dict["R_idx"] = R_idx_renumbered[None,]
    output_dict["R_idx_original"] = input_dict["R_idx"][None,]
    output_dict["chain_labels"] = input_dict["chain_labels"][None,]
    output_dict["S"] = input_dict["S"][None,]
    output_dict["chain_mask"] = input_dict["chain_mask"][None,]
    output_dict["mask"] = input_dict["mask"][None,]
    output_dict["X"] = input_dict["X"][None,]
    if "xyz_37" in list(input_dict):
        output_dict["xyz_37"] = input_dict["xyz_37"][None,]
        output_dict["xyz_37_m"] = input_dict["xyz_37_m"][None,]
    return output_dict