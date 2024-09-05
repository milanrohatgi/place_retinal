from Bio.PDB import PDBParser, Superimposer, PDBIO
from Bio.PDB.Polypeptide import is_aa
from Bio import SeqIO
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
from Bio.PDB import Atom, Residue
import sys
from Bio.PDB import PDBIO
from Bio.PDB import PDBParser, NeighborSearch

# MSA Locations of 24 residues and lysine
POSITIONS = [
    429,
    465,
    469,
    597,
    599,
    600,
    603,
    604,
    607,
    651,
    652,
    655,
    687,
    690,
    691,
    694,
    774,
    777,
    778,
    781,
    828,
    832,
    835,
    836,
]

msa_index = 835

BR_reference_path = "/home/groups/deissero/mrohatgi/retinal/BR_true.pdb"
output_path = "/home/groups/deissero/mrohatgi/retinal/final_structures/"
temp_alignment_path = "/home/groups/deissero/mrohatgi/retinal/aligned_target.pdb"
msa_file = "/home/groups/deissero/mrohatgi/retinal/cleaned_updated_alignment.txt"
visualizations_path = "/home/groups/deissero/mrohatgi/retinal/visuals"

def get_positional_vectors():
    parser = PDBParser()
    structure = parser.get_structure("rhodopsin", BR_reference_path)

    lys_residue_id = 216
    retinal_residue_name = "RET"

    lysine_residue = None
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[1] == lys_residue_id:
                    lysine_residue = residue

    retinal_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == retinal_residue_name:
                    retinal_atoms.extend(residue.get_atoms())

    nz_atom = lysine_residue['NZ']
    nz_coord = nz_atom.get_vector()

    positional_vectors = {}
    for atom in retinal_atoms:
        atom_coord = atom.get_vector()
        vector = atom_coord - nz_coord
        positional_vectors[atom.get_name()] = vector

    return positional_vectors


def align_target(target):
    parser = PDBParser()
    ref_structure = parser.get_structure("reference", BR_reference_path)
    target_structure = parser.get_structure("target", target)

    ref_atoms = []
    target_atoms = []

    for ref_res, tgt_res in zip(ref_structure.get_residues(), target_structure.get_residues()):
        if is_aa(ref_res) and is_aa(tgt_res):
            ref_atoms.append(ref_res["CA"])
            target_atoms.append(tgt_res["CA"])

    super_imposer = Superimposer()
    super_imposer.set_atoms(ref_atoms, target_atoms)
    super_imposer.apply(target_structure.get_atoms())

    io = PDBIO()
    io.set_structure(target_structure)
    io.save(temp_alignment_path)


def get_sequence(target_structure_name):
    target_sequence = None

    fasta_file = msa_file

    target_header = target_structure_name.split("/")[-1].split(".")[0]

    for record in SeqIO.parse(fasta_file, "fasta"):
        if record.id == target_header:
            target_sequence = str(record.seq)

    return target_sequence


def get_lysine(target_sequence):
    pdb_index = 0  # Location of lysine in the PDB
    for i in range(0, msa_index + 1):
        if target_sequence[i] != '-':
            pdb_index += 1

    parser = PDBParser()
    structure = parser.get_structure("rhodopsin", temp_alignment_path)

    lys_residue_id = pdb_index

    lysine_residue = None
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[1] == lys_residue_id:
                    lysine_residue = residue

    return lysine_residue


def get_pocket(target_sequence):
    residues = []
    for i in range(len(POSITIONS)):
        index = 0
        for i in range(0, POSITIONS[i] + 1):
            if target_sequence[i] != '-':
                index += 1
        residues.append(index)

    parser = PDBParser()
    structure = parser.get_structure("rhodopsin", temp_alignment_path)

    final = []
    for target_residue in residues:
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[1] == target_residue:
                        final.append((chain.id, target_residue))

    pdb_residues = final

    target_residues = pdb_residues

    coordinates = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if (chain.id, residue.id[1]) in target_residues:
                    for atom in residue:
                        coordinates.append(atom.coord)

    return coordinates


def get_retinal_axis_pca(atom_coords):
    centered_coords = atom_coords - np.mean(atom_coords, axis=0)

    pca = PCA(n_components=3)
    pca.fit(centered_coords)

    axis_vector = pca.components_[0]

    axis_vector = axis_vector / np.linalg.norm(axis_vector)

    return axis_vector


def objective(P, points, reference):
    distances = [np.linalg.norm(P - p) for p in points]
    closest_distance = min(distances)
    angle = np.dot(retinal_axis, (P - anchor) / np.linalg.norm(P - anchor))
    
    return -closest_distance - 10  * angle


def constraint_distance(P, anchor, X):
    return np.linalg.norm(P - anchor) - X


def point_inside_hull(P, hull):
    delaunay = Delaunay(hull.points)
    if delaunay.find_simplex(P) >= 0:
        return 1
    else:
        return -1

def constraint_within_distance(P, reference):
    return 3.5 - np.linalg.norm(P - reference)

def optimize_point(anchor, points, X, reference):
    hull = ConvexHull(points)

    initial_guess = anchor + np.array(list(positional_vectors['C3']))

    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)

    bounds = [(min_bounds[i], max_bounds[i]) for i in range(3)]

    constraints = [
        {'type': 'eq', 'fun': lambda P: constraint_distance(P, anchor, X)},
        {'type': 'ineq', 'fun': lambda P: constraint_within_distance(P, reference)}
    ]

    result = minimize(objective, initial_guess, args=(points,reference), constraints=constraints, bounds=bounds, method='SLSQP')

    return result.x if result.success else None

def transform_vectors(optimal_point, anchor):
    transform_vector = optimal_point - anchor

    for key, value in positional_vectors.items():
        new = np.array(list(value))
        positional_vectors[key] = new

    pos_C3 = np.array(list(positional_vectors['C3'])) / np.linalg.norm(np.array(list(positional_vectors['C3'])))
    trans_vec = transform_vector / np.linalg.norm(transform_vector)
    # trans_vec = retinal_axis

    rotation_vector = np.cross(pos_C3, trans_vec)
    rotation_angle = np.arccos(np.dot(pos_C3, trans_vec))

    rotation = R.from_rotvec(rotation_angle * rotation_vector / np.linalg.norm(rotation_vector))

    transformed_vectors = {}
    for key, vector in positional_vectors.items():
        transformed_vectors[key] = rotation.apply(vector)

    return transformed_vectors

def place_retinal(lysine_residue, retinal_vectors):
    nz_atom = lysine_residue['NZ']
    nz_coord = nz_atom.get_coord()

    # Create a new residue for retinal
    retinal_residue = Residue.Residue((' ', 999, ' '), "RET", ' ')

    # Place each atom based on the vectors
    for atom_name, vector in retinal_vectors.items():
        vector_np = np.array(list(vector))
        new_atom_coord = nz_coord + vector_np
        new_atom = Atom.Atom(atom_name, new_atom_coord, 1.0, 1.0, " ", atom_name, 0, "C")
        retinal_residue.add(new_atom)

    return retinal_residue


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("bad arguments")
        sys.exit(1)

    target_structure_name = sys.argv[1]

    positional_vectors = get_positional_vectors()
    align_target(target_structure_name)
    target_sequence = get_sequence(target_structure_name)
    lysine_residue = get_lysine(target_sequence)
    pdb_points = get_pocket(target_sequence)

    if lysine_residue.get_resname() == 'LYS':
        retinal_axis = get_retinal_axis_pca(pdb_points)
        if np.dot(retinal_axis, np.array(list(positional_vectors['C3'])) / np.linalg.norm(
                np.array(list(positional_vectors['C3'])))) < 0:
            retinal_axis = -1 * retinal_axis

        anchor = lysine_residue['NZ'].get_coord()
        points = pdb_points
        X = np.linalg.norm(np.array(positional_vectors['C3']))

        reference = anchor + retinal_axis * np.linalg.norm(np.array(positional_vectors['C3']))

        optimal_point = optimize_point(anchor, points, X, reference)
        if optimal_point is None:
            print("No placement found")
            sys.exit(1)

        transformed_vectors = transform_vectors(optimal_point, anchor)

        structure = lysine_residue.get_parent().get_parent().get_parent()
        retinal_residue = place_retinal(lysine_residue, transformed_vectors)
        lysine_residue.get_parent().add(retinal_residue)

        io = PDBIO()
        io.set_structure(structure)
        io.save(output_path + target_structure_name.split('/')[-1] + "_with_retinal.pdb")
    else:
        print("Broken MSA! Taking a different approach")
        parser = PDBParser()
        structure = parser.get_structure("rhodopsin", BR_reference_path)

        lys_residue_id = 216
        retinal_residue_name = "RET"

        lysine_residue = None
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[1] == lys_residue_id:
                        lysine_residue_BR = residue

        anchor = lysine_residue_BR['NZ'].get_coord()

        parser = PDBParser()
        structure = parser.get_structure("rhodopsin", temp_alignment_path)

        all_atoms = [atom for atom in structure.get_atoms()]

        ns = NeighborSearch(all_atoms)
        close_atoms = ns.search(anchor, 2.0)

        nz_atoms_within_2_angstroms = [atom for atom in close_atoms if atom.get_name() == 'NZ']

        if len(nz_atoms_within_2_angstroms) == 0:
            print("Bad structure :(")
            sys.exit(1)

        new_anchor = nz_atoms_within_2_angstroms[0]

        new_residue = new_anchor.get_parent()

        retinal_residue = place_retinal(new_residue, positional_vectors)
        new_residue.get_parent().add(retinal_residue)

        io = PDBIO()
        io.set_structure(structure)
        io.save(output_path + target_structure_name.split('/')[-1] + "_with_retinal_NO_MSA.pdb")












