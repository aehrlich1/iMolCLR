import csv
import math
import random
import signal

import networkx as nx
import numpy as np
import torch
import torch_geometric.data
from networkx.algorithms.components import node_connected_component
from rdkit import Chem
from rdkit.Chem.BRICS import BRICSDecompose, FindBRICSBonds, BreakBRICSBonds
from rdkit.Chem.rdchem import BondType
from rdkit.Chem.rdchem import Mol
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from sklearn.utils import shuffle

ATOM_MASK_CONSTANT = 119
BOND_LIST: list = [
    BondType.SINGLE,
    BondType.DOUBLE,
    BondType.TRIPLE,
    BondType.AROMATIC
]
BOND_DIR_LIST: list = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


class TimeoutError(Exception):
    pass


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def collate_fn(batch):
    gis, gjs, mols, atom_nums, frag_mols, frag_indices = zip(*batch)

    frag_mols = [j for i in frag_mols for j in i]

    # gis = Batch().from_data_list(gis)
    # gjs = Batch().from_data_list(gjs)
    gis = Batch.from_data_list(gis)
    gjs = Batch.from_data_list(gjs)

    gis.motif_batch = torch.zeros(gis.x.size(0), dtype=torch.long)
    gjs.motif_batch = torch.zeros(gjs.x.size(0), dtype=torch.long)

    curr_indicator = 1
    curr_num = 0
    for N, indices in zip(atom_nums, frag_indices):
        for idx in indices:
            curr_idx = np.array(list(idx)) + curr_num
            gis.motif_batch[curr_idx] = curr_indicator
            gjs.motif_batch[curr_idx] = curr_indicator
            curr_indicator += 1
        curr_num += N

    return gis, gjs, mols, frag_mols


# TODO: Refactor read_smiles method
def read_smiles(data_path) -> list[str]:
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            smiles = row[-1]
            smiles_data.append(smiles)
    return smiles_data


def get_fragment_indices(mol):
    bonds = mol.GetBonds()
    edges = []
    for bond in bonds:
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    molGraph = nx.Graph(edges)

    BRICS_bonds = list(FindBRICSBonds(mol))
    break_bonds = [b[0] for b in BRICS_bonds]
    break_atoms = [b[0][0] for b in BRICS_bonds] + [b[0][1] for b in BRICS_bonds]
    molGraph.remove_edges_from(break_bonds)

    indices = []
    for atom in break_atoms:
        n = node_connected_component(molGraph, atom)
        if len(n) > 3 and n not in indices:
            indices.append(n)
    indices = set(map(tuple, indices))
    return indices


def get_fragments(mol):
    try:
        with timeout(seconds=20):

            ref_indices = get_fragment_indices(mol)

            frags = list(BRICSDecompose(mol, returnMols=True))
            mol2 = BreakBRICSBonds(mol)

            extra_indices = []
            for i, atom in enumerate(mol2.GetAtoms()):
                if atom.GetAtomicNum() == 0:
                    extra_indices.append(i)
            extra_indices = set(extra_indices)

            frag_mols = []
            frag_indices = []
            for frag in frags:
                indices = mol2.GetSubstructMatches(frag)
                # if len(indices) >= 1:
                #     idx = indices[0]
                #     idx = set(idx) - extra_indices
                #     if len(idx) > 3:
                #         frag_mols.append(frag)
                #         frag_indices.append(idx)
                if len(indices) == 1:
                    idx = indices[0]
                    idx = set(idx) - extra_indices
                    if len(idx) > 3:
                        frag_mols.append(frag)
                        frag_indices.append(idx)
                else:
                    for idx in indices:
                        idx = set(idx) - extra_indices
                        if len(idx) > 3:
                            for ref_idx in ref_indices:
                                if (tuple(idx) == ref_idx) and (idx not in frag_indices):
                                    frag_mols.append(frag)
                                    frag_indices.append(idx)

            return frag_mols, frag_indices

    except:
        print('timeout!')
        return [], [set()]


# TODO: there must be a package that does this for you
# e.g https://anaconda.org/conda-forge/openbabel
def get_graph(mol: Mol) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return a tuple consisting of an edge_index and edge attributes.
    The edge_index or edge_set has the dimensions (2, 2E) where E is the number of edges/bonds.
    Since we are only dealing with undirected graphs row 1 is the same as row 2 expect that each
    pair is swapped.

    [[row 1,
     row 2]]

    Example:
    Given the molecule propane with the following graph:

    C(0)-C(1)-C(2)

    it's edge_set is given by:

    [[0, 1, 1, 2],
     [1, 0, 2, 1]]


    edge_attributes: captures for each bond the bond_type (0 = single, 1 = double, etc.)
    and the bond_direction (). Once again, since we have a undirected graph, every second entry is
    copied and flipped.

    The dimension of edge_attr is: (2E, 2).


    :param mol:
    :return:
    """
    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BOND_DIR_LIST.index(bond.GetBondDir())
        ])
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BOND_DIR_LIST.index(bond.GetBondDir())
        ])

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

    return edge_index, edge_attr


def _mask_subgraph_idx(mol: Mol) -> tuple[list[int], list[int]]:
    masked_nodes = _mask_nodes_idx(mol)
    masked_edges = _mask_edges_idx(mol)
    return masked_nodes, masked_edges


def _mask_nodes_idx(mol: Mol) -> list[int]:
    num_atoms: int = mol.GetNumAtoms()
    num_mask_nodes: int = max([1, math.floor(0.25 * num_atoms)])
    masked_nodes = random.sample(list(range(num_atoms)), num_mask_nodes)
    return masked_nodes


def _mask_edges_idx(mol: Mol) -> list[int]:
    num_bonds: int = mol.GetNumBonds()
    num_mask_edges: int = max([0, math.floor(0.25 * num_bonds)])
    masked_edges_single = random.sample(list(range(num_bonds)), num_mask_edges)
    masked_edges: list[int] = [2 * i for i in masked_edges_single] + [2 * i + 1 for i in masked_edges_single]
    return masked_edges


def create_molecule(mol: Mol) -> tuple[torch.Tensor, int, int]:
    """
    Create a molecule from a Mol

    :param mol:
    :return:
    """
    atom_types: list[int] = []
    chirality_idx: list[int] = []

    for atom in mol.GetAtoms():
        atom_types.append(atom.GetAtomicNum())
        chirality_idx.append(int(atom.GetChiralTag()))

    x_atoms = torch.tensor(atom_types, dtype=torch.long).view(-1, 1)
    x_bonds = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    molecule = torch.cat([x_atoms, x_bonds], dim=-1)

    return molecule, mol.GetNumAtoms(), mol.GetNumBonds()


def _augment_molecule(mol: Mol) -> torch_geometric.data.Data:
    augmented_molecule, num_atoms, num_bonds = create_molecule(mol)
    masked_nodes, masked_edges = _mask_subgraph_idx(mol)
    augmented_molecule[masked_nodes, 0] = torch.tensor([ATOM_MASK_CONSTANT])

    num_masked_edges: int = max([0, math.floor(0.25 * num_bonds)])
    edge_index, edge_attr = get_graph(mol)
    augmented_edge_index = torch.zeros((2, 2 * (num_bonds - num_masked_edges)), dtype=torch.long)
    edge_attr_i = torch.zeros((2 * (num_bonds - num_masked_edges), 2), dtype=torch.long)
    count = 0

    for bond_idx in range(2 * num_bonds):
        if bond_idx not in masked_edges:
            augmented_edge_index[:, count] = edge_index[:, bond_idx]
            edge_attr_i[count, :] = edge_attr[bond_idx, :]
            count += 1

    pyg_graph = Data(x=augmented_molecule, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_graph


class MoleculeDataset(Dataset):
    def __init__(self, smiles_data):
        self.smiles_data = smiles_data

    def __getitem__(self, idx):
        mol: Mol = Chem.MolFromSmiles(self.smiles_data[idx])
        data_i = _augment_molecule(mol)
        data_j = _augment_molecule(mol)
        num_atoms = mol.GetNumAtoms()
        frag_mols = get_fragments(mol)
        frag_indices = get_fragment_indices(mol)

        return data_i, data_j, mol, num_atoms, frag_mols, frag_indices

    def __len__(self) -> int:
        return len(self.smiles_data)


class MoleculeDatasetWrapper:
    def __init__(self, batch_size: int, num_workers: int, valid_size: float, data_path: str):
        self.data_path: str = data_path
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.valid_size: float = valid_size

    def get_data_loaders(self) -> tuple[DataLoader, DataLoader]:
        smiles_data: list[str] = read_smiles(self.data_path)
        smiles_data = shuffle(smiles_data, random_state=0)

        num_train = len(smiles_data)
        split: int = int(np.floor(self.valid_size * num_train))

        valid_smiles: list[str] = smiles_data[:split]
        train_smiles: list[str] = smiles_data[split:]
        del smiles_data

        print(f"Training set size: {len(train_smiles)}")
        print(f"Validation set size: {len(valid_smiles)}")

        train_dataset: MoleculeDataset = MoleculeDataset(train_smiles)
        valid_dataset: MoleculeDataset = MoleculeDataset(valid_smiles)

        train_loader: DataLoader = DataLoader(
            train_dataset, batch_size=self.batch_size, collate_fn=collate_fn,
            num_workers=self.num_workers, drop_last=True, shuffle=True
        )
        valid_loader: DataLoader = DataLoader(
            valid_dataset, batch_size=self.batch_size, collate_fn=collate_fn,
            num_workers=self.num_workers, drop_last=True
        )

        return train_loader, valid_loader
