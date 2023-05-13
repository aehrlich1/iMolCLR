import csv
import math
import random
import signal
from copy import deepcopy

import networkx as nx
import numpy as np
import torch
from networkx.algorithms.components import node_connected_component
from rdkit import Chem
from rdkit.Chem.BRICS import BRICSDecompose, FindBRICSBonds, BreakBRICSBonds
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import Mol
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from sklearn.utils import shuffle

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE,
    BT.DOUBLE,
    BT.TRIPLE,
    BT.AROMATIC
]
BONDDIR_LIST = [
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


class MoleculeDataset(Dataset):
    def __init__(self, smiles_data):
        super(Dataset).__init__()
        self.smiles_data = smiles_data

    def __getitem__(self, idx):
        mol: Mol = Chem.MolFromSmiles(self.smiles_data[idx])
        # mol = Chem.AddHs(mol)

        n: int = mol.GetNumAtoms()
        m: int = mol.GetNumBonds()

        type_idx: list[int] = []
        chirality_idx: list[int] = []

        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))  # TODO: what is the point of that?
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))  # TODO: what is the point here again? The respective class is already returned?

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        # TODO: there must be a package that does this for you
        # e.g https://anaconda.org/conda-forge/openbabel
        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        # random mask a subgraph of the molecule
        # TODO: convert atom masking and bond deletion percentage to a parameter
        num_mask_nodes: int = max([1, math.floor(0.25 * n)])
        num_mask_edges: int = max([0, math.floor(0.25 * m)])
        mask_nodes_i: list[int] = random.sample(list(range(n)), num_mask_nodes)
        mask_nodes_j: list[int] = random.sample(list(range(n)), num_mask_nodes)
        mask_edges_i_single = random.sample(list(range(m)), num_mask_edges)
        mask_edges_j_single = random.sample(list(range(m)), num_mask_edges)
        mask_edges_i = [2 * i for i in mask_edges_i_single] + [2 * i + 1 for i in mask_edges_i_single]
        mask_edges_j = [2 * i for i in mask_edges_j_single] + [2 * i + 1 for i in mask_edges_j_single]

        x_i = deepcopy(x)
        for atom_idx in mask_nodes_i:
            x_i[atom_idx] = torch.tensor([len(ATOM_LIST), 0])
        edge_index_i = torch.zeros((2, 2 * (m - num_mask_edges)), dtype=torch.long)
        edge_attr_i = torch.zeros((2 * (m - num_mask_edges), 2), dtype=torch.long)
        count = 0
        for bond_idx in range(2 * m):
            if bond_idx not in mask_edges_i:
                edge_index_i[:, count] = edge_index[:, bond_idx]
                edge_attr_i[count, :] = edge_attr[bond_idx, :]
                count += 1
        data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i)

        x_j = deepcopy(x)
        for atom_idx in mask_nodes_j:
            x_j[atom_idx, :] = torch.tensor([len(ATOM_LIST), 0])
        edge_index_j = torch.zeros((2, 2 * (m - num_mask_edges)), dtype=torch.long)
        edge_attr_j = torch.zeros((2 * (m - num_mask_edges), 2), dtype=torch.long)
        count = 0
        for bond_idx in range(2 * m):
            if bond_idx not in mask_edges_j:
                edge_index_j[:, count] = edge_index[:, bond_idx]
                edge_attr_j[count, :] = edge_attr[bond_idx, :]
                count += 1
        data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j)

        frag_mols, frag_indices = get_fragments(mol)

        return data_i, data_j, mol, n, frag_mols, frag_indices

    def __len__(self) -> int:
        return len(self.smiles_data)


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

        valid_smiles = smiles_data[:split]
        train_smiles = smiles_data[split:]
        del smiles_data

        print(f"Training set size: {len(train_smiles)}")
        print(f"Validation set size: {len(valid_smiles)}")

        train_dataset = MoleculeDataset(train_smiles)
        valid_dataset = MoleculeDataset(valid_smiles)

        train_loader: DataLoader = DataLoader(
            train_dataset, batch_size=self.batch_size, collate_fn=collate_fn,
            num_workers=self.num_workers, drop_last=True, shuffle=True
        )
        valid_loader: DataLoader = DataLoader(
            valid_dataset, batch_size=self.batch_size, collate_fn=collate_fn,
            num_workers=self.num_workers, drop_last=True
        )

        return train_loader, valid_loader
