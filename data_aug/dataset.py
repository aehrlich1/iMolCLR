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

ATOM_MASK_CONSTANT = 118
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


def read_smiles(data_path) -> list[str]:
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            smiles = row[-1]
            smiles_data.append(smiles)
    return smiles_data


def get_graph(mol: Mol) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return a tuple consisting of an edge_index and edge attributes.
    The edge_index or edge_set has the dimensions (2, 2E) where E is the number of edges/bonds.
    Since we are only dealing with undirected graphs row 1 is the same as row 2 expect that each
    pair is swapped.

    [[row 1,
     row 2]]

    Example
    ----------
    Given the molecule propane with the following graph:

    C(0)-C(1)-C(2)

    it's edge_set is given by:

    [[0, 1, 1, 2],
     [1, 0, 2, 1]]


    edge_attributes: captures for each bond the bond_type (0 = single, 1 = double, etc.)
    and the bond_direction (). Once again, since we have a undirected graph, every second entry is
    copied and flipped.

    The dimension of edge_attr is: (2E, 2).
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


def get_node_feature_matrix(mol: Mol) -> torch.Tensor:
    """
    Generate the node features (atom type, chirality) for a given mol.

    Args:
        mol (Mol): rdkit.Chem.rdchem.Mol

    Returns:
        torch.Tensor: A tensor of shape: (NUM_ATOMS, 2). The first column
        contains the atom types, the second contains the atom chirality.
    """

    atom_types: list[int] = []
    chirality_idx: list[int] = []

    for atom in mol.GetAtoms():
        atom_types.append(atom.GetAtomicNum())
        chirality_idx.append(int(atom.GetChiralTag()))

    atoms = torch.tensor(atom_types, dtype=torch.long).view(-1, 1)
    bonds = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    node_features = torch.cat([atoms, bonds], dim=-1)

    return node_features


class MoleculeDataset(Dataset):
    def __init__(self, smiles_data):
        self.smiles_data = smiles_data

    def __getitem__(self, idx):
        mol: Mol = Chem.MolFromSmiles(self.smiles_data[idx])
        data_i: torch_geometric.data.Data = self._augment_mol(mol)
        data_j: torch_geometric.data.Data = self._augment_mol(mol)
        num_atoms: int = mol.GetNumAtoms()
        frag_mols = self._get_fragments(mol)
        frag_indices = self._get_fragment_indices(mol)

        return data_i, data_j, mol, num_atoms, frag_mols, frag_indices

    def __len__(self) -> int:
        return len(self.smiles_data)

    def _augment_mol(self, mol: Mol) -> torch_geometric.data.Data:
        """
        Perform a two step augmentation on the molecule:
        1. Mask randomly 25% of the nodes to an arbitrary integer
        ATOM_MASK_CONSTANT.
        2. Mask randomly 25% of the edges by removing them.

        Parameters
        ----------
        mol : Mol
            The molecule to be augmented.

        Returns
        -------
        torch_geometric.data.Data
            The augmented molecule represented as a graph.
        """

        node_feature_matrix: torch.Tensor = get_node_feature_matrix(mol)
        self._mask_nodes(node_feature_matrix)

        edge_index_masked, edge_attr_masked = get_graph(mol)
        edge_index_masked, edge_attr_masked = self._mask_edges(
            edge_index_masked, edge_attr_masked)

        pyg_graph = Data(x=node_feature_matrix,
                         edge_index=edge_index_masked, edge_attr=edge_attr_masked)
        return pyg_graph

    def _mask_nodes(self, node_feature_matrix: torch.Tensor) -> None:
        """
        Performs a random masking of 25% of the nodes in the provided feature matrix.
        The masking process replaces the atom number with a predefined constant:
        ATOM_MASK_CONSTANT.

        Parameters
        ----------
        node_feature_matrix : torch.Tensor
            Dim: (NUM_BONDS, 2) The first column contains the atoms types.
            The second columns contains information about the chirality for that atom.
        """

        num_atoms: int = node_feature_matrix.shape[0]
        num_mask_nodes: int = max([1, math.floor(0.25 * num_atoms)])
        masked_nodes = random.sample(list(range(num_atoms)), num_mask_nodes)
        node_feature_matrix[masked_nodes, 0] = torch.tensor(
            [ATOM_MASK_CONSTANT])

    def _mask_edges(self, edge_index, edge_attr) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a bond deletion on the provided edge_index (edge_set) and edge_attr.

        Args:
            edge_index (_type_): _description_
            edge_attr (_type_): _description_
        """

        num_bonds: int = int(edge_index.shape[1]/2)
        num_masked_bonds: int = max([0, math.floor(0.25 * num_bonds)])

        rng = np.random.default_rng()
        bonds_to_mask: np.array = rng.choice(
            num_bonds, size=num_masked_bonds, replace=False)
        edges_to_mask: np.array = np.concatenate(
            [2 * bonds_to_mask, 2 * bonds_to_mask + 1])
        edges_to_mask.sort()

        edge_index = np.delete(edge_index, edges_to_mask, 1)
        edge_attr = np.delete(edge_attr, edges_to_mask, 0)

        return edge_index, edge_attr

    def _get_fragments(self, mol):
        try:
            with timeout(seconds=20):

                ref_indices = self._get_fragment_indices(mol)

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

    def _get_fragment_indices(self, mol):
        bonds = mol.GetBonds()
        edges = []
        for bond in bonds:
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        molGraph = nx.Graph(edges)

        BRICS_bonds = list(FindBRICSBonds(mol))
        break_bonds = [b[0] for b in BRICS_bonds]
        break_atoms = [b[0][0] for b in BRICS_bonds] + [b[0][1]
                                                        for b in BRICS_bonds]
        molGraph.remove_edges_from(break_bonds)

        indices = []
        for atom in break_atoms:
            n = node_connected_component(molGraph, atom)
            if len(n) > 3 and n not in indices:
                indices.append(n)
        indices = set(map(tuple, indices))
        return indices


class MoleculeDatasetWrapper:
    def __init__(self, batch_size: int, num_workers: int, valid_size: float, data_file: str, data_dir: str):
        self.data_path: str = data_dir + data_file
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.valid_size: float = valid_size

    def get_data_loaders(self) -> tuple[DataLoader, DataLoader]:
        train_smiles, test_smiles = self._shuffle_smiles()

        print(f"Training set size: {len(train_smiles)}")
        print(f"Testing set size: {len(test_smiles)}")

        train_dataset: MoleculeDataset = MoleculeDataset(train_smiles)
        valid_dataset: MoleculeDataset = MoleculeDataset(test_smiles)

        train_loader: DataLoader = DataLoader(
            train_dataset, batch_size=self.batch_size, collate_fn=self._collate_fn,
            num_workers=self.num_workers, drop_last=True, shuffle=True
        )
        test_loader: DataLoader = DataLoader(
            valid_dataset, batch_size=self.batch_size, collate_fn=self._collate_fn,
            num_workers=self.num_workers, drop_last=True
        )

        return train_loader, test_loader

    def _collate_fn(self, batch):
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

    def _shuffle_smiles(self):
        smiles_data: list[str] = read_smiles(self.data_path)
        smiles_data = shuffle(smiles_data, random_state=0)

        num_train = len(smiles_data)
        split: int = int(np.floor(self.valid_size * num_train))

        train_smiles: list[str] = smiles_data[split:]
        test_smiles: list[str] = smiles_data[:split]
        del smiles_data
        return train_smiles, test_smiles
