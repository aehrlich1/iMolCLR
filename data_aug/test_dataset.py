import unittest
from dataset import read_smiles
from dataset import MoleculeDatasetWrapper
from dataset import MoleculeDataset
from dataset import get_graph
from dataset import create_molecule
import torch
from rdkit import Chem


class TestDatasetMethods(unittest.TestCase):

    def test_read_smiles(self):
        smiles_data = read_smiles('../data/test_pubchem-100-clean.txt')
        print(smiles_data)
        self.assertEqual('O=C(C)Oc1ccccc1C(=O)O', smiles_data[0])
        self.assertEqual('Cc1coc(SCc2csc(C(=O)NN)n2)n1', smiles_data[-1])

    def test_get_data_loaders(self):
        dataset = MoleculeDatasetWrapper(batch_size=5, num_workers=10, valid_size=0.2, data_path='../data/test_pubchem-100-clean.txt')
        train_loader, valid_loader = dataset.get_data_loaders()
        self.assertEqual(16, len(train_loader))
        self.assertEqual(4, len(valid_loader))

    def test_molecule_dataset_getitem(self):
        smiles_data = read_smiles('../data/test_pubchem-100-clean.txt')
        molecule_dataset = MoleculeDataset(smiles_data)
        n: int = molecule_dataset[0][3]
        self.assertEqual(13, n)

    def test_get_graph(self):
        smiles_data = read_smiles('../data/test_pubchem-100-clean.txt')
        smiles_aspirin = smiles_data[0]
        mol = Chem.MolFromSmiles(smiles_aspirin)
        edge_set, edge_attr = get_graph(mol)
        self.assertTrue(len(edge_set), 13*2)
        self.assertTrue(torch.all(edge_attr[:, 1] == 0))

    def test_get_graph_2(self):
        mol = Chem.MolFromSmiles('CC(=O)C')
        edge_set, edge_attr = get_graph(mol)

    def test_create_molecule(self):
        smiles_data = read_smiles('../data/test_pubchem-100-clean.txt')
        smiles_aspirin = smiles_data[0]
        mol = Chem.MolFromSmiles(smiles_aspirin)
        molecule, num_atoms, num_bonds = create_molecule(mol)


if __name__ == '__main__':
    unittest.main()
