import unittest
from dataset import *
import torch
from rdkit import Chem
import os


class TestDatasetMethods(unittest.TestCase):
    toluene_smiles: str = 'Cc1ccccc1'
    ethane_smiles: str = 'CC'
    aspirin_smiles: str = 'O=C(C)Oc1ccccc1C(=O)O'
    acetone_smiles: str = 'CC(=O)C'
    L_alanine_smiles: str = 'N[C@@H](C)C(=O)O'

    def test_read_smiles(self):
        print('CURRENT WORKING DIRECTORY')
        print(os.getcwd())
        smiles_data = read_smiles('./data/test_pubchem-100.txt')
        self.assertEqual('O=C(C)Oc1ccccc1C(=O)O', smiles_data[0])
        self.assertEqual('Cc1coc(SCc2csc(C(=O)NN)n2)n1', smiles_data[-1])

    def test_get_data_loaders(self):
        dataset = MoleculeDatasetWrapper(
            batch_size=5, num_workers=10, valid_size=0.2, data_dir='./data/', data_file='test_pubchem-100.txt')
        train_loader, valid_loader = dataset.get_data_loaders()
        self.assertEqual(16, len(train_loader))
        self.assertEqual(4, len(valid_loader))

    def test_molecule_dataset_getitem(self):
        smiles_data = read_smiles('./data/test_pubchem-100.txt')
        molecule_dataset = MoleculeDataset(smiles_data)
        n: int = molecule_dataset[0][3]
        self.assertEqual(13, n)

    def test_molecule_dataset_getitem_2(self):
        smiles_data: list = [self.L_alanine_smiles]
        molecule_dataset = MoleculeDataset(smiles_data)
        n: int = molecule_dataset[0]
        # self.assertEqual(13, n)

    def test_get_graph(self):
        l_alanine_mol = Chem.MolFromSmiles(self.L_alanine_smiles)
        edge_set, edge_attr = get_graph(l_alanine_mol)
        self.assertTrue(len(edge_set), 6*2)
        self.assertTrue(torch.all(edge_attr[:, 1] == 0))

    def test_get_graph_2(self):
        mol = Chem.MolFromSmiles('CC(=O)C')
        edge_set, edge_attr = get_graph(mol)

    def test_create_molecule(self):
        l_alanine_mol = Chem.MolFromSmiles(self.L_alanine_smiles)
        molecule = get_node_feature_matrix(l_alanine_mol)
        atom_list = molecule.T[0].tolist()
        chirality_list = molecule.T[1].tolist()
        self.assertEqual(atom_list, [7, 6, 6, 6, 8, 8])
        self.assertEqual(chirality_list, [0, 1, 0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
