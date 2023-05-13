import unittest
from dataset import read_smiles
from dataset import MoleculeDatasetWrapper
from dataset import MoleculeDataset


class TestDatasetMethods(unittest.TestCase):

    def test_read_smiles(self):
        smiles_data = read_smiles('../data/test_pubchem-100-clean.txt')
        print(smiles_data)
        self.assertEqual('O=C(C)Oc1ccccc1C(=O)O', smiles_data[0])
        self.assertEqual('Cc1coc(SCc2csc(C(=O)NN)n2)n1', smiles_data[-1])

    def test_get_data_loaders(self):
        dataset = MoleculeDatasetWrapper(batch_size=5, num_workers=12, valid_size=0.2, data_path='../data/test_pubchem-100-clean.txt')
        train_loader, valid_loader = dataset.get_data_loaders()
        self.assertEqual(16, len(train_loader))
        self.assertEqual(4, len(valid_loader))

    def test_molecule_dataset_getitem(self):
        smiles_data = read_smiles('../data/test_pubchem-100-clean.txt')
        molecule_dataset = MoleculeDataset(smiles_data)
        n: int = molecule_dataset[0][3]
        self.assertEqual(13, n)


if __name__ == '__main__':
    unittest.main()
