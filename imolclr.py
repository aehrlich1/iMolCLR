import os
import shutil
from datetime import datetime

import sys
import pprint
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from data_aug.dataset import MoleculeDatasetWrapper
from models.ginet import GINet
from utils.nt_xent import NTXentLoss
from utils.weighted_nt_xent import WeightedNTXentLoss


class iMolCLR:
    def __init__(self, dataset: MoleculeDatasetWrapper, config):
        dir_name = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('./runs', dir_name)

        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter(log_dir=log_dir)

        self.dataset: MoleculeDatasetWrapper = dataset
        self.nt_xent_criterion = NTXentLoss(self.device, **config['loss'])
        self.weighted_nt_xent_criterion = WeightedNTXentLoss(self.device, **config['loss'])

        self.n_iter: int = 0
        self.valid_n_iter: int = 0
        self.best_valid_loss: float = np.inf
        self.epochs: int = config['epochs']
        self.model_checkpoints_folder: str = os.path.join(self.writer.log_dir, 'checkpoints')
        self._save_config_file(self.model_checkpoints_folder)

    def start(self):
        train_loader, test_loader = self.dataset.get_data_loaders()

        model = GINet(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)
        print(f"\nModel \n-------------------------------\n{model}\n")

        optimizer = torch.optim.Adam(
            model.parameters(), self.config['optim']['init_lr'],
            weight_decay=self.config['optim']['weight_decay']
        )
        print(f"Optimizer \n-------------------------------\n{optimizer}\n")

        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs - 9, eta_min=0, last_epoch=-1)

        torch.cuda.empty_cache()

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}\n-------------------------------")
            self._train_loop(train_loader, optimizer, model, scheduler)
            self._test_model(test_loader, model, self.model_checkpoints_folder, epoch)
            
            if (epoch + 1) % 5 == 0:
                self._save_model(model, epoch)

            # warmup for the first 10 epochs
            if epoch >= self.config['warmup'] - 1:
                scheduler.step()

    def _train_loop(self, train_loader, optimizer, model, scheduler):
        size: int = len(train_loader.dataset)
        n_batches: int = int(size/self.config["batch_size"]) - 1

        model.train()
        for batch, (g1, g2, mols, _) in enumerate(train_loader):
            g1 = g1.to(self.device)
            g2 = g2.to(self.device)

            # get the representations and the projections
            __, z1_global, z1_sub = model(g1)  # [N,C]
            __, z2_global, z2_sub = model(g2)  # [N,C]

            # normalize projection feature vectors
            z1_global = F.normalize(z1_global, dim=1)
            z2_global = F.normalize(z2_global, dim=1)
            loss_global = self.weighted_nt_xent_criterion(z1_global, z2_global, mols)

            # normalize projection feature vectors
            z1_sub = F.normalize(z1_sub, dim=1)
            z2_sub = F.normalize(z2_sub, dim=1)
            loss_sub = self.nt_xent_criterion(z1_sub, z2_sub)

            loss = loss_global + self.config['loss']['lambda_2'] * loss_sub

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            self.n_iter += 1

            if batch % 5 == 0:
                print(f"Batch: [{batch}/{n_batches}]")
                loss, current = loss.item(), (batch + 1) * len(g1)
                self._log_loss(scheduler, loss_global, loss_sub, loss, current, size)

    def _test_model(self, train_loader, model, model_checkpoints_folder, epoch):
        valid_loss_global, valid_loss_sub = self._test(model, train_loader)
        valid_loss = valid_loss_global + 0.5 * valid_loss_sub
        print("Test Loss: \n")
        print(f" Loss: {valid_loss}\n")
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

        self.writer.add_scalar('valid_loss_global', valid_loss_global, global_step=self.valid_n_iter)
        self.writer.add_scalar('valid_loss_sub', valid_loss_sub, global_step=self.valid_n_iter)
        self.writer.add_scalar('valid_loss', valid_loss, global_step=self.valid_n_iter)
        self.valid_n_iter += 1

    def _test(self, model, valid_loader):
        model.eval()
        with torch.no_grad():
            valid_loss_global, valid_loss_sub = 0.0, 0.0
            counter = 0
            for g1, g2, mols, frag_mols in valid_loader:
                g1 = g1.to(self.device)
                g2 = g2.to(self.device)

                # get the representations and the projections
                __, z1_global, z1_sub = model(g1)  # [N,C]
                __, z2_global, z2_sub = model(g2)  # [N,C]

                # normalize projection feature vectors
                z1_global = F.normalize(z1_global, dim=1)
                z2_global = F.normalize(z2_global, dim=1)
                loss_global = self.weighted_nt_xent_criterion(
                    z1_global, z2_global, mols)

                # normalize projection feature vectors
                z1_sub = F.normalize(z1_sub, dim=1)
                z2_sub = F.normalize(z2_sub, dim=1)
                loss_sub = self.nt_xent_criterion(z1_sub, z2_sub)

                valid_loss_global += loss_global.item()
                valid_loss_sub += loss_sub.item()

                counter += 1

            valid_loss_global /= counter
            valid_loss_sub /= counter

        model.train()
        return valid_loss_global, valid_loss_sub

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _save_config_file(self, model_checkpoints_folder: str):
        if not os.path.exists(model_checkpoints_folder):
            os.makedirs(model_checkpoints_folder)
            shutil.copy('./config/config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

    def _save_model(self, model, epoch):
        torch.save(model.state_dict(), os.path.join(self.model_checkpoints_folder, f'Model_{epoch}.Pth'))

    def _log_loss(self, scheduler, loss_global, loss_sub, loss, current, size):
        self.writer.add_scalar('loss_global', loss_global, global_step=self.n_iter)
        self.writer.add_scalar('loss_sub', loss_sub, global_step=self.n_iter)
        self.writer.add_scalar('loss', loss, global_step=self.n_iter)
        self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=self.n_iter)
        print(f" loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join(self.config['resume_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model


def main(data_dir: str):
    with open("./config/config.yaml", "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    print("\nConfig \n-------------------------------\n")
    pprint.pprint(config)
    print("\n")
    dataset = MoleculeDatasetWrapper(config['batch_size'], **config['dataset'], data_dir=data_dir)

    molclr = iMolCLR(dataset, config)
    molclr.start()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print('Argument List:', str(sys.argv))
        DATA_DIR = str(sys.argv[1])
    else:
        DATA_DIR = './data/local/'
    main(DATA_DIR)
