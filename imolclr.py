import os
import shutil
from datetime import datetime

import csv
import sys
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

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False


def read_smiles(data_path):
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            smiles = row[-1]
            smiles_data.append(smiles)
    return smiles_data


class iMolCLR:
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        dir_name = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('./runs', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(self.device, **config['loss'])
        self.weighted_nt_xent_criterion = WeightedNTXentLoss(
            self.device, **config['loss'])

    def train(self):

        train_loader, valid_loader = self.dataset.get_data_loaders()

        model = GINet(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)
        print(model)

        optimizer = torch.optim.Adam(
            model.parameters(), self.config['optim']['init_lr'],
            weight_decay=self.config['optim']['weight_decay']
        )
        print('Optimizer:', optimizer)

        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.config['epochs'] - 9, eta_min=0, last_epoch=-1)

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        model_checkpoints_folder = os.path.join(
            self.writer.log_dir, 'checkpoints')

        self._save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        torch.cuda.empty_cache()

        for epoch_counter in range(self.config['epochs']):
            for bn, (g1, g2, mols, frag_mols) in enumerate(train_loader):
                optimizer.zero_grad()

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

                loss = loss_global + self.config['loss']['lambda_2'] * loss_sub

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self._log_loss(scheduler, n_iter, epoch_counter,
                                   bn, loss_global, loss_sub, loss)

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                self._validate_model(valid_loader, model, model_checkpoints_folder,
                                     valid_n_iter, best_valid_loss, epoch_counter, bn)

            if (epoch_counter + 1) % 5 == 0:
                self._save_model(
                    model, model_checkpoints_folder, epoch_counter)

            # warmup for the first 10 epochs
            if epoch_counter >= self.config['warmup'] - 1:
                scheduler.step()

    def _get_device(self):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _save_config_file(model_checkpoints_folder):
        if not os.path.exists(model_checkpoints_folder):
            os.makedirs(model_checkpoints_folder)
            shutil.copy('./config/config.yaml',
                        os.path.join(model_checkpoints_folder, 'config.yaml'))

    def _save_model(self, model, model_checkpoints_folder, epoch_counter):
        torch.save(model.state_dict(),
                   os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(str(epoch_counter))))

    def _log_loss(self, scheduler, n_iter, epoch_counter, bn, loss_global, loss_sub, loss):
        self.writer.add_scalar(
            'loss_global', loss_global, global_step=n_iter)
        self.writer.add_scalar(
            'loss_sub', loss_sub, global_step=n_iter)
        self.writer.add_scalar(
            'loss', loss, global_step=n_iter)
        self.writer.add_scalar(
            'cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
        print(epoch_counter, bn, loss_global.item(),
              loss_sub.item(), loss.item())

    def _validate_model(self, valid_loader, model, model_checkpoints_folder, valid_n_iter, best_valid_loss, epoch_counter, bn):
        valid_loss_global, valid_loss_sub = self._validate(
            model, valid_loader)
        valid_loss = valid_loss_global + 0.5 * valid_loss_sub
        print(epoch_counter, bn, valid_loss_global,
              valid_loss_sub, valid_loss, '(validation)')
        if valid_loss < best_valid_loss:
            # save the model weights
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(
                model_checkpoints_folder, 'model.pth'))

        self.writer.add_scalar(
            'valid_loss_global', valid_loss_global, global_step=valid_n_iter)
        self.writer.add_scalar(
            'valid_loss_sub', valid_loss_sub, global_step=valid_n_iter)
        self.writer.add_scalar(
            'valid_loss', valid_loss, global_step=valid_n_iter)
        valid_n_iter += 1

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join(
                self.config['resume_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(
                checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss_global, valid_loss_sub = 0.0, 0.0
            counter = 0
            for bn, (g1, g2, mols, frag_mols) in enumerate(valid_loader):
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


def main():
    config = yaml.load(open("./config/config.yaml", "r"),
                       Loader=yaml.FullLoader)
    print(config)
    dataset = MoleculeDatasetWrapper(
        config['batch_size'], **config['dataset'], data_dir=DATA_DIR)

    molclr = iMolCLR(dataset, config)
    molclr.train()


if __name__ == "__main__":
    global DATA_DIR
    if (len(sys.argv) > 1):
        print('Argument List:', str(sys.argv))
        DATA_DIR = str(sys.argv[1])
    else:
        DATA_DIR = './data/'
    main()
