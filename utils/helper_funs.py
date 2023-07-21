import os
import yaml
import torch


def save_config_file(log_dir: str, config):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        with open(os.path.join(log_dir, 'config_finetune.yaml'), 'w') as config_file:
            yaml.dump(config, config_file)


def get_device(device_config):
    if torch.cuda.is_available() and device_config != 'cpu':
        device = device_config
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    print("Running on:", device)

    return device
