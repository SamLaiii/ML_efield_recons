import tempfile
from pathlib import Path

import argparse
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

from training_function import CustomDataset,  psnr_loss,  multi_domain_loss
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import os

from hyperCNNmodel import DualBranchAutoencoder as CNN
# from hyperLNN import LNN_DualBranchAE as LNN
import numpy as np

def get_data_loaders(config):
    """Load data and create data loaders.
    save_folder_name: str,where the data is saved
    """
    current_directory = os.getcwd()

    save_folder = config["save_folder_name"]

    if not os.path.exists(save_folder):
        os.makedirs(current_directory + '/' + save_folder) 

    train_save_path_noised = os.path.join(save_folder, 'dc2_train_noised_signals.npz')
    train_save_path_clean = os.path.join(save_folder, 'dc2_train_clean_signals.npz')

    valid_save_path_noised = os.path.join(save_folder, 'dc2_validation_noised_signals.npz')
    valid_save_path_clean = os.path.join(save_folder, 'dc2_validation_clean_signals.npz')

    test_save_path_noised = os.path.join(save_folder, 'dc2_test_noised_signals.npz')
    test_save_path_clean = os.path.join(save_folder, 'dc2_test_clean_signals.npz')

    os.makedirs(save_folder, exist_ok=True)

    try:
        train_noised_signals = np.load(train_save_path_noised)['signals']
        train_clean_signals = np.load(train_save_path_clean)['signals']
        valid_noised_signals = np.load(valid_save_path_noised)['signals']
        valid_clean_signals = np.load(valid_save_path_clean)['signals']   
        test_noised_signals = np.load(test_save_path_noised)['signals']
        test_clean_signals = np.load(test_save_path_clean)['signals']     
        print(f'Successfully loaded signals')

        print(f'shape of train noised signals = {np.shape(train_noised_signals)}')
        print(f'Shape of train clean signals = {np.shape(train_clean_signals)}')

        print(f'shape of validation noised signals = {np.shape(valid_noised_signals)}')
        print(f'Shape of validation clean signals = {np.shape(valid_clean_signals)}')

        print(f'shape of test noised signals = {np.shape(test_noised_signals)}')
        print(f'Shape of test clean signals = {np.shape(test_clean_signals)}')

    except FileNotFoundError:
        print(f'Error: Signal files not found in {save_folder}')
        raise

    train_total_samples = np.shape(train_noised_signals)[1]
    train_indices = np.arange(train_total_samples)
    print(f'number of train sample = {train_total_samples}')

    valid_total_samples = np.shape(valid_noised_signals)[1]
    valid_indices = np.arange(valid_total_samples)

    test_total_samples = np.shape(test_noised_signals)[1]
    test_indices = np.arange(test_total_samples)

    train_dataset = CustomDataset(train_noised_signals, train_clean_signals, indices=train_indices)
    valid_dataset = CustomDataset(valid_noised_signals, valid_clean_signals, indices=valid_indices)
    test_dataset = CustomDataset(test_noised_signals, test_clean_signals,  indices=test_indices)

    train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 4), shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=config.get("batch_size", 4), shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=True)
    return train_loader, valid_loader, test_loader

def get_criterion(name):
    if name == "mse":
        return nn.MSELoss()
    elif name == "psnr":
        return psnr_loss  # Make sure this is defined/imported
    elif name == "multi":
        return multi_domain_loss  # Make sure this is defined/imported
    else:
        raise ValueError(f"Unknown criterion: {name}")
    
    
def train_validate(config, checkpoint_dir=None):
    # Initialize the model with hyperparameters from config
    model = CNN(config["model_config"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    # Define the optimizer with hyperparameters from config
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config["lr"], 
        weight_decay=config["weight_decay"]
    )

    # Choose the criterion
    criterion = get_criterion(config["criterion"])

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    # Load data
    train_loader, valid_loader, _  = get_data_loaders(config)

    for epoch in range(start_epoch, config["epochs"]):
        total_train_loss = 0.0
        model.train()
        for noisy_data, clean_data in train_loader:
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            optimizer.zero_grad()
            outputs = model(noisy_data)
            loss = criterion(outputs, clean_data)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_valid_loss = 0.0
        with torch.no_grad():
            for noisy_data, clean_data in valid_loader:
                noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
                outputs = model(noisy_data)
                loss = criterion(outputs, clean_data)
                total_valid_loss += loss.item()

        avg_valid_loss = total_valid_loss / len(valid_loader)

        # Send the current validation loss back to Tune
        checkpoint_data = {
                    "epoch": epoch,
                    "net_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report({"loss":avg_valid_loss}, checkpoint = checkpoint)