import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def create_trace(ts, tssig, omega, sigma):
    """Generate a trace using the sine and exponential functions."""
    return np.sin(2. * np.pi * (ts - tssig) * omega) * np.exp(-(ts - tssig)**2 / sigma)


def save_trace_to_csv(ts, trace, idat, directory):
    """Save the generated trace data to a CSV file."""
    df = pd.DataFrame({'Time': ts, 'Trace': trace})
    filename = os.path.join(directory, f'trace_data_{idat}.csv')
    df.to_csv(filename, index=False)
    print(f'Data saved to {filename}')


def generate_dataset(ts, ndat, directory, save_csv):
    """Generate a dataset of traces along with their parameters."""
    dataset = []
    parameters = []
    for idat in range(ndat):
        tssig = random.uniform(512 - 100, 512 + 100)
        omega = random.uniform(0.01, 0.1)
        sigma = random.uniform(200, 800)
        trace = create_trace(ts, tssig, omega, sigma)
        dataset.append([trace])
        parameters.append((tssig, omega, sigma))
        if save_csv:
            save_trace_to_csv(ts, trace, idat, directory)
    return torch.tensor(np.array(dataset), dtype=torch.float32), parameters


def visualize_dataset(ts, dataset_tensor, parameters, ndat):
    """Visualize the generated dataset using matplotlib."""
    for i in range(ndat):
        fig, axs = plt.subplots(figsize=(16, 4))
        trace = np.array(dataset_tensor)[i, 0, :]
        label = f'tssig={parameters[i][0]:.2f}, omega={parameters[i][1]:.5f}, sigma={parameters[i][2]:.2f}'
        axs.plot(ts, trace, color='r', label=label)
        axs.legend(loc='upper right')
        axs.set_title('Trace')
        axs.set(xlabel='time', ylabel='Amplitude')
        plt.show()


def create_dataloaders(train_tensor, valid_tensor, test_tensor, batch_size):
    """Create data loaders for training, validation, and testing."""
    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_tensor, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_tensor, batch_size=1, shuffle=True)
    return train_loader, valid_loader, test_loader


def prepare_denoise_data(ndat_train=1024, ndat_valid=128, ndat_test=5, save_csv=False,
                         directory=os.getcwd(), batch_size=1, plot=False, split=True, test_samples=2):
    """Prepare the data for denoising."""
    nts = 1024
    ts = np.arange(0, nts, 1)
    total_train = ndat_train + test_samples
    train_ds_tensor, train_params = generate_dataset(ts, ndat_train, directory, save_csv)
    valid_ds_tensor, valid_params = generate_dataset(ts, ndat_valid, directory, save_csv)

     # Separate test samples from training data
    if split == True:
        test_ds_tensor = train_ds_tensor[-test_samples:]
        train_ds_tensor = train_ds_tensor[:-test_samples]

    if split == False:
        test_ds_tensor, test_params = generate_dataset(ts, ndat_test, directory, save_csv)
        
    if plot:
        visualize_dataset(ts, test_ds_tensor, test_params, ndat_test)
    return create_dataloaders(train_ds_tensor, valid_ds_tensor, test_ds_tensor, batch_size)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class NoisyDataset(Dataset):
    """A dataset that adds noise to the original data based on SNR."""
    
    def __init__(self, original_dataset, snr_value):
        self.original_dataset = original_dataset
        self.snr_value = abs(snr_value)  # Ensure the SNR value is positive
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        pure_data = self.original_dataset[idx]
        if not isinstance(pure_data, torch.Tensor):
            pure_data = torch.tensor(pure_data, dtype=torch.float32)
        
        signal_abs = torch.abs(pure_data)
        S = torch.max(signal_abs)
        
        # Ensure N is never negative
        N = S / self.snr_value  # Prevent division by zero or negative values
        
        noise = torch.normal(0, N.item(), pure_data.size())
        noisy_data = pure_data + noise
        # noisy_data = torch.clamp(noisy_data, -1., 1.)
        
        return pure_data, noisy_data, self.snr_value


#modeltrain1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device set to : {device}')

snr_values = [0.5,0.6,0.7,0.8,0.9,1,2,3]
num_epochs = 200
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader, valid_loader, test_loader = prepare_denoise_data(ndat_train=5000, ndat_valid=1, ndat_test=1, batch_size=10, split=True, test_samples=2)

# Generate dataset
model.train()
for snr_value in snr_values:
    clean_train_loader = DataLoader(train_loader.dataset, batch_size=10, shuffle=True)
    noisy_train_loader = DataLoader(NoisyDataset(train_loader.dataset, snr_value=snr_value), batch_size=10, shuffle=True)

    average_train_loss_per_epoch = []
    for epoch in range(num_epochs):
        total_loss = 0
        for clean_data, noisy_data, _ in noisy_train_loader:
            clean_data, noisy_data = clean_data.to(device), noisy_data.to(device)
            optimizer.zero_grad()
            outputs = model(noisy_data)
            loss = criterion(outputs, clean_data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        average_train_loss_per_epoch.append(train_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss}, for snr_value = {snr_value}')

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), average_train_loss_per_epoch, label=f'Training Loss, SNR = {snr_value}', color='blue')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training Loss Over Epochs', fontsize=16)
    plt.legend()
    plt.show()
    
print('Training for all snr_values is complete') 

snr_values = [0.5,0.6,0.7,0.8,0.9,1,2]
model.eval()
with torch.no_grad():
    for snr_value in snr_values:
        noisy_test_loader = DataLoader(NoisyDataset(test_loader.dataset, snr_value=snr_value), batch_size=1, shuffle=False)
        for clean_data, noisy_data, _ in noisy_test_loader:
            clean_data, noisy_data, = clean_data.to(device), noisy_data.to(device)
            denoised_output = model(noisy_data)

            plt.figure(figsize=(18, 6))
            
            # Subplot for Pure and Denoised data
            plt.subplot(2, 1, 1)
            plt.plot(clean_data.cpu().squeeze(), label='Pure', color='blue')
            plt.plot(denoised_output.cpu().squeeze(), label=f'Denoised SNR = {snr_value}', linestyle='--', color='orange')
            plt.title(f'Comparison of Pure and Denoised Data at SNR = {snr_value}')
            plt.legend()
            
            # Subplot for Noisy data
            plt.subplot(2, 1, 2)
            plt.plot(noisy_data.cpu().squeeze(), label=f'Noisy SNR = {snr_value}', color='red')
            plt.title(f'Noisy Data at SNR = {snr_value}')
            plt.legend()
            
            # plt.tight_layout()
            plt.show()


