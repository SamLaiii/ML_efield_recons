"""
Functions for checking the metrics

"""
import os
import sys# Set the PYTHONPATH
os.environ['PYTHONPATH'] = '/home/923714256/grand'
print(os.environ['PYTHONPATH'])
sys.path.append('/home/923714256/grand')


import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader 
import numpy as np
from scipy.signal import hilbert
import torch.nn.functional as F
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, noised_signals, clean_signals, indices=None):
        """
        Args:
            noised_signals: Tuple of lists containing noised X, Y, Z signal components.
            clean_signals: Tuple of lists containing clean X, Y, Z signal components.
            indices: Array-like list of indices specifying which samples to include.
        """
        self.indices = indices if indices is not None else list(range(len(noised_signals[0])))

        self.noised_signals = noised_signals
        self.clean_signals = clean_signals


    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]

        # Properly access the sample data
        noised_x = self.noised_signals[0][actual_idx]
        noised_y = self.noised_signals[1][actual_idx]
        noised_z = self.noised_signals[2][actual_idx]
        clean_x = self.clean_signals[0][actual_idx]
        clean_y = self.clean_signals[1][actual_idx]
        clean_z = self.clean_signals[2][actual_idx]

        # Convert to PyTorch tensors
        noised_signals = np.stack([noised_x, noised_y, noised_z], axis=0)
        clean_signals = np.stack([clean_x, clean_y, clean_z], axis=0)

        return torch.tensor(noised_signals, dtype=torch.float32), torch.tensor(clean_signals, dtype=torch.float32)

class CustomDataset_for_new_dc2(Dataset):
    def __init__(self, noised_signals, clean_signals, indices=None):
        """
        Args:
            noised_signals: np.ndarray of shape (3, N, 1024)
            clean_signals: np.ndarray of shape (3, N, 1024)
            indices: Array-like list of indices specifying which samples to include.
        """
        # Transpose to (N, 3, 1024)
        self.noised_signals = np.transpose(noised_signals, (1, 0, 2))
        self.clean_signals = np.transpose(clean_signals, (1, 0, 2))
        self.indices = indices if indices is not None else list(range(self.noised_signals.shape[0]))

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        noised = self.noised_signals[actual_idx]  # shape: (3, 1024)
        clean = self.clean_signals[actual_idx]    # shape: (3, 1024)
        return torch.tensor(noised, dtype=torch.float32), torch.tensor(clean, dtype=torch.float32)

def split_indices(n, train_frac=0.8, valid_frac=0.1):
    """
    Split indices into training, validation, and test sets.
    default: 80% are train data, 10% are validation data
    
    """
    indices = np.arange(n)
    np.random.shuffle(indices)

    train_size = int(n * train_frac)
    valid_size = int(n * valid_frac)

    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices = indices[train_size + valid_size:]

    return train_indices, valid_indices, test_indices

def get_peak_amplitude(signal):
    """
    Function to get peak amplitude of a signal using Hilbert transform
    
    return peak amplitude
    """
    hilbert_amp = np.abs(hilbert(signal))  # Compute Hilbert transform and get amplitude
    peak_amplitude = np.max(hilbert_amp)  # Find peak amplitude
    return peak_amplitude

def calculate_psnr_with_peak(original_signal, reconstructed_signal):
    """
    Function to calculate PSNR using peak amplitude of the original signal
    
    return psnr
    """
    peak_amplitude = get_peak_amplitude(original_signal)  # Get peak amplitude of original signal
    mse_loss = np.mean((original_signal - reconstructed_signal) ** 2)  # Calculate MSE
    if mse_loss == 0:
        return float('inf')  # Return infinity if MSE is zero to indicate perfect reconstruction
    max_i = peak_amplitude  # Use peak amplitude as MAX_I for PSNR calculation
    with np.errstate(divide='ignore'):
        psnr_value = 10 * np.log10((max_i ** 2) / mse_loss)  # Calculate PSNR
    return psnr_value

def peak_to_peak_ratio(original, reconstructed):
    """
    Peak to peak ratio metrics
    
    return ratio 
    """
    original_amp = np.abs(hilbert(original))
    reconstructed_amp = np.abs(hilbert(reconstructed))
    max_original_amp = np.max(original_amp)
    if max_original_amp == 0:
        return float('inf')  # Return infinity if max_original_amp is zero to avoid division by zero
    ratio = np.abs((np.max(original_amp) - np.max(reconstructed_amp))) / max_original_amp
    return ratio

def psnr(target, ref, scale):
    target_data = np.array(target)
    ref_data = np.array(ref)
    diff = ref_data - target_data
    rmse = np.sqrt(np.mean(diff ** 2))
    max_pixel = scale
    psnr = 10 * np.log10(max_pixel**2 / rmse)
    return psnr

def psnr_loss(input, target, device='cpu'):
    """
    Psnr loss that use in the training loop plis

    return -psnr
    """
    # Ensure input is on the correct device and compute MSE loss
    mse_loss = F.mse_loss(input.to(device), target.to(device))
    
    # Detach the tensor, move it to CPU, and convert to NumPy array for get_peak_amplitude
    input_detached = input.detach().cpu().numpy()
    
    # Calculate peak amplitude using the detached array
    peak_amplitude = get_peak_amplitude(input_detached)
    
    # No need to move peak_amplitude to a device, as it's now a scalar value and will be used as such
    psnr = 10 * torch.log10((peak_amplitude**2) / mse_loss)
    
    return - psnr

def plot_metrics(epochs, training_losses, validation_losses, validation_psnr, learning_rates, validation_peak_to_peak, save_folder):
    """
    Plot four metrics versus epochs and save the figures to a specified folder.

    Training Loss and validation loss versus epochs
    Validation PSNR versus epochs
    Peak to Peak ratio versus epochs
    Learning rate versus epochs
    
    save_folder for saving the metrics into the folder, string: name of the file
    """

    plt.figure(figsize=(16, 9))

    # Plotting Training and Validation Loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, training_losses, label='Training loss')
    plt.plot(epochs, validation_losses, label='Validation loss', color='orange')
    plt.title('Training and Validation loss vs Epochs', fontsize = 20)
    plt.xlabel('Epochs', fontsize = 20)
    plt.ylabel('Loss', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(fontsize = 20)
    
    # Plotting Validation PSNR
    plt.subplot(2, 1, 2)
    plt.plot(epochs, validation_psnr, label='Validation PSNR', color='green')
    plt.title('PSNR vs Epochs', fontsize = 20)
    plt.xlabel('Epochs', fontsize = 20)
    plt.ylabel('PSNR (dB)', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(fontsize = 20)
    
    # Plotting Learning Rate
    plt.subplot(4, 1, 3)
    plt.plot(epochs, learning_rates, label='Learning Rate', color='cyan')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate vs Epochs')
    plt.legend()

    # Plotting Peak-to-Peak Amplitude
    plt.subplot(4, 1, 4)
    plt.plot(epochs, validation_peak_to_peak, label='Validation Peak-to-Peak', color='magenta')
    plt.xlabel('Epochs')
    plt.ylabel('Peak-to-Peak Amplitude')
    plt.title('Peak-to-Peak Amplitude vs Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'metrics.png'))
    plt.close()

def calculate_snr(clean_array, noisy_array):
    """Calculate the SNR of a signal."""
    snr = np.max(clean_array) / np.std(noisy_array)
    return snr

def multi_domain_loss(clean, pred, fft_weight=0.1):
    # time-domain psnr + log-magnitude STFT psnr loss (frequency domain)
    psnr_time = psnr_loss(pred, clean, device= 'cpu')

    clean_fft = torch.fft.rfft(clean, dim=-1)
    pred_fft  = torch.fft.rfft(pred , dim=-1)
    psnr_freq = psnr_loss(torch.log1p(torch.abs(pred_fft)),
                       torch.log1p(torch.abs(clean_fft)), device= 'cpu')
    return psnr_time + fft_weight * psnr_freq