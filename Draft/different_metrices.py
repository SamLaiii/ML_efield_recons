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
import torch.nn.functional as F
import numpy as np
from scipy.signal import hilbert
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.optim.lr_scheduler import CyclicLR

######################################Functions for data generation ###############################
# Function to generate synthetic trace data for simulation
def create_trace(ts, tssig, omega, sigma):
    """Generate a trace using the sine and exponential functions."""
    return np.sin(2. * np.pi * (ts - tssig) * omega) * np.exp(-(ts - tssig)**2 / sigma)


# Function to save trace data to CSV file
def save_trace_to_csv(ts, trace, idat, directory):
    """Save the generated trace data to a CSV file."""
    df = pd.DataFrame({'Time': ts, 'Trace': trace})
    filename = os.path.join(directory, f'trace_data_{idat}.csv')
    df.to_csv(filename, index=False)
    print(f'Data saved to {filename}')

# Function to generate a dataset of traces
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


# Function to visualize dataset traces
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

# Function to prepare data using scikit-learn's train_test_split
def prepare_denoise_data_scikit(ndat_total, ndat_train, ndat_valid, ndat_test, save_csv=False, directory=os.getcwd(), batch_size=1, plot=False):
    """Prepare the data for denoising using scikit-learn with specific numbers for training, validation, and test sets."""
    nts = 1024
    ts = np.arange(0, nts, 1)
    
    # Ensure the total number of data points is correctly specified
    assert ndat_total == ndat_train + ndat_valid + ndat_test, "Total data size does not match the sum of train, valid, and test sizes."
    
    # Generate the full dataset
    full_ds_tensor, full_params = generate_dataset(ts, ndat_total, directory, save_csv)

    # First split: separate out the test dataset
    train_valid_ds_tensor, test_ds_tensor, train_valid_params, test_params = train_test_split(
        full_ds_tensor, full_params, test_size=ndat_test, random_state=42
    )
    
    # Calculate the proportion of the validation set relative to the sum of training and validation sets
    valid_size_proportion = ndat_valid / (ndat_train + ndat_valid)

    # Second split: separate out the validation dataset from the combined training and validation dataset
    train_ds_tensor, valid_ds_tensor, train_params, valid_params = train_test_split(
        train_valid_ds_tensor, train_valid_params, test_size=valid_size_proportion, random_state=42
    )

    if plot:
        # Plot a subset of the test dataset
        visualize_dataset(ts, test_ds_tensor, test_params, min(ndat_test, 10))

    # Create data loaders
    train_loader = DataLoader(train_ds_tensor, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds_tensor, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds_tensor, batch_size=1, shuffle=False)

    return train_loader, valid_loader, test_loader

########################################## Functions of training, metrice ###################################

def get_peak_amplitude(signal):
    """
    Function to get peak amplitude of a signal using Hilbert transform
    
    return peak amplitude
    """

    hilbert_amp = np.abs(hilbert(signal))  # Compute Hilbert transform and get amplitude
    peakamplitude = np.max(hilbert_amp)  # Find peak amplitude
    return peakamplitude

def calculate_psnr_with_peak(original_signal, reconstructed_signal):
    """
    Function to calculate PSNR using peak amplitude of the original signal
    
    return psnr
    """

    peak_amplitude = get_peak_amplitude(original_signal)  # Get peak amplitude of original signal
    mse_loss = np.mean((original_signal - reconstructed_signal) ** 2)  # Calculate MSE
    max_i = peak_amplitude  # Use peak amplitude as MAX_I for PSNR calculation
    psnr_value = 10 * np.log10((max_i ** 2) / mse_loss)  # Calculate PSNR
    return psnr_value

def peak_to_peak_ratio(original, reconstructed):
    """
    peak to peak ratio metrices
    
    return ratio 
    """

    original_amp = np.abs(hilbert(original))
    reconstructed_amp = np.abs(hilbert(reconstructed))
    ratio = np.abs((np.max(original_amp) - np.max(reconstructed_amp))) / np.max(original_amp)
    return ratio


def psnr_loss(input, target, device='cpu'):
    """
    Psnr loss that use in the training loop

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
    
    return -psnr


def train_and_validate_model(model, noisy_train_loader, noisy_valid_loader, criterion, optimizer, scheduler, num_epochs=100, device='cpu'):
    """
    Training Loop for a model
    
    return 
    
    training loss, validation loss, validation psnr, learning rate, validation peak to peak
    """

    training_losses, validation_losses, validation_psnr, learning_rates, validation_peak_to_peak = [], [], [], [], []
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for clean_data, noisy_data in noisy_train_loader:
            clean_data, noisy_data = clean_data.to(device), noisy_data.to(device)
            optimizer.zero_grad()
            outputs = model(noisy_data)
            loss = criterion(outputs, clean_data)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(noisy_train_loader)
        training_losses.append(avg_train_loss)

        model.eval()
        total_valid_loss, total_psnr, total_peak_to_peak_ratio = 0, 0, 0
        with torch.no_grad():
            for clean_data, noisy_data in noisy_valid_loader:

                clean_data, noisy_data = clean_data.to(device), noisy_data.to(device)
                outputs = model(noisy_data)
                loss = criterion(outputs, clean_data)
                total_valid_loss += loss.item()

                psnr_value = calculate_psnr_with_peak(clean_data.detach().cpu().numpy(), outputs.detach().cpu().numpy())

                total_psnr += psnr_value

                ratio = peak_to_peak_ratio(clean_data.detach().cpu().numpy(), outputs.detach().cpu().numpy())

                total_peak_to_peak_ratio += ratio

        avg_valid_loss = total_valid_loss / len(noisy_valid_loader)

        validation_losses.append(avg_valid_loss)

        avg_psnr = total_psnr / len(noisy_valid_loader)

        validation_psnr.append(avg_psnr)

        avg_peak_to_peak_ratio = total_peak_to_peak_ratio / len(noisy_valid_loader)

        validation_peak_to_peak.append(avg_peak_to_peak_ratio)

        learning_rates.append(scheduler.get_last_lr()[0])

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_valid_loss}, Validation PSNR: {avg_psnr}, Validation Peak-to-Peak: {avg_peak_to_peak_ratio}, Learning Rate: {learning_rates[-1]}')
    
    return training_losses, validation_losses, validation_psnr, learning_rates, validation_peak_to_peak


def get_reconstructed_signals(model, loader, device):
    model.eval()  # Set model to evaluation mode
    reconstructed_signals = []  # List to store reconstructed signals
    original_signals = []  # List to store original signals
    with torch.no_grad():  # Disabling gradient calculation
        for clean_data, noisy_data in loader:  # Iterate over data
            clean_data, noisy_data = clean_data.to(device), noisy_data.to(device)  # Move data to the specified device
            reconstructed_signal = model(noisy_data)  # Forward pass: compute the model output
            reconstructed_signals.append(reconstructed_signal.cpu().numpy())  # Store reconstructed signal
            original_signals.append(clean_data.cpu().numpy())  # Store original signal
    return np.array(original_signals), np.array(reconstructed_signals)  # Return arrays of original and reconstructed signals

def plot_metrics(epochs, training_losses, validation_losses, validation_psnr, learning_rates, validation_peak_to_peak):
    """
    plot four metrices versus epochs

    Training Loss and validation loss versus epochs

    Validation PSNR versus epochs

    Peak to Peak ratio versus epochs

    Learning rate versus epochs
    """
    plt.figure(figsize=(25, 16))
    # Plotting Training and Validation Loss 
    plt.subplot(4, 1, 1)
    plt.plot(epochs, training_losses, label='Training loss')
    plt.plot(epochs, validation_losses, label='Validation loss', color='orange')
    plt.title('Training and Validation loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plotting Validation PSNR
    plt.subplot(4, 1, 2)
    plt.plot(epochs, validation_psnr, label='Validation PSNR', color='green')
    plt.title('PSNR vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    
    plt.subplot(4, 1, 3)
    plt.plot(epochs, learning_rates, label='Learning Rate', color='cyan')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate vs Epochs')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(epochs, validation_peak_to_peak, label='Validation Peak-to-Peak', color='magenta')
    plt.xlabel('Epochs')
    plt.ylabel('Peak-to-Peak Amplitude')
    plt.title('Peak-to-Peak Amplitude vs Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

def visualize_denoised_signal(model, test_loader, device, snr_values, mse_values, psnr, peak_to_peak):
    """
    plot the denoised signal from the noise signals, and the expected signal.
    
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disabling gradient calculation
        for snr_value in snr_values:  # Iterate over specified SNR values
            for clean_data, noisy_data in test_loader:  # Iterate over test data
                clean_data, noisy_data = clean_data.to(device), noisy_data.to(device)  # Move data to the specified device
                denoised_output = model(noisy_data)  # Forward pass: compute the model output
                plt.figure(figsize=(25, 16))  # Set figure size
                plt.subplot(2, 1, 1)  # First subplot for clean vs denoised signal
                plt.plot(clean_data.cpu().squeeze(), label='Pure', color = 'blue')  # Plot clean signal
                plt.plot(denoised_output.cpu().squeeze(), label=f'DenoisedSNR = {snr_value}'
                         , linestyle ='--', color='orange')  # Plot denoised signal
                plt.legend()  # Show legend
                plt.title(f'Denoised vs Pure Signal at SNR = {snr_value}')  # Title
                plt.subplot(2, 1, 2)  # Second subplot for noisy signal
                plt.plot(noisy_data.cpu().squeeze(), label=f'Noisy SNR = {snr_value}\n\nMSE = {mse_values}\n\n psnr = {psnr}\n\n peak to peak = {peak_to_peak}', color='red')  # Plot noisy signal
                plt.legend()  # Show legend
                plt.title(f'Noisy Signal at SNR = {snr_value}')  # Title
                plt.show()  # Display the plot

def plot_loss_vs_psnr(mse_values, psnr_values, title='Validation loss vs PSNR', xlabel='Validation Loss', ylabel='PSNR (dB)'):
    """
    Plot Training Loss versus Peak Signal-to-Noise Ratio (PSNR).
    """
    plt.figure(figsize=(10, 6))  # Set the figure size for better readability
    plt.scatter(mse_values, psnr_values, color='blue')  # Plot MSE vs PSNR as line plot
    plt.title('validation Loss vs PSNR')  # Set the title of the plot
    plt.xlabel(f'validation loss')  # Set the x-axis label
    plt.ylabel(f'PSNR')  # Set the y-axis label
    plt.show()  # Display the plot



############################# Simple autoencoder neural network model definition #########################################
    
class ResidualBlock(nn.Module):
    """
    encoder: 2 1-d convolution layers in one block. There are 3 blocks in the encoder.

    decoder: 2 1-d convotranspose layers in one block.  There are 3 blocks in the decoder.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)

        # Adjust channels in skip connection if necessary
        self.adjust_channels = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        # Apply the skip connection
        if self.adjust_channels is not None:
            identity = self.adjust_channels(identity)

        out += identity
        out = self.relu(out)
        return out
    
class DecoderResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, output_padding):
        super(DecoderResidualBlock, self).__init__()
        self.conv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=output_padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)

        # Adjust channels in skip connection if necessary
        self.adjust_channels = nn.ConvTranspose1d(in_channels, out_channels, 1, stride=2, output_padding=output_padding) if in_channels != out_channels else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        # Apply the skip connection
        if self.adjust_channels is not None:
            identity = self.adjust_channels(identity)

        out += identity
        out = self.relu(out)
        return out
    
class Autoencoder(nn.Module):
    def __init__(self, input_size=1024, kernel_size=3):
        super(Autoencoder, self).__init__()
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        padding = kernel_size // 2

        # Encoder with Residual Blocks
        self.encoder = nn.Sequential(
            ResidualBlock(1, 4, kernel_size, stride=1, padding=padding),
            nn.MaxPool1d(2, stride=2),
            ResidualBlock(4, 8, kernel_size, stride=1, padding=padding),
            nn.MaxPool1d(2, stride=2),
            ResidualBlock(8, 16, kernel_size, stride=1, padding=padding),
            nn.MaxPool1d(2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            DecoderResidualBlock(16, 8, kernel_size, padding=kernel_size//2, output_padding=1),
            DecoderResidualBlock(8, 4, kernel_size, padding=kernel_size//2, output_padding=1),
            nn.ConvTranspose1d(4, 1, kernel_size, stride=2, padding=kernel_size//2, output_padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        if x.size(2) != 1024:
            # Adjust the size to 1024
            x = F.pad(x, (0, 1024 - x.size(2)))
        return x

# Dataset class that adds noise to the original data
class NoisyDataset(Dataset):
    """
    A dataset that adds noise to the original data based on SNR.
    return both pure dataset and noise dataset.
    """
    
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
        S = torch.max(signal_abs) # Signal strength
        N = S / self.snr_value  # Noise level
        noise = torch.normal(0, N.item(), pure_data.size()) # Generate noise
        noisy_data = pure_data + noise # Add noise to the signal
        return pure_data, noisy_data
        
        

##################### parameters for Training and evaluation loop ##########################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Determine device (GPU or CPU)
print(f'Device set to : {device}')
kernel_sizes = [3]  # List of kernel sizes to try
snr_values = [3]  # SNR values to simulate noisy conditions
num_epochs = 100

####################### Training ##################################
# for snr_value in snr_values:
    # for kernel_size in kernel_sizes:
    #     print(f'snr_value={snr_value}')
    #     model = Autoencoder(kernel_size=kernel_size).to(device)  # Initialize the model with given kernel size
    #     base_lr = 0.0005  # The lower bound of the learning rate range for the cycle.
    #     max_lr = 0.006 
    #     optimizer = optim.Adam(model.parameters(), lr=base_lr)  # Initialize the optimizer with given learning rate
    #     scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, 
    #                  step_size_up=5, step_size_down=20, 
    #                  mode='triangular', cycle_momentum=False)
        
    #     criterion = psnr_loss #### Instead of using MSE, use psnr to calculate the loss in training 
    #     ### if want to use MSE, use nn.mseloss()

    #     train_loader, valid_loader, test_loader = prepare_denoise_data_scikit(
    #             ndat_total=4100, ndat_train=2048, ndat_valid=2048, ndat_test=4, 
    #             save_csv=False, directory=os.getcwd(), batch_size=1, plot=False
    #         )
    #     test_dataset = test_loader.dataset  # Get test dataset

    #     noisy_test_dataset = NoisyDataset(test_dataset, snr_value=snr_value)  # Create noisy test dataset with given SNR value
            
    #     noisy_train_loader = DataLoader(NoisyDataset(train_loader.dataset, snr_value=snr_value), batch_size=5, shuffle=True)  # Noisy training data loader

    #     noisy_valid_loader = DataLoader(NoisyDataset(valid_loader.dataset, snr_value=snr_value), batch_size=5, shuffle=True)  # Noisy validation data loader

    #     noisy_test_loader = DataLoader(noisy_test_dataset, batch_size=1, shuffle=True)  # Noisy test data loader

    #     # Train and validate the model
    #     train_losses, valid_losses, validation_psnr, learning_rates, validation_peak_to_peak = train_and_validate_model(
    #     model, noisy_train_loader, noisy_valid_loader, criterion, optimizer, scheduler = scheduler, num_epochs= num_epochs , device=device
    #     )

    #     # Calculate the psnr with noise signals and original signal
    #     original_signals, reconstructed_signals = get_reconstructed_signals(model, noisy_test_loader, device)  # Get original and reconstructed signals

    #     psnr_values = []  # List to store PSNR values for each test sample

    #     for i in range(len(original_signals)):

    #         original_signal = original_signals[i].squeeze()  # Flatten signal
                
    #         reconstructed_signal = reconstructed_signals[i].squeeze()  # Flatten signal

    #         psnr_value = calculate_psnr_with_peak(original_signal, reconstructed_signal)  # Calculate PSNR

    #         psnr_values.append(psnr_value)  # Store PSNR value

    #     average_psnr = np.mean(psnr_values)  # Calculate average PSNR across all test samples

    #     print(f"Learning Rate: {base_lr}, SNR Value: {snr_value}, Kernel Size: {kernel_size}")

    #     print(f"Average PSNR on Test Set: {average_psnr} dB")  # Print average PSNR

    #     epochs = range(1, num_epochs + 1)

    #     plot_metrics(epochs, train_losses, valid_losses, validation_psnr, learning_rates= learning_rates, validation_peak_to_peak = validation_peak_to_peak)

    #     visualize_denoised_signal(model, noisy_test_loader, device, snr_values=[snr_value], mse_values=valid_losses[-1], psnr=validation_psnr[-1], peak_to_peak=validation_peak_to_peak[-1] )

    #     plot_loss_vs_psnr(valid_losses, validation_psnr)