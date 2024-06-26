"""
Functions for checking the metrics

"""
import os
import sys
import numpy as np
from scipy import hilbert
import matplotlib.pyplot as plt
from sim2root.Common.IllustrateSimPipe import *
import grand.dataio.root_trees as groot

def traces(directory, nb_event=100, min_primary_energy=1e9, min_zenith=85, max_zenith=88, plot=False, xmin=0, xmax=8192, ymin=0, ymax=8192, zmin=0, zmax=8192 ):
    """
    directory: path for the data file
    nb_event: number of events want to select
    min_primary_energy: minimum energy to filter out the data
    min_zenith, max_zenith: minimum and maximum zenith of the data in the data filter
    
    """
    time = []
    x_dataset = []
    y_dataset = []
    z_dataset = []
    
    d_input = groot.DataDirectory(directory)
    
    tvoltage_l0 = d_input.tvoltage_l0 
    tshower_l0 = d_input.tshower_l0
    trunefieldsim_l0 = d_input.trunefieldsim_l0
    tefield_l0 = d_input.tefield_l0
    trun_l0 = d_input.trun_l0
    # Get the list of events
    events_list = tvoltage_l0.get_list_of_events()
    nb_events = len(events_list)
    
    print('Number of events:', nb_events) 
    
    # If there are no events in the file, exit
    if nb_events == 0:
        sys.exit("There are no events in the file! Exiting.")
        
    event_counter = 0
    max_events_to_store = nb_event
    previous_run = None    
    
    for event_number, run_number in events_list:
        assert isinstance(event_number, int)
        assert isinstance(run_number, int)
        
        if event_counter < max_events_to_store:
            tshower_l0.get_event(event_number, run_number)
            zenith = tshower_l0.zenith
            energy_primary = tshower_l0.energy_primary
            # Filter events based on zenith angle
            if energy_primary > min_primary_energy:
                if min_zenith <= zenith <= max_zenith:
                    tvoltage_l0.get_event(event_number, run_number)
                    tefield_l0.get_event(event_number, run_number)

                    if previous_run != run_number:                          # Load only for new run.
                        trun_l0.get_run(run_number)                         # Update run info to get site latitude and longitude.       
                        trunefieldsim_l0.get_run(run_number)       
                        previous_run = run_number

                    trace_voltage = np.asarray(tvoltage_l0.trace, dtype=np.float32) #### modify here if you want ADC_l1
                    event_counter += 1

                    du_id = np.asarray(tefield_l0.du_id) # Used for printing info and saving in voltage tree.

                    # t0 calculations
                    event_second = tshower_l0.core_time_s
                    event_nano = tshower_l0.core_time_ns
                    t0_voltage_L0 = (tvoltage_l0.du_seconds-event_second)*1e9 - event_nano + tvoltage_l0.du_nanoseconds 
                    t_pre_L0 = trunefieldsim_l0.t_pre

                    trace_shape = trace_voltage.shape
                    nb_du = trace_shape[0]
                    sig_size = trace_shape[-1]

                    event_dus_indices = tefield_l0.get_dus_indices_in_run(trun_l0)
                    dt_ns_l0 = np.asarray(trun_l0.t_bin_size)[event_dus_indices]

                    for du_idx in range(nb_du):
                        trace_voltage_x = trace_voltage[du_idx, 0, x_min: x_max]
                        trace_voltage_y = trace_voltage[du_idx, 1, y_min: y_max]
                        trace_voltage_z = trace_voltage[du_idx, 2, z_min: z_max]
                        trace_voltage_time = np.arange(0, len(trace_voltage_z)) * dt_ns_l0[du_idx] - t_pre_L0
                        x_dataset.append(trace_voltage_x)
                        y_dataset.append(trace_voltage_y)
                        z_dataset.append(trace_voltage_z)

                        if plot:
                            fig, axs = plt.subplots(1, 1, figsize=(8, 6))
                            axs.plot(trace_voltage_time, trace_voltage_x, alpha=0.5, label="polarization N")
                            axs.plot(trace_voltage_time, trace_voltage_y, alpha=0.5, label="polarization E")
                            axs.plot(trace_voltage_time, trace_voltage_z, alpha=0.5, label="polarization V")
                            axs.legend()
                            axs.set_title(f"Voltage antenna {du_idx}")
                            axs.set_xlabel("Time in ns")
                            axs.set_ylabel("Voltage in uV")
                            plt.show()
                            plt.close(fig)
        else:
            break
    
    print("Processing complete for specified number of events!")
    return time, x_dataset, y_dataset, z_dataset


class CustomDataset(Dataset):
    def __init__(self, noised_signals, clean_signals, indices=None):
        """
        Args:
            noised_signals: Tuple of lists containing noised X, Y, Z signal components.
            clean_signals: Tuple of lists containing clean X, Y, Z signal components.
            indices: Array-like list of indices specifying which samples to include.
        """
        self.indices = indices if indices is not None else list(range(len(noised_signals[0])))

        # Ensure we access the signals using indices correctly
        self.noised_signals = noised_signals
        self.clean_signals = clean_signals

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Fetch the correct index for the current sample
        actual_idx = self.indices[idx]

        # Properly access the sample data
        noised_signal = np.stack([self.noised_signals[i][actual_idx] for i in range(3)], axis=0)
        clean_signal = np.stack([self.clean_signals[i][actual_idx] for i in range(3)], axis=0)

        return torch.tensor(noised_signal, dtype=torch.float32), torch.tensor(clean_signal, dtype=torch.float32)
    
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

def plot_metrics(epochs, training_losses, validation_losses, validation_psnr, learning_rates, validation_peak_to_peak, save_folder):
    """
    Plot four metrics versus epochs and save the figures to a specified folder.

    Training Loss and validation loss versus epochs
    Validation PSNR versus epochs
    Peak to Peak ratio versus epochs
    Learning rate versus epochs
    
    save_folder for saving the metrics into the folder, string: name of the file
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
    plt.savefig(os.path.join(save_folder, 'training_validation_loss.png'))
    
    # Plotting Validation PSNR
    plt.subplot(4, 1, 2)
    plt.plot(epochs, validation_psnr, label='Validation PSNR', color='green')
    plt.title('PSNR vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.savefig(os.path.join(save_folder, 'validation_psnr.png'))
    
    # Plotting Learning Rate
    plt.subplot(4, 1, 3)
    plt.plot(epochs, learning_rates, label='Learning Rate', color='cyan')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate vs Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_folder, 'learning_rate.png'))

    # Plotting Peak-to-Peak Amplitude
    plt.subplot(4, 1, 4)
    plt.plot(epochs, validation_peak_to_peak, label='Validation Peak-to-Peak', color='magenta')
    plt.xlabel('Epochs')
    plt.ylabel('Peak-to-Peak Amplitude')
    plt.title('Peak-to-Peak Amplitude vs Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_folder, 'peak_to_peak_amplitude.png'))

    plt.tight_layout()
    plt.close()