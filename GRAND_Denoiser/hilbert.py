import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
import os

def peak_amplitude(dataloader, model, device='cpu', min_snr=1, max_snr=1e3, save_folder=''):
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    # Initialize dictionaries to store data per channel
    peak_amplitudes = {'X Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                       'Y Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                       'Z Channel': {'Clean': [], 'Noisy': [], 'Denoised': []}}
    snr_values = {'X Channel': [], 'Y Channel': [], 'Z Channel': []}
    channel_names = ['X Channel', 'Y Channel', 'Z Channel']
    
    with torch.no_grad():
        for noisy_data, clean_data in dataloader:
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            denoised_output = model(noisy_data)
            
            batch_size = noisy_data.size(0)
            for i in range(batch_size):
                for idx, channel in enumerate(channel_names):  # idx: 0,1,2
                    clean_np = clean_data[i, idx].cpu().numpy()
                    noisy_np = noisy_data[i, idx].cpu().numpy()
                    denoised_np = denoised_output[i, idx].cpu().numpy()
                    
                    if np.std(noisy_np) != 0:
                        snr = np.max(clean_np) / np.std(noisy_np)
                    else:
                        snr = float('inf')
                    
                    if  max_snr > snr > min_snr:
                        # Calculate envelopes
                        envelope_clean = np.abs(hilbert(clean_np))
                        envelope_noisy = np.abs(hilbert(noisy_np))
                        envelope_denoised = np.abs(hilbert(denoised_np))
                        
                        # Store peak amplitudes
                        peak_amplitudes[channel]['Clean'].append(np.max(envelope_clean))
                        peak_amplitudes[channel]['Noisy'].append(np.max(envelope_noisy))
                        peak_amplitudes[channel]['Denoised'].append(np.max(envelope_denoised))
                        
                        snr_values[channel].append(snr)
    
    # Plotting per channel and saving individual figures
    for idx, channel in enumerate(channel_names):
        clean = np.array(peak_amplitudes[channel]['Clean'])
        noisy = np.array(peak_amplitudes[channel]['Noisy'])
        denoised = np.array(peak_amplitudes[channel]['Denoised'])
        snr_vals = np.array(snr_values[channel])
                # Check if arrays are not empty
        if clean.size == 0 or noisy.size == 0 or denoised.size == 0:
            print(f"No data to plot for {channel}. Skipping...")
            continue
        
        # Calculating MSE between scatter points and x=y line
        errors_noisy = noisy - clean
        mse_noisy = np.mean(errors_noisy ** 2)
        
        errors_denoised = denoised - clean
        mse_denoised = np.mean(errors_denoised ** 2)
        
        plt.figure(figsize=(16, 10))
        
        # Noisy vs Clean
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(clean, noisy, c=snr_vals, cmap='viridis', alpha=0.6)
        plt.plot([clean.min(), clean.max()], [clean.min(), clean.max()], 'r--', label=f'x=y (MSE: {mse_noisy:.2f})')
        cbar = plt.colorbar(scatter, label='SNR')
        cbar.ax.tick_params(labelsize = 18)
        plt.xlabel('Peak Amplitude of Clean Data (Counts)', fontsize = 18)
        plt.ylabel('Peak Amplitude of Noisy Data (Counts)', fontsize = 18)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.title(f'Noisy vs Clean - {channel}', fontsize = 18)
        plt.legend(fontsize = 18)
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log')
        
        # Denoised vs Clean
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(clean, denoised, c=snr_vals, cmap='viridis', alpha=0.6)
        plt.plot([clean.min(), clean.max()], [clean.min(), clean.max()], 'r--', label=f'x=y (MSE: {mse_denoised:.2f})')
        cbar = plt.colorbar(scatter, label='SNR')
        cbar.ax.tick_params(labelsize = 18)
        plt.xlabel('Peak Amplitude of Clean Data (Counts)', fontsize = 18)
        plt.ylabel('Peak Amplitude of Denoised Data (Counts)', fontsize = 18)
        plt.title(f'Denoised vs Clean - {channel}', fontsize = 18)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.legend(fontsize = 18)
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log')
        
        plt.tight_layout()
        # Save the figure for the current channel
        plt.savefig(os.path.join(save_folder, f'Peak_Amplitude_{channel.replace(" ", "_")}_leftMSE={mse_noisy:.2f}_rightMSE={mse_denoised:.2f}.png'))
        plt.close()
    print('Peak amplitude graphs have been saved individually for each channel.')


def peak_time(dataloader, model, device='cpu', min_snr=1, max_snr=1e3, save_folder=''):
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    # Initialize dictionaries to store data per channel
    peak_times = {'X Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                  'Y Channel': {'Clean': [], 'Noisy': [], 'Denoised': []},
                  'Z Channel': {'Clean': [], 'Noisy': [], 'Denoised': []}}
    snr_values = {'X Channel': [], 'Y Channel': [], 'Z Channel': []}
    channel_names = ['X Channel', 'Y Channel', 'Z Channel']
    
    with torch.no_grad():
        for noisy_data, clean_data in dataloader:
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            denoised_output = model(noisy_data)
            
            batch_size = noisy_data.size(0)
            for i in range(batch_size):
                for idx, channel in enumerate(channel_names):
                    clean_np = clean_data[i, idx].cpu().numpy()
                    noisy_np = noisy_data[i, idx].cpu().numpy()
                    denoised_np = denoised_output[i, idx].cpu().numpy()
                    
                    timing = np.arange(clean_np.size)
                    
                    if np.std(noisy_np) != 0:
                        snr = np.max(clean_np) / np.std(noisy_np)
                    else:
                        snr = float('inf')
                    
                    if max_snr > snr > min_snr:
                        # Calculate envelopes
                        envelope_clean = np.abs(hilbert(clean_np))
                        envelope_noisy = np.abs(hilbert(noisy_np))
                        envelope_denoised = np.abs(hilbert(denoised_np))
                        
                        # Find peak times
                        peak_time_clean = timing[np.argmax(envelope_clean)]
                        peak_time_noisy = timing[np.argmax(envelope_noisy)]
                        peak_time_denoised = timing[np.argmax(envelope_denoised)]
                        
                        # Store peak times
                        peak_times[channel]['Clean'].append(peak_time_clean)
                        peak_times[channel]['Noisy'].append(peak_time_noisy)
                        peak_times[channel]['Denoised'].append(peak_time_denoised)
                        
                        snr_values[channel].append(snr)
    
    # Plotting per channel and saving individual figures
    for idx, channel in enumerate(channel_names):
        clean = np.array(peak_times[channel]['Clean'])
        noisy = np.array(peak_times[channel]['Noisy'])
        denoised = np.array(peak_times[channel]['Denoised'])
        snr_vals = np.array(snr_values[channel])
        
        # Calculating MSE between scatter points and x=y line
        errors_noisy = noisy - clean
        mse_noisy = np.mean(errors_noisy ** 2)
        
        errors_denoised = denoised - clean
        mse_denoised = np.mean(errors_denoised ** 2)
        
        plt.figure(figsize=(16,10))
        
        # Noisy vs Clean
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(clean, noisy, c=snr_vals, cmap='viridis', alpha=0.6)
        plt.plot([clean.min(), clean.max()], [clean.min(), clean.max()], 'r--', label=f'x=y (MSE: {mse_noisy:.2f})')
        cbar = plt.colorbar(scatter, label='SNR')
        cbar.ax.tick_params(labelsize = 18)
        plt.xlabel('Peak Time Bins of Clean Data', fontsize = 18 )
        plt.ylabel('Peak Time Bins of Noisy Data', fontsize = 18 )
        plt.title(f'Noisy vs Clean - {channel}', fontsize = 18)
        plt.legend(fontsize = 18)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.grid(True)
        y_limits = [min(noisy.min(), clean.min()), max(noisy.max(), clean.max())]
        plt.ylim(y_limits)

        # Denoised vs Clean
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(clean, denoised, c=snr_vals, cmap='viridis', alpha=0.6)
        plt.plot([clean.min(), clean.max()], [clean.min(), clean.max()], 'r--', label=f'x=y (MSE: {mse_denoised:.2f})')
        cbar = plt.colorbar(scatter, label='SNR')
        cbar.ax.tick_params(labelsize = 18)
        plt.xlabel('Peak Time of Clean Data (ns)', fontsize = 18 )
        plt.ylabel('Peak Time of Denoised Data (ns)', fontsize = 18 )
        plt.title(f'Denoised vs Clean - {channel}', fontsize =18)
        plt.legend(fontsize = 18)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.grid(True) 
        plt.ylim(y_limits)
        plt.tight_layout()
        # Save the figure for the current channel
        plt.savefig(os.path.join(save_folder, f'Peak_Time_{channel.replace(" ", "_")}_leftMSE={mse_noisy:.2f}_rightMSE={mse_denoised:.2f}.png'))
        plt.close()
    print('Peak time graphs have been saved individually for each channel.')
