# import the model
from model import Autoencoder

### import the traces functions from the training_function
from training_function import traces
from training_function import CustomDataset
from training_function import split_indices
from training_function import plot_metrics
from training_function import psnr_loss
from training_function import get_peak_amplitude
from training_function import calculate_psnr_with_peak
from training_function import psnr_loss
from training_function import calculate_psnr_with_peak

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import hilbert
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.optim.lr_scheduler import CyclicLR
import matplotlib as mpl
from torch.utils.data import Dataset, DataLoader

save_folder = "Test" #### Name of the file 

if not os.path.exists(save_folder):
    os.makedirs(save_folder) 

print("Empty Folder is created")   
mpl.rcParams['figure.max_open_warning'] = 50

##### data preparation
directory = "ZHAireS/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000/" #voltage_29-24992_L0_0000.root    

noised_time, noised_trace_x, noised_trace_y, noised_trace_z = traces(directory, nb_event=1000, min_primary_energy=1e9, min_zenith=85, max_zenith=88, plot=False, xmin=0, xmax=8192, ymin=0, ymax=8192, zmin=0, zmax=8192 )

print(f'shape of noised_time:{np.shape(noised_time)}')
print(f'shape of noised_trace_x:{np.shape(noised_trace_x)}')
print(f'shape of noised_trace_y:{np.shape(noised_trace_y)}')
print(f'shape of noised_trace_z:{np.shape(noised_trace_z)}')
        
    
    
NJ_directory = "ZHAireS-NJ/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000NJ" 

clean_time, clean_trace_x, clean_trace_y, clean_trace_z = traces(NJ_directory, nb_event=1000, min_primary_energy=1e9, min_zenith=85, max_zenith=88, plot=False, xmin=0, xmax=8192, ymin=0, ymax=8192, zmin=0, zmax=8192 )

print(f'shape of clean_time:{np.shape(clean_time)}')
print(f'shape of clean_trace_x:{np.shape(clean_trace_x)}')
print(f'shape of clean_trace_y:{np.shape(clean_trace_y)}')
print(f'shape of clean_trace_z:{np.shape(clean_trace_z)}')



noised_signals = (noised_trace_x, noised_trace_y, noised_trace_z)
clean_signals = (clean_trace_x, clean_trace_y, clean_trace_z)
total_samples = len(noised_trace_x)
train_indices, valid_indices, test_indices = split_indices(total_samples)

train_dataset = CustomDataset(noised_signals, clean_signals, indices=train_indices)
valid_dataset = CustomDataset(noised_signals, clean_signals, indices=valid_indices)
test_dataset = CustomDataset(noised_signals, clean_signals, indices=test_indices)

# Creating DataLoader instances for each dataset
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4,shuffle=False)   



##### Training Loop

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device set to : {device}')
num_epochs = 100
base_lr = 0.0001
max_lr = 0.006 
model = Autoencoder(kernel_size=3).to(device)
optimizer = optim.AdamW(model.parameters(), lr=base_lr,)
criterion = nn.MSELoss()    #### The criterion could be changed to criterion = psnr_loss 
# scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, 
#                      step_size_up=5, step_size_down=20, 
#                      mode='triangular', cycle_momentum=False)  # uncomment it if there are problems with gradient descent in learning.


training_losses, validation_losses, validation_psnr, learning_rates, validation_peak_to_peak = [], [], [], [], []
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for noisy_data, clean_data in train_loader: 
        noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)

        optimizer.zero_grad()
        outputs = model(noisy_data)
        # outputs = (outputs * clean_data_var) + clean_data_mean

        loss = criterion(outputs, clean_data)
        loss.backward()
        optimizer.step()
        # scheduler.step()  
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader.dataset)
    training_losses.append(avg_train_loss)
    
    model.eval()
    total_valid_loss, total_psnr, total_peak_to_peak_ratio = 0, 0, 0
    for noisy_data, clean_data in valid_loader:  # Assuming validation_dataset is a DataLoader
        clean_data, noisy_data = clean_data.to(device), noisy_data.to(device)

        with torch.no_grad():
            outputs = model(noisy_data)
            loss = criterion(outputs, clean_data)
            total_valid_loss += loss.item()
            psnr_value = calculate_psnr_with_peak(clean_data.cpu().numpy(), outputs.cpu().numpy())
            total_psnr += psnr_value
            ratio = peak_to_peak_ratio(clean_data.cpu().numpy(), outputs.cpu().numpy())
            total_peak_to_peak_ratio += ratio

    avg_valid_loss = total_valid_loss / len(valid_loader.dataset)
    validation_losses.append(avg_valid_loss)
    avg_psnr = total_psnr / len(valid_loader.dataset)
    validation_psnr.append(avg_psnr)
    avg_peak_to_peak_ratio = total_peak_to_peak_ratio / len(valid_loader.dataset)
    validation_peak_to_peak.append(avg_peak_to_peak_ratio)
    learning_rates.append(scheduler.get_last_lr()[0])

    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}, Validation PSNR: {avg_psnr:.2f}, Validation Peak-to-Peak: {avg_peak_to_peak_ratio:.2f}, Learning Rate: {learning_rates[-1]:.6f}')
    
epochs = range(1, num_epochs + 1)

plot_metrics(epochs, training_losses, validation_losses, validation_psnr, learning_rates= learning_rates, validation_peak_to_peak = validation_peak_to_peak, save_folder = save_folder)

model_path = 'Test.pth' 
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

print("training is complete" )

#### if you want to use different test dataset, uncomment below
# test_directory = "ZHAireS/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000/" #voltage_29-24992_L0_0000.root    

# test_noised_time, test_noised_trace_x, test_noised_trace_y, test_noised_trace_z = traces(test_directory, nb_event=1000, min_primary_energy=1e9, min_zenith=85, max_zenith=88, plot=False, xmin=500, xmax=4596, ymin=1000, ymax=4796, zmin=0, zmax=4096 )

# print(f'shape of test noised_time:{np.shape(test_noised_time)}')
# print(f'shape of test noised_trace_x:{np.shape(test_noised_trace_x)}')
# print(f'shape of test noised_trace_y:{np.shape(test_noised_trace_y)}')
# print(f'shape of test noised_trace_z:{np.shape(test_noised_trace_z)}')
        
    
    
# test_NJ_directory = "ZHAireS-NJ/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000NJ" 
# test_clean_time, test_clean_trace_x, test_clean_trace_y, test_clean_trace_z = traces(test_NJ_directory, nb_event=1000, min_primary_energy=1e9, min_zenith=85, max_zenith=88, plot=False, xmin=500, xmax=4596, ymin=1000, ymax=4796, zmin=0, zmax=4096 )

# print(f'shape of test clean_time:{np.shape(test_clean_time)}')
# print(f'shape of test clean_trace_x:{np.shape(test_clean_trace_x)}')
# print(f'shape of test clean_trace_y:{np.shape(test_clean_trace_y)}')
# print(f'shape of test clean_trace_z:{np.shape(test_clean_trace_z)}')



# noised_signals = (test_noised_trace_x, test_noised_trace_y, test_noised_trace_z)
# clean_signals = (test_clean_trace_x, test_clean_trace_y, test_clean_trace_z)
# total_samples = len(test_noised_trace_x)
# train_indices, valid_indices, test_indices = split_indices(total_samples)

# train_dataset = CustomDataset(noised_signals, clean_signals, indices=train_indices)
# valid_dataset = CustomDataset(noised_signals, clean_signals, indices=valid_indices)
# test_dataset = CustomDataset(noised_signals, clean_signals, indices=test_indices)

# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=4,shuffle=False)   
################################


device = torch.device("cpu")
model = model.to(device)  # Move model to the specified device
model.eval()  # Set model to evaluation mode

#### compare the reconstructed traces with noised, original traces. 
with torch.no_grad():  
    count = 0  # To count the number of images saved
    for noisy_data, clean_data in test_loader:
        if count >= 100:  # Stop after saving 100 images
            break
        
        noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
        denoised_output = model(noisy_data)
        sample_idx = 0  # Index of the sample to plot
        channel_names = ['X Channel', 'Y Channel', 'Z Channel']
        
        for channel_idx in range(3):  # Assuming 3 channels: X, Y, Z
            clean_np = clean_data[sample_idx, channel_idx].cpu().numpy()
            noisy_np = noisy_data[sample_idx, channel_idx].cpu().numpy()
            denoised_np = denoised_output[sample_idx, channel_idx].cpu().numpy()
            snr = np.max(clean_np) / np.std(noisy_np)

            if snr > 0.1:
                plt.figure(figsize=(25, 16))  # Set figure size for each channel
                
                # Metrics calculations
                mse_value = np.mean((clean_np - denoised_np) ** 2)
                psnr_value = psnr(clean_np, denoised_np, np.max(clean_np))

                # Plot the clean signal
                plt.subplot(2, 1, 1)
                plt.plot(clean_np, label=f'Pure - {channel_names[channel_idx]}', color='blue')
                plt.plot(denoised_np, label=f'Denoised - MSE: {mse_value:.2f}, PSNR: {psnr_value:.2f}', linestyle='--', color='orange')
                plt.legend(fontsize=15)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.xlabel('Time (ns)',fontsize = 15)
                plt.ylabel('Voltage (µV)',fontsize = 15)
                plt.title(f'Denoised vs Pure Signal - {channel_names[channel_idx]}', fontsize=15)
                plt.xlim(1500, 2500)

                # Plot the noisy signal
                plt.subplot(2, 1, 2)
                plt.plot(noisy_np, label=f'Noisy signal - {channel_names[channel_idx]}, SNR = {snr:.2f}', color='red')
                plt.legend(fontsize=15)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.xlabel('Time (ns)',fontsize = 15)
                plt.ylabel('Voltage (µV)',fontsize = 15)
                plt.title(f'Noisy Signal - {channel_names[channel_idx]}, SNR:{snr:.2f}', fontsize=15)
                plt.xlim(1500, 2500)
                
                # Save the figure
                plt.tight_layout()
                plt.savefig(os.path.join(save_folder, f'sample_{count:03d}_channel_{channel_idx}.png'))
                plt.close()
                
                count += 1  # Increment the count

                if count >= 100:  # Stop after saving 100 images
                    break
        if count >= 100:  # Stop after saving 100 images
            break
            
            
####### hilbert 

# # Prepare lists to store peak times and peak amplitude for plots

peak_times_noisy = { 'X Channel': [], 'Y Channel': [], 'Z Channel': [] }
peak_times_clean = { 'X Channel': [], 'Y Channel': [], 'Z Channel': [] }
peak_times_denoised = { 'X Channel': [], 'Y Channel': [], 'Z Channel': [] }


peak_amplitudes_noisy = { 'X Channel': [], 'Y Channel': [], 'Z Channel': [] }
peak_amplitudes_clean = { 'X Channel': [], 'Y Channel': [], 'Z Channel': [] }
peak_amplitudes_denoised = { 'X Channel': [], 'Y Channel': [], 'Z Channel': [] }

snr_values = {'X Channel': [], 'Y Channel': [], 'Z Channel': []}
with torch.no_grad():
    for noisy_data, clean_data in train_loader:
        noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
        denoised_output = model(noisy_data)
        
        
        sample_idx = 0  # Index of the sample to plot
        channel_names = ['X Channel', 'Y Channel', 'Z Channel']
        
        for channel_idx in range(3): 
            # Convert tensors to numpy for processing
            clean_np = clean_data[sample_idx, channel_idx].cpu().numpy()
            noisy_np = noisy_data[sample_idx, channel_idx].cpu().numpy()
            denoised_np = denoised_output[sample_idx, channel_idx].cpu().numpy()
            timing = np.array([i for i in range(clean_np.size)])
            snr = np.max(clean_np) / np.std(noisy_np)
            if snr>0.01:
                
                # Calculate Hilbert envelopes and find peaks
                envelope_clean = np.abs(hilbert(clean_np))
                envelope_noisy = np.abs(hilbert(noisy_np))
                envelope_denoised = np.abs(hilbert(denoised_np))

                peak_amplitude_noisy = np.max(envelope_noisy)
                peak_amplitude_clean = np.max(envelope_clean)
                peak_amplitude_denoised = np.max(envelope_denoised)

                peak_time_noisy = timing[np.argmax(envelope_noisy)]
                peak_time_clean = timing[np.argmax(envelope_clean)]
                peak_time_denoised = timing[np.argmax(envelope_denoised)]

                # Store peak times for later plotting
                peak_times_clean[channel_names[channel_idx]].append(peak_time_clean)
                peak_times_noisy[channel_names[channel_idx]].append(peak_time_noisy)
                peak_times_denoised[channel_names[channel_idx]].append(peak_time_denoised)

                peak_amplitudes_clean[channel_names[channel_idx]].append(peak_amplitude_clean)
                peak_amplitudes_noisy[channel_names[channel_idx]].append(peak_amplitude_noisy)
                peak_amplitudes_denoised[channel_names[channel_idx]].append(peak_amplitude_denoised)
                
                snr_values[channel_names[channel_idx]].append(snr)
                
# # Now plot the results
plt.figure(figsize=(15, 18)) 

for i, channel in enumerate(channel_names):
    # Plotting peak Amplitude comparison with noisy
    plt.subplot(3, 2, 2*i+1)
    scatter = plt.scatter(peak_amplitudes_clean[channel], peak_amplitudes_noisy[channel], 
                          c=snr_values[channel], cmap='viridis', alpha=0.6, label='Noisy vs Clean')
    plt.plot([min(peak_amplitudes_clean[channel]), max(peak_amplitudes_clean[channel])], [min(peak_amplitudes_clean[channel]), max(peak_amplitudes_clean[channel])], 'blue', linestyle='--', linewidth=2, label='x=y')
    # plt.xlim(1660,1680)
    # plt.ylim(1660,1680)
    plt.colorbar(scatter, label='SNR')
    plt.xlabel('Peak Amplitude of Clean Data(µV)')
    plt.ylabel('Peak Amplitude of Noisy Data(µV)')
    plt.title(f'Noisy vs Clean - {channel}')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    # Plotting peak Amplitude comparison with denoised
    plt.subplot(3, 2, 2*i+2)
    scatter = plt.scatter(peak_amplitudes_clean[channel], peak_amplitudes_denoised[channel], 
                          c=snr_values[channel], cmap='viridis', alpha=0.6, label='Denoised vs Clean')
    plt.plot([min(peak_amplitudes_clean[channel]), max(peak_amplitudes_clean[channel])], [min(peak_amplitudes_clean[channel]), max(peak_amplitudes_clean[channel])], 'blue', linestyle='--', linewidth=2, label='x=y')
    # plt.xlim(1660,1680)
    # plt.ylim(1660,1680)
    plt.colorbar(scatter, label='SNR')
    plt.xlabel('Peak Amplitude of Clean Data (µV)')
    plt.ylabel('Peak Amplitude of Denoised Data (µV)')
    plt.title(f'Denoised vs Clean - {channel}')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_folder, f'Peak_Amplitude.png'))
plt.close()

plt.figure(figsize=(15, 18))  
for i, channel in enumerate(channel_names):
    # Plotting peak times comparison with noisy
    plt.subplot(3, 2, 2*i+1)
    scatter = plt.scatter(peak_times_clean[channel], peak_times_noisy[channel], 
                          c=snr_values[channel], cmap='viridis', alpha=0.6, label='Noisy vs Clean')
    plt.plot([min(peak_times_clean[channel]), max(peak_times_clean[channel])], 
             [min(peak_times_clean[channel]), max(peak_times_clean[channel])], 
             'b--', label='x = y')  # 'b--' specifies a blue dashed line
    plt.xlim(1650,1680)
    plt.ylim(1650,1680)
    plt.colorbar(scatter, label='SNR')
    plt.xlabel('Peak Times of Clean Data (ns)')
    plt.ylabel('Peak Times of Noisy Data (ns)')
    plt.title(f'Noisy vs Clean - {channel}')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    # Plotting peak times comparison with denoised
    plt.subplot(3, 2, 2*i+2)
    scatter = plt.scatter(peak_times_clean[channel], peak_times_denoised[channel], 
                          c=snr_values[channel], cmap='viridis', alpha=0.6, label='Denoised vs Clean')
    plt.plot([min(peak_times_clean[channel]), max(peak_times_clean[channel])], 
             [min(peak_times_clean[channel]), max(peak_times_clean[channel])], 
             'b--', label='x = y')  # 'b--' specifies a blue dashed line
    plt.xlim(1650,1680)
    plt.ylim(1650,1680)
    plt.colorbar(scatter, label='SNR')
    plt.xlabel('Peak Times of Clean Data (ns)')
    plt.ylabel('Peak Times of Denoised Data (ns)')
    plt.title(f'Denoised vs Clean - {channel}')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_folder, f'Peak_Time.png'))
plt.close()


print(f'Saved {count} images to the folder {save_folder}')