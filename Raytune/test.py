from training_function import psnr
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np 
import os 

def test(testloader, 
         model, 
         num_images = 200, 
         device="cpu", 
         min_snr =3, 
         max_snr = 1e3, 
         save_folder =''):
    

    with torch.no_grad():  
        device = torch.device(device)
        model = model.to(device)
        model.eval() 
        count = 0  # To count the number of images saved
        for noisy_data, clean_data in testloader:
            if count >= num_images: 
                break
            
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            denoised_output = model(noisy_data)
            sample_idx = 0  # Index of the sample to plot
            channel_names = ['X Channel', 'Y Channel', 'Z Channel']
            
            for channel_idx in range(3): 
                clean_np = clean_data[sample_idx, channel_idx].cpu().numpy()
                noisy_np = noisy_data[sample_idx, channel_idx].cpu().numpy()
                denoised_np = denoised_output[sample_idx, channel_idx].cpu().numpy()
                snr = np.max(clean_np) / np.std(noisy_np)

                if max_snr > snr > min_snr:
                    plt.figure(figsize=(25, 16))  # Set figure size for each channel
                    
                    # Metrics calculations
                    mse_value = np.mean((clean_np - denoised_np) ** 2)
                    psnr_value = psnr(clean_np, denoised_np, np.max(clean_np))

                    # Plot the clean signal
                    plt.subplot(2, 1, 1)

                    plt.plot(clean_np, label=f'Pure - {channel_names[channel_idx]}', color='blue')
                    plt.plot(denoised_np, label=f'Denoised - MSE: {mse_value:.2f}, PSNR: {psnr_value:.2f}', linestyle='--', color='orange')
                    plt.xlabel(f'Time Bin[ns]',fontsize = 24)
                    plt.ylabel(f'Counts',fontsize = 24)
                    plt.title(f'Pure Signal - {channel_names[channel_idx]}', fontsize=24)
                        # plt.xlim(200, 712)
                    plt.legend(fontsize=24)
                    plt.xticks(fontsize=24)
                    plt.yticks(fontsize=24)

                    # Plot the noisy signal
                    plt.subplot(2, 1, 2)
                    plt.plot(noisy_np, label=f'Noisy signal - {channel_names[channel_idx]}, SNR = {snr:.2f}', color='red')
                    plt.xlabel('Time Bin(ns)',fontsize = 24)
                        # plt.xlim(200, 712)
                    plt.ylabel('Counts',fontsize = 24)
                    plt.title(f'Noisy Signal - {channel_names[channel_idx]}, SNR:{snr:.2f}', fontsize=24)
                    plt.legend(fontsize=24)
                    plt.xticks(fontsize=24)
                    plt.yticks(fontsize=24)

                    # Save the figure
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_folder, f'sample_{count:03d}_channel_{channel_idx}_snr_{snr:.2f}.png'))
                    plt.close()
                    
            count += 1  # Increment the count

            if count >= num_images:  # Stop after saving 100 images
                break
        # if count >= num_images:  # Stop after saving 100 images
        #     break
    print('test is completed') 