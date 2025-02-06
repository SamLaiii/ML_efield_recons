import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import hilbert
from torch.utils.data import Dataset, DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt

from old_model import Autoencoder
from CNNModel import DualBranchAutoencoder

from training_function import traces, CustomDataset, split_indices, psnr_loss
from train import train_validate
from test import test 
from hilbert import peak_amplitude, peak_time
# /home/923714256/0422_simulation/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000
# /home/923714256/0422_simulation/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000NJ

def main(args):
    current_directory = os.getcwd()
    save_folder = args.save_folder_name

    if not os.path.exists(save_folder):
        os.makedirs(current_directory + '/' + save_folder) 

    print("Empty Folder is created")   
    mpl.rcParams['figure.max_open_warning'] = 50
    voltage_type = False
    adc_type = False
    efield_type = False
    band_filter_voltage_mode = False
    band_filter_adc_mode = False
    band_filter_efield_mode = False
    if args.trace_type == 'voltage':
        voltage_type = True
        band_filter_voltage_mode = True 
    if args.trace_type == 'adc':
        adc_type = True 
        band_filter_adc_mode = True
    if args.trace_type == 'efield':
        efield_type = True 
        band_filter_efield_mode = True

    ##### data preparation
    time, frequency, noised_trace_x, clean_trace_x, noised_trace_y, clean_trace_y, noised_trace_z, clean_trace_z = traces(
        args.directory, # directory for Jitter folder
        args.NJ_directory, # directory for Non-Jitter folder
        nb_events=args.nb_events, # number of events 
        mpe=args.mpe, # maximum potential energy 
        min_zenith=args.min_zenith, # minimum zenith for data selection
        max_zenith=args.max_zenith, # maximum zenith for data selection
        band_filter = band_filter_voltage_mode, # filter voltage frequency outside between 50 ~ 200 MHz 
        band_filter_adc =  band_filter_adc_mode, # filter voltage frequency outside between 50 ~ 200 MHz 
        band_filter_efield = band_filter_efield_mode, # filter voltage frequency outside between 50 ~ 200 MHz 
        voltage = voltage_type, # choose voltage data as training set
        ADC = adc_type, # choose adc data as training set
        efield = efield_type) # choose efield data as training set
    
    print(f'type of traces to train: {args.trace_type} ')
    print(f'shape of time:{np.shape(time)}')
    print(f'shape of frequency{np.shape(frequency)}')
    print(f'shape of noised_trace_x:{np.shape(noised_trace_x)}')
    print(f'shape of noised_trace_y:{np.shape(noised_trace_y)}')
    print(f'shape of noised_trace_z:{np.shape(noised_trace_z)}')        
    print(f'shape of clean_trace_x:{np.shape(clean_trace_x)}')
    print(f'shape of clean_trace_y:{np.shape(clean_trace_y)}')
    print(f'shape of clean_trace_z:{np.shape(clean_trace_z)}')

    noised_signals = (noised_trace_x, noised_trace_y, noised_trace_z)
    clean_signals = (clean_trace_x, clean_trace_y, clean_trace_z)

#    plot_snr_distribution(clean_signals= clean_signals, noised_signals=noised_signals, save_folder= args.save_folder_name)


    total_samples = len(noised_trace_x)
    train_indices, valid_indices, test_indices = split_indices(total_samples)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device set to : {device}')
    train_dataset = CustomDataset(noised_signals, clean_signals, indices=train_indices)
    valid_dataset = CustomDataset(noised_signals, clean_signals, indices=valid_indices)
    test_dataset = CustomDataset(noised_signals, clean_signals,  indices=test_indices)

    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=4, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, num_workers=4, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=True)

    model = DualBranchAutoencoder().to(device)
    print(model)

    base_lr = 0.0001

    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay = 0.00005)

    # Set the criterion based on the argument
    if args.criterion == 'mse':
        criterion = nn.MSELoss()
    elif args.criterion == 'psnr':
        criterion = psnr_loss 

    train_validate(train_loader = train_loader, 
                   valid_loader = valid_loader, 
                   model = model, 
                   optimizer = optimizer,
                   criterion = criterion, 
                   num_epochs=args.epochs, 
                   device=device, 
                   save_folder=save_folder)

    torch.save(model.state_dict(), os.path.join(args.save_folder_name, 'best_model.pth'))

    if args.test_mode: ### Test Other types of traces 
        time, frequency, noised_trace_x, clean_trace_x, noised_trace_y, clean_trace_y, noised_trace_z, clean_trace_z = traces(
            args.directory, 
            args.NJ_directory, 
            nb_events = args.nb_events, 
            mpe = args.mpe, 
            min_zenith = args.min_zenith, 
            max_zenith = args.max_zenith, 
            voltage = False, 
            ADC = True, 
            efield = False 
        )
        print(f'type of traces to test: ADC')
        print(f'shape of noised_time:{np.shape(time)}')
        print(f'shape of noised_trace_x:{np.shape(noised_trace_x)}')
        print(f'shape of noised_trace_y:{np.shape(noised_trace_y)}')
        print(f'shape of noised_trace_z:{np.shape(noised_trace_z)}')        
        print(f'shape of clean_trace_x:{np.shape(clean_trace_x)}')
        print(f'shape of clean_trace_y:{np.shape(clean_trace_y)}')
        print(f'shape of clean_trace_z:{np.shape(clean_trace_z)}')
   
        noised_signals = (noised_trace_x, noised_trace_y, noised_trace_z)
        clean_signals = (clean_trace_x, clean_trace_y, clean_trace_z)
        total_samples = len(noised_trace_x)
        train_indices, valid_indices, test_indices = split_indices(total_samples)
        test_dataset = CustomDataset(noised_signals, clean_signals, indices=test_indices)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

        test(time=time, 
             frequency=frequency, 
             testloader=test_loader, 
             model=model, 
             device="cpu", 
             min_snr=args.min_snr, 
             max_snr = args.max_snr,
             save_folder=save_folder,
             voltage = False, 
             ADC = True, 
             efield = False)


        peak_amplitude(dataloader = test_loader, 
                       model = model, 
                       device="cpu", 
                       min_snr=args.min_snr, 
                       max_snr = args.max_snr,
                       save_folder = save_folder)
        

        peak_time(dataloader= test_loader, 
                  model=model, 
                  device="cpu", 
                  min_snr=args.min_snr, 
                  max_snr = args.max_snr,
                  save_folder=save_folder)
        
        print(f'All process are complete.')
    
    else:
        print(f'type of traces to test : {args.trace_type} ')
        test(time=time, 
             frequency=frequency, 
             testloader=test_loader, 
             model=model, 
             device="cpu", 
             min_snr=args.min_snr, 
             max_snr = args.max_snr,
             save_folder=save_folder, 
             voltage=voltage_type, 
             ADC = adc_type, 
             efield = efield_type)
        

        peak_amplitude(dataloader = test_loader, 
                       model = model, 
                       device = "cpu", 
                       min_snr = args.min_snr, 
                       max_snr = args.max_snr,
                       save_folder = save_folder)
        
        peak_time(dataloader = test_loader, 
                  model = model, 
                  device = "cpu", 
                  min_snr = args.min_snr,
                  max_snr = args.max_snr, 
                  save_folder = save_folder)
        
        print(f'All process are complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some traces.')
    parser.add_argument('--directory', type=str, required=True, help='Path to the main directory')

    parser.add_argument('--NJ_directory', type=str, required=True, help='Path to the NJ directory')

    parser.add_argument('--nb_events', type=int, default=1000, help='Number of events')

    parser.add_argument('--criterion', default='mse', choices=['mse', 'psnr'], help='Loss function: either MSE or PSNR')

    parser.add_argument('--mpe', type=float, default=1e9, help='MPE value')

    parser.add_argument('--epochs', type = float, default = 100, help ='nums of epochs')

    parser.add_argument('--min_zenith', type=float, default= 70, help='Minimum zenith angle')

    parser.add_argument('--max_zenith', type=float, default= 89, help='Maximum zenith angle')

    parser.add_argument('--save_folder_name', type=str, required=True, help='Folder to save results')

    parser.add_argument('--test_mode',default= False, help='Whether to run in test mode')
    
    parser.add_argument('--min_snr', type = float, default = 3, help ='minimum of snr for the data display')

    parser.add_argument('--max_snr', type = float, default = 10, help ='maximum of snr for the data display')

    parser.add_argument('--trace_type', default = 'adc', choices= ['voltage','adc','efield'], help = 'Choose one of the trace type of training and testing')
    args = parser.parse_args()
    main(args)