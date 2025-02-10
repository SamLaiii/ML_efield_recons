import argparse
import os
import numpy as np
from training_function import traces




def main(args): 
##### data preparation
    current_directory = os.getcwd()
    save_folder = args.save_folder_name
    print("Empty Folder is created")   

    if not os.path.exists(save_folder):
        os.makedirs(current_directory + '/' + save_folder) 
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

    time, frequency, noised_trace_x, clean_trace_x, noised_trace_y, clean_trace_y, noised_trace_z, clean_trace_z = traces(
        args.directory, 
        args.NJ_directory, 
        nb_events=args.nb_events, 
        mpe=args.mpe, 
        min_zenith=args.min_zenith, 
        max_zenith=args.max_zenith, 
        band_filter = band_filter_voltage_mode,
        band_filter_adc =  band_filter_adc_mode,
        band_filter_efield = band_filter_efield_mode,
        voltage = voltage_type,
        ADC = adc_type,
        efield = efield_type)
    
        
    print(f'type of traces to train: {args.trace_type} ')
    print(f'shape of time:{np.shape(time)}')
    print(f'shape of frequency{np.shape(frequency)}')
    print(f'shape of noised_trace_x:{np.shape(noised_trace_x)}')
    print(f'shape of noised_trace_y:{np.shape(noised_trace_y)}')
    print(f'shape of noised_trace_z:{np.shape(noised_trace_z)}')        
    print(f'shape of clean_trace_x:{np.shape(clean_trace_x)}')
    print(f'shape of clean_trace_y:{np.shape(clean_trace_y)}')
    print(f'shape of clean_trace_z:{np.shape(clean_trace_z)}')

    save_path_noised = os.path.join(args.save_folder_name, 'dc2_noised_signals.npz')
    save_path_clean = os.path.join(args.save_folder_name, 'dc2_clean_signals.npz')

    os.makedirs(save_folder, exist_ok = True)

    if not os.path.exists(save_path_noised):
        noised_signals = np.stack((noised_trace_x, noised_trace_y, noised_trace_z))
        np.savez_compressed(save_path_noised, signals = noised_signals, frequency = frequency, time = time)
        print(f'dc2_noised_signals.npz is created')
    else:
        print(f'dc2_noised_signals.npz already exists')

    if not os.path.exists(save_path_clean):
        clean_signals = np.stack((clean_trace_x, clean_trace_y, clean_trace_z))
        np.savez_compressed(save_path_clean,  signals = clean_signals, frequency = frequency, time = time)
        print(f'dc2_clean_signals.npz is created')
    else:
        print(f'dc2_clean_signals.npz already exists')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some traces.')
    parser.add_argument('--directory', type=str, required=True, help='Path to the main directory')

    parser.add_argument('--NJ_directory', type=str, required=True, help='Path to the NJ directory')

    parser.add_argument('--nb_events', type=int, default=1000, help='Number of events')

    parser.add_argument('--mpe', type=float, default=1e9, help='MPE value')
    
    parser.add_argument('--min_zenith', type=float, default= 70, help='Minimum zenith angle')

    parser.add_argument('--max_zenith', type=float, default= 89, help='Maximum zenith angle')

    parser.add_argument('--save_folder_name', type=str, required=True, help='Folder to save results')

    parser.add_argument('--trace_type', default = 'adc', choices= ['voltage','adc','efield'], help = 'Choose one of the trace type for saving as npz')
    args = parser.parse_args()
    main(args)