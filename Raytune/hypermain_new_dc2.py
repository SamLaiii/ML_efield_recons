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
from hyperCNNmodel import DualBranchAutoencoder as CNN
# from hyperLNN import LNN_DualBranchAE as LNN
from training_function import CustomDataset, split_indices, plot_metrics, psnr_loss, get_peak_amplitude, calculate_psnr_with_peak, peak_to_peak_ratio, psnr, multi_domain_loss
from hypertrain_new_dc2 import train_validate
from pathlib import Path
import random
from test import test 
from hilbert import peak_amplitude, peak_time
# /Users/923714256/0422_simulation/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000
# /Users/923714256/0422_simulation/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000NJ

from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

from hypertrain_new_dc2 import get_data_loaders



def main(number_samples=10, gpus_per_trial=2):
    current_directory = os.getcwd()
    save_folder = args.save_folder_name

    if not os.path.exists(save_folder):
        os.makedirs(current_directory + '/' + save_folder) 

    print("Empty Folder is created")   
    mpl.rcParams['figure.max_open_warning'] = 50
    config = {
        "lr": tune.loguniform(1e-6, 1e-3),
        "weight_decay": tune.loguniform(1e-5, 5e-3),
        "epochs": args.epochs,
        "criterion": tune.choice(["mse", "psnr", "multi"]),
        "save_folder_name": save_folder,
        "model_config" :{
            "time_branch": {
                "conv_channels": tune.choice([16, 32]),  # Sample a random float from a normal distribution with 
                "res_channels": tune.choice([(16, 32), (32,64),(64, 128), (128, 256)])  # Random tuple for res_channels
                            },
            "freq_branch": {
                "conv_channels": tune.choice([16, 32]),  # Random number between 16 and 64
                "res_channels":tune.choice([(16, 32), (32,64), (64, 128), (128,256)])  # Random tuple for res_channels
                            },
            "decoder_channels": tune.choice([[128, 64, 32, 3], [256, 128, 64, 3], [64 , 32, 16 ,3 ], [32, 16 ,8 ,3]])
                            }
    }

        #     "model_config" :{
    #         "time_branch": {
    #             "conv_channels": tune.qrandint(4, 16, 2),  # Sample a random float from a normal distribution with 
    #             "res_channels": (tune.qrandint(8, 32, 2), tune.qrandint(16, 64, 2))  # Random tuple for res_channels
    #         },
    #         "freq_branch": {
    #             "conv_channels": tune.qrandint(4, 16, 2),  # Random number between 16 and 64
    #             "res_channels": (tune.qrandint(8, 32, 2), tune.qrandint(16, 64, 2))   # Random tuple for res_channels
    #         },
    #         "decoder_channels": [
    #              tune.qrandint(128, 256,2), 
    #              tune.qrandint(64, 128, 2),
    #              tune.qrandint(32, 64 , 2),
    #              3]
    # }
    
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=args.epochs,
        grace_period=1,
        reduction_factor=2
    )

    result = tune.run(
        train_validate,
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=config,
        num_samples = args.number_samples,
        scheduler=scheduler
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    if 'validation_psnr' in best_trial.last_result:
        print(f"Best trial final validation PSNR: {best_trial.last_result['validation_psnr']}")

    best_train_model = CNN(best_trial.config["model_config"])
    torch.save(best_train_model.state_dict(), os.path.join(args.save_folder_name, 'best_model.pth'))


    best_checkpoint = result.get_best_checkpoint(trial=best_trial, mode="max")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_train_model.load_state_dict(best_checkpoint_data["net_state_dict"])
    
    #test
    ##### data preparation


    _ ,_ , testloader = get_data_loaders(best_trial.config)

    test(testloader= testloader, 
        model= best_train_model, 
        num_images = 200,
        min_snr = args.min_snr,
        max_snr = args.max_snr, 
        save_folder =save_folder)

    peak_amplitude(dataloader=testloader, 
                    model = best_train_model, 
                    device = "cpu", 
                    min_snr = args.min_snr, 
                    max_snr = args.max_snr,
                    save_folder=save_folder)
    
    peak_time(dataloader = testloader, 
              model = best_train_model, 
              device = "cpu", 
              min_snr = args.min_snr,
              max_snr = args.max_snr, 
              save_folder = save_folder)

    print("Process is all finished")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Hyperparameter tuning with Ray Tune.')

    parser.add_argument('--criterion', default='multi_loss', choices=['mse', 'psnr','multi_loss'], help='Loss function: either MSE or PSNR')

    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')

    parser.add_argument('--save_folder_name', type=str, required = True, help='Folder to save results')

    # parser.add_argument('--trace_type', default = 'adc', choices =['voltage', 'efield','adc'], help='type of traces, whether choose voltage or 1')

    parser.add_argument('--min_snr', type = float, default = 1, help ='minimum of snr for the data display')

    parser.add_argument('--max_snr', type = float, default = 1e3, help ='maximum of snr for the data display')

    parser.add_argument('--number_samples', type=int, default=36, help='Number of samples for tune training')

    args = parser.parse_args()

    main(args)


# PYTHONPATH=/home/923714256/grand python hypermain.py  --save_folder "dual_model_resnet_4_layers_encoder_psnr_3_SNR_1_50_epoch_adc_for_overleaf_hypertune"  