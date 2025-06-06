# ML_efield_recons
This work will investigate the use of ML to perform denoising of traces and/or E-field reconstruction
1. Run Python data.file --directory (directory of Jitter dc2 simulation) --NJ_directory (directory of Non-Jitter dc2 simulation) --save_folder (directory of npz data file want to save) in the GRAND_Denoiser.
2. Run python main.py --save_folder (directory of training results want to save) GRAND_Denoiser.


# Raytune
Raytune is introduced for finding the best hyperparameters of the model. 
1. Make sure the saved folder contain dc2_train_noised_signals.npz, dc2_train_clean_signals.npz, dc2_validation_noised_signals.npz, dc2_validation_clean_signals.npz, dc2_test_noised_signals.npz, dc2_test_clean_signals.npz.  
2. Run python hypermain_new_dc2.py --save_folder "/saved/folder/that/containing/npz/file" 