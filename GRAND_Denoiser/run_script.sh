#!/bin/bash
#SBATCH --job-name=dualautoencoder        # Job name
#SBATCH --partition=gpucluster            # Partition name
#SBATCH --time=02:00:00                   # Time limit hrs:min:sec
#SBATCH --output=dualautoencoder_%j.out   # Standard output and error log
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task

sinfo -o "%n %c %m"  # Shows nodes, their CPU count, and memory
sinfo -N -l  # Detailed node information

# Check NVIDIA GPU status using srun
srun --partition=gpucluster nvidia-smi

# Activate conda
source /Users/923714256/miniconda3/bin/activate
conda activate grandlib

# Check if the environment was activated successfully
if [ $? -ne 0 ]; then
    echo "Failed to activate the Conda environment."
    exit 1 
fi

# Verify the active environment and Python version
echo "Active Conda Environment: $(conda info --envs | grep '*' | awk '{print $1}')"
python --version

# Set PYTHONPATH to include the necessary directories (Should be run in the git branch: dev_sim2root)
export PYTHONPATH=/Users/923714256/grand

# Run the main Python script with appropriate arguments
srun --partition=gpucluster python main.py --directory /Users/923714256/0422_simulation/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000 --NJ_directory /Users/923714256/0422_simulation/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000NJ --save_folder "ADC_100_epochs"

# Deactivate the conda environment
conda deactivate