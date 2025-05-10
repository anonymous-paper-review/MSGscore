#!/bin/bash

# Set base directory
BASE_DIR="/app/host_volume/MSGscore_git/MSGscore" # Adjust this if your script is located elsewhere
VBENCH_DIR="${BASE_DIR}/VBench"
EVALCRAFTER_DIR="${BASE_DIR}/EvalCrafter"

echo "Setting up VBench and EvalCrafter environments..."

# Function to check if conda environment exists
env_exists() {
    conda info --envs | grep -q "^$1 "
}

cd $VBENCH_DIR

# Create conda environment for VBench
if env_exists "vbench"; then
    echo "Removing existing vbench environment to ensure clean installation..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda deactivate
    conda env remove -n vbench -y
    echo "Existing vbench environment removed."
fi

echo "Creating conda environment for VBench..."
conda create -n vbench python=3.9 -y

echo "Installing VBench dependencies..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vbench

# Install PyTorch with updated versions
pip install torch==2.0.1 torchvision==0.15.2
pip install torchmetrics==1.2.0

# Install other dependencies
pip install -r requirements.txt

# Additional common dependencies that might be needed
pip install matplotlib tqdm scipy pandas scikit-learn

conda deactivate
echo "VBench environment setup complete."





# Setup EvalCrafter
echo "Setting up EvalCrafter environment..."

cd $EVALCRAFTER_DIR

# Create conda environment for EvalCrafter using environment.yml
echo "Creating/Updating EvalCrafter environment from environment.yml..."
source $(conda info --base)/etc/profile.d/conda.sh

# Check if environment exists and remove if it does (to ensure clean install)
if env_exists "EvalCrafter"; then
    echo "Removing existing EvalCrafter environment to ensure clean installation..."
    conda deactivate
    conda env remove -n EvalCrafter -y
fi

# Create environment from yml file
conda env create -f environment.yml
conda activate EvalCrafter

# Install the package in development mode
pip install -r requirements.txt

# Compile the RAFT networks
echo "Compiling RAFT networks..."
cd ./metrics/RAFT/networks/resample2d_package
python setup.py install --user
cd $EVALCRAFTER_DIR

# Download model weights
echo "Downloading model weights..."
cd checkpoints
bash download.sh
cd $EVALCRAFTER_DIR

conda deactivate
echo "EvalCrafter environment setup complete."





echo "Setup completed successfully!"
