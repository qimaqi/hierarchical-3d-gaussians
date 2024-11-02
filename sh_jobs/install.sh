#!/bin/bash
#SBATCH --job-name=single_gpu_pose_est
#SBATCH --nodelist=gcp-eu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100-40g:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-gpu=48G
#SBATCH --output=./joblogs/image/cifar_vgg_exps_%j.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/image/cifar_vgg_exps_%j.error     # Redirect stderr to a separate error log file
#SBATCH --time=48:00:00


# cuda
# srun --nodes=1 -G1 --ntasks=1 --cpus-per-task=8 --mem=48G  --time=04:00:00  --gpus=l4-24g:1  --nodelist=gcpl4-eu-3  --pty bash
micromamba create -n h3dgs python=3.12
export CUDA_HOME=/opt/modules/nvidia-cuda-11.8.0
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.8.0/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.8.0/bin:$PATH
export PATH=/opt/modules/gcc-10.5.0/bin:$PATH
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install trimesh
pip install hydra-core
micromamba install conda-forge::colmap
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

source ~/.bashrc 
micromamba activate h3dgs