#!/bin/bash
#SBATCH --job-name=single_gpu_pose_est
#SBATCH --nodelist=gcpl4-eu-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=l4-24g:1
#SBATCH --cpus-per-task=12
#SBATCH --output=./joblogs/h3dgs_small_city_demo_%j.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/h3dgs_small_city_demo__%j.error     # Redirect stderr to a separate error log file
#SBATCH --time=48:00:00


# cuda
export CUDA_HOME=/opt/modules/nvidia-cuda-11.8.0
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.8.0/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.8.0/bin:$PATH

# pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
export PATH=/opt/modules/gcc-10.5.0/bin:$PATH
DATASET_DIR=/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug
# /data/work-gcp-europe-west4-a/qimaqi/datasets/small_city
source ~/.bashrc 
micromamba activate h3dgs
cd ..
python scripts/full_train.py --project_dir ${DATASET_DIR}



# python scripts/full_train.py --project_dir /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap

# MODEL=mambaout_tiny
# python3 validate.py /data/imagenet  --model $MODEL -b 128 \
#   --pretrained

# srun --nodes=1 -G1 --ntasks=1 --cpus-per-task=12 --mem=48G  --time=04:00:00  --gpus=l4-24g:1  --nodelist=gcpl4-eu-3  --pty bash -i 
