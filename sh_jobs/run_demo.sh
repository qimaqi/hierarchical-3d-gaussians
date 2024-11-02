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
DATASET_DIR=/data/work2-gcp-europe-west4-a/qimaqi/datasets/small_city_chunks/
# /data/work-gcp-europe-west4-a/qimaqi/datasets/small_city
source ~/.bashrc 
micromamba activate auto-vpik4ilrdqdq
cd ..
# python scripts/full_train.py --project_dir ${DATASET_DIR}

# python train_coarse.py -s ${DATASET_DIR}/camera_calibration/aligned -i ${DATASET_DIR}/camera_calibration/rectified/images --skybox_num 100000 --position_lr_init 0.00016 --position_lr_final 0.0000016 --model_path ${DATASET_DIR}/output/scaffold


# python train_coarse.py -s ${DATASET_DIR}/camera_calibration/aligned -i ${DATASET_DIR}/camera_calibration/rectified/images --skybox_num 100000 --position_lr_init 0.00016 --position_lr_final 0.0000016 --model_path ${DATASET_DIR}/output/scaffold

python -u train_single.py -s [project/chunks/chunk_name] --model_path [output/chunks/chunk_name] -i [project/rectified/images] -d [project/rectified/depths] --alpha_masks [project/rectified/masks] --scaffold_file [output/scaffold/point_cloud/iteration_30000] --skybox_locked --bounds_file [project/chunks/chunk_name]    

# python scripts/full_train.py --project_dir /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap

# MODEL=mambaout_tiny
# python3 validate.py /data/imagenet  --model $MODEL -b 128 \
#   --pretrained

# srun --nodes=1 -G1 --ntasks=1 --cpus-per-task=12 --mem=48G  --time=04:00:00  --gpus=l4-24g:1  --nodelist=gcpl4-eu-3  --pty bash -i 
