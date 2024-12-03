#!/bin/bash
#SBATCH --job-name=genreate_large_city_pcd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=./joblogs/generate_chunks_%j.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/generate_chunks_%j.error     # Redirect stderr to a separate error log file
#SBATCH --time=24:00:00


module load eth_proxy
module load stack/2024-06
module load cuda/12.1.1
module load gcc/12.2.0
module load cmake/3.27.7
source /cluster/work/cvl/qimaqi/miniconda3/etc/profile.d/conda.sh 
conda deactivate
conda activate tool

cd /cluster/work/cvl/qimaqi/cvpr_2025_city/hierarchical-3d-gaussians/
# python generatr_big_city_pcd.py --VIEW_NAME=aerial 

RAW_PHOTO_PATH=/cluster/work/cvl/qimaqi/cvpr_2025/datasets/urban3d/colmap_results/residence/photos
CAMERA_POSE_PATH=/cluster/work/cvl/qimaqi/cvpr_2025/datasets/urban3d/colmap_results/residence/residence-pixsfm

# python copy_images.py --image_path $RAW_PHOTO_PATH --dataset_path $CAMERA_POSE_PATH
# python preprocess/generate_chunks.py --project_dir /cluster/work/cvl/qimaqi/cvpr_2025/datasets/urban3d/colmap_results/residence/residence-pixsfm/train/ --min_n_cams 20 --chunk_size 100  --skip_bundle_adjustment

python preprocess/generate_chunks.py --project_dir /cluster/work/cvl/qimaqi/cvpr_2025/datasets/urban3d/colmap_results/residence/residence-pixsfm/train/ --min_n_cams 20 --chunk_size 100  --skip_bundle_adjustment


# srun --ntasks=8 --mem-per-cpu=4G --gpus=rtx_3090:1  --time=240 --pty bash -i
# srun --ntasks=16 --mem-per-cpu=8G --time=240 --pty bash -i