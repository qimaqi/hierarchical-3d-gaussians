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

# colmap convert to txt 
# /cluster/work/cvl/qimaqi/cvpr_2025/datasets/small_city_train/small_city/camera_calibration/aligned/sparse/0

# colmap model_converter --input_path=/cluster/work/cvl/qimaqi/cvpr_2025/datasets/small_city_train/small_city/camera_calibration/aligned/sparse/0 --output_path=/cluster/work/cvl/qimaqi/cvpr_2025/datasets/small_city_train/small_city/camera_calibration/aligned/sparse/known --output_type=BIN


