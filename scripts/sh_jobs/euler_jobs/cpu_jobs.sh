#!/bin/bash
#SBATCH --job-name=aerial_collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=./joblogs/cpu_jobs_colmap_aerial_%j.log      # Redirect stdout to a log file
#SBATCH --time=24:00:00

module load eth_proxy
module load stack/2024-06
module load cuda/12.1.1
module load gcc/12.2.0
module load cmake/3.27.7
source /cluster/work/cvl/qimaqi/miniconda3/etc/profile.d/conda.sh 
conda deactivate
conda activate tool

cd /cluster/work/cvl/qimaqi/cvpr_2025_city/hierarchical-3d-gaussians/scripts
# python matrixcity_street_to_colmap_all.py --start_idx=0 --end_idx=5000
python matrixcity_to_colmap.py --start_idx=0 --end_idx=-1  --city_name=big_city --view_name=aerial --collect --overwrite