#!/bin/bash
#SBATCH --job-name=street_10000_15000
#SBATCH --nodelist=gcpl4-eu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=./joblogs/generate_pcd_%j.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/generate_pcd_%j.error     # Redirect stderr to a separate error log file
#SBATCH --time=24:00:00

source ~/.bashrc
micromamba activate city
# python matrixcity_street_to_colmap_all.py --start_idx=10000 --end_idx=15000

python matrixcity_aerial_block_to_colmap.py --start_idx=4000 --end_idx=6400