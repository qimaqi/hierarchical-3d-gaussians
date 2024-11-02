#!/bin/bash
#SBATCH --job-name=street_0_5000
#SBATCH --nodelist=gcpl4-eu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=./joblogs/aerial_small_city_%j.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/generate_pcd_%j.error     # Redirect stderr to a separate error log file
#SBATCH --time=64:00:00

source ~/.bashrc
micromamba activate city
# python matrixcity_street_to_colmap_all.py --start_idx=0 --end_idx=5000
python matrixcity_sanity_check.py --city_name=small_city --view_name=street 
# srun --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=64G  --time=04:00:00  --nodelist=gcpl4-eu-2  --pty bash -i cpu tasks