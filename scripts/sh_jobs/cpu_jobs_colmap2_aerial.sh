#!/bin/bash
#SBATCH --job-name=street_3k_6k
#SBATCH --nodelist=gcpl4-eu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=./joblogs/aerial_small_city_%j.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/aerial_small_city_%j.error     # Redirect stderr to a separate error log file
#SBATCH --time=24:00:00

source ~/.bashrc
micromamba activate city
cd ..
# python matrixcity_street_to_colmap_all.py --start_idx=0 --end_idx=5000
python matrixcity_to_colmap.py --start_idx=6000 --end_idx=7000 --city_name=small_city --view_name=aerial