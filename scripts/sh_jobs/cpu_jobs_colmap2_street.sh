#!/bin/bash
#SBATCH --job-name=street_10k_15k
#SBATCH --nodelist=gcpl4-eu-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=./joblogs/street_small_city_10k_15k_%j.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/street_small_city_10k_15k_%j.error     # Redirect stderr to a separate error log file
#SBATCH --time=24:00:00

source ~/.bashrc
micromamba activate city
cd ..
# python matrixcity_street_to_colmap_all.py --start_idx=0 --end_idx=5000
python matrixcity_to_colmap.py --start_idx=10000 --end_idx=15000 --city_name=small_city --view_name=street