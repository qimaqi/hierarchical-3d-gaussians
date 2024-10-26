#!/bin/bash
#SBATCH --job-name=street_20000_25000
#SBATCH --nodelist=gcpl4-eu-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=./joblogs/street_20000_25000_%j.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/street_20000_25000_%j.error     # Redirect stderr to a separate error log file
#SBATCH --time=64:00:00

source ~/.bashrc
micromamba activate city
python matrixcity_street_to_colmap_all.py --start_idx=20000 --end_idx=25000
