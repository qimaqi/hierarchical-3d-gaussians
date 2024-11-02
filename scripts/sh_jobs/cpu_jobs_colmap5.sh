#!/bin/bash
#SBATCH --job-name=street_25000_31000
#SBATCH --nodelist=gcpl4-eu-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --output=./joblogs/street_25000_31000_%j.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/street_25000_31000_%j.error     # Redirect stderr to a separate error log file
#SBATCH --time=6:00:00

source ~/.bashrc
micromamba activate city
python matrixcity_street_to_colmap_all.py --start_idx=25000 --end_idx=31000
