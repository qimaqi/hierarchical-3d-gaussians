#!/bin/bash
#SBATCH --job-name=genreate_large_city_pcd
#SBATCH --nodelist=gcpl4-eu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=./joblogs/generate_pcd_%j.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/generate_pcd_%j.error     # Redirect stderr to a separate error log file
#SBATCH --time=64:00:00

source ~/.bashrc
micromamba activate tool
python generatr_big_city_pcd.py 