#!/bin/bash
#SBATCH --job-name=unzip
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=./joblogs/unzip_%j.log      # Redirect stdout to a log file
#SBATCH --time=24:00:00


# Directory containing the .tar.gz files
SOURCE_DIR="."
# Directory where all extracted files will be merged
# Loop through each .tar.gz file in the source directory
for file in "$SOURCE_DIR"/*.tar*; 
do
  # Extract the .tar.gz file
  tar -xvf "$file" 

  # Find the name of the extracted directory  extracted_dir=$(tar -tf "$file" | head -1 | cut -f1 -d"/")

  # Move contents of extracted directory to the destination directorymv "$SOURCE_DIR/$extracted_dir"/* "$DEST_DIR"/

  # Remove the empty extracted directory rmdir "$SOURCE_DIR/$extracted_dir"
done
