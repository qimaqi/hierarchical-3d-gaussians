#!/bin/bash
#SBATCH --job-name=single_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=./joblogs/smallcity_demo_%j.log      # Redirect stdout to a log file
#SBATCH --time=48:00:00


module load eth_proxy
module load stack/2024-06
module load cuda/12.1.1
module load gcc/12.2.0
module load cmake/3.27.7
source /cluster/work/cvl/qimaqi/miniconda3/etc/profile.d/conda.sh 
conda deactivate
conda activate h3dgs

export CUDA_HOME=/cluster/software/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/cuda-12.1.1-5znnrjb5x5xr26nojxp3yhh6v77il7ie/
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd /cluster/work/cvl/qimaqi/cvpr_2025_city/hierarchical-3d-gaussians/
# preprocess:

DATASET_DIR=/cluster/work/cvl/qimaqi/cvpr_2025/datasets/small_city_train/small_city


# python scripts/full_train.py --project_dir ${DATASET_DIR}


# CHUNK_NAMES=('0_0' '0_1'  '1_0'  '1_1')


# for CHUNK_NAME in $CHUNK_NAMES
# do 
#     python -u train_single.py -s ${DATASET_DIR}/camera_calibration/chunks/${CHUNK_NAME} --model_path ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME} -i ${DATASET_DIR}/camera_calibration/rectified/images -d ${DATASET_DIR}/camera_calibration/rectified/depths --scaffold_file ${DATASET_DIR}/output/scaffold/point_cloud/iteration_30000 --skybox_locked --bounds_file ${DATASET_DIR}/camera_calibration/chunks/${CHUNK_NAME}    

#     submodules/gaussianhierarchy/build/GaussianHierarchyCreator ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME}/point_cloud/iteration_30000/point_cloud.ply ${DATASET_DIR}/camera_calibration/chunks/${CHUNK_NAME}  ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME}/ ${DATASET_DIR}/output/scaffold/point_cloud/iteration_30000

#     # post 
#     python -u train_post.py -s ${DATASET_DIR}/camera_calibration/chunks/${CHUNK_NAME} --model_path ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME} --hierarchy ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME}/hierarchy.hier --iterations 15000 --feature_lr 0.0005 --opacity_lr 0.01 --scaling_lr 0.001 --save_iterations -1 -i ${DATASET_DIR}/camera_calibration/rectified/images --alpha_masks ${DATASET_DIR}/camera_calibration/rectified/masks --scaffold_file ${DATASET_DIR}/output/scaffold/point_cloud/iteration_30000 --skybox_locked --bounds_file ${DATASET_DIR}/camera_calibration/chunks/${CHUNK_NAME}
      
# done

# python scripts/full_train.py --project_dir /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap

# MODEL=mambaout_tiny
# python3 validate.py /data/imagenet  --model $MODEL -b 128 \
#   --pretrained

# srun --nodes=1 -G1 --ntasks=1 --cpus-per-task=12 --mem=48G  --time=04:00:00  --gpus=l4-24g:1  --nodelist=gcpl4-eu-3  --pty bash -i 


# DATASET_DIR=/cluster/work/cvl/qimaqi/cvpr_2025/datasets/small_city_train/small_city/
# /cluster/work/cvl/qimaqi/cvpr_2025/datasets/small_city_eval
# /cluster/work/cvl/qimaqi/cvpr_2025/datasets/small_city_train/small_city


# python scripts/full_train.py --project_dir ${DATASET_DIR} --extra_training_args '--exposure_lr_init 0.0 --eval' 

# python render_hierarchy.py -s /cluster/work/cvl/qimaqi/cvpr_2025/datasets/small_city_eval/aligned --model_path ${DATASET_DIR}/output --hierarchy ${DATASET_DIR}/output/merged.hier --out_dir ${DATASET_DIR}/output/renders --eval --scaffold_file ${DATASET_DIR}/output/scaffold/point_cloud/iteration_30000

# 3s for tau0, 1.1s for tau6

# one trunk?
python render_hierarchy.py -s /cluster/work/cvl/qimaqi/cvpr_2025/datasets/small_city_train/small_city/camera_calibration/chunks/0_0 --model_path /cluster/work/cvl/qimaqi/cvpr_2025/datasets/small_city_train/small_city/output/ --hierarchy /cluster/work/cvl/qimaqi/cvpr_2025/datasets/small_city_train/small_city/output/hierarchy.hier_opt --out_dir ${OUTPUT_DIR} --eval

# python preprocess/copy_file_to_chunks.py