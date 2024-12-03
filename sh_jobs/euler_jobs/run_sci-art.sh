#!/bin/bash
#SBATCH --job-name=single_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_3090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=./joblogs/urbsn3d_sciart_%j.log      # Redirect stdout to a log file
#SBATCH --time=24:00:00


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

cd /cluster/work/cvl/qimaqi/cvpr_2025_city/hierarchical-3d-gaussians/scripts

# python copy_images.py --image_path /cluster/work/cvl/qimaqi/cvpr_2025/datasets/urban3d/Sci-Art/photos --dataset_path /cluster/work/cvl/qimaqi/cvpr_2025/datasets/urban3d/sci-art-pixsfm

cd /cluster/work/cvl/qimaqi/cvpr_2025_city/hierarchical-3d-gaussians/
# preprocess:
# srun --nodes=1 --ntasks=8 --cpus-per-task=1 --mem-per-cpu=8G --time=240 --gpus=rtx_3090:1 --pty bash -i 
# srun --nodes=1 --ntasks=24 --cpus-per-task=1 --mem-per-cpu=8G --time=240 --pty bash -i 
DATASET_DIR=/cluster/work/cvl/qimaqi/cvpr_2025/datasets/urban3d/colmap_results/sciart/train/

# python scripts/copy_images.py --image_path $RAW_PHOTO_PATH --dataset_path $CAMERA_POSE_PATH

# conda activate tool
# python preprocess/generate_chunks.py --project_dir /cluster/work/cvl/qimaqi/cvpr_2025/datasets/urban3d/colmap_results/sciart/train/ --min_n_cams 50 --max_n_cams 2500 --chunk_size 200  --skip_bundle_adjustment


# python train_coarse.py -s ${DATASET_DIR}/camera_calibration/aligned -i ${DATASET_DIR}/camera_calibration/rectified/images --skybox_num 100000 --position_lr_init 0.00016 --position_lr_final 0.0000016 --model_path ${DATASET_DIR}/output/scaffold --port 6012 --iterations 60000 
# CHUNK_NAMES=('2_0')

# for CHUNK_NAME in "${CHUNK_NAMES[@]}"; do 
#     python -u train_single.py -s ${DATASET_DIR}/camera_calibration/chunks/${CHUNK_NAME} --model_path ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME} -i ${DATASET_DIR}/camera_calibration/rectified/images  --scaffold_file ${DATASET_DIR}/output/scaffold/point_cloud/iteration_60000 --skybox_locked --bounds_file ${DATASET_DIR}/camera_calibration/chunks/${CHUNK_NAME}  --port 6008 --iterations 60000 --densify_until_iter 30000

#     submodules/gaussianhierarchy/build/GaussianHierarchyCreator ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME}/point_cloud/iteration_30000/point_cloud.ply ${DATASET_DIR}/camera_calibration/chunks/${CHUNK_NAME}  ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME}/ ${DATASET_DIR}/output/scaffold/point_cloud/iteration_60000

#     # post 
#     python -u train_post.py -s ${DATASET_DIR}/camera_calibration/chunks/${CHUNK_NAME} --model_path ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME} --hierarchy ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME}/hierarchy.hier --iterations 15000 --feature_lr 0.0005 --opacity_lr 0.01 --scaling_lr 0.001  -i ${DATASET_DIR}/camera_calibration/rectified/images --scaffold_file ${DATASET_DIR}/output/scaffold/point_cloud/iteration_60000 --skybox_locked --bounds_file ${DATASET_DIR}/camera_calibration/chunks/${CHUNK_NAME} --port 6008 
# done


submodules/gaussianhierarchy/build/GaussianHierarchyMerger /cluster/work/cvl/qimaqi/cvpr_2025/datasets/urban3d/colmap_results/sciart/train/output/trained_chunks "0" /cluster/work/cvl/qimaqi/cvpr_2025/datasets/urban3d/colmap_results/sciart/train/camera_calibration/chunks /cluster/work/cvl/qimaqi/cvpr_2025/datasets/urban3d/colmap_results/sciart/train/output/merged.hier 1_0 2_0 

