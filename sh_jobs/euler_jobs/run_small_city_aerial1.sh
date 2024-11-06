#!/bin/bash
#SBATCH --job-name=single_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=8G
#SBATCH --output=./joblogs/aerial_split1_%j.log      # Redirect stdout to a log file
#SBATCH --time=48:00:00
#SBATCH --gpus=rtx_3090:1

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


# preprocess:
# colmap model_converter --input_path=/cluster/work/cvl/qimaqi/cvpr_2025/datasets/MatrixCity/colmap_aerial/camera_calibration/rectified/sparse/known --output_path=/cluster/work/cvl/qimaqi/cvpr_2025/datasets/MatrixCity/colmap_aerial/camera_calibration/rectified/sparse/0 --output_type=TXT


# python preprocess/generate_chunks.py --project_dir /cluster/work/cvl/qimaqi/cvpr_2025/datasets/MatrixCity/colmap_street/
cd /cluster/work/cvl/qimaqi/cvpr_2025_city/hierarchical-3d-gaussians/

# python train_coarse.py -s ${DATASET_DIR}/camera_calibration/aligned -i ${DATASET_DIR}/camera_calibration/rectified/images --skybox_num 0 --position_lr_init 0.0 --position_lr_final 0.0 --model_path ${DATASET_DIR}/output/scaffold --port 6016 

CHUNK_NAMES=('2_5' '2_6' '2_7' '3_1' '3_2' '3_3' '3_4' '3_5' '3_6' '3_7' '4_0' '4_1')


# 0_2  1_2  1_5  2_2  2_5  3_1  3_4  3_7  4_2  4_5  5_0  5_3  5_6  6_1  6_4  6_7  7_2  7_5  8_0  8_3  8_6  9_3
# 0_3  1_3  1_6  2_3  2_6  3_2  3_5  4_0  4_3  4_6  5_1  5_4  5_7  6_2  6_5  7_0  7_3  7_6  8_1  8_4  9_1  9_4
# 0_4  1_4  2_1  2_4  2_7  3_3  3_6  4_1  4_4  4_7  5_2  5_5  6_0  6_3  6_6  7_1  7_4  7_7  8_2  8_5  9_2  9_5
# srun --nodes=1 --ntasks=16 --cpus-per-task=1 --mem-per-cpu=8G --time=240 --pty bash -i

# DATASET_DIR=/cluster/work/cvl/qimaqi/cvpr_2025/datasets/MatrixCity/colmap_street/

# python train_coarse.py -s ${DATASET_DIR}/camera_calibration/aligned -i ${DATASET_DIR}/camera_calibration/rectified/images --skybox_num 100000 --position_lr_init 0.0 --position_lr_final 0.0 --model_path ${DATASET_DIR}/output/scaffold --port 6011

DATASET_DIR=/cluster/work/cvl/qimaqi/cvpr_2025/datasets/MatrixCity/colmap_aerial/

for CHUNK_NAME in "${CHUNK_NAMES[@]}"; do 
    python -u train_single.py -s ${DATASET_DIR}/camera_calibration/chunks/${CHUNK_NAME} --model_path ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME} -i ${DATASET_DIR}/camera_calibration/rectified/images  --scaffold_file ${DATASET_DIR}/output/scaffold/point_cloud/iteration_30000 --skybox_locked --bounds_file ${DATASET_DIR}/camera_calibration/chunks/${CHUNK_NAME}  --port 6016 -d ${DATASET_DIR}/camera_calibration/rectified/depths --iterations 60000 --densify_until_iter 30000

    submodules/gaussianhierarchy/build/GaussianHierarchyCreator ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME}/point_cloud/iteration_30000/point_cloud.ply ${DATASET_DIR}/camera_calibration/chunks/${CHUNK_NAME}  ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME}/ ${DATASET_DIR}/output/scaffold/point_cloud/iteration_30000

    # post 
    python -u train_post.py -s ${DATASET_DIR}/camera_calibration/chunks/${CHUNK_NAME} --model_path ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME} --hierarchy ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME}/hierarchy.hier --iterations 15000 --feature_lr 0.0005 --opacity_lr 0.01 --scaling_lr 0.001  -i ${DATASET_DIR}/camera_calibration/rectified/images --scaffold_file ${DATASET_DIR}/output/scaffold/point_cloud/iteration_30000 --skybox_locked --bounds_file ${DATASET_DIR}/camera_calibration/chunks/${CHUNK_NAME} --port 6016 -d ${DATASET_DIR}/camera_calibration/rectified/depths    
done




# # merge final hierarchy
# submodules/gaussianhierarchy/build/GaussianHierarchyMerger /cluster/work/cvl/qimaqi/cvpr_2025/datasets/urban3d/colmap_results/residence/residence-pixsfm/train/output/trained_chunks "0" /cluster/work/cvl/qimaqi/cvpr_2025/datasets/urban3d/colmap_results/residence/residence-pixsfm/train/camera_calibration/chunks cluster/work/cvl/qimaqi/cvpr_2025/datasets/urban3d/colmap_results/residence/residence-pixsfm/train/output/merged.hier ["0_0","1_0","2_0","3_0"]

