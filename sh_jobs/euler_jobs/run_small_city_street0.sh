#!/bin/bash
#SBATCH --job-name=single_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=./joblogs/smallcity_0_%j.log      # Redirect stdout to a log file
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


# preprocess:
# python preprocess/generate_chunks.py --project_dir /cluster/work/cvl/qimaqi/cvpr_2025/datasets/MatrixCity/colmap_street/
cd /cluster/work/cvl/qimaqi/cvpr_2025_city/hierarchical-3d-gaussians/
# python preprocess/generate_chunks.py --project_dir /cluster/work/cvl/qimaqi/cvpr_2025/datasets/MatrixCity/colmap_street/ --min_n_cams 20 --chunk_size 200  --skip_bundle_adjustment
CHUNK_NAMES=(0_1  0_2  0_3  0_4  0_5  1_1  1_2  1_3  1_4  1_5  1_6)
# ("0_2")
# 
DATASET_DIR=/cluster/work/cvl/qimaqi/cvpr_2025/datasets/MatrixCity/colmap_street/
# python make_depth_scale_dummy.py --chunks_dir=/cluster/work/cvl/qimaqi/cvpr_2025/datasets/MatrixCity/colmap_street/camera_calibration/chunks_mc/
# python train_coarse.py -s ${DATASET_DIR}/camera_calibration/aligned -i ${DATASET_DIR}/camera_calibration/rectified/images --skybox_num 100000 --position_lr_init 0.0 --position_lr_final 0.0 --model_path ${DATASET_DIR}/output/scaffold --port 6010 

# CHUNK_NAME=0_2 
# python -u train_single.py -s ${DATASET_DIR}/camera_calibration/chunks_mc/${CHUNK_NAME} --model_path ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME} -i ${DATASET_DIR}/camera_calibration/rectified/images  --scaffold_file ${DATASET_DIR}/output/scaffold/point_cloud/iteration_30000 --skybox_locked --bounds_file ${DATASET_DIR}/camera_calibration/chunks_mc/${CHUNK_NAME}  --port 6010 -d ${DATASET_DIR}/camera_calibration/rectified/depths --iterations 60000 --densify_until_iter 30000

# # python train_post.py -s ${CHUNK_DIR} --model_path ${OUTPUT_DIR} --hierarchy ${OUTPUT_DIR}/hierarchy.hier --iterations 15000 --feature_lr 0.0005 --opacity_lr 0.01 --scaling_lr 0.001 --eval

# python render_hierarchy.py -s ${DATASET_DIR}/camera_calibration/chunks_mc/${CHUNK_NAME} --model_path ${DATASET_DIR}/output/ --hierarchy ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME}/hierarchy.hier_opt --out_dir ${DATASET_DIR}/output/render_0_2 --eval



for CHUNK_NAME in "${CHUNK_NAMES[@]}"; do 
    python -u train_single.py -s ${DATASET_DIR}/camera_calibration/chunks_mc/${CHUNK_NAME} --model_path ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME} -i ${DATASET_DIR}/camera_calibration/rectified/images  --scaffold_file ${DATASET_DIR}/output/scaffold/point_cloud/iteration_30000 --skybox_locked --bounds_file ${DATASET_DIR}/camera_calibration/chunks_mc/${CHUNK_NAME}  --port 6010 -d ${DATASET_DIR}/camera_calibration/rectified/depths --iterations 60000 --densify_until_iter 30000

    submodules/gaussianhierarchy/build/GaussianHierarchyCreator ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME}/point_cloud/iteration_30000/point_cloud.ply ${DATASET_DIR}/camera_calibration/chunks_mc/${CHUNK_NAME}  ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME}/ ${DATASET_DIR}/output/scaffold/point_cloud/iteration_30000

    # post 
    python -u train_post.py -s ${DATASET_DIR}/camera_calibration/chunks_mc/${CHUNK_NAME} --model_path ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME} --hierarchy ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME}/hierarchy.hier --iterations 15000 --feature_lr 0.0005 --opacity_lr 0.01 --scaling_lr 0.001  -i ${DATASET_DIR}/camera_calibration/rectified/images --scaffold_file ${DATASET_DIR}/output/scaffold/point_cloud/iteration_30000 --skybox_locked --bounds_file ${DATASET_DIR}/camera_calibration/chunks_mc/${CHUNK_NAME} --port 6010 -d ${DATASET_DIR}/camera_calibration/rectified/depths    
done

# let's visualize the results of 0_2

# python render_hierarchy.py -s ${CHUNK_DIR} --model_path ${OUTPUT_DIR} --hierarchy ${OUTPUT_DIR}/hierarchy.hier_opt --out_dir ${OUTPUT_DIR} --eval


# merge final hierarchy
# submodules/gaussianhierarchy/build/GaussianHierarchyMerger /cluster/work/cvl/qimaqi/cvpr_2025/datasets/urban3d/colmap_results/residence/residence-pixsfm/train/output/trained_chunks "0" /cluster/work/cvl/qimaqi/cvpr_2025/datasets/urban3d/colmap_results/residence/residence-pixsfm/train/camera_calibration/chunks cluster/work/cvl/qimaqi/cvpr_2025/datasets/urban3d/colmap_results/residence/residence-pixsfm/train/output/merged.hier ["0_0","1_0","2_0","3_0"]

