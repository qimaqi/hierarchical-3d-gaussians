#!/bin/bash
#SBATCH --job-name=single_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
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
# cd /cluster/work/cvl/qimaqi/cvpr_2025_city/hierarchical-3d-gaussians/preprocess

# conda activate tool
# python json_to_colmap.py --base_dir=/cluster/work/cvl/qimaqi/cvpr_2025/datasets/MatrixCity/big_city/colmap_aerial/camera_calibration/rectified --view_name=aerial --pc_path=/cluster/work/cvl/qimaqi/cvpr_2025/datasets/MatrixCity/big_city_pc/big_city_pointcloud_aerial/Block_all_.ply

# cp -r /cluster/work/cvl/qimaqi/cvpr_2025/datasets/MatrixCity/big_city/colmap_aerial/camera_calibration/rectified/sparse/ /cluster/work/cvl/qimaqi/cvpr_2025/datasets/MatrixCity/big_city/colmap_aerial/camera_calibration/aligned 


# colmap model_converter --input_path /cluster/work/cvl/qimaqi/cvpr_2025/datasets/MatrixCity/big_city/colmap_aerial/camera_calibration/aligned/sparse/mc --output_path=/cluster/work/cvl/qimaqi/cvpr_2025/datasets/MatrixCity/big_city/colmap_aerial/camera_calibration/aligned/sparse/mc --output_type=BIN

cd /cluster/work/cvl/qimaqi/cvpr_2025_city/hierarchical-3d-gaussians/

# python preprocess/generate_chunks_mc.py --mc_dir /cluster/work/cvl/qimaqi/cvpr_2025/datasets/MatrixCity/big_city/colmap_aerial/  --min_n_cams 100 --chunk_size 200
# cd /cluster/work/cvl/qimaqi/cvpr_2025_city/hierarchical-3d-gaussians/
# python preprocess/generate_chunks.py --project_dir /cluster/work/cvl/qimaqi/cvpr_2025/datasets/MatrixCity/colmap_street/ --min_n_cams 20 --chunk_size 200  --skip_bundle_adjustment
# CHUNK_NAMES=(0_10 0_11 0_12 0_13 0_14 0_15 0_16 0_17 0_18 0_19 0_20 0_21 0_22 0_23 0_24 0_25 0_7 0_8 0_9 10_0 10_1 10_10 10_11 10_12 10_13 10_14 10_15 10_16 10_17 10_18 10_19 10_2 10_20 10_21 10_22 10_23 10_24 10_25 10_3 10_4 10_5 10_6 10_7 10_8 10_9 11_0 11_1 11_10 11_11 11_12 11_13 11_14 11_15 11_16 11_17 11_18 11_19 11_2 11_20 11_21 11_22 11_23 11_24 11_25 11_3 11_4 11_5 11_6 11_7 11_8 11_9 12_0 12_1 12_10 12_11 12_12 12_13 12_14 12_15 12_16 12_17 12_18 12_19 12_2 12_20 12_21 12_22 12_23 12_24 12_25 12_3 12_4 12_5 12_6 12_7 12_8 12_9 13_0 13_1 13_10 13_11 13_12 13_13 13_14 13_15 13_16 13_17 13_18 13_19 13_2 13_20 13_21 13_22 13_23 13_24 13_25 13_3 13_4 13_5 13_6 13_7 13_8 13_9 14_0 14_1 14_10 14_11 14_12 14_13 14_14 14_15 14_16 14_17 14_18 14_19 14_2 14_20 14_21 14_22 14_23 14_24 14_25 14_3 14_4 14_5 14_6 14_7 14_8 14_9 15_0 15_1 15_10 15_11 15_12 15_13 15_14 15_15 15_16 15_17 15_18 15_19 15_2 15_20 15_21 15_22 15_23 15_24 15_25 15_3 15_4 15_5 15_6 15_7 15_8 15_9 16_0 16_1 16_10 16_11 16_12 16_13 16_14 16_15 16_16 16_17 16_18 16_19 16_2 16_20 16_21 16_22 16_23 16_24 16_25 16_3 16_4 16_5 16_6 16_7 16_8 16_9 17_0 17_1 17_10 17_11 17_12 17_13 17_14 17_15 17_16 17_17 17_18 17_19 17_2 17_20 17_21 17_22 17_23 17_24 17_25 17_3 17_4 17_5 17_6 17_7 17_8 17_9 18_0 18_1 18_10 18_11 18_12 18_13 18_14 18_15 18_16 18_17 18_18 18_19 18_2 18_20 18_21 18_22 18_23 18_24 18_25 18_3 18_4 18_5 18_6 18_7 18_8 18_9 19_0 19_1 19_10 19_11 19_12 19_13 19_14 19_15 19_16 19_17 19_18 19_19 19_2 19_20 19_21 19_22 19_23 19_24 19_25 19_3 19_4 19_5 19_6 19_7 19_8 19_9 1_10 1_11 1_12 1_13 1_14 1_15 1_16 1_17 1_18 1_19 1_20 1_21 1_22 1_23 1_24 1_25 1_7 1_8 1_9 20_0 20_1 20_10 20_11 20_12 20_13 20_14 20_15 20_16 20_17 20_18 20_19 20_2 20_20 20_21 20_22 20_23 20_24 20_25 20_3 20_4 20_5 20_6 20_7 20_8 20_9 21_0 21_1 21_10 21_11 21_12 21_13 21_14 21_15 21_16 21_17 21_18 21_19 21_2 21_20 21_21 21_22 21_23 21_24 21_25 21_3 21_4 21_5 21_6 21_7 21_8 21_9 2_10 2_11 2_12 2_13 2_14 2_15 2_16 2_17 2_18 2_19 2_20 2_21 2_22 2_23 2_24 2_25 2_7 2_8 2_9 3_10 3_11 3_12 3_13 3_14 3_15 3_16 3_17 3_18 3_19 3_20 3_21 3_22 3_23 3_24 3_25 3_7 3_8 3_9 4_10 4_11 4_12 4_13 4_14 4_15 4_16 4_17 4_18 4_19 4_2 4_20 4_21 4_22 4_23 4_24 4_25 4_3 4_4 4_5 4_6 4_7 4_8 4_9 5_10 5_11 5_12 5_13 5_14 5_15 5_16 5_17 5_18 5_19 5_2 5_20 5_21 5_22 5_23 5_24 5_25 5_3 5_4 5_5 5_6 5_7 5_8 5_9 6_0 6_1 6_10 6_11 6_12 6_13 6_14 6_15 6_16 6_17 6_18 6_19 6_2 6_20 6_21 6_22 6_23 6_24 6_25 6_3 6_4 6_5 6_6 6_7 6_8 6_9 7_0 7_1 7_10 7_11 7_12 7_13 7_14 7_15 7_16 7_17 7_18 7_19 7_2 7_20 7_21 7_22 7_23 7_24 7_25 7_3 7_4 7_5 7_6 7_7 7_8 7_9 8_0 8_1 8_10 8_11 8_12 8_13 8_14 8_15 8_16 8_17 8_18 8_19 8_2 8_20 8_21 8_22 8_23 8_24 8_25 8_3 8_4 8_5 8_6 8_7 8_8 8_9 9_0 9_1 9_10 9_11 9_12 9_13 9_14 9_15 9_16 9_17 9_18 9_19 9_2 9_20 9_21 9_22 9_23 9_24 9_25 9_3 9_4 9_5 9_6 9_7 9_8 9_9)

# CHUNK_NAMES=(0_10 0_11 0_12 0_13 0_14 0_15 0_16 0_17 0_18 0_19 0_20 0_21 0_22 0_23 0_24 0_25 0_7 0_8 0_9)
CHUNK_NAMES=(5_9)
DATASET_DIR=/cluster/work/cvl/qimaqi/cvpr_2025/datasets/MatrixCity/big_city/colmap_aerial/
# python train_coarse.py -s ${DATASET_DIR}/camera_calibration/aligned -i ${DATASET_DIR}/camera_calibration/rectified/images --skybox_num 0 --position_lr_init 0.0 --position_lr_final 0.0 --model_path ${DATASET_DIR}/output/scaffold --port 6007

for CHUNK_NAME in "${CHUNK_NAMES[@]}"; do 
    python -u train_single.py -s ${DATASET_DIR}/camera_calibration/chunks_mc_bak/${CHUNK_NAME} --model_path ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME} -i ${DATASET_DIR}/camera_calibration/rectified/images  --scaffold_file ${DATASET_DIR}/output/scaffold/point_cloud/iteration_30000 --skybox_locked --bounds_file ${DATASET_DIR}/camera_calibration/chunks_mc_bak/${CHUNK_NAME}  --port 6007 -d ${DATASET_DIR}/camera_calibration/rectified/depths --iterations 60000 --densify_until_iter 30000

    submodules/gaussianhierarchy/build/GaussianHierarchyCreator ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME}/point_cloud/iteration_30000/point_cloud.ply ${DATASET_DIR}/camera_calibration/chunks_mc/${CHUNK_NAME}  ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME}/ ${DATASET_DIR}/output/scaffold/point_cloud/iteration_30000

    # post 
    python -u train_post.py -s ${DATASET_DIR}/camera_calibration/chunks_mc/${CHUNK_NAME} --model_path ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME} --hierarchy ${DATASET_DIR}/output/trained_chunks/${CHUNK_NAME}/hierarchy.hier --iterations 15000 --feature_lr 0.0005 --opacity_lr 0.01 --scaling_lr 0.001  -i ${DATASET_DIR}/camera_calibration/rectified/images --scaffold_file ${DATASET_DIR}/output/scaffold/point_cloud/iteration_30000 --skybox_locked --bounds_file ${DATASET_DIR}/camera_calibration/chunks_mc/${CHUNK_NAME} --port 6007 -d ${DATASET_DIR}/camera_calibration/rectified/depths    
done

# let's visualize the results of 0_2

# python render_hierarchy.py -s ${CHUNK_DIR} --model_path ${OUTPUT_DIR} --hierarchy ${OUTPUT_DIR}/hierarchy.hier_opt --out_dir ${OUTPUT_DIR} --eval


# merge final hierarchy
# submodules/gaussianhierarchy/build/GaussianHierarchyMerger /cluster/work/cvl/qimaqi/cvpr_2025/datasets/urban3d/colmap_results/residence/residence-pixsfm/train/output/trained_chunks "0" /cluster/work/cvl/qimaqi/cvpr_2025/datasets/urban3d/colmap_results/residence/residence-pixsfm/train/camera_calibration/chunks cluster/work/cvl/qimaqi/cvpr_2025/datasets/urban3d/colmap_results/residence/residence-pixsfm/train/output/merged.hier ["0_0","1_0","2_0","3_0"]

