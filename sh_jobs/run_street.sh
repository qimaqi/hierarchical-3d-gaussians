#!/bin/bash
#SBATCH --job-name=single_gpu_pose_est
#SBATCH --nodelist=gcpl4-eu-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=l4-24g:1
#SBATCH --output=./joblogs/h3dgs_matrix_city_street_%j.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/h3dgs_matrix_city_street_%j.error     # Redirect stderr to a separate error log file
#SBATCH --time=12:00:00


# cuda
# pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
export CUDA_HOME=/opt/modules/nvidia-cuda-11.8.0
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.8.0/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.8.0/bin:$PATH
export PATH=/opt/modules/gcc-10.5.0/bin:$PATH
DATASET_DIR=/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug
# /data/work-gcp-europe-west4-a/qimaqi/datasets/small_city
source ~/.bashrc 
micromamba activate h3dgs
cd ..
# python scripts/full_train.py --project_dir ${DATASET_DIR}

python train_coarse.py -s /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_all_th50/camera_calibration/aligned -i /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_all_th50/camera_calibration/rectified/images --position_lr_init 0 --position_lr_final 0 --model_path /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_all_th50/output/scaffold_nodepth --port 6009 --iterations 50000 --skybox_num 100000

# python train_coarse.py -s /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug/camera_calibration/aligned -i /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug/camera_calibration/rectified/images  --position_lr_init 0 --position_lr_final 0 --model_path /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug/output/scaffold_depth --port 6011 --depths /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug/camera_calibration/rectified/depths

# python train_coarse.py -s /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug/camera_calibration/aligned -i /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug/camera_calibration/rectified/images --position_lr_init 0 --position_lr_final 0 --model_path /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug/output/scaffold_nodepth_sky --port 6009 --iterations 50000 --skybox_num 100000

# python -u train_single.py -s  ${DATASET_DIR}/camera_calibration/chunks/2_3 --model_path  ${DATASET_DIR}/output/chunks/2_3 -i /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug/camera_calibration/rectified/images -d /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug/camera_calibration/rectified/depths  --scaffold_file /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug/output/scaffold_nodepth_sky/point_cloud/iteration_50000 --skybox_locked --bounds_file  ${DATASET_DIR}/camera_calibration/chunks/2_3 --port 6012

# submodules/gaussianhierarchy/build/GaussianHierarchyCreator ${DATASET_DIR}/output/chunks/2_3/point_cloud/iteration_30000/point_cloud.ply ${DATASET_DIR}/camera_calibration/chunks/2_3  ${DATASET_DIR}/output/chunks/2_3/


# python train_post.py -s ${DATASET_DIR}/camera_calibration/chunks/2_3 --model_path ${DATASET_DIR}/output/chunks/2_3 --hierarchy ${DATASET_DIR}/output/chunks/2_3/hierarchy.hier --iterations 15000 --feature_lr 0.0005 --opacity_lr 0.01 --scaling_lr 0.001 --eval

# python render_hierarchy.py -s ${DATASET_DIR}/camera_calibration/chunks/2_3 --model_path ${DATASET_DIR}/output/chunks/2_3 --hierarchy ${DATASET_DIR}/output/chunks/2_3/hierarchy.hier_opt --out_dir ${DATASET_DIR}/output/chunks/2_3 --eval


# submodules/gaussianhierarchy/build/GaussianHierarchyCreator ${OUTPUT_DIR}/point_cloud/iteration_30000/point_cloud.ply ${CHUNK_DIR}  ${OUTPUT_DIR} 




# python scripts/full_train.py --project_dir /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap

# MODEL=mambaout_tiny
# python3 validate.py /data/imagenet  --model $MODEL -b 128 \
#   --pretrained

# srun --nodes=1 -G1 --ntasks=1 --cpus-per-task=12 --mem=48G  --time=04:00:00  --gpus=l4-24g:1  --nodelist=gcpl4-eu-3  --pty bash -i 
