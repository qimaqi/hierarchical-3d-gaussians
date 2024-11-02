#!/bin/bash
cd /home/qi_ma/cvpr_2025_remote_sensing/hierarchical-3d-gaussians/
export CUDA_HOME=/opt/modules/nvidia-cuda-11.8.0
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.8.0/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.8.0/bin:$PATH
export PATH=/opt/modules/gcc-10.5.0/bin:$PATH
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install trimesh
pip install hydra-core
pip install easydict==1.13
micromamba install conda-forge::colmap
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

