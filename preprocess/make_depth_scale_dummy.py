#
# Copyright (C) 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import argparse
import cv2
from joblib import delayed, Parallel
import json
from read_write_model import *

def get_scales(key, cameras, images, args):
    image_meta = images[key]
    scale = 1.0
    offset = 0.0
    n_remove = len(image_meta.name.split('.')[-1]) + 1
    return {"image_name": image_meta.name[:-n_remove], "scale": scale, "offset": offset}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--chunks_dir', required=True)
    # parser.add_argument('--depths_dir', required=True)
    parser.add_argument('--model_type', default="bin")
    args = parser.parse_args()

    chunk_names = os.listdir(args.chunks_dir)
    if 'sparse' in chunk_names:
        chunk_names = ['']
    for chunk_name in chunk_names:
        args.base_dir = os.path.join(args.chunks_dir, chunk_name)
        print("args.base_dir", args.base_dir)

                
        cam_intrinsics, images_metas, points3d = read_model(os.path.join(args.base_dir, "sparse", "0"), ext=f".{args.model_type}")

        # pts_indices = np.array([points3d[key].id for key in points3d])
        # pts_xyzs = np.array([points3d[key].xyz for key in points3d])
        # points3d_ordered = np.zeros([pts_indices.max()+1, 3])
        # points3d_ordered[pts_indices] = pts_xyzs

        # depth_param_list = [get_scales(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas]
        depth_param_list = Parallel(n_jobs=-1, backend="threading")(
            delayed(get_scales)(key, cam_intrinsics, images_metas, args) for key in images_metas
        )

        depth_params = {
            depth_param["image_name"]: {"scale": depth_param["scale"], "offset": depth_param["offset"]}
            for depth_param in depth_param_list if depth_param != None
        }

        with open(f"{args.base_dir}/sparse/0/depth_params.json", "w") as f:
            json.dump(depth_params, f, indent=2)

        print(0)
