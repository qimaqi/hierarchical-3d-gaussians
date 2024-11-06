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
import os
import random
from read_write_model import *
import json

def get_nb_pts(image_metas):
    n_pts = 0
    for key in image_metas:
        pts_idx = image_metas[key].point3D_ids
        if(len(pts_idx) > 5):
            n_pts = max(n_pts, np.max(pts_idx))

    return n_pts + 1

if __name__ == '__main__':
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--depths_dir', required=True)
    parser.add_argument('--chunk_size', default=100, type=float)
    parser.add_argument('--min_padd', default=0.2, type=float)
    parser.add_argument('--lapla_thresh', default=1, type=float, help="Discard images if their laplacians are < mean - lapla_thresh * std") # 1
    parser.add_argument('--min_n_cams', default=100, type=int) # 100
    parser.add_argument('--max_n_cams', default=3000, type=int) # 1500
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--add_far_cams', default=True)
    parser.add_argument('--model_type', default="bin")

    args = parser.parse_args()

    # eval
    test_file = f"{args.base_dir}/test.txt"
    if os.path.exists(test_file):
        with open(test_file, 'r') as file:
            test_cam_names_list = file.readlines()
            blending_dict = {name[:-1] if name[-1] == '\n' else name: {} for name in test_cam_names_list}

    cam_intrinsics, images_metas, points3d = read_model(args.base_dir, ext=f".{args.model_type}")

    # data basic info
    xyzs = np.zeros((len(points3d), 3), dtype=np.float32)
    indices = np.zeros(len(points3d), dtype=np.int32)
    errors = np.zeros(len(points3d), dtype=np.float32)
    colors = np.zeros((len(points3d), 3), dtype=np.float32)
    idx = 0
    for key in points3d:
        # print("points3d[key].point2D_idxs",type(points3d[key].point2D_idxs), np.shape(points3d[key].point2D_idxs))
        # different points data properties
        xyzs[idx] = points3d[key].xyz # 3d world points
        indices[idx] = points3d[key].id # 
        errors[idx] = points3d[key].error
        colors[idx] = points3d[key].rgb
        # n_images[idx] = len(points3d[key].image_ids)
        # # indices_2d[idx] = points3d[key].point2D_idxs
        # indices_2d.append(points3d[key].point2D_idxs)
        # # image_ids[idx] = points3d[key].image_ids
        # image_ids.append(points3d[key].image_ids)
        idx +=1

    print("images_metas keys", len(images_metas.keys())) 
    print("xyzs range", np.min(xyzs[:, 0]), np.max(xyzs[:, 0]), np.min(xyzs[:, 1]), np.max(xyzs[:, 1]), np.min(xyzs[:, 2]), np.max(xyzs[:, 2]))


    cam_centers = np.array([
        -qvec2rotmat(images_metas[key].qvec).astype(np.float32).T @ images_metas[key].tvec.astype(np.float32)
        for key in images_metas
    ])


    global_bbox = np.stack([cam_centers.min(axis=0), cam_centers.max(axis=0)])
    global_bbox[0, :2] -= args.min_padd * args.chunk_size
    global_bbox[1, :2] += args.min_padd * args.chunk_size
    extent = global_bbox[1] - global_bbox[0]
    padd = np.array([args.chunk_size - extent[0] % args.chunk_size, args.chunk_size - extent[1] % args.chunk_size])
    global_bbox[0, :2] -= padd / 2
    global_bbox[1, :2] += padd / 2

    global_bbox[0, 2] = -1e12
    global_bbox[1, 2] = 1e12

    # def get_var_of_laplacian(key):
    #     image = cv2.imread(os.path.join(args.images_dir, images_metas[key].name))
    #     if image is not None:
    #         gray = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)
    #         return cv2.Laplacian(gray, cv2.CV_32F).var()
    #     else:
    #         return 0   
        
    # if args.lapla_thresh > 0: 
    #     laplacians = Parallel(n_jobs=-1, backend="threading")(
    #         delayed(get_var_of_laplacian)(key) for key in images_metas
    #     )
    #     laplacians_dict = {key: laplacian for key, laplacian in zip(images_metas, laplacians)}

    excluded_chunks = []
    chunks_pcd = {}
    
    def make_chunk(i, j, n_width, n_height):
        # in_path = f"{args.base_dir}/chunk_{i}_{j}"
        # if os.path.exists(in_path):
        print(f"chunk {i}_{j}")
        # corner_min, corner_max = bboxes[i, j, :, 0], bboxes[i, j, :, 1]
        corner_min = global_bbox[0] + np.array([i * args.chunk_size, j * args.chunk_size, 0])
        corner_max = global_bbox[0] + np.array([(i + 1) * args.chunk_size, (j + 1) * args.chunk_size, 1e12])
        corner_min[2] = -1e12
        corner_max[2] = 1e12
        
        corner_min_for_pts = corner_min.copy()
        corner_max_for_pts = corner_max.copy()
        if i == 0:
            corner_min_for_pts[0] = -1e12
        if j == 0:
            corner_min_for_pts[1] = -1e12
        if i == n_width - 1:
            corner_max_for_pts[0] = 1e12
        if j == n_height - 1:
            corner_max_for_pts[1] = 1e12

        valid_cam = np.all(cam_centers < corner_max, axis=-1) * np.all(cam_centers > corner_min, axis=-1) # cam

        box_center = (corner_max + corner_min) / 2
        extent = (corner_max - corner_min) / 2
        acceptable_radius = 2
        extended_corner_min = box_center - acceptable_radius * extent
        extended_corner_max = box_center + acceptable_radius * extent

        print("corner pos", corner_max_for_pts, corner_min_for_pts, corner_max, corner_min)

        for cam_idx, key in enumerate(images_metas):
            # If within chunk
            if np.all(cam_centers[cam_idx] < corner_max) and np.all(cam_centers[cam_idx] > corner_min):
                valid_cam[cam_idx] = True
                # print("get valid cam", n_pts)
            # If within 2x of the chunk
            elif np.all(cam_centers[cam_idx] < extended_corner_max) and np.all(cam_centers[cam_idx] > extended_corner_min):
                valid_cam[cam_idx] = random.uniform(0, 1) > 0.5
                # add soming camera in extend to valid
                # print("get valid cam in extend area", n_pts)
            # All distances
            # if (not valid_cam[cam_idx]) and args.add_far_cams:
            #     valid_cam[cam_idx] = random.uniform(0, 0.5) < (float(n_pts) / len(image_points3d))
            #     # print("get valid cam in all area", n_pts)
            # if in the chunk, we assume to be valid

        # get xyz which in the extended_corner_min and extended_corner_max
        valid_pts = np.arange(len(points3d))[(xyzs < corner_max_for_pts).all(axis=-1) * (xyzs > corner_min_for_pts).all(axis=-1)]
        print(f"{len(valid_pts)} valid points", valid_pts.shape)
        new_indices = indices[valid_pts]
        new_xyzs = xyzs[valid_pts]
        new_errors = errors[valid_pts]
        new_colors = colors[valid_pts]
        new_colors = np.clip(new_colors, 0, 255).astype(np.uint8)

            
        print(f"{valid_cam.sum()} valid cameras after visibility-base selection")
        # if args.lapla_thresh > 0:
        #     chunk_laplacians = np.array([laplacians_dict[key] for cam_idx, key in enumerate(images_metas) if valid_cam[cam_idx]])
        #     laplacian_mean = chunk_laplacians.mean()
        #     laplacian_std_dev = chunk_laplacians.std()
        #     for cam_idx, key in enumerate(images_metas):
        #         if valid_cam[cam_idx] and laplacians_dict[key] < (laplacian_mean - args.lapla_thresh * laplacian_std_dev):
        #             # print("image", key, "is blurry")
        #             # print("images_metas", images_metas.keys(), len(images_metas.keys()))
        #             # image = cv2.imread(f"{args.base_dir}/images/{images_metas[str(cam_idx)]['name']}")
        #             # cv2.imshow("blurry", image)
        #             # cv2.waitKey(0)
        #             # cv2.imwrite(f"{args.output_path}/blurry/{images_metas[key]['name']}", image)
        #             # cv2.imwrite(f"./blurry/{images_metas[key].name}", image)
        #             valid_cam[cam_idx] = False

        #     print(f"{valid_cam.sum()} after Laplacian")

        if valid_cam.sum() > args.max_n_cams:
            for _ in range(valid_cam.sum() - args.max_n_cams):
                remove_idx = random.randint(0, valid_cam.sum() - 1)
                remove_idx_glob = np.arange(len(valid_cam))[valid_cam][remove_idx]
                valid_cam[remove_idx_glob] = False

            print(f"{valid_cam.sum()} after random removal")

        valid_keys = [key for idx, key in enumerate(images_metas) if valid_cam[idx]]
        
        if valid_cam.sum() > args.min_n_cams:# or init_valid_cam.sum() > 0:
            out_path = os.path.join(args.output_path, f"{i}_{j}")
            out_colmap = os.path.join(out_path, "sparse", "0")
            os.makedirs(out_colmap, exist_ok=True)

            # must remove sfm points to use colmap triangulator in following steps
            images_out = {}
            for key in valid_keys:
                image_meta = images_metas[key]
                images_out[key] = Image(
                    id = key,
                    qvec = image_meta.qvec,
                    tvec = image_meta.tvec,
                    camera_id = image_meta.camera_id,
                    name = image_meta.name,
                    xys = image_meta.xys,
                    point3D_ids = image_meta.point3D_ids
                    # xys = [],
                    # point3D_ids = []
                )
                # here it create image file with no 3d points

                if os.path.exists(test_file) and image_meta.name in blending_dict:
                    n_pts = np.isin(image_meta.point3D_ids, new_indices).sum()
                    blending_dict[image_meta.name][f"{i}_{j}"] = str(n_pts)

            # sanity check
            # print("new_image_ids[idx]", new_image_ids[idx].shape)
            points_out = {
                new_indices[idx] : Point3D(
                        id=new_indices[idx],
                        xyz= new_xyzs[idx],
                        rgb=new_colors[idx],
                        error=new_errors[idx],
                        image_ids=np.array([]),
                        point2D_idxs=np.array([]),
   
                    )
                for idx in range(len(new_xyzs))
            }

            write_model(cam_intrinsics, images_out, points_out, out_colmap, f".{args.model_type}")

            with open(os.path.join(out_path, "center.txt"), 'w') as f:
                f.write(' '.join(map(str, (corner_min + corner_max) / 2)))
            with open(os.path.join(out_path, "extent.txt"), 'w') as f:
                f.write(' '.join(map(str, corner_max - corner_min)))
        else:
            excluded_chunks.append([i, j])
            print("Chunk excluded",valid_cam.sum() )

    extent = global_bbox[1] - global_bbox[0]
    n_width = round(extent[0] / args.chunk_size)
    n_height = round(extent[1] / args.chunk_size)

    for i in range(n_width):
        for j in range(n_height):
            make_chunk(i, j, n_width, n_height)

    if os.path.exists(test_file):
        with open(f"{args.base_dir}/blending_dict.json", "w") as f:
            json.dump(blending_dict, f, indent=2)