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
    parser.add_argument('--chunk_size', default=100, type=float)
    parser.add_argument('--min_padd', default=0.2, type=float)
    parser.add_argument('--lapla_thresh', default=1, type=float, help="Discard images if their laplacians are < mean - lapla_thresh * std") # 1
    parser.add_argument('--min_n_cams', default=100, type=int) # 100
    parser.add_argument('--max_n_cams', default=1500, type=int) # 1500
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--add_far_cams', default=True)
    parser.add_argument('--model_type', default="bin")
    parser.add_argument('--depth_dir')

    args = parser.parse_args()

    # eval
    test_file = f"{args.base_dir}/test.txt"
    if os.path.exists(test_file):
        with open(test_file, 'r') as file:
            test_cam_names_list = file.readlines()
            blending_dict = {name[:-1] if name[-1] == '\n' else name: {} for name in test_cam_names_list}

    cam_intrinsics, images_metas, points3d = read_model(args.base_dir, ext=f".{args.model_type}")

    # data basic info
    print("images_metas keys", len(images_metas.keys())) 
    print("points3d keys", len(points3d.keys()))


    cam_centers = np.array([
        -qvec2rotmat(images_metas[key].qvec).astype(np.float32).T @ images_metas[key].tvec.astype(np.float32)
        for key in images_metas])

    qvec_w2c = [(images_metas[key].qvec).astype(np.float32) for key in images_metas]
    tvec_w2c = [(images_metas[key].tvec).astype(np.float32) for key in images_metas]
    qvec_w2c = np.array(qvec_w2c)
    tvec_w2c = np.array(tvec_w2c)
    print("cam_centers", cam_centers.shape) # (100, 3)
    # print("qvec_w2c", qvec_w2c.shape) # (100, 4)
    # print("tvec_w2c", tvec_w2c.shape) # (100, 3)
    # [w x y z ] to [x y z w]
    qvec_w2c = qvec_w2c[:, [1, 2, 3, 0]]
    from scipy.spatial.transform import Rotation
    rotmat_w2c = Rotation.from_quat(qvec_w2c).as_matrix()
    # print("rotmat_w2c", rotmat_w2c.shape) # (100, 3, 3)
    

    # save cam centers to ply file for visualization
    # import trimesh
    # mesh = trimesh.Trimesh(vertices=cam_centers)
    # mesh.export('./cam_centers.ply')

    # get 4x4 pose from qvec and tvec


    n_pts = get_nb_pts(images_metas)
    print("Number of points: ", n_pts) #

    xyzs = np.zeros([n_pts, 3], np.float32)
    errors = np.zeros([n_pts], np.float32) + 9e9
    indices = np.zeros([n_pts], np.int64)
    n_images = np.zeros([n_pts], np.int64)
    colors = np.zeros([n_pts, 3], np.float32)

    idx = 0    
    # parse the points3d to get the xyzs, indices, errors, colors, n_images
    # each points3d have a key name
    # map all information to each list
    for key in points3d:
        # different points data properties
        xyzs[idx] = points3d[key].xyz # 3d world points
        indices[idx] = points3d[key].id # 
        errors[idx] = points3d[key].error
        colors[idx] = points3d[key].rgb
        n_images[idx] = len(points3d[key].image_ids)
        idx +=1
    # id do not matter, a lot of sparse exist

    mask = errors < 1e1
    print(f"Number of points with error < 1e1: {mask.sum()}") # 47936
    # how to add more points here

    # mask *= n_images > 3
    # xyzsC, colorsC, errorsC, indicesC, n_imagesC = xyzs[mask], colors[mask], errors[mask], indices[mask], n_images[mask]

    # save xyzs to ply file for visualization
    # import trimesh 
    # mesh = trimesh.Trimesh(vertices=xyzsC, vertex_colors=colorsC)
    # mesh.export('points_colmap.ply')

    # indicesC.max, the length of number of points
    # points3d_ordered = np.zeros([indicesC.max()+1, 3])
    # points3d_ordered[indicesC] = xyzsC # reoreder is masked out which have less errors
    images_points3d = {}

    '''
    main function to modify the images_metas
    '''
    # this operation give every image key its 3d points we can replace with depth map 

    def load_world_points_for_frame(depth_map_path, rot_w2c, t_w2c, intrinscs, type='aerial'):
        os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
        # TODO aerial intrinsic differnet from street
        if 'aerial' in depth_map_path:
            intrinsics =  np.array([[2317.6449482429634/4, 0, 960.0/4],
                                    [0, 2317.6449482429634/4, 540.0/4],
                                    [0, 0, 1]])
        elif 'street' in depth_map_path:
            intrinsics = np.array([[500., 0, 500.0],
                                    [0, 500., 500.0],
                                    [0, 0, 1]])
        else:
            raise ValueError("Unknown type")

        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        H,W, _  = depth_map.shape
        depth_map = depth_map[...,0]
        # using cv2 nearest interpoate to resize the depth map
        if 'aerial' in depth_map_path:
            depth_map = cv2.resize(depth_map, (W//4,H//4), interpolation=cv2.INTER_NEAREST)

        w2c_pose = np.eye(4)
        w2c_pose[:3, :3] = rot_w2c
        w2c_pose[:3, 3] = t_w2c

        c2w_pose = np.linalg.inv(w2c_pose)
        c2w_pose = c2w_pose[:3,:4]

        # 1. get the pixel coordinate
        H,W = depth_map.shape
        x = np.arange(W)
        y = np.arange(H)
        xx,yy = np.meshgrid(x,y)
        xx = xx.flatten()
        yy = yy.flatten()

        # 2. get the depth value
        depths = depth_map.flatten()

        points = np.vstack([xx,yy,np.ones_like(xx)])
        # 3xN
        points = np.multiply(points, depths)
        # 3xN
        points = np.dot(np.linalg.inv(intrinsics), points)
        # 3xN
        points = np.dot(c2w_pose, np.vstack([points, np.ones_like(xx)]))
        # 3xN
        points = points[:3,:]
        # Nx3
        points = points.T
        # print(points.shape)
        return points

    for count, key in enumerate(images_metas):
        depth_name = (str(images_metas[key].name)[-8:]).replace(".png", ".exr")
        depth_name = os.path.join(args.depth_dir, depth_name)
        # print("depth_name", depth_name)
        if os.path.exists(depth_name):
            points_3d = load_world_points_for_frame(depth_name, rotmat_w2c[count], tvec_w2c[count], cam_intrinsics)
        images_points3d[key] = points_3d
        # pts_idx = images_metas[key].point3D_ids 
        # mask = pts_idx >= 0 # what is the mask means here >= 0
        # mask *= pts_idx < len(points3d_ordered)
        # pts_idx = pts_idx[mask]
        # if len(pts_idx) > 0:
        #     image_points3d = points3d_ordered[pts_idx]
        #     mask = (image_points3d != 0).sum(axis=-1)
        #     # images_metas[key]["points3d"] = image_points3d[mask>0]
        #     # each images, add the 3d world points
        #     images_points3d[key] = image_points3d[mask>0]
        # else:
        #     # images_metas[key]["points3d"] = np.array([])
        #     images_points3d[key] = np.array([])


    global_bbox = np.stack([cam_centers.min(axis=0), cam_centers.max(axis=0)])
    global_bbox[0, :2] -= args.min_padd * args.chunk_size
    global_bbox[1, :2] += args.min_padd * args.chunk_size
    extent = global_bbox[1] - global_bbox[0]
    padd = np.array([args.chunk_size - extent[0] % args.chunk_size, args.chunk_size - extent[1] % args.chunk_size])
    global_bbox[0, :2] -= padd / 2
    global_bbox[1, :2] += padd / 2

    global_bbox[0, 2] = -1e12
    global_bbox[1, 2] = 1e12

    def get_var_of_laplacian(key):
        image = cv2.imread(os.path.join(args.images_dir, images_metas[key].name))
        if image is not None:
            gray = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_32F).var()
        else:
            return 0   
        
    if args.lapla_thresh > 0: 
        laplacians = Parallel(n_jobs=-1, backend="threading")(
            delayed(get_var_of_laplacian)(key) for key in images_metas
        )
        laplacians_dict = {key: laplacian for key, laplacian in zip(images_metas, laplacians)}

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

        # mask = np.all(xyzsC < corner_max_for_pts, axis=-1) * np.all(xyzsC > corner_min_for_pts, axis=-1)
        # new_xyzs = xyzsC[mask]
        # new_colors = colorsC[mask]
        # new_indices = indicesC[mask]
        # new_errors = errorsC[mask]

        # new_colors = np.clip(new_colors, 0, 255).astype(np.uint8)

        valid_cam = np.all(cam_centers < corner_max, axis=-1) * np.all(cam_centers > corner_min, axis=-1)

        # camera in this chunk
        print(f"{valid_cam.sum()} cameras in chunk")

        # extend for a overlap
        box_center = (corner_max + corner_min) / 2
        extent = (corner_max - corner_min) / 2
        acceptable_radius = 2
        extended_corner_min = box_center - acceptable_radius * extent
        extended_corner_max = box_center + acceptable_radius * extent

        for cam_idx, key in enumerate(images_metas):
            # if not valid_cam[cam_idx]:
            image_points3d =  images_points3d[key]
            # print("image_points3d", image_points3d.shape)
            n_pts = (np.all(image_points3d < corner_max_for_pts, axis=-1) * np.all(image_points3d > corner_min_for_pts, axis=-1)).sum() if len(image_points3d) > 0 else 0

            # If within chunk, means this view see a lot enough points inside this chunk
            if np.all(cam_centers[cam_idx] < corner_max) and np.all(cam_centers[cam_idx] > corner_min):
                valid_cam[cam_idx] = n_pts > 50
            # If within 2x of the chunk
            elif np.all(cam_centers[cam_idx] < extended_corner_max) and np.all(cam_centers[cam_idx] > extended_corner_min):
                valid_cam[cam_idx] = n_pts > 50 and random.uniform(0, 1) > 0.5
            # All distances
            if (not valid_cam[cam_idx]) and n_pts > 10 and args.add_far_cams:
                valid_cam[cam_idx] = random.uniform(0, 0.5) < (float(n_pts) / len(image_points3d))
            
        print(f"{valid_cam.sum()} valid cameras after visibility-base selection")
        if args.lapla_thresh > 0:
            chunk_laplacians = np.array([laplacians_dict[key] for cam_idx, key in enumerate(images_metas) if valid_cam[cam_idx]])
            laplacian_mean = chunk_laplacians.mean()
            laplacian_std_dev = chunk_laplacians.std()
            for cam_idx, key in enumerate(images_metas):
                if valid_cam[cam_idx] and laplacians_dict[key] < (laplacian_mean - args.lapla_thresh * laplacian_std_dev):
                    # print("image", key, "is blurry")
                    # print("images_metas", images_metas.keys(), len(images_metas.keys()))
                    # image = cv2.imread(f"{args.base_dir}/images/{images_metas[str(cam_idx)]['name']}")
                    # cv2.imshow("blurry", image)
                    # cv2.waitKey(0)
                    # cv2.imwrite(f"{args.output_path}/blurry/{images_metas[key]['name']}", image)
                    # cv2.imwrite(f"./blurry/{images_metas[key].name}", image)
                    valid_cam[cam_idx] = False

            print(f"{valid_cam.sum()} after Laplacian")

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
                    xys = [],
                    point3D_ids = []
                )

                # if os.path.exists(test_file) and image_meta.name in blending_dict:
                #     n_pts = np.isin(image_meta.point3D_ids, new_indices).sum()
                #     blending_dict[image_meta.name][f"{i}_{j}"] = str(n_pts)


            # points_out = {
            #     new_indices[idx] : Point3D(
            #             id=new_indices[idx],
            #             xyz= new_xyzs[idx],
            #             rgb=new_colors[idx],
            #             error=new_errors[idx],
            #             image_ids=np.array([]),
            #             point2D_idxs=np.array([])
            #         )
            #     for idx in range(len(new_xyzs))
            # }

            # create empty points3d
            points_out = {}

            write_model(cam_intrinsics, images_out, points_out, out_colmap, f".{args.model_type}")

            with open(os.path.join(out_path, "center.txt"), 'w') as f:
                f.write(' '.join(map(str, (corner_min + corner_max) / 2)))
            with open(os.path.join(out_path, "extent.txt"), 'w') as f:
                f.write(' '.join(map(str, corner_max - corner_min)))
        else:
            excluded_chunks.append([i, j])
            print("Chunk excluded")

    extent = global_bbox[1] - global_bbox[0]
    n_width = round(extent[0] / args.chunk_size)
    n_height = round(extent[1] / args.chunk_size)
    

    for i in range(n_width):
        for j in range(n_height):
            make_chunk(i, j, n_width, n_height)

    if os.path.exists(test_file):
        with open(f"{args.base_dir}/blending_dict.json", "w") as f:
            json.dump(blending_dict, f, indent=2)