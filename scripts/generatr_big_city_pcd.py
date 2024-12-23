import os  
import numpy as np
import json
import PIL
import cv2 
from tqdm import tqdm
from colmap_conversion_utils import gl2world_to_cv2world
from PIL import Image
# get the test image, depth and pose
import open3d as o3d


def load_depth(depth_path, is_float16=True):
    image = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0] #(H, W)
    if is_float16==True:
        invalid_mask=(image==65504)
    else:
        invalid_mask=None
    image = image / 100 # cm -> m
    return image, invalid_mask

def arg_parse():
    import argparse
    parser = argparse.ArgumentParser(description='MatrixCity pointcloud generation')
    parser.add_argument('--VIEW_NAME', type=str, default='aerial', help='VIEW_NAME')
    parser.add_argument('--merge', action='store_true', help='merge all pointclouds', default=False)
    return parser.parse_args()

def main():
    args = arg_parse()
    input_dir = '/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/big_city'
    VIEW_NAME = args.VIEW_NAME
    OUT_NAME = "big_city_pointcloud" + '_' + VIEW_NAME
    merge_only=args.merge

    city_street_list = ["bottom_area", "left_area",  "right_area",  "top_area"]
    city_aerial_list = ["big_high_block_1", "big_high_block_2", "big_high_block_3", "big_high_block_4", "big_high_block_5" , "big_high_block_6"]


    city_street_pcd_save_path = '/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/big_city_pointcloud/point_cloud_ds20/street/' + OUT_NAME
    city_aerial_pcd_save_path = '/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/big_city_pointcloud/point_cloud_ds20/aerial/' + OUT_NAME

    if VIEW_NAME == "street":
        city_list = city_street_list
        city_pcd_save_path = city_street_pcd_save_path

    else:
        city_list = city_aerial_list
        city_pcd_save_path = city_aerial_pcd_save_path

    task_split = ['train', 'test']


    c2ws_all = []
    imgs_path_all = []
    depth_path_all = []


    # using aerial now
    if VIEW_NAME == "aerial": 
        intrinsic_matrix = np.array([[2317.6449482429634, 0, 960.0], [0, 2317.6449482429634, 540.0], [0, 0, 1]])
        fl_x = 2317.6449482429634
        fl_y = 2317.6449482429634
        cx = 960.0
        cy = 540.0
        high_w = 1920
        high_h = 1080
        ratio = 0.2
        depth_max = 600

    elif VIEW_NAME == "street":
        intrinsic_matrix = np.array([[500.0, 0, 500.0], [0, 500.0, 500.0], [0, 0, 1]])
        fl_x = 500.0
        fl_y = 500.0
        cx = 500.0
        cy = 500.0
        high_w = 1000
        high_h = 1000
        ratio = 0.1
        depth_max = 150

    # frames_block_A = meta_file['frames']
    for task_i in task_split:
        for block_name in city_list:
            if task_i == 'test':
                block_name = block_name + '_test'
            
            transform_path_sub = os.path.join(input_dir, VIEW_NAME , task_i,  block_name, f'transforms_correct.json')
            
            with open(transform_path_sub, "r") as f:
                meta_block_i = json.load(f)   
                meta_block_frame_i = meta_block_i['frames']
                tmp_image_list = []
                tmp_poses_list = []
                tmp_depth_list = []

                for count_i, frame in enumerate(meta_block_frame_i):
                    img_idx = frame["frame_index"]
                    img_path_abs = os.path.join(input_dir, VIEW_NAME , task_i,  block_name, str(img_idx).zfill(4)+'.png')
                    depth_path_abs = os.path.join(input_dir.replace('big_city', 'big_city_depth'), VIEW_NAME , task_i,  block_name+'_depth', str(img_idx).zfill(4)+'.exr')
                    if os.path.exists(img_path_abs) and os.path.exists(depth_path_abs):
                        # imgs_path_all.append(img_path_abs)
                        # depth_path_all.append(depth_path_abs)
                        # check if image and depth can be opened or not
                        c2w=np.array(frame["rot_mat"])
                        # check unit of distance (m or 100m)
                        c2w[3,3]=1
                        rot_mat = c2w[:3, :3]
                        # check rot mat is valid or not
                        composed_mat = rot_mat*rot_mat.T
                        if not np.allclose(composed_mat, np.eye(3)):
                            c2w[:3,:3]*=100 # bug of data
                        assert not np.allclose(composed_mat, np.eye(3)), f"composed_mat {composed_mat}"
                        c2w = gl2world_to_cv2world(c2w)
                        rot_mat = c2w[:3, :3]
                        z_axis = rot_mat[:, 2]
                        if z_axis[-1] > 0.9:
                            continue

                        # c2ws_all.append(c2w.tolist())
                        # we only consider the image and depth if the pose is not point to up

                        tmp_image_list.append(img_path_abs)
                        tmp_depth_list.append(depth_path_abs)
                        tmp_poses_list.append(c2w.tolist())
                        if count_i % 100 == 0:
                            # for debug use
                            print("add img_path_abs", img_path_abs)
                            print("add depth_path_abs", depth_path_abs)
                    else:
                        raise ValueError(f"img_path_abs or depth path abs not exists, {img_path_abs}, {depth_path_abs}")

                # save to transform_path_sub_up
                # random sample 10% poses 
                sample_num = int(len(tmp_poses_list) * ratio)
                sample_idx = np.random.choice(len(tmp_poses_list), sample_num, replace=False)
                for idx in sample_idx:
                    imgs_path_all.append(tmp_image_list[idx])
                    depth_path_all.append(tmp_depth_list[idx])
                    c2ws_all.append(tmp_poses_list[idx]) 
                


    c2ws_all = np.stack(c2ws_all) #[B,4,4]
    centers = c2ws_all[:, :3, 3]

    print("centers x min", np.min(centers[:,0]), "centers x max", np.max(centers[:,0]))
    print("centers y min", np.min(centers[:,1]), "centers y max", np.max(centers[:,1]))
    print("centers z min", np.min(centers[:,2]), "centers z max", np.max(centers[:,2]))

    # next step, we go through the all poses and depth, image pair
    os.makedirs(os.path.join(city_pcd_save_path, 'tmp'), exist_ok=True)

    if not merge_only:
        for global_step, (image_path_i, depth_path_i, pose_i) in tqdm(enumerate(zip(imgs_path_all, depth_path_all, c2ws_all)), total=len(imgs_path_all)):

            # we first resize image and depth to half to decrease the points number 
            img = Image.open(image_path_i).convert("RGB")
            img = img.resize((high_w//2, high_h//2))
            img = np.array(img)
            depth, _ = load_depth(depth_path_i)
            depth = cv2.resize(depth, (high_w//2,high_h//2 ), interpolation=cv2.INTER_NEAREST)
            # then we convert the depth to pointcloud
            W, H = high_w//2 , high_h//2
            intrinsic_matrix = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])
            intrinsic_matrix[:2] = intrinsic_matrix[:2] / 2
            intrinsic_matrix[2, 2] = 1
            intrinsic_matrix = intrinsic_matrix.astype(np.float32)
            depth = depth.astype(np.float32)
            # rgb color of the pointcloud
            color = img.reshape(-1,3) # (H*W, 3)
            x = np.arange(W)
            y = np.arange(H)
            xx,yy = np.meshgrid(x,y)
            xx = xx.flatten()
            yy = yy.flatten()

            # 2. get the depth value
            depth = depth.flatten() # (H*W,)

            points = np.vstack([xx,yy,np.ones_like(xx)])
            # 3xN
            points = np.multiply(points, depth)
            # 3xN
            points = np.dot(np.linalg.inv(intrinsic_matrix), points)
            # 3xN
            points = np.dot(pose_i, np.vstack([points, np.ones_like(xx)]))
            # 3xN
            points = points[:3,:]
            # Nx3
            points = points.T # (H*W, 3)

            # mask some far away points
            mask = depth > depth_max
            points = points[~mask]
            color = color[~mask]
            
            # save to ply file using trimesh
            points_world = points
            # color = color.astype(np.int8)
            # save to ply file using o3d
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points_world)
            point_cloud.colors = o3d.utility.Vector3dVector(color/255.0)

            # point_cloud = trimesh.PointCloud(vertices=points_world, colors=color)
            output_path = os.path.join(city_pcd_save_path, 'tmp' ,f"{str(global_step).zfill(8)}.ply")
            point_cloud = point_cloud.voxel_down_sample(voxel_size=0.5)
            o3d.io.write_point_cloud(output_path, point_cloud)
            # close the file


        # merage all .ply to one file
        ply_list = os.listdir(os.path.join(city_pcd_save_path, 'tmp'))
        ply_list = [os.path.join(city_pcd_save_path,'tmp', ply_i) for ply_i in ply_list]
        ply_list = [ply_i for ply_i in ply_list if ply_i.endswith('.ply')]

        # new_ply_points = []
        # new_ply_colors = []
        os.makedirs(os.path.join(city_pcd_save_path, 'merge'), exist_ok=True)
        merged_pcd = o3d.geometry.PointCloud()
        merge_every_step = 100
        merge_count = 0
        for ply_i in ply_list:
            point_cloud = o3d.io.read_point_cloud(ply_i)
            merged_pcd+=point_cloud
            merge_count+=1
            if merge_count % merge_every_step == 0:
                print("merged_pcd num", len(merged_pcd.points))
                # we merge some points and colors if it is too close
                # combined_points = np.vstack(new_ply_points)
                # combined_colors = np.vstack(new_ply_colors)

                # Create Open3D point cloud and save it
                pcd = merged_pcd.voxel_down_sample(voxel_size=0.5)
                output_file = os.path.join(city_pcd_save_path, 'merge' ,f'Block_{merge_count}.ply')
                o3d.io.write_point_cloud(output_file, pcd)
                print(pcd)
                print(f"Point cloud saved to {output_file}") 
                merged_pcd = o3d.geometry.PointCloud()

        # we merge some points and colors if it is too close

        # combined_points = np.vstack(new_ply_points)
        # combined_colors = np.vstack(new_ply_colors)
    else:
        # Create Open3D point cloud and save it
        ply_list = os.listdir(os.path.join(city_pcd_save_path,'merge'))
        ply_list = [os.path.join(city_pcd_save_path,'merge' , ply_i) for ply_i in ply_list]
        ply_list = [ply_i for ply_i in ply_list if ply_i.endswith('.ply')]

        merged_pcd = o3d.geometry.PointCloud()
        merge_count = 0
        for ply_i in tqdm(ply_list):
            point_cloud = o3d.io.read_point_cloud(ply_i)
            # point_cloud = point_cloud.voxel_down_sample(voxel_size=0.5)
            # point_cloud = point_cloud.voxel_down_sample(voxel_size=5)
            merged_pcd+=point_cloud
            merge_count+=1

            if merge_count % 50 == 0:
                print("merged_pcd num", len(merged_pcd.points))
                merged_pcd = merged_pcd.voxel_down_sample(voxel_size=1.0)
                print(merged_pcd)

            # in case too big, every 100 step we dowmsample once

        print("origianl pointcloud num", len(merged_pcd.points))
        pcd = merged_pcd.voxel_down_sample(voxel_size=0.5)
        output_file = os.path.join(city_pcd_save_path, 'Block_all.ply')
        o3d.io.write_point_cloud(output_file, pcd)
        print(pcd)
        print(f"Point cloud saved to {output_file}") 


if __name__ == '__main__':
    main()