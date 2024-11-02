import os  
import numpy as np
import json
import PIL
import cv2 
import trimesh
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

def main():
    input_dir = '/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/big_city'
    VIEW_NAME = "street"
    OUT_NAME = "big_city_pointcloud"
    city_street_list = ["bottom_area", "left_area",  "right_area",  "top_area"]
    city_aerial_list = ["big_high_block_1", "big_high_block_2", "big_high_block_3", "big_high_block_4", "big_high_block_5" , "big_high_block_6"]
    merge_only=False

    city_street_pcd_save_path = '/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/big_city_pointcloud/point_cloud_ds20/street/'
    city_aerial_pcd_save_path = '/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/big_city_pointcloud/point_cloud_ds20/aerial/'

    task_split = ['train','test']

    high_angle_x = 1.5707963705062866
    fl_x = 500.0
    fl_y = 500.0
    # high_cx = meta_file['cx']
    w = float(1000)
    h = float(1000)
    # fl_x = float(.5 * w / np.tan(.5 * high_angle_x))
    # fl_y = fl_x
    cx = w / 2
    cy = h / 2
    # high_cy = meta_file['cy']
    high_w = 1000.0
    high_h = 1000.0
    print("fl_x", fl_x, "fl_y", fl_y, "cx", cx, "cy", cy, "w", w, "h", h)

    intrinsic_matrix = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])

    c2ws_all = []
    imgs_path_all = []
    depth_path_all = []

    new_transforms = {}
    new_transforms['camera_angle_x'] = high_angle_x
    new_transforms['fl_x'] = fl_x
    new_transforms['fl_y'] = fl_y
    new_transforms['cx'] = cx
    new_transforms['cy'] = cy
    new_transforms['w'] = high_w
    new_transforms['h'] = high_h
    new_transforms['frames'] = {'train':{}, 'test':{}}

    # frames_block_A = meta_file['frames']
    for task_i in task_split:
        for street_name in city_street_list:
            if task_i == 'test':
                street_name = street_name + '_test'
            new_transforms['frames'][task_i][street_name] = []
            
            transform_path_sub = os.path.join(input_dir, VIEW_NAME , task_i,  street_name, f'transforms.json')
            
            with open(transform_path_sub, "r") as f:
                meta_block_i = json.load(f)   
                meta_block_frame_i = meta_block_i['frames']
                tmp_poses_list = []
                tmp_images_list = []
                tmp_depths_list = []
                for count_i, frame in enumerate(meta_block_frame_i):
                    img_idx = frame["frame_index"]
                    img_path_abs = os.path.join(input_dir, VIEW_NAME , task_i,  street_name, str(img_idx).zfill(4)+'.png')
                    depth_path_abs = os.path.join(input_dir.replace('big_city', 'big_city_depth'), VIEW_NAME , task_i,  street_name+'_depth', str(img_idx).zfill(4)+'.exr')
                    if os.path.exists(img_path_abs) and os.path.exists(depth_path_abs):
                        try:
                            img = Image.open(img_path_abs).convert("RGB")
                            depth = cv2.imread(depth_path_abs, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0] #(H, W)
                            # if depth all 65504, ignore this frame
                            if np.all(depth==65504):
                                continue
                        except:
                            # fail in open image or depth, continue
                            print("fail in open image or depth, continue")
                            continue
                        
        
                        tmp_images_list.append(img_path_abs)
                        tmp_depths_list.append(depth_path_abs)
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
                        tmp_poses_list.append(c2w.tolist()) 
                        if count_i % 100 == 0:
                            # for debug use
                            print("add img_path_abs", img_path_abs)
                            print("add depth_path_abs", depth_path_abs)

                        frame_data = {'images': os.path.join(VIEW_NAME , task_i,  street_name, str(img_idx).zfill(4)+'.png'), 'depths': os.path.join(input_dir.replace('big_city', 'big_city_depth'), VIEW_NAME , task_i,  street_name+'_depth', str(img_idx).zfill(4)+'.exr'), 'rot_mat': c2w.tolist()}
                        new_transforms['frames'][task_i][street_name].append(frame_data)
                            
                    else:
                        raise ValueError(f"img_path_abs or depth path abs not exists, {img_path_abs}, {depth_path_abs}")
                    
                # before we downsampling, we do sanity check but load all images and depth

                # random sample 10% poses 
                sample_num = int(len(tmp_poses_list) * 0.1)
                sample_idx = np.random.choice(len(tmp_poses_list), sample_num, replace=False)
                for idx in sample_idx:
                    imgs_path_all.append(tmp_images_list[idx])
                    depth_path_all.append(tmp_depths_list[idx])
                    c2ws_all.append(tmp_poses_list[idx]) 

    new_transforms_path = './big_city_valid_transforms.json'
    with open(new_transforms_path, 'w') as f:
        json.dump(new_transforms, f, indent=4)

    c2ws_all = np.stack(c2ws_all) #[B,4,4]
    centers = c2ws_all[:, :3, 3]

    print("centers x min", np.min(centers[:,0]), "centers x max", np.max(centers[:,0]))
    print("centers y min", np.min(centers[:,1]), "centers y max", np.max(centers[:,1]))
    print("centers z min", np.min(centers[:,2]), "centers z max", np.max(centers[:,2]))


    # next step, we go through the all poses and depth, image pair
    if not merge_only:
        os.makedirs(os.path.join(city_street_pcd_save_path, 'tmp'), exist_ok=True)
        for global_step, (image_path_i, depth_path_i, pose_i) in tqdm(enumerate(zip(imgs_path_all, depth_path_all, c2ws_all)), total=len(imgs_path_all)):

            # we first resize image and depth to half to decrease the points number 
            img = Image.open(image_path_i).convert("RGB")
            img = img.resize((500, 500), PIL.Image.BILINEAR)
            img = np.array(img)
            depth, _ = load_depth(depth_path_i)
            depth = cv2.resize(depth, (500, 500), interpolation=cv2.INTER_NEAREST)
            # then we convert the depth to pointcloud
            W, H = 500, 500
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
            mask = depth > 100
            points = points[~mask]
            color = color[~mask]
            
            # save to ply file using trimesh
            points_world = points
            color = color / 255.0

            if points_world.shape[0] == 0:
                print("points_world.shape[0] == 0")
                print("depth", depth.min(), depth.max())
            # save to ply file
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points_world)
            point_cloud.colors = o3d.utility.Vector3dVector(color)

            # point_cloud = trimesh.PointCloud(vertices=points_world, colors=color)
            output_path = os.path.join(city_street_pcd_save_path,'tmp' , f"{str(global_step).zfill(8)}.ply")
            point_cloud = point_cloud.voxel_down_sample(voxel_size=0.5)
            o3d.io.write_point_cloud(output_path, point_cloud)
    
     # merage all .ply to one file
        ply_list = os.listdir(os.path.join(city_street_pcd_save_path, 'tmp'))
        ply_list = [os.path.join(city_street_pcd_save_path,'tmp', ply_i) for ply_i in ply_list]
        ply_list = [ply_i for ply_i in ply_list if ply_i.endswith('.ply')]

        # new_ply_points = []
        # new_ply_colors = []
        os.makedirs(os.path.join(city_street_pcd_save_path, 'merge'), exist_ok=True)
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
                output_file = os.path.join(city_street_pcd_save_path, 'merge' ,f'Block_{merge_count}.ply')
                o3d.io.write_point_cloud(output_file, pcd)
                print(pcd)
                print(f"Point cloud saved to {output_file}") 
                merged_pcd = o3d.geometry.PointCloud()

        # we merge some points and colors if it is too close

        # combined_points = np.vstack(new_ply_points)
        # combined_colors = np.vstack(new_ply_colors)
    else:
        # Create Open3D point cloud and save it
        ply_list = os.listdir(os.path.join(city_street_pcd_save_path,'merge'))
        ply_list = [os.path.join(city_street_pcd_save_path,'merge' , ply_i) for ply_i in ply_list]
        ply_list = [ply_i for ply_i in ply_list if ply_i.endswith('.ply')]

        merged_pcd = o3d.geometry.PointCloud()
        merge_count = 0
        for ply_i in tqdm(ply_list):
            point_cloud = o3d.io.read_point_cloud(ply_i)
            point_cloud = point_cloud.voxel_down_sample(voxel_size=0.5)
            # point_cloud = point_cloud.voxel_down_sample(voxel_size=5)
            merged_pcd+=point_cloud
            merge_count+=1

            if merge_count % 50 == 0:
                print("merged_pcd num", len(merged_pcd.points))
                merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.5)
                print(merged_pcd)

            # in case too big, every 100 step we dowmsample once

        print("origianl pointcloud num", len(merged_pcd.points))
        pcd = merged_pcd.voxel_down_sample(voxel_size=0.5)
        output_file = os.path.join(city_street_pcd_save_path, 'Block_all.ply')
        o3d.io.write_point_cloud(output_file, pcd)
        print(pcd)
        print(f"Point cloud saved to {output_file}") 


if __name__ == '__main__':
    main()