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
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--city_name', type=str, default='small_city', help='the city name')
    parser.add_argument('--view_name', type=str, default='street', help='the view name')
    parser.add_argument('--merge_only', action='store_true', help='if we only merge the pointcloud')
    return parser.parse_args()


def main():
    args = arg_parse()

    input_dir = '/work/qimaqi/datasets/MatrixCity'
    # city_name = 'small_city'
    input_city_dir = os.path.join(input_dir, args.city_name)
    VIEW_NAME = args.view_name
    output_demo_dir = os.path.join(input_dir, 'demo_data', args.city_name, VIEW_NAME)
    os.makedirs(output_demo_dir, exist_ok=True)

    small_city_street_list = ["small_city_road_down" , "small_city_road_horizon" , "small_city_road_outside",   "small_city_road_vertical"]
    small_city_aerial_list = ["block_1",  "block_10",  "block_2",  "block_3",  "block_4",  "block_5",  "block_6",  "block_7",  "block_8",  "block_9", ]

    
    big_city_street_list = ["bottom_area", "left_area",  "right_area",  "top_area"]
    big_city_aerial_list = ["big_high_block_1", "big_high_block_2", "big_high_block_3", "big_high_block_4", "big_high_block_5" , "big_high_block_6"]

    # city_street_pcd_save_path = '/cluster/scratch/qimaqi/matrix_city_proj/big_city_pointcloud/street/'
    # city_aerial_pcd_save_path = '/cluster/scratch/qimaqi/matrix_city_proj/big_city_pointcloud/aerial/'

    task_split = ['train', 'test']

    street_dict = { "camera_angle_x": 1.5707963705062866, "fl_x": 499.9999781443055, "fl_y": 499.9999781443055, "cx": 500.0, "cy": 500.0, "w": 1000.0, "h": 1000.0,}
    # aerial property:
    aerial_dict = {"fl_x": 2317.6449482429634,"fl_y": 2317.6449482429634,"cx": 960.0, "cy": 540.0, "w": 1920.0, "h": 1080.0}

    street_intrinsics_matrix = np.array([[street_dict['fl_x'], 0, street_dict['cx']], [0, street_dict['fl_y'], street_dict['cy']], [0, 0, 1]])

    aerial_intrinsic_matrix = np.array([[aerial_dict['fl_x'], 0, aerial_dict['cx']], [0, aerial_dict['fl_y'], aerial_dict['cy']], [0, 0, 1]])

    if args.city_name == 'small_city':
        city_street_list = small_city_street_list
        city_aerial_list = small_city_aerial_list

    elif args.city_name == 'big_city':
        city_street_list = big_city_street_list
        city_aerial_list = big_city_aerial_list

    if args.view_name == 'street':
        intrinsic_matrix = street_intrinsics_matrix
        data_list = city_street_list

    elif args.view_name == 'aerial':
        intrinsic_matrix = aerial_intrinsic_matrix
        data_list = city_aerial_list

    c2ws_all = []
    imgs_path_all = []
    depth_path_all = []

    # frames_block_A = meta_file['frames']
    for task_i in task_split:
        for street_name in data_list:
            if task_i == 'test':
                street_name = street_name + '_test'
            
            transform_path_sub = os.path.join(input_city_dir, VIEW_NAME , task_i,  street_name, f'transforms.json')

  
            transforms_correct = {'frames': [], 'intrinsics': intrinsic_matrix.tolist()}
            
            with open(transform_path_sub, "r") as f:
                meta_block_i = json.load(f)   
                meta_block_frame_i = meta_block_i['frames']
                tmp_image_list = []
                tmp_poses_list = []
                tmp_depth_list = []

                for count_i, frame in tqdm(enumerate(meta_block_frame_i), total=len(meta_block_frame_i)):
                    img_idx = frame["frame_index"]
                    img_path_abs = os.path.join(input_city_dir, VIEW_NAME , task_i,  street_name, str(img_idx).zfill(4)+'.png')
                    depth_path_abs = os.path.join(input_city_dir.replace(args.city_name, args.city_name+'_depth'), VIEW_NAME , task_i,  street_name+'_depth', str(img_idx).zfill(4)+'.exr')
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


                        tmp_image_list.append(img_path_abs)
                        tmp_depth_list.append(depth_path_abs)
                        tmp_poses_list.append(c2w.tolist())
                        if count_i % 10 == 0:
                            # for debug use
                            # print("add img_path_abs", img_path_abs)
                            # print("add depth_path_abs", depth_path_abs)
                            image_demo = Image.open(img_path_abs).convert('RGB')
                            # resize to 128x128
                            image_demo = image_demo.resize((128, 128))
                            image_demo.save(os.path.join(output_demo_dir, f'{task_i}_{street_name}_{img_idx}.png'))
          
                    else:
                        raise ValueError(f"img_path_abs or depth path abs not exists, {img_path_abs}, {depth_path_abs}")




if __name__ == '__main__':
    main()