# licence : MIT
# Authors: Qi Ma
# Date: 2024-09
# Contact: qimaqi@ethz.ch
# colmap model_converter --input_path=/work/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap --output_path=/work/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/rectified/sparse/0 --output_type=BIN

# usage: this code change scenes in matrix city both aerial and street, (both train and test) to colmap format
# python matrixcity_to_colmap.py --input_dir /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city --output_dir /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/
# 
# python preprocess/auto_reorient_npts.py --input_path=/data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/rectified/sparse/0 --output_path=/data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/aligned/sparse/0  --upscale=1

import os
import argparse
import json 
import numpy as np
import PIL

import numpy as np
import os
from scipy.spatial.transform import Rotation
import numpy as np
import os
import shutil
from read_write_model import read_images_binary,write_images_binary, Image
import cv2 
from PIL import Image as ImagePIL
import trimesh 
from matplotlib import pyplot as plt
from tqdm import tqdm
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from colmap_conversion_utils import ImageDepth2Colmap

def load_depth(depth_path, is_float16=True):
    image = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0] #(H, W)
    if is_float16==True:
        invalid_mask=(image==65504)
    else:
        invalid_mask=None
    image = image / 100 # cm -> m
    return image, invalid_mask

def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def gl2world_to_cv2world(gl2world):
    cv2gl = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
    cv2world = gl2world @ cv2gl

    return cv2world




def save_cameras_txt(cameras, output_path):
    """
    Save camera intrinsics to cameras.txt in COLMAP format.
    
    Args:
    - cameras (list): List of dictionaries with camera model parameters.
    - output_path (str): Path to save the cameras.txt file.
    """
    camera_file_path = os.path.join(output_path, 'cameras.txt')
    with open(camera_file_path, 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        
        for i, camera_params in enumerate(cameras):
            camera_id = i + 1
            model_id = camera_params['model']
            width = camera_params['width']
            height = camera_params['height']
            params = ' '.join(map(str, camera_params['params']))  # fx, fy, cx, cy, distortion params
            f.write(f'{camera_id} {model_id} {width} {height} {params}\n')
    
    print(f"Saved cameras to {camera_file_path}")




def matrix_to_quaternion_and_translation(matrix):
    """Convert 4x4 camera-to-world matrix to quaternion and translation."""
    rotation_matrix = matrix[:3, :3]
    translation = matrix[:3, 3]
    
    # Convert rotation matrix to quaternion (x, y, z, w)
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
    # change x y z w to w x y z
    quaternion = np.roll(quaternion, 1)

    return quaternion, translation


def arg_parser():
    parser = argparse.ArgumentParser(description='MatrixCity to Colmap')
    parser.add_argument('--input_root', type=str, default='/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/', help='Input directory')
    parser.add_argument('--city_name', type=str, default='small_city', help='Output directory')
    parser.add_argument('--view_name', type=str, default='street', help='View name')
    args = parser.parse_args()
    return args


def main():
    args = arg_parser()
    input_root = args.input_root
    city_name = args.city_name
    view_name = args.view_name
    # input_dir = os.path.join(input_root, city_name)

    if city_name == 'small_city':
        city_street_list = ["small_city_road_down", "small_city_road_horizon",  "small_city_road_outside",  "small_city_road_vertical"]
        city_aerial_list = ["block_1", "block_2", "block_3",  "block_4", "block_5" , "block_6", "block_7", "block_8", "block_9" , "block_10"]

    elif city_name == 'big_city':
        city_street_list = ["bottom_area", "left_area",  "right_area",  "top_area"]
        city_aerial_list = ["big_high_block_1", "big_high_block_2", "big_high_block_3", "big_high_block_4", "big_high_block_5" , "big_high_block_6"]

    task_split = ['train', 'test']

    if view_name == 'street':
        city_list = city_street_list
        fl_x = 500.0
        fl_y = 500.0
        cx = 500.0
        cy = 500.0
        width = 1000.0
        height = 1000.0

    elif view_name == 'aerial':
        city_list = city_aerial_list
        fl_x = 2317.6449482429634
        fl_y = 2317.6449482429634
        cx = 960.0
        cy = 540.0
        width = 1920.0
        height = 1080.0


    # frames_block_A = meta_file['frames']
    for task_i in task_split:
        for block_name in city_list:
            if task_i == 'test':
                block_name = block_name + '_test'
            
            transform_path_sub = os.path.join(input_root, city_name, view_name , task_i,  block_name, f'transforms.json')
            transform_path_sub_correct_path = os.path.join(input_root, city_name, view_name , task_i,  block_name, f'transforms_correct.json')

            if os.path.exists(transform_path_sub_correct_path):
                print("transform_path_sub_correct_path exists", transform_path_sub_correct_path)
                continue

            transform_path_sub_correct = {'intrinsics': [fl_x, fl_y, cx, cy], 'width': width, 'height': height, 'frames': []}
            
            with open(transform_path_sub, "r") as f:
                meta_block_i = json.load(f)   
                meta_block_frame_i = meta_block_i['frames']
                for count_i, frame in enumerate(meta_block_frame_i):
                    img_idx = frame["frame_index"]
                    img_path_abs = os.path.join(input_root, city_name, view_name , task_i,  block_name, str(img_idx).zfill(4)+'.png')
                    depth_path_abs = os.path.join(input_root, city_name + '_depth', view_name , task_i,  block_name+'_depth', str(img_idx).zfill(4)+'.exr')
                    if os.path.exists(img_path_abs) and os.path.exists(depth_path_abs):
    
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
                        if count_i % 100 == 0:
                            # for debug use
                            print("add img_path_abs", img_path_abs)
                            print("add depth_path_abs", depth_path_abs)
                        z_axis = c2w[:3, :3][:3, 2]
                        if z_axis[-1] > 0.9:
                            continue

                        try:
                            # try load images and depth 
                            img = ImagePIL.open(img_path_abs).convert('RGB')
                            depth =  cv2.imread(depth_path_abs, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0]
                            # if all depth is 65504, then skip
                            if np.all(depth==65504):
                                continue
                        except Exception as e:
                            print("error", e)
                            continue
                    else:
                        raise ValueError(f"img_path_abs or depth path abs not exists, {img_path_abs}, {depth_path_abs}")
                
                    transform_path_sub_correct['frames'].append(frame)

            with open(transform_path_sub_correct_path, "w") as f:
                json.dump(transform_path_sub_correct, f, indent=4)


if __name__ == '__main__':
    main()