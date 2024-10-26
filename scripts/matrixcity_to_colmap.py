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

# 
import numpy as np
from scipy.spatial.transform import Rotation
import os
import shutil
from read_write_model import read_images_binary,write_images_binary, Image
import cv2 
from PIL import Image as ImagePIL

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

def save_images_txt(image_folder, poses, cameras, output_path):
    """
    Save image extrinsics and filenames to images.txt in COLMAP format.
    
    Args:
    - image_folder (str): Path to the folder containing images.
    - poses (list): List of 4x4 camera-to-world transformation matrices.
    - cameras (list): List of dictionaries with camera model parameters for each image.
    - output_path (str): Path to save the images.txt file.
    """
    image_file_path = os.path.join(output_path, 'images.txt')
    with open(image_file_path, 'w') as f:
        f.write('# Image list with one line of data per image:\n')
        f.write('# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n')
                
        # Here, the first two lines define the information of the first image, and so on. The reconstructed pose of an image is specified as the projection from world to the camera coordinate system of an image using a quaternion (QW, QX, QY, QZ) and a translation vector (TX, TY, TZ). The quaternion is defined using the Hamilton convention, which is, for example, also used by the Eigen library. The coordinates of the projection/camera center are given by -R^t * T, where R^t is the inverse/transpose of the 3x3 rotation matrix composed from the quaternion and T is the translation vector. The local camera coordinate system of an image is defined in a way that the X axis points to the right, the Y axis to the bottom, and the Z axis to the front as seen from the image.


        for i, image_name in enumerate(sorted(os.listdir(image_folder))):
            image_path = os.path.join(image_folder, image_name)
            
            # Extract the corresponding pose for this image
            pose = poses[i]
            # change pose to world to camera poses 
            pose = np.linalg.inv(pose)

            quaternion, translation = matrix_to_quaternion_and_translation(pose)
            
            # Write image data in COLMAP format
            image_id = i + 1
            camera_id = 1  # Assuming each image has a unique camera, change if needed
            qw, qx, qy, qz = quaternion  # Quaternion (w, x, y, z)
            tx, ty, tz = translation  # Translation (tx, ty, tz)
            f.write(f'{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n')

            # adding also keypoints here 
            # so for each image, we project the 3D points to 2D, and check if depth in z is close or not, if close, then this is a keypoint





    
    print(f"Saved images to {image_file_path}")





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
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()
    return args


def main():
    args = arg_parser()
    input_dir = args.input_dir
    HIGH_NAME = "aerial"
    ROAD_NAME = "street"
    OUT_NAME = "aerial_block1_debug"
    output_dir = os.path.join(args.output_dir, OUT_NAME)


    # debug first, let's try first with only block_1
    # for small city
    blocks_used = ['block_1']
    block_pose_name_used = ['block_A']

    # road_metas={}
    # assume all situations share the same intri
    for task in ["train"]: # , "test"
        c2ws_high = []
        imgs_path_high = []
        with open(os.path.join(input_dir, HIGH_NAME , "pose",  'block_all', f'transforms_{task}.json'), "r") as f:
            meta_high_train = json.load(f)
        for block in blocks_used:
            # with open(os.path.join(input_dir, HIGH_NAME , "train",  block, 'transforms.json'), "r") as f:
            #     meta_high_train = json.load(f)
            high_angle_x = meta_high_train['camera_angle_x']
            high_fl_x = meta_high_train['fl_x']
            high_fl_y = meta_high_train['fl_y']
            high_cx = meta_high_train['cx']
            high_cy = meta_high_train['cy']
            high_w = meta_high_train['w']
            high_h = meta_high_train['h']
            high_frames = meta_high_train['frames']
            print("len(high_frames)", len(high_frames))
            for count_i, frame in enumerate(high_frames):
                img_path = frame["file_path"]
                block_name = img_path.split("/")[-2]
                if block == block_name:
                    c2w=np.array(frame["transform_matrix"])
                    c2w[3,3]=1
                    # times translation 100
                    c2w[:3, 3] = c2w[:3, 3] * 100

                    c2ws_high.append(c2w.tolist()) 
                    
                    img_path_relative = HIGH_NAME+img_path.replace("../../", "/")
                    img_path_abs = os.path.join(input_dir, img_path_relative)
                    if os.path.exists(img_path_abs):
                        imgs_path_high.append(img_path_abs)
                        
                        if count_i % 100 == 0:
                            # for debug use
                            print("add img_path_abs", img_path_abs)

                # raise ValueError("block not in img_path")


        c2ws_high=np.stack(c2ws_high) #[B,4,4]

        camera_intrinsics_0 = {'model': 'PINHOLE', 'width': int(high_w//2), 'height': int(high_h//2), 'params': [high_fl_x//2, high_fl_y//2, high_cx//2, high_cy//2]}
        # camera_intrinsics_0 = {'model': 'PINHOLE', 'width': high_w, 'height': high_h, 'params': [high_fl_x, high_fl_y, high_cx, high_cy]}

        # save to pycolmap format, image.bin, camera.bin, points3D.bin
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # save images
        os.makedirs(os.path.join(output_dir, 'camera_calibration', 'rectified' ,'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'camera_calibration', 'rectified' ,'masks'), exist_ok=True)
        mask_dis_tmp_path = os.path.join(output_dir, 'camera_calibration', 'rectified' ,'masks', f"tmp.png")
        mask_array = np.ones((int(high_h//2), int(high_w//2)), dtype=np.uint8)
        mask_array = mask_array * 255
        for i, img_path in enumerate(imgs_path_high):
            dst_path = os.path.join(output_dir, 'camera_calibration', 'rectified' ,'images', f"{str(i).zfill(8)}.png")
            if not os.path.exists(dst_path):                
                img = ImagePIL.open(img_path)
                img = img.resize((int(high_w//2), int(high_h//2)))
                img.save(dst_path)
            # 
            #     shutil.copy(img_path, dst_path)

            # create same size all white mask 
            
            # mask_dis_path = os.path.join(output_dir, 'camera_calibration', 'rectified' ,'masks', f"{str(i).zfill(8)}.png")
            # # print("mask_dis_path", mask_dis_path)
            # # write one example 

            # cv2.imwrite(mask_dis_tmp_path, mask_array)
            # if not os.path.exists(mask_dis_path):
            #     shutil.copy(mask_dis_tmp_path, mask_dis_path)
            # delete tmp mask
        # os.remove(mask_dis_tmp_path)

            # shutil.copy(img_path, os.path.join(args.output_dir, 'tmp', f"{str(i).zfill(8)}.png"))

        os.makedirs(os.path.join(output_dir, 'camera_calibration', 'rectified' ,'sparse'), exist_ok=True)
        save_cameras_txt([camera_intrinsics_0],  os.path.join(output_dir, 'camera_calibration', 'rectified', 'sparse'))
        save_images_txt(os.path.join(output_dir, 'camera_calibration', 'rectified' ,'images'), c2ws_high, [camera_intrinsics_0], os.path.join(output_dir, 'camera_calibration', 'rectified' ,'sparse'))

        # # save also a transform.json file
        # new_transforms_path = os.path.join(output_dir, 'camera_calibration', 'rectified' ,'transforms.json')
        # new_transforms = {}
        # for i, c2w in enumerate(c2ws_high):
        #     new_transforms[str(i).zfill(8)] = c2w.tolist()
        # with open(new_transforms_path, 'w') as f:
        #     json.dump(new_transforms, f, indent=4)



    # create masks 
    # with open(os.path.join(input_dir, ROAD_NAME,f"transforms_train.json"), "r") as f:
    #     meta_road_train = json.load(f)
    # road_angle_x = meta_road_train['camera_angle_x']
    # road_fl_x = meta_road_train['fl_x']
    # road_fl_y = meta_road_train['fl_y']
    # road_cx = meta_road_train['cx']
    # road_cy = meta_road_train['cy']
    # road_w = meta_road_train['w']
    # road_h = meta_road_train['h']
    # road_frames = meta_road_train['frames']
    # c2ws_road = []
    # for frame in road_frames:
    #     c2w=np.array(frame["transform_matrix"])
    #     c2w[3,3]=1
    #     c2ws_road.append(c2w.tolist())
    # c2ws_road=np.stack(c2ws_road) #[B,4,4]

    # camera_intrinsics_1 = {'model': 'PINHOLE', 'width': road_w, 'height': road_h, 'params': [road_fl_x, road_fl_y, road_cx, road_cy]}



if __name__ == '__main__':
    main()