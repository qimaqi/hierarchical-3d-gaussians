
import time
import numpy as np
import os
import json
import torch 
import cv2
from tqdm import tqdm
import argparse
from read_write_model import *
import trimesh
from scipy.spatial.transform import Rotation

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
    parser.add_argument('--base_dir', type=str, default='/cluster/work/cvl/qimaqi/cvpr_2025/datasets/MatrixCity/small_city/colmap_street/camera_calibration/rectified/', help='Input directory')
    parser.add_argument('--view_name', type=str, default='street', help='view name')
    parser.add_argument('--pc_path', type=str, help='City name')
    args = parser.parse_args()
    return args


def main():
    args = arg_parser()
    view_name = args.view_name
    sparse_path = os.path.join(args.base_dir, 'sparse/mc/')
    transform_path = os.path.join(args.base_dir, 'sparse/known/transforms.json')
    splits = ['train', 'test']
    with open(transform_path, 'r') as f:
        transform_dict = json.load(f)

    if view_name == 'street':
        fl_x = 500.0
        fl_y = 500.0
        cx = 500.0
        cy = 500.0
        width = 1000.0
        height = 1000.0

    elif view_name == 'aerial':
        fl_x = 2317.6449482429634
        fl_y = 2317.6449482429634
        cx = 960.0
        cy = 540.0
        width = 1920.0
        height = 1080.0

    
    print('Creating sparse directory at {}'.format(sparse_path))
    if not os.path.exists(sparse_path):
        os.mkdir(sparse_path)

    
    # create cameras.txt under sparse_path
    cameras_path = os.path.join(sparse_path, 'cameras.txt')
    print('Creating cameras.txt at {}'.format(cameras_path))
    with open(cameras_path, 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write('# Number of cameras: 1\n')
        WIDTH = width
        HEIGHT = height
        f.write('1 PINHOLE {} {} {} {} {} {}\n'.format(WIDTH, HEIGHT, fl_x, fl_y, 
                                                    cx, cy))


    # create images.txt under sparse_path
    images_path = os.path.join(sparse_path, 'images.txt')
    image_id_count = 1
    print('Creating images.txt at {}'.format(images_path))
    with open(images_path, 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        f.write('# Number of images: {}\n'.format(len(transform_dict['train']) + len(transform_dict['test'])))

        for split_i in splits:
            for block_i in transform_dict[split_i]:
                for frame_i in transform_dict[split_i][block_i]:
                    for key, frame_data in frame_i.items():
                        # frame_pose = frame_i[key]
                        image_id = image_id_count
                        camera_id = 1 if view_name == 'street' else 2
                        file_path = key + '.png'
                        frame_pose = np.array(frame_data).reshape(4, 4)
                        pose_w2c = np.linalg.inv(frame_pose)

                        quaternion, translation = matrix_to_quaternion_and_translation(pose_w2c)
                        qw, qx, qy, qz = quaternion  # Quaternion (w, x, y, z)
                        tx, ty, tz = translation  # Translation (tx, ty, tz)


                        f.write('{} {} {} {} {} {} {} {} {} {}\n'.format(image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, file_path))
                        f.write('\n')
                        image_id_count += 1

    # create points3D.txt under sparse_path
    pc_path = args.pc_path
    points3D_path = os.path.join(sparse_path, 'points3D.txt')
    loaded_ply = trimesh.load(pc_path)
    vertices = loaded_ply.vertices
    colors = loaded_ply.visual.vertex_colors[:, :3]
    pc_id = 1
    print('Creating points3D.txt at {}'.format(points3D_path))
    with open(points3D_path, 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        f.write('# Number of points: {}\n'.format(len(vertices)))

        for i, vertex in enumerate(vertices):
            x, y, z = vertex
            r, g, b = colors[i]
            error = 0.0
            track = ''
            f.write('{} {} {} {} {} {} {} {} {}\n'.format(pc_id, x, y, z, r, g, b, error, track))
            f.write('\n')
            pc_id += 1


     

if __name__ == '__main__':
    main()