
import time
import numpy as np
import os
import json
import torch 
import cv2
from tqdm import tqdm
import argparse
from read_write_model import *



def arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--base_dir', type=str, default='/cluster/work/cvl/qimaqi/cvpr_2025/datasets/small_city_train/small_city/camera_calibration/aligned/sparse/0')
    parser.add_argument('--split', action='store_true', default=False)
    parser.add_argument('--model_type', type=str, default='bin')
    return parser.parse_args()

# using this code to convert urban3d data and small city demo data to transform json 

def main():
    args = arg_parser()

    transform_dict = {}

    current_model = 'train'
    cam_intrinsics, images_metas, points3d = read_model(args.base_dir, ext=f".{args.model_type}", ignore_points3D=True)
    # CameraModel = collections.namedtuple(
    #     "CameraModel", ["model_id", "model_name", "num_params"]
    # )
    # reoreder_split_dict['mean'] = position_mean.tolist()
    # reoreder_split_dict['largest_side'] = largest_side
    # reoreder_split_dict['focal_length'] = focal_length
    # reoreder_split_dict['principle_point'] = principle_point
    # reoreder_split_dict['image_size'] = image_size

    print("cam_intrinsics", cam_intrinsics[1].model)
    intrinsics = {}
    intrinsics['model'] = cam_intrinsics[1].model 
    intrinsics['width'] = cam_intrinsics[1].width
    intrinsics['height'] = cam_intrinsics[1].height
    intrinsics['params'] = cam_intrinsics[1].params.tolist()
    # print("images_metas", images_metas.keys())
    # transform_dict['intrinsics'] = cam_intrinsics
    transform_dict['intrinsics'] = intrinsics
    transform_dict['train'] = []
    transform_dict['test'] = []

    translation_list = []
    total_list = []
    name_list = []
    for image_id, image_meta in images_metas.items():
        # transform_dict[image_id] = image_meta
        # print("image_meta id ", image_meta.id)
        # print("image_meta qvec", image_meta.qvec)
        # print("image_meta tvec", image_meta.tvec)
        # print("image_meta camera_id", image_meta.camera_id)
        # print("image_meta name", image_meta.name)
        rotmat = image_meta.qvec2rotmat()
        poses_w2c = np.concatenate([rotmat, image_meta.tvec[:, None]], axis=1)
        poses_w2c = np.concatenate([poses_w2c, np.array([[0, 0, 0, 1]])], axis=0)
        poses_c2w = np.linalg.inv(poses_w2c)

        translation_list.append(poses_c2w[:3, -1].tolist())
        data_dict = {image_meta.name: poses_c2w.tolist()}
        if not args.split:
            transform_dict[current_model].append(data_dict)
        else:
            total_list.append(data_dict)
            name_list.append(image_meta.name)

    if args.split:  
        reorder_idx = sorted(range(len(name_list)), key=lambda k: name_list[k])
        reorder_name = [name_list[i] for i in reorder_idx]
        # print("reorder name",reorder_name)
        # get 10% test data by get from interval
        test_idx = reorder_idx[::10]
        train_idx = [i for i in reorder_idx if i not in test_idx]
        transform_dict['train'] = [total_list[i] for i in train_idx]
        transform_dict['test'] = [total_list[i] for i in test_idx]


        

    position = np.array(translation_list)
    position_mean = position.mean(axis=0)
    min_bbx, max_bbx = position.min(axis=0), position.max(axis=0)
    largest_side = (max_bbx - min_bbx).max() / 2

    transform_dict['pos_mean'] = position_mean.tolist()
    transform_dict['pos_size'] = largest_side

    with open(args.base_dir + '/transforms.json', 'w') as f:
        json.dump(transform_dict, f, indent=4)


# reoreder_split_dict = {'train': {}, 'test': {}}
# reoreder_split_dict['mean'] = position_mean.tolist()
# reoreder_split_dict['largest_side'] = largest_side
# reoreder_split_dict['focal_length'] = focal_length
# reoreder_split_dict['principle_point'] = principle_point
# reoreder_split_dict['image_size'] = image_size





if __name__ == '__main__':
    main()

