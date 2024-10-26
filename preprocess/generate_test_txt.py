import os  
import numpy as np

import numpy as np
import argparse
import cv2
from joblib import delayed, Parallel
import os
import random
from read_write_model import *
import json
import shutil

base_dir = '/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug/camera_calibration/chunks/2_3/sparse/0/'
raw_image_dir = '/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug/camera_calibration/rectified/images/'
tgt_image_dir = '/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug/camera_calibration/chunks/2_3/images/'
os.makedirs(tgt_image_dir, exist_ok=True)
cam_intrinsics, images_metas, points3d = read_model(base_dir, ext=f".bin")

global_count = 0
test_name = []
for key in images_metas:
    image_meta = images_metas[key]
    image_name = image_meta.name
    print(image_name)

    if global_count % 10 == 0:
        test_name.append(image_name)
        shutil.copy(os.path.join(raw_image_dir, image_name), os.path.join(tgt_image_dir, image_name))

    global_count += 1


np.savetxt(os.path.join(base_dir, 'test.txt'), test_name, fmt='%s')

