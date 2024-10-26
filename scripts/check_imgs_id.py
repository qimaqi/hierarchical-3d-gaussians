import os 
import numpy as np 

# image txt path
image_txt_path = '/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug/camera_calibration/rectified/sparse/known/images.txt'

# read txt line by line
with open(image_txt_path, 'r') as f:
    lines = f.readlines()

image_id_list = []
for line_num, line in enumerate(lines):
    image_id = line.split(' ')[0]
    if line_num>1 and line_num%2 == 0:
        print("image_id", image_id)
        image_id_list.append(image_id)

image_id_list = np.array(image_id_list)
# check if there is repeated id
unique_image_id_list = np.unique(image_id_list)
print("length of ids", len(image_id_list))
print("length of unique ids", len(unique_image_id_list))
# find all ids with repeated name
# for unique_id in unique_image_id_list:
#     if len(np.where(image_id_list == unique_id)[0]) > 1:
#         print("repeated id", unique_id)