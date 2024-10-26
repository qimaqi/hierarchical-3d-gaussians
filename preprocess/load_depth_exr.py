
import numpy as np
import cv2 
import os 
import json 
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

# Open the .exr file
file_path = '/data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city_depth_float32/aerial/train/block_1_depth/0062.exr'

depth_map = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

# Check if the depth map is loaded correctly
if depth_map is None:
    raise Exception('Could not load the .exr file.')

# You can now use `depth` as a depth map
print(depth_map.shape, depth_map.min(), depth_map.max())
H,W, _  = depth_map.shape
# using cv2 nearest interpoate to resize the depth map
depth_map = cv2.resize(depth_map, (W//4,H//4), interpolation=cv2.INTER_NEAREST)
depth_map = depth_map[...,0]
print(depth_map.shape, depth_map.min(), depth_map.max())

# save to visualize
depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imwrite('depth_map.png', depth_map_norm)

# using camera pose to change the depth map to the world coordinate

c2ws = [
                [
                    -4.3711387287537207e-10,
                    0.0070710680447518826,
                    -0.007071067579090595,
                    -808.6956176757812
                ],
                [
                    -0.009999999776482582,
                    -5.844237316310341e-10,
                    3.374863583038845e-11,
                    -266.0
                ],
                [
                    -3.89386078936127e-10,
                    0.007071067579090595,
                    0.0070710680447518826,
                    150.0
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
        ]

c2ws = np.array(c2ws) 
c2ws = c2ws[:3,:4]
print(c2ws)
# 3x3 matrix
intrinsics =  np.array([[2317.6449482429634/4, 0, 960.0/4],
                        [0, 2317.6449482429634/4, 540.0/4],
                        [0, 0, 1]])


# project depth map to 3d points in local coordinate
# 1. get the pixel coordinate
# 2. get the depth value
# 3. project to 3d points

# 1. get the pixel coordinate
H,W = depth_map.shape
x = np.arange(W)
y = np.arange(H)
xx,yy = np.meshgrid(x,y)
xx = xx.flatten()
yy = yy.flatten()

# 2. get the depth value
depths = depth_map.flatten()

# 2: colorize each point with rgb value ToDO

# 3. project to 3d points
# x = (u - cx) * depth / fx
# y = (v - cy) * depth / fy
# z = depth
# 3xN
points = np.vstack([xx,yy,np.ones_like(xx)])
# 3xN
points = np.multiply(points, depths)
# 3xN
points = np.dot(np.linalg.inv(intrinsics), points)
# 3xN
points = np.dot(c2ws, np.vstack([points, np.ones_like(xx)]))
# 3xN
points = points[:3,:]
# Nx3
points = points.T
print(points.shape)


# save to ply file 
# import open3d  
import trimesh 
# pcd = open3d.geometry.PointCloud()
# pcd.points = open3d.utility.Vector3dVector(points.T)
# open3d.io.write_point_cloud("points.ply", pcd)

# create a mesh
mesh = trimesh.Trimesh(vertices=points.T)
mesh.export('points.ply')







#   "fl_x": 2317.6449482429634,
#   "fl_y": 2317.6449482429634,
#   "k1": 0,
#   "k2": 0,
#   "k3": 0,
#   "k4": 0,
#   "p1": 0,
#   "p2": 0,
#   "cx": 960.0,
#   "cy": 540.0,
#   "w": 1920.0,
#   "h": 1080.0,
