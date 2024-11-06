import trimesh
import numpy as np
import os

# srun --nodes=1 --ntasks=24 --cpus-per-task=1 --mem-per-cpu=8G --time=240 --pty bash -i 

output_path = '/cluster/work/cvl/qimaqi/cvpr_2025/datasets/MatrixCity/colmap_aerial/camera_calibration/rectified/'

points3d_xyz = trimesh.load('/cluster/work/cvl/qimaqi/cvpr_2025/datasets/MatrixCity/small_city_pc/point_cloud_ds20/aerial/Block_all_.ply')
POINT3D_ID = np.arange(len(points3d_xyz.vertices)) + 1
color3d = points3d_xyz.visual.vertex_colors
print("===== color3d ======", color3d.shape)
color3d = color3d[:, :3]

# srun --nodes=1 --ntasks=8 --cpus-per-task=1 --mem-per-cpu=8G --time=240--pty bash -i 
all_files = os.listdir(os.path.join(output_path,'sparse', 'known'))
images_file = [file for file in all_files if file.startswith('images_')]
points3d_file = [file for file in all_files if file.startswith('points3D_')]
# merge images
images_file = sorted(images_file)
new_image_file = os.path.join(output_path,'sparse', 'known', 'images.txt')
with open (new_image_file, 'w') as f_new:
    f_new.write('# Image list with one line of data per image:\n')
    f_new.write('# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n')
    for i, file in enumerate(images_file):
        with open(os.path.join(output_path,'sparse', 'known', file), 'r') as f_read:
            for line in f_read:
                if line.startswith('#'):
                    continue
                f_new.write(line)

# merge points3d
new_points3d_file = os.path.join(output_path,'sparse', 'known', 'points3D.txt')

points3d_file = sorted(points3d_file)
track_summary =  [[] for _ in range(len(points3d_xyz.vertices))]
for i, file in enumerate(points3d_file):
    with open(os.path.join(output_path,'sparse', 'known', file), 'r') as f_read:
        for line in f_read:
            if line.startswith('#'):
                continue
            line = line.strip().split()
            point3d_id = int(line[0])
            track = line[8:]
            # print("track", track)
            track = [[int(track[i]), int(track[i+1])] for i in range(0, len(track), 2)]
            track_summary[point3d_id-1] += track

with open(new_points3d_file, 'w') as f:
    f.write('# 3D point list with one line of data per point:\n')
    f.write('# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
    for i, point3d_id in enumerate(POINT3D_ID):
        # only save points if track is not empty
        point3d = points3d_xyz.vertices[i]
        #self.points3d['XYZ'][i]
        r, g, b =color3d[i]  #self.color3d[i]
        error = 0
        track =  track_summary[i]
        if len(track) > 0:
            track_str = ' '.join([f'{t[0]} {t[1]}' for t in track])
            f.write(f'{point3d_id} {point3d[0]} {point3d[1]} {point3d[2]} {r} {g} {b} {error} {track_str}\n')

