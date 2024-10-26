source ~/.bashrc 
micromamba activate h3dgs

'''
Demo 1, using small city block1 train data
'''
# step1: convert images and poses to colmap format .txt
# python matrixcity_to_colmap.py --input_dir /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city --output_dir /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/ note that we have to map the normaliized 100 meter unit to 1 meter unit

# step2: create a empty points3D.txt file in /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_block1_debug/camera_calibration/rectified/sparse/
# nano points3D.txt
# colmap model_converter --input_path=/data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_block1_debug/camera_calibration/rectified/sparse/known --output_path=/data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_block1_debug/camera_calibration/rectified/sparse/known --output_type=BIN

# optional step 3 might not be needed
# optional step 3: run colmap to convert this known pos


# step 4.2 do not use colmap, change the way of make chunk
python preprocess/make_chunk_known_depth.py --base_dir /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_block1_debug/camera_calibration/aligned/sparse/known --images_dir /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_block1_debug/camera_calibration/rectified/images --output_path /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_block1_debug/camera_calibration/raw_chunks --depth_dir /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city_depth_float32/aerial/train/block_1_depth 



python train_coarse.py -s /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_block1_debug/ -i /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_block1_debug/camera_calibration/rectified/images --skybox_num 100000 --position_lr_init 0.00016 --position_lr_final 0.0000016 --model_path /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_block1_debug/camera_calibration/output/scaffold


----------------------------------------------------------------------
# step 4: extract feature for block 0 
# step 4.3 we mannual create points3D.txt file
# 3D point list with one line of data per point:
#  POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
### street
# 1. python matrixcity_street_to_colmap.py   ,save in  /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug
# 2. try following standard, first convert to Bin
# colmap model_converter --input_path=/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug/camera_calibration/rectified/sparse/known --output_path=/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug/camera_calibration/rectified/sparse/0 --output_type=BIN
# cp -r rectified/sparse/ aligned/
#3.  python preprocess/generate_chunks.py --project_dir /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug
python preprocess/generate_chunks.py --project_dir /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug --min_n_cams 50 --skip_bundle_adjustment
# python preprocess/make_depth_scale_dummy.py --chunks_dir /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug/camera_calibration/chunks/ --depths_dir=/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug/camera_calibration/rectified/depths/
# python preprocess/make_depth_scale_dummy.py --chunks_dir /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug/camera_calibration/aligned/ --depths_dir=/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_A_debug/camera_calibration/rectified/depths/


## street all
# colmap model_converter --input_path=/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_all_th50/camera_calibration/rectified/sparse/known --output_path=/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_all_th50/camera_calibration/rectified/sparse/0 --output_type=BIN
# cp -r rectified/sparse/ aligned/
#3.   python preprocess/generate_chunks.py --project_dir /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/steet_blocks_all_th50 --min_n_cams 20 --chunk_size 100  --skip_bundle_adjustment




### aerial
# colmap model_converter --input_path=/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_A_blocks_debug/camera_calibration/rectified/sparse/known --output_path=/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_A_blocks_debug/camera_calibration/rectified/sparse/0 --output_type=BIN
# cp -r rectified/sparse/ aligned/
#3.  python preprocess/generate_chunks.py --project_dir /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_A_blocks_debug --min_n_cams 10 --chunk_size 200  --skip_bundle_adjustment

# python preprocess/make_depth_scale_dummy.py --chunks_dir /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_A_blocks_debug/camera_calibration/chunks/ --depths_dir=/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_A_blocks_debug/camera_calibration/rectified/depths/

# python preprocess/make_depth_scale_dummy.py --chunks_dir /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_A_blocks_debug/camera_calibration/aligned/ --depths_dir=/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_A_blocks_debug/camera_calibration/rectified/depths/

# debug
# python preprocess/prepare_chunk.py --raw_chunk /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_A_blocks_debug/camera_calibration/raw_chunks/1_2 --out_chunk /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_A_blocks_debug/camera_calibration/chunks/1_2 --images_dir /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_A_blocks_debug/camera_calibration/rectified/images --skip_bundle_adjustment


# prepare chunk
## do this for each chunk
# python preprocess/prepare_chunk_all.py --chunks_dir /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_A_blocks_debug/camera_calibration/ --skip_bundle_adjustment --images_dir /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_A_blocks_debug/camera_calibration/rectified/images


# finish chunk, check mono depth





-------



mkdir known
mv *.txt known/
colmap feature_extractor \
    --database_path /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_block1_debug/camera_calibration/rectified/sparse/database.db \
    --image_path /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_block1_debug/camera_calibration/rectified/images \
    --SiftExtraction.use_gpu=false 

# step 5: optional replace the model with known intrinsics
colmap exhaustive_matcher --database_path /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_block1_debug/camera_calibration/rectified/sparse/database.db --SiftMatching.use_gpu=false

# step 6: match features 


# step 7: triangulate points
colmap point_triangulator \
    --database_path /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_block1_debug/camera_calibration/rectified/sparse/database.db   \
    --image_path /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_block1_debug/camera_calibration/rectified/images \
    --input_path /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_block1_debug/camera_calibration/rectified/sparse/known \
    --output_path /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_block1_debug/camera_calibration/rectified/sparse/0



# step 8: follow the preprocess steps, copy to align  
cp -r rectified/sparse/ aligned/

# step 9: generate chunks
 python preprocess/generate_chunks.py --project_dir /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_block1_debug/
 
python preprocess/make_chunk_known_depth.py --base_dir /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_block1_debug/camera_calibration/aligned/sparse/0 --images_dir /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_block1_debug/camera_calibration/rectified/images --output_path /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_block1_debug/camera_calibration/raw_chunks
# get error chunk exclude
# because points3D have problem, read make chunk code
# based on distance of camera, git  18_12 chunck
# goal get images_points3d from known pose and depth images

# Here, the first two lines define the information of the first image, and so on. The reconstructed pose of an image is specified as the projection from world to the camera coordinate system of an image using a quaternion (QW, QX, QY, QZ) and a translation vector (TX, TY, TZ). The quaternion is defined using the Hamilton convention, which is, for example, also used by the Eigen library. The coordinates of the projection/camera center are given by -R^t * T, where R^t is the inverse/transpose of the 3x3 rotation matrix composed from the quaternion and T is the translation vector. The local camera coordinate system of an image is defined in a way that the X axis points to the right, the Y axis to the bottom, and the Z axis to the front as seen from the image.







# colmap model_converter --input_path=/work/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap --output_path=/work/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/rectified/sparse/0 --output_type=BIN

# usage: this code change scenes in matrix city both aerial and street, (both train and test) to colmap format
# python matrixcity_to_colmap.py --input_dir /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city --output_dir /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/
# 
# python preprocess/auto_reorient_npts.py --input_path=/data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/rectified/sparse/1 --output_path=/data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/aligned/sparse/0  --upscale=1

# python preprocess/generate_chunks.py --project_dir /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/
# python preprocess/make_chunk.py --base_dir /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/aligned/sparse/0  --images_dir /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/rectified/images  --output_path /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/raw_chunks --min_n_cams 30


# python preprocess/prepare_chunk.py --raw_chunk /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/raw_chunks/0_0 --out_chunk /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/chunks/0_0 --images_dir /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/rectified/images --skip_bundle_adjustment


# python train_coarse.py -s /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/aligned/ -i /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/rectified/images --skybox_num 100000 --position_lr_init 0.00016 --position_lr_final 0.0000016 --model_path /work/qimaqi/datasets/small_city/output/scaffold


#  python preprocess/concat_chunks_info.py --base_dir=/data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/chunks --dest_dir=/data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/aligned

--depths_dir <project/rectified/depths> --preprocess_dir <path to hierarchical_gaussians/preprocess_dir>




colmap feature_extractor \
    --database_path /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/rectified/sparse/database.db \
    --image_path /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/rectified/images \
    --SiftExtraction.use_gpu=false 




# Modifying the database is possible in many ways, but an easy option is to use the provided scripts/python/database.py script. Otherwise, you can skip this step and simply continue as follows:

colmap sequential_matcher --database_path /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/rectified/sparse/database.db --SiftMatching.use_gpu=false

colmap point_triangulator \
    --database_path /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/rectified/sparse/database.db  \
    --image_path /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/rectified/images \
    --input_path /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/rectified/sparse/0/ \
    --output_path /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/rectified/sparse/1/
