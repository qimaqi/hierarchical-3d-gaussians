colmap sequential_matcher --database_path /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/rectified/sparse/database.db --SiftMatching.use_gpu=false

colmap point_triangulator \
    --database_path /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/rectified/sparse/database.db  \
    --image_path /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/rectified/images \
    --input_path /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/rectified/sparse/0 \
    --output_path /data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/rectified/sparse/1
