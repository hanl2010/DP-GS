import os

scans = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
# scans = [24]
dataset_root = DTU_DATA_ROOT
DTU = Path_to_the_GT_DTU_point_clouds
sfm_type = "colmap_sfm"
data_type = "small_overlap"

for scan in scans:
    dataset_dir = os.path.join(dataset_root, data_type)
    input_mesh = f"output/DTU/scan{scan}_{sfm_type}_{data_type}/train/ours_10000/fuse_post.ply"

    cmd = ("python scripts/dtu_eval/evaluate_single_scene.py "
           f"--input_mesh {input_mesh} "
           f"--scan_id {scan} "
           f"--dataset_dir {dataset_dir} "
           f"--DTU {DTU} "
           f"--data_type {data_type} ")

    print(cmd)
    os.system(cmd)