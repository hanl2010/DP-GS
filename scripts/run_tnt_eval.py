import os

scenes = ['Truck', 'Caterpillar', 'Barn', 'Ignatius']
gt_data_path = GT_DATA_PATH

out_base_path=f'output/TNT'
n_views = N_VIEWS # 5, 10 or 20


excluded_gpus = set()
mem_threshold = 0.2
base_dir = os.path.abspath(os.getcwd())


jobs = scenes

for scene in scenes:
    mesh_path = f"{out_base_path}/{scene}_mast3r_sfm_{n_views}views/train/ours_10000"
    # require open3d==0.9
    cmd = (f'/data14_2/hanl/miniconda3/envs/python37/bin/python scripts/tnt_eval/run.py --dataset-dir {gt_data_path}/{scene} '
           f'--traj-path {gt_data_path}/{scene}/{scene}_COLMAP_SfM.log --ply-path {mesh_path}/fuse_post.ply '
           f'--out-dir {mesh_path}')
    print(cmd)
    os.system(cmd)


