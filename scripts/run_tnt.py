import os
import argparse

parser = argparse.ArgumentParser(description='Run training.')
parser.add_argument('--scene_path', type=str, required=True, help='Path for scene')
parser.add_argument("--scene_name", type=str, required=True, help='Name of scene')
parser.add_argument('--n_views', type=int, required=True, help='Number of views')
parser.add_argument('--output_path', type=str, default="output", help='Path for output')
parser.add_argument("--dataset_name", type=str, default="TNT", help='Name of the dataset')
parser.add_argument('--resolution', type=int, default=2, help='Resolution for training')
parser.add_argument("--iterations", type=int, default=10000, help='Number of training iterations')
parser.add_argument('--sfm_type', type=str, default="mast3r_sfm", help='Type of SfM to use (colmap_sfm or mast3r_sfm)')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
args = parser.parse_args()

scene_name = args.scene_name
model_path = os.path.join(args.output_path, args.dataset_name, f"{scene_name}_{args.sfm_type}_{args.n_views}views")
mast3r_data_dir = os.path.join(args.scene_path, f"mast3r_{args.n_views}views")

base_dir = os.path.abspath(os.getcwd())
cmd = (f"CUDA_VISIBLE_DEVICES={args.gpu_id} python run_mast3r.py "
       f"--scene_path {args.scene_path} "
       f"--output_dir {mast3r_data_dir} "
       f"--n_images {args.n_views} ")
print(cmd)
os.chdir(os.path.join(base_dir, "mast3r"))
os.system(cmd)
os.chdir(base_dir)

cmd = (f"CUDA_VISIBLE_DEVICES={args.gpu_id} python test.py "
       f"--data_dir {mast3r_data_dir} "
       f"--get_normal "
       f"--resolution {args.resolution} ")
print(cmd)
os.chdir(os.path.join(base_dir, "Metric3D"))
os.system(cmd)
os.chdir(base_dir)


cmd = (
       f'CUDA_VISIBLE_DEVICES={args.gpu_id} python train.py '
       f'-s {mast3r_data_dir} '
       f'-m {model_path} '       
       f'--dataset_name {args.dataset_name} '
       f'--resolution {args.resolution} '
       f'--iterations {args.iterations} '
       f'--sfm_type mast3r_sfm '
       f'--depth_smooth_iter_start {args.iterations}'
       )
print(cmd)
os.system(cmd)


cmd = (f'CUDA_VISIBLE_DEVICES={args.gpu_id} python render_and_mesh_extract.py '
       f'--model_path {model_path} '
       f'--dataset_name {args.dataset_name} '
       f'--skip_train '
       f'--skip_test '
       )
print(cmd)
os.system(cmd)


