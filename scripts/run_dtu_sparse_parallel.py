import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
import argparse

parser = argparse.ArgumentParser(description='Run training.')
parser.add_argument('--data_path', type=str, required=True, help='Path for scene')
parser.add_argument("--data_type", type=str, required=True, choices=['small_overlap', 'large_overlap'])
parser.add_argument('--output_path', type=str, default="output", help='Path for output')
parser.add_argument("--dataset_name", type=str, default="DTU", help='Name of the dataset')
parser.add_argument('--resolution', type=int, default=2, help='Resolution for training')
parser.add_argument("--iterations", type=int, default=10000, help='Number of training iterations')
parser.add_argument('--sfm_type', type=str, default="colmap_sfm", choices=["colmap_sfm", "mast3r_sfm"], help='Type of SfM to use')
parser.add_argument("--single_view_weight_from_iter", type=int, default=3000)
parser.add_argument("--multi_view_weight_from_iter", type=int, default=3000)
args = parser.parse_args()

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]

dataset_name = args.dataset_name
data_path = args.data_path
data_type = args.data_type
output_path = args.output_path
resolution = args.resolution
iterations = args.iterations
sfm_type = args.sfm_type  ## colmap_sfm or mast3r_sfm


if data_type == "large_overlap":
    depth_smooth_weight = 0.0
    depth_propagation_iters = 5
else:
    depth_smooth_weight = 0.03
    depth_propagation_iters = 10

mem_threshold = 0.1
excluded_gpus = set({})
jobs = scenes

def train_block(gpu_id, scene):
    scene_name = f"scan{scene}"
    scene_path = os.path.join(data_path, scene_name)
    model_path = os.path.join(output_path, dataset_name, f"{scene_name}_{sfm_type}_{data_type}")

    # base_dir = os.path.abspath(os.getcwd())
    # if sfm_type == "mast3r_sfm":
    #     cmd = (f"CUDA_VISIBLE_DEVICES={gpu_id} python run_mast3r.py "
    #            f"--scene_path {scene_path} "
    #            f"--output_dir {scene_path}/mast3r "
    #            f"--use_all_images")
    #     print(cmd)
    #     os.chdir(os.path.join(base_dir, "mast3r"))
    #     os.system(cmd)
    #     os.chdir(base_dir)
    #
    #
    #
    # cmd = (f"CUDA_VISIBLE_DEVICES={gpu_id} python test.py "
    #        f"--data_dir {scene_path} "
    #        f"--get_normal "
    #        f"--resolution {resolution} ")
    # print(cmd)
    # os.chdir(os.path.join(base_dir, "Metric3D"))
    # os.system(cmd)
    # os.chdir(base_dir)


    cmd = (
        f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py '
        f'-s {scene_path} '
        f'-m {model_path} '
        f'--dataset_name {dataset_name} '
        f'--resolution {resolution} '
        f'--iterations {iterations} '
        f'--sfm_type {sfm_type} '
        f'--opacity_decay '
        f'--depth_smooth_weight {depth_smooth_weight} '
        f'--depth_propagation_iters {depth_propagation_iters} '
    )
    print(cmd)
    os.system(cmd)


    if scene_name in ["scan37", "scan55", "scan63", "scan97"]:
        num_cluster = 4
    else:
        num_cluster = 1

    cmd = (f'CUDA_VISIBLE_DEVICES={gpu_id} python render_and_mesh_extract.py '
           f'--model_path {model_path} '
           f'--dataset_name {dataset_name} '
           f'--skip_test '
           f'--skip_train '
           f'--num_cluster {num_cluster} '
           f'--voxel_size 0.002 '
           )
    print(cmd)
    os.system(cmd)

    return True


def worker(gpu_id, scene):
    print(f"Starting job on GPU {gpu_id} with scene {scene}\n")
    train_block(gpu_id, scene)
    print(f"Finished job on GPU {gpu_id} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.


def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=mem_threshold, maxLoad=0.9))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)
            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        if len(jobs) > 0:
            print("No GPU available at the moment. Retrying in 30 seconds.")
            time.sleep(30)
        else:
            time.sleep(10)

    print("All blocks have been processed.")

## Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=10) as executor:
    dispatch_jobs(jobs, executor)