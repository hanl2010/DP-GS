import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time

# scenes = ["Barn","Caterpillar", "Ignatius", "Truck", "Courthouse", "Meetingroom"]
scenes = ["Barn","Caterpillar", "Ignatius", "Truck"]

dataset_name = "TNT"
# dataset_name = "TNT_sparse"
data_base_path='/home/hanl/hanl/data/public_dataset/TNT_GOF/'
# data_base_path='/data10_1/hanl/code/02_paper_code/MAtCha/output/tnt'
out_base_path=f'output/{dataset_name}'
iterations = 10000
resolution = 2
n_views = 10
sfm_type = "mast3r_sfm"


excluded_gpus = {0}
mem_threshold = 0.2
base_dir = os.path.abspath(os.getcwd())

jobs = scenes


def train_block(gpu_id, scene):

    mast3r_data_dir = os.path.join(data_base_path, f"{scene}", f"mast3r_{n_views}views")

    # model_path = f"{out_base_path}/{scene}"
    model_path = os.path.join(out_base_path, f"{scene}_{sfm_type}_{n_views}views")

    cmd = (
           f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py '
           f'-s {mast3r_data_dir} '
           f'-m {model_path} '       
           f'--dataset_name {dataset_name} '
           f'--resolution {resolution} '
           f'--iterations {iterations} '
           f'--depth_smooth_iter_start {iterations} '
           f'--sfm_type mast3r_sfm '
           )
    print(cmd)
    os.system(cmd)

    cmd = (f'CUDA_VISIBLE_DEVICES={gpu_id} python render_and_mesh_extract.py '
           f'--model_path {model_path} '
           f'--dataset_name {dataset_name} '
           f'--skip_train '
           f'--skip_test '
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
              all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=mem_threshold))
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
                     print("No GPU available at the moment. Retrying in 1 minutes.")
                     time.sleep(60)
              else:
                     time.sleep(10)

       print("All blocks have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=10) as executor:
    dispatch_jobs(jobs, executor)
