<h1 align="center"> Sparse-View Surface Reconstruction using Gaussian Splatting through High-Confidence Depth Propagation with Normal Priors </h1>

<p align="center">
    <strong>Liang Han</strong>
    ·
    <strong>Bangcai Wei</strong>
    ·
    <a href="https://junshengzhou.github.io/"><strong>Junsheng Zhou</strong></a>
    ·
    <a href="https://yushen-liu.github.io/"><strong>Yu-Shen Liu</strong></a>
    ·
    <a href="https://h312h.github.io/"><strong>Zhizhong Han</strong></a>
</p>

<h2 align="center">ECCV 2026</h2>


## [Project page](https://hanl2010.github.io/DP-GS/) |  [Paper](https://arxiv.org/abs/2607.03765)
This is the official repo for the implementation of **Sparse-View Surface Reconstruction using Gaussian Splatting through High-Confidence Depth Propagation with Normal Priors**.

## Usage

### Setup

1. Clone this repository

```shell
git clone https://github.com/hanl2010/DP-GS.git
```
2. Setup Environment

```shell
conda create -n DP_GS python=3.10
conda activate DP_GS

pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Install the MASt3R-SfM dependencies
```shell
cd mast3r/asmk/cython
cythonize *.pyx
cd ..
pip install .
cd ..
cd ../dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../../
```
3. Download the MASt3R-SfM checkpoint
```shell
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P ./mast3r/checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P ./mast3r/checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P ./mast3r/checkpoints/
```
4. Download the Metric3D checkpoint
- Download the Metric3D checkpoint from [Metric3D](https://github.com/YvanYin/Metric3D) ([weight link](https://drive.google.com/file/d/1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr/view?usp=drive_link))
- Put it in the *Metric3D/weight* folder.

### DTU Dataset
- Download the processed dataset: [DTU](https://drive.google.com/file/d/11NBPiVAnjB7HxMIE56rl_2Zk1it0IkWp/view?usp=drive_link).
- The complete dataset can be downloaded from [2DGS](https://surfsplatting.github.io/).
- You need to download the ground truth point clouds from the [DTU dataset](https://roboimagedata.compute.dtu.dk/?page_id=36).

### TNT Dataset
- Download the dataset from [official webiste](https://www.tanksandtemples.org/download/). You need to download the reconstruction, alignment and cropfiles.

### Training and Evaluation 

- **DTU dataset**

Training on a single scene
```shell
python train.py -s <DATA_ROOT>/small_overlap/scan24 -m output/DTU/scan24 --dataset_name DTU --resolution 2 --iterations 10000 --sfm_type colmap_sfm --opacity_decay 
```

Evaluate single scene
```shell
python scripts/dtu_eval/evaluate_single_scene.py --input_mesh <MESH_PATH> --scan_id <SCAN_ID> --dataset_dir <DTU_DATA_PATH> --DTU <GT_POINTS_PATH> --data_type <DATA_TYPE>
```

Training on multiple scenes
```shell
python scripts/run_dtu_sparse_parallel.py --data_path <DATA_ROOT>/small_overlap --data_type small_overlap
```

Evaluate all scenes
```shell
python scripts/run_dtu_eval.py 
```


- **TNT dataset**

Training
```shell
python scripts/run_tnt.py --scene_path <SCENE_PATH> --scene_name <SCENE_NAME> --n_views <5, 10 or 20>     
```

Evaluation
```shell
python scripts/run_tnt_eval.py
```


## Citation

Cite as below if you find this repository is helpful to your project:

```
@article{han2026dpgs,
  title={Sparse-View Surface Reconstruction using Gaussian Splatting through High-Confidence Depth Propagation with Normal Priors},
  author={Han, Liang and Wei, Bangcai and Zhou, Junsheng and Liu, Yu-Shen and Han, Zhizhong},
  journal={arXiv preprint arXiv:2607.03765},
  year={2026}
}
```

## Acknowledgement
This project is built upon [PGSR](https://github.com/zju3dv/PGSR) and [IronDepth](https://github.com/baegwangbin/IronDepth). Thanks for these great project.
