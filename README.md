<p align="center">

  <h2 align="center">LoopSplat: Loop Closure by Registering 3D Gaussian Splats</h2>
  <p align="center">
    <a href="https://www.zhuliyuan.net/"><strong>Liyuan Zhu</strong></a><sup>1</sup>
    ·
    <a href="https://unique1i.github.io/"><strong>Yue Li</strong></a><sup>2</sup>
    ·
    <a href="https://scholar.google.com/citations?user=phiETm4AAAAJ&hl=en/"><strong>Erik Sandström</strong></a><sup>3</sup>
    ·
    <a href="https://shengyuh.github.io/"><strong>Shengyu Huang</strong></a><sup>3</sup>
    ·
    <a href="https://prs.igp.ethz.ch/group/people/person-detail.schindler.html/"><strong>Konrad Schindler</strong></a><sup>3</sup>
    ·
    <a href="https://ir0.github.io/"><strong>Iro Armeni</strong></a><sup>1</sup>
</p>

<p align="center">
    <sup>1</sup>Stanford University · <sup>2</sup>University of Amsterdam · <sup>3</sup>ETH Zurich
</p>
   <h3 align="center">

   [![arXiv](https://img.shields.io/badge/arXiv-2408.10154-blue?logo=arxiv&color=%23B31B1B)](https://arxiv.org/abs/2408.10154) [![ProjectPage](https://img.shields.io/badge/Project_Page-LoopSplat-blue)](https://loopsplat.github.io/)
  <div align="center"></div>
</p>



<p align="center">
  <a href="">
    <img src="./assets/loopsplat.gif" width="100%">
  </a>
</p>

## 📃 Description
<p align="center">
  <a href="">
    <img src="./assets/architecture.png" width="100%">
  </a>
</p>

**LoopSplat** is a coupled RGB-D SLAM system that uses Gaussian splats as a unified scene representation for tracking, mapping, and maintaining global consistency. In the front-end, it continuously estimates the camera position while constructing the scene using Gaussian splats submaps. When the camera traverses beyond a predefined threshold, the current submap is finalized, and a new one is initiated. Concurrently, the back-end loop closure module monitors for location revisits. Upon detecting a loop, the system generates a pose graph, incorporating loop edge constraints derived from our proposed 3DGS registration. Subsequently, pose graph optimization (PGO) is executed to refine both camera poses and submaps, ensuring overall spatial coherence.

# 🛠️ Setup
The code has been tested on:

- Ubuntu 22.04 LTS, Python 3.10.14, CUDA 12.2, GeForce RTX 4090/RTX 3090
- CentOS Linux 7, Python 3.12.1, CUDA 12.4, A100/A6000

## 📦 Repository

Clone the repo with `--recursive` because we have submodules:

```
git clone --recursive git@github.com:GradientSpaces/LoopSplat.git
cd LoopSplat
```

## 💻 Installation
Make sure that gcc and g++ paths on your system are exported:

```
export CC=<gcc path>
export CXX=<g++ path>
```

To find the <i>gcc path</i> and <i>g++ path</i> on your machine you can use <i>which gcc</i>.


Then setup environment from the provided conda environment file,

```
conda env create -f environment.yml
conda activate loop_splat
```

You will also need to install <i>hloc</i> for loop detection and 3DGS registration.
```
cd thirdparty/Hierarchical-Localization
python -m pip install -e .
cd ../..
```

We tested our code on RTX3090 and RTX A6000 GPUs respectively and Ubuntu22 and CentOS7.5.

## 🚀 Usage

Here we elaborate on how to load the necessary data, configure Gaussian-SLAM for your use-case, debug it, and how to reproduce the results mentioned in the paper.

  <!-- <details>
  <summary><b>Downloading the Data</b></summary> -->
  ### Downloading the Datasets
  We tested our code on Replica, TUM_RGBD, ScanNet, and ScanNet++ datasets. We also provide scripts for downloading Replica and TUM_RGBD in `scripts` folder. Install git lfs before using the scripts by running ```git lfs install```.

  For reconstruction evaluation on <b>Replica</b>, we follow [Co-SLAM](https://github.com/JingwenWang95/neural_slam_eval?tab=readme-ov-file#datasets) mesh culling protocal, please use their code to processing the meshes first.

  For downloading ScanNet, follow the procedure described on <a href="http://www.scan-net.org/">here</a>.<br>
  <b>Pay attention! </b> There are some frames in ScanNet with `inf` poses, we filter them out using the jupyter notebook `scripts/scannet_preprocess.ipynb`. Please change the path to your ScanNet data and run the cells.

  For downloading ScanNet++, follow the procedure described on <a href="https://kaldir.vc.in.tum.de/scannetpp/">here</a>.<br>
  
  The config files are named after the sequences that we used for our method.
  <!-- </details> -->

  ### Running the code
  Start the system with the command:

  ```
  python run_slam.py configs/<dataset_name>/<config_name> --input_path <path_to_the_scene> --output_path <output_path>
  ```
  
  You can also configure input and output paths in the config yaml file.
  

  ### Reproducing Results

  You can reproduce the results for a single scene by running:

  ```
  python run_slam.py configs/<dataset_name>/<config_name> --input_path <path_to_the_scene> --output_path <output_path>
  ```

  If you are running on a SLURM cluster, you can reproduce the results for all scenes in a dataset by running the script:
  ```
  ./scripts/reproduce_sbatch.sh
  ``` 
  Please note the evaluation of ```depth_L1``` metric requires reconstruction of the mesh, which in turns requires headless installation of open3d if you are running on a cluster.
  
  ## 📧 Contact
  If you have any questions regarding this project, please contact Liyuan Zhu (liyzhu@stanford.edu). If you want to use our intermediate results for qualitative comparisons, please reach out to the same email.

# ✏️ Acknowledgement
Our implementation is heavily based on <a href="https://vladimiryugay.github.io/gaussian_slam/index.html">Gaussian-SLAM</a> and <a href="https://github.com/muskie82/MonoGS">MonoGS</a>. We thank the authors for their open-source contributions. If you use the code that is based on their contribution, please cite them as well. We thank [Jianhao Zheng](https://jianhao-zheng.github.io/) for the help with datasets and [Yue Pan](https://github.com/YuePanEdward) for the fruitful discussion.<br>

# 🎓 Citation

If you find our paper and code useful, please cite us:

```bib
@misc{zhu2024_loopsplat,
      title={LoopSplat: Loop Closure by Registering 3D Gaussian Splats}, 
      author={Liyuan Zhu and Yue Li and Erik Sandström and Shengyu Huang and Konrad Schindler and Iro Armeni},
      year={2024},
      eprint={2408.10154},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
