# Transforming NeRFusion: Fusion, ScanNet++, Depth and Distortion Loss for Superior Volume Reconstruction

## Introduction

This project extends the NeRFusion framework to work with the new ScanNet++ dataset. Key improvements include the reimplementation of the GRUFusion block, which enhances multi-view information fusion for local volume reconstruction. This allows the system to better synthesize and interpret data from different perspectives. Additionally, depth and distortion loss components have been added during training to optimize learning and improve model accuracy. The fusion process is also visualized by comparing input views with the updated radiance field and extracting the mesh, providing a clear understanding of the process. These enhancements aim to make the NeRFusion framework more robust and versatile, paving the way for advanced 3D reconstruction tasks.

This is a re-development of the original NeRFusion code based heavily on [nerf_pl](https://github.com/kwea123/nerf_pl), [NeuralRecon](https://github.com/zju3dv/NeuralRecon), [MVSNeRF](https://github.com/apchenstu/mvsnerf). We thank the authors for sharing their code. 


## Installation

#### Weights & Biases
In order for logging to work, please export your WANDB_API_KEY 

### Requirements
All the codes are tested in the following environment:
* Linux (Ubuntu 20.04 or above)
* 32GB RAM (in order to load full size images)
* NVIDIA GPU with Compute Compatibility >= 75 and VRAM >= 6GB, CUDA >= 11.3

### Dependencies
```shell
# Ubuntu 18.04 and above is recommended.
sudo apt install libsparsehash-dev  # you can try to install sparsehash with conda if you don't have sudo privileges.
conda env create -f environment.yaml
conda activate nerfusion
```
<!-- Follow instructions in [torchsparse](https://github.com/mit-han-lab/torchsparse) to install torchsparse. -->

* Python libraries
    * Install `pytorch>=1.11.0` by `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113`
    * Install `torch-scatter` following their [instruction](https://github.com/rusty1s/pytorch_scatter#installation)
    * Install `tinycudann` following their [instruction](https://github.com/NVlabs/tiny-cuda-nn#requirements) (compilation and pytorch extension)
    * Install `apex` following their [instruction](https://github.com/NVIDIA/apex#linux)
    * Install `torchsparse` following their [instruction](https://github.com/mit-han-lab/torchsparse#installation)
    * Install core requirements by `pip install -r requirements.txt`

* Cuda extension: Upgrade `pip` to >= 22.1 and run `pip install models/csrc/` (please run this each time you `pull` the code)

## Data Preparation
We follow the same data organization as the original NeRF, which expects camera parameters to be provided in a `transforms.json` file. We also support data from NSVF, NeRF++, colmap and ScanNet.

### Pretrained Model on ScanNet
Download the [pretrained NeuralRecon weights](https://drive.google.com/file/d/1zKuWqm9weHSm98SZKld1PbEddgLOQkQV/view?usp=sharing) and put it under 
`PROJECT_PATH/checkpoints/release`.
You can also use [gdown](https://github.com/wkentaro/gdown) to download it in command line:
```bash
mkdir checkpoints && cd checkpoints
gdown --id 1zKuWqm9weHSm98SZKld1PbEddgLOQkQV
```

### Training 

To run training on a given dataset, the data should be organized in the original NeRF-style:

```
data
├── transforms.json
├── images
│   ├── 0000.jpg
    ├── 0001.jpg
    ├── ...
```

### Per-Scene Optimization
The following script trains models from scratch and automatically uploads metrics and artifacts to Weights & Biases.

```bash
python train.py --dataset_name DATASET_NAME --root_dir DIR_TO_SCANNET_SCENE --exp_name EXP_NAME
```


## Training Procedure

Please download and organize the datasets in the following manner:
```
├──data/
    ├──DTU/
    ├──google_scanned_objects/
    ├──ScanNet/
    ├──ScanNetPP/
```

For google scanned objects, we used [renderings](https://drive.google.com/file/d/1w1Cs0yztH6kE3JIz7mdggvPGCwIKkVi2/view?usp=sharing) from IBRNet. Download with:

```
gdown https://drive.google.com/uc?id=1w1Cs0yztH6kE3JIz7mdggvPGCwIKkVi2
unzip google_scanned_objects_renderings.zip
```

For DTU and ScanNet, please use the official toolkits for downloading and processing of the data, and unpack the root directory to the `data` folder mentioned above. Train with:

```bash
python train.py --train_root_dir DIR_TO_DATA --exp_name EXP_NAME
```

See `opt.py` for more options.


# How to run our contributions for the ML for 3D project

## Inference using Fusion

The following command will generate and extract the global feature volumes created by the GRUFusion module, leveraging the pre-trained weights of NeuralRecon.

```bash
python train_fusion.py --cfg ./config/test.yaml
```

Once the global feature volume is available, you can run fusion based scene-reconstruction on any scannet scene by including the 
--use_gru_fusion flag.

## Inference using Depth Loss

Depth loss is added by default, but can be deactivated using --skip_depth_loading.

## Inference using Distortion Loss

Use flag --distortion_loss_w and specify the weight (0 by defautl). Good values are 1e-3 for real scene and 1e-2 for synthetic scene.

## Inference on Scannet++

Follow the procedures outlines above. Specify the dataset name as 
--dataset_name scannetpp. Note that training is done on DSLR images that first need to be undistorted using the scannetpp-toolkit.

## Using Weights & Biases Sweep agents

All experiments are automatically tracked in Weights and Biases. To deactivate this use the --debug flag.

Use flag --use_sweep to leverage wandb sweep agents for hyperparameter tuning (default: False).

## Real-time visualization of model checkpoints / learned scenes

Make sure you have the following library installed:
`conda install -c conda-forge libstdcxx-ng`

Then execute the following command:

`python show_gui.py --dataset_name scannet --root_dir data/scannet_official/scans/scene0000_00 --ckpt_path ckpts/scannet/test_scannet_8frames/epoch=29.ckpt`


* w and s can be used to move forward and backward instead of using the mouse scroll.
* q and e can be used to move up and down, and a and d can be used to move left and right.
* Use right-click instead of left-click to control rotation.