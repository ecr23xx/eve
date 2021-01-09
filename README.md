# Efficient 3D Video Engine Using Frame Redundancy

Gao Peng, [Bo Pang](https://bopang1996.github.io), [Cewu Lu](https://www.mvig.org)

Code for our WACV 2021 paper [Efficient 3D Video Engine Using Frame Redundancy](https://openaccess.thecvf.com/content/WACV2021/papers/Peng_Efficient_3D_Video_Engine_Using_Frame_Redundancy_WACV_2021_paper.pdf).

## Data

Download semantic KITTI from its official website, and organize the folder as follow:

```
data/semanticKITTI
├── semantic-kitti-all.yaml
├── semantic-kitti.yaml
└── sequences
```

## Install

Install following dependencies first

* PyTorch == 1.3
* MinkowskiEngine == 0.2.4

Then run

```
$ pip install -r requirements.txt
$ python setup.py build develop
```

## Getting started

Train

```
$ export NUM_GPUS=4
$ python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
    train_net.py \
    --config-file=config_files/semantic_kitti/minkunet.yaml \
    --transfer
```

Test

```
$ export NUM_GPUS=4
$ python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
    train_net.py \
    --config-file=config_files/semantic_kitti/minkunet.yaml \
    MODEL.WEIGHT logs/semantic_kitti/minkunet/model_best.pth
```

## Citation

If this project helps you in your research or project, please cite this paper:

```
@inproceedings{gao2021eve,
  Author    = {Gao Peng and Bo Pang and Cewu Lu},
  Title     = {{Efficient 3D Video Engine Using Frame Redundancy}},
  Booktitle = {{WACV}},
  Year      = {2021}
}
```
