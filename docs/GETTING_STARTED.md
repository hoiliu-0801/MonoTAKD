# Getting Started
The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs), 
and the model configs are located within [tools/cfgs](../tools/cfgs) for different datasets. 


## Dataset Preparation
Currently we provide the dataloader of KITTI dataset and NuScenes dataset.  

### KITTI Dataset
* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):
* If you would like to use the depth maps for trainval set, download the precomputed [depth maps](https://drive.google.com/file/d/1qFZux7KC_gJ0UHEg-qGJKqteE9Ivojin/view?usp=sharing) for the KITTI trainval set
* Download the [KITTI Raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php) and put in into data/kitti/raw/KITTI_Raw
* (optional) If you want to use the [sparse depth maps](https://www.cvlibs.net/datasets/kitti/eval_depth_all.php) for KITTI Raw, download it and put it into data/kitti/raw/depth_sparse

```
OpenPCDet
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
│   │   │── raw
|   |   |   |——calib & KITTI_Raw & (optional: depth_sparse)
├── pcdet
├── tools
```

* Generate the data infos by running the following command (kitti train, kitti val, kitti test): 
```python 
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

* Generate the data infos by running the following command (kitti train + eigen clean, unlabeled):
```python 
python -m pcdet.datasets.kitti.kitti_dataset_cmkd create_kitti_infos_unlabel tools/cfgs/dataset_configs/kitti_dataset.yaml
```

### Nuscenes Dataset
<!-- Please refer to [this link](https://github.com/Cc-Hy/CMKD-MV). -->

## Pretrained Models
****
If you would like to use some pretrained models, download them and put them into ../checkpoints
```
OpenPCDet
├── checkpoints
|   ├── second_teacher.pth
|   ├── ···
├── data
├── pcdet
├── tools
```

### STUDENT MODEL

* Train with multiple GPUs or multiple machines
```
CUDA_VISIBLE_DEVICES=0,1 python train_TAKD.py --cfg cfgs/kitti_models/TAKD/TAKD-scd/kitti_R50_scd_TAKD.yaml --pretrained_lidar_model ../checkpoints/monotakd_kitti_teacher.pth --pretrained_img_model ../checkpoints/monotakd_kitti_TA-2161.pth
```

* Test model
```
python test_TAKD.py --cfg cfgs/kitti_models/TAKD/TAKD-scd/kitti_R50_scd_TAKD.yaml --ckpt ../checkpoints/monotakd_kitti_student-8015e10.pth
```

### TEACHER MODEL
* Train Teacher **with** pre-trained weights. 
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg cfgs/kitti_models/second_teacher.yaml --ckpt ../checkpoints/monotakd_kitti_teacher.pth --pretrained_model ../checkpoints/monotakd_kitti_teacher.pth
```

* Train Teacher **without** pre-trained weights. 
```
python train.py --cfg cfgs/kitti_models/second.yaml  
```

* Test Teacher:
```
CUDA_VISIBLE_DEVICES=0,1 python test.py --cfg cfgs/kitti_models/second_teacher.yaml --ckpt ../checkpoints/scd-teacher-kitti.pth --save_to_file
```

### TENSORBOARD 
tensorboard --logdir ../output/kitti_models/TAKD/TAKD-scd/kitti_R50_scd_TAKD/
