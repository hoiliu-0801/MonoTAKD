## Our code will be released soon.

## Framework Overview
![image](/docs/framework.png)

## BEV Features Generation
![image](/docs/BEV%20generation.png)

## Use TAKD

### Installation

Please follow [INSTALL](docs/INSTALL.md) to install CMKD.

### Getting Started

Please follow [GETTING_START](docs/GETTING_STARTED.md) to train or evaluate the models.

## Models

### KITTI
<!-- 
|   | Teacher Model|  Car Easy@R40|	Car Moderate@R40	|Car Hard@R40	 | Model | Teacher Model |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| [CMKD-R50 (kitti train + eigen clean)](tools/cfgs/kitti_models/CMKD/CMKD-scd/cmkd_kitti_eigen_R50_scd_V2.yaml)| [SECOND](tools/cfgs/kitti_models/CMKD/CMKD-scd/second_teacher.yaml) |  33.36  | 21.61  | 17.97  |  [model](https://drive.google.com/file/d/1A9rdGUdLkqOWVt8IbZfF1s0FHotQUIK_/view?usp=share_link)   | [model](https://drive.google.com/file/d/1SYbReQHNjOsWQ-zxM6mn3dq9Iyi98wAg/view?usp=share_link) |
| [CMKD-R50 (kitti train)](tools/cfgs/kitti_models/CMKD/CMKD-scd/cmkd_kitti_R50_scd_V2.yaml)|[SECOND](tools/cfgs/kitti_models/CMKD/CMKD-scd/second_teacher.yaml)|  24.02  | 15.80  | 13.22  |  [model](https://drive.google.com/file/d/1weEb8DkAHKNa4HPgzM_Pbc7FLr-Yiuii/view?usp=share_link)  | [model](https://drive.google.com/file/d/1SYbReQHNjOsWQ-zxM6mn3dq9Iyi98wAg/view?usp=share_link) |
| [CMKD-R50 (kitti train + eigen clean)](tools/cfgs/kitti_models/CMKD/CMKD-ctp/cmkd_kitti_eigen_R50_ctp_V2.yaml)|[CenterPoint](tools/cfgs/kitti_models/CMKD/CMKD-ctp/centerpoint_teacher.yaml)|  29.78  | 21.17  | 18.41  |  [model](https://drive.google.com/file/d/1fhXf5UZ0fat9ihdApCTuAVUE8ozttNej/view?usp=share_link)  |[model](https://drive.google.com/file/d/1Oqmnl6Kctg5BRKHEgtAw7ef9Lw3eyPky/view?usp=share_link)|
| [CMKD-R50 (kitti train)](tools/cfgs/kitti_models/CMKD/CMKD-ctp/cmkd_kitti_R50_ctp_V2.yaml)|[CenterPoint](tools/cfgs/kitti_models/CMKD/CMKD-ctp/centerpoint_teacher.yaml)|  22.56  | 16.02  | 13.52  |  [model](https://drive.google.com/file/d/1tuZdy_S4EYeGaH8Mu5nDMOTu8b1PpfG6/view?usp=share_link)  |[model](https://drive.google.com/file/d/1Oqmnl6Kctg5BRKHEgtAw7ef9Lw3eyPky/view?usp=share_link)|
| [CMKD-R50 (kitti train + eigen clean)](tools/cfgs/kitti_models/CMKD/CMKD-pp/cmkd_kitti_eigen_R50_pp_V2.yaml)    |[PointPillar](tools/cfgs/kitti_models/CMKD/CMKD-pp/pointpillar_teacher.yaml)|  32.25  | 21.47  | 18.21  |  [model](https://drive.google.com/file/d/1yX70t4pyTTaJr0X9lzivp0JEwB4uwOTz/view?usp=share_link)  | [model](https://drive.google.com/file/d/1JvpBqNCcJjfASs86q7Qp3772eaJU3wnL/view?usp=share_link)|
| [CMKD-R50 (kitti train)](tools/cfgs/kitti_models/CMKD/CMKD-pp/cmkd_kitti_R50_pp_V2.yaml)|[PointPillar](tools/cfgs/kitti_models/CMKD/CMKD-pp/pointpillar_teacher.yaml)|  23.84  | 16.44 | 13.58  | [model](https://drive.google.com/file/d/1tHTLoBi2m5OqpTVM9biY4ExOZ40PfTIB/view?usp=share_link)  |[model](https://drive.google.com/file/d/1JvpBqNCcJjfASs86q7Qp3772eaJU3wnL/view?usp=share_link)|
 -->


### Waymo
Coming Soon
                  

### Nuscenes
<!-- |   |  mAP |	NDS |Model | 
|---|:---:|:---:|:---:|
| BEVDet-R50|  30.7  | 38.2  | - |
| BEVDet-R50 + CMKD|  34.7  | 42.6  | - |
 -->




=======
# MonoTAKD
Paper : MonoTAKD: Teaching assistant knowledge distillation for monocular 3D object detection.

Train
python train_TAKD.py --cfg cfgs/kitti_models/TAKD/TAKD-scd/kitti_R50_scd_TAKD.yaml --pretrained_lidar_model ../checkpoints/scd-teacher-kitti.pth --pretrained_img_model ../checkpoints/cmkd-scd-2161.pth

Test
python test_TAKD.py --cfg cfgs/kitti_models/TAKD/TAKD-scd/kitti_R50_scd_TAKD.yaml --ckpt ../checkpoints/twcc_8015_checkpoint_epoch_10.pth

Train Teacher:
python train.py --cfg cfgs/kitti_models/second_teacher.yaml --ckpt ../checkpoints/scd-teacher-kitti.pth --pretrained_model ../checkpoints/scd-teacher-kitti.pth 
python train.py --cfg cfgs/kitti_models/second.yaml  (w/o pretrained)

Test Teacher:
python test.py --cfg cfgs/kitti_models/second_teacher.yaml --ckpt ../checkpoints/scd-teacher-kitti.pth --save_to_file

Tensorboard:
tensorboard --logdir ../output/kitti_models/TAKD/TAKD-scd/kitti_R50_scd_TAKD/
>>>>>>> 9d852a3f193097eb414f776304cb2d8b7fb5e1de
