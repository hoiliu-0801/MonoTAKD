# MonoTAKD: Teaching Assistant Knowledge Distillation for Monocular 3D Object Detection

## Paper
[MonoTAKD: Teaching Assistant Knowledge Distillation for Monocular 3D Object Detection](https://arxiv.org/pdf/2404.04910) (arXiv, Supplimentary Included)

<!-- [MonoTAKD: Teaching Assistant Knowledge Distillation for Monocular 3D Object Detection]() (CVPR2025, Supplimentary Included) -->

## Introduction
This is the official implementation of MonoTAKD which utilizes [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) for the KITTI dataset.

<!-- [another version]() is implemented with [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) for Nuscenes dataset.  -->

## News
**[2025.4.6] Initial Release**
* Release code and pre-trained models for the KITTI dataset.
* Visualization utils are provided to visualize detection results in both camera perspective and BEV perspective. [Demo]() images & videos are included in this release.

<!-- **[2023.2.14] We have several updates.** -->

**Notice: Due to the short schedule, instructions and pre-trained models will be released and adjusted gradually in the near future. Please let us know if there are any issues and bugs.**

---

## Framework Overview
<!-- ![image](/docs/framework.png) -->
![image](./docs/imgs/framework.png)

## BEV Feature Generation
<!-- ![image](/docs/BEV%20generation.png) -->
![image](./docs/imgs/vis_bev.png)

## MonoTAKD DEMO
### Detection in CAMERA perspective
<img src="https://github.com/hoiliu-0801/MonoTAKD/blob/main/demo/3d.gif" width = "80%">
<br>
<img src="https://github.com/hoiliu-0801/MonoTAKD/blob/main/demo/seq_329.gif" width = "80%">

### Detection in BEV perspective
<img src="https://github.com/hoiliu-0801/MonoTAKD/blob/main/demo/bev.gif" width = "80%">

### Detection with CAMERA & BEV Side-By-Side
<img src="https://github.com/hoiliu-0801/MonoTAKD/blob/main/demo/cam_bev_demo.gif" width = "80%">
---

## Performance
### KITTI
Performance on the KITTI *test* set car category as [`AP_3D` / `AP_BEV`].
|   | Teacher | TA | Student | Easy| Moderate | Hard |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| [MonoTAKD](tools/cfgs/kitti_models/CMKD/CMKD-scd/cmkd_kitti_eigen_R50_scd_V2.yaml)| [SECOND]() | [CaDDN]() | [model](https://drive.google.com/file/d/1S4Uehq7ix1CE2BXwL9SmaDsrtOiNZUIN/view?usp=drive_link) |  **27.91** / 38.75  | **19.43** / 27.76 | **16.51** / 24.14 | 
| [MonoTAKD_*Lite*]()| - | - | - | - | - | - | 
| [MonoTAKD_*Raw*]() | - | - | - | - | - | - |

<!-- [model](https://drive.google.com/file/d/1S4Uehq7ix1CE2BXwL9SmaDsrtOiNZUIN/view?usp=drive_link) -->


### Nuscenes
|   | mAP | NDS | Model | 
|---|:---:|:---:|:---:|
| BEVDet-R50        | - | - | - |
| BEVDet-R50 + TAKD | - | - | - |

---

## Setting Up MonoTAKD

### Installation

Please follow [INSTALL](docs/INSTALL.md) to install MonoTAKD.

### Getting Started

Please follow [GETTING_START](docs/GETTING_STARTED.md) to train or evaluate the models.
