- [Visualize MonoTAKD Detection Results](#visualize-monotakd-detection-results)
  - [Get Started with VIS-KITTI](#get-started-with-vis-kitti)
  - [Main Script Args](#main-script-args)
    - [Command Examples](#command-examples)
  - [MonoTAKD DEMO](#monotakd-demo)
    - [Detection in CAMERA perspective](#detection-in-camera-perspective)
    - [Detection in BEV perspective](#detection-in-bev-perspective)
    - [Detection with CAMERA \& BEV Side-By-Side](#detection-with-camera--bev-side-by-side)

---

# Visualize MonoTAKD Detection Results
The following visualizations utilizes tool that was adapted from [Visualize-KITTI-Objects-In-Videos](https://github.com/HengLan/Visualize-KITTI-Objects-in-Videos). This tool is modified to provide visualization of MonoTAKD. It can be used to visualize objects of KITTI in camera image, point cloud and bird's eye view.

## Get Started with [VIS-KITTI](https://github.com/christinewoo/VIS-KITTI.git)
```
(base)$ cd vis-kitti
```
* Create a new conda environment (note: vtk is incompatible with Python 3.8)
```
(base)$ conda create -n vis python=3.7 -y 
(base)$ conda activate vis
```

* Install required packages 
```
(vis)$ pip install opencv-python
(vis)$ pip install pillow
(vis)$ pip install scipy
```

* Install mayavi and vtk
```
(vis)$ conda install mayavi -c conda-forge

OR 

(vis)$ pip install vtk==8.1.2
(vis)$ pip install mayavi
```

## Main Script Args
```
usage: visualize.py [-h] [--dataset_path DATASET_PATH]
                         [--vis_data_type {camera,pointcloud,bev}] [--fov]
                         [--vis_box] [--box_type {2d,3d}] [--save_img]
                         [--save_path SAVE_PATH]

optional arguments:
  -h, --help                        show this help message and exit
  --dataset_path DATASET_PATH       set path to KITTI, a default dataset is provided
  --vis_data_type {camera,pc,bev}   show object in camera, pointcloud or birds eye view
  --fov                             only show front view of pointcloud
  --vis_box                         show object box or not
  --box_type {2d,3d}                for vis in camera, show 2d or 3d object box
  --save_img                        save visualization result or not
  --save_path SAVE_PATH             path to save visualization result
```

### Command Examples
* Visualize 3D Bounding Box of detected objects in CAMERA and save the visualization to images
```
(vis)$ python visualize.py --dataset_path=path_to_KITTI --vis_data_type='camera' --vis_box --box_type='3d' --save_img
```
* Visualize objects in BEV
```
(vis)$ python visualize.py --dataset_path=path_to_KITTI --vis_data_type='bev' --vis_box
```

## MonoTAKD DEMO
### Detection in CAMERA perspective
<img src="https://github.com/hoiliu-0801/MonoTAKD/blob/main/demo/3d.gif" width = "80%">
<br>
<img src="https://github.com/hoiliu-0801/MonoTAKD/blob/main/demo/seq_329.gif" width = "80%">

### Detection in BEV perspective
<img src="https://github.com/hoiliu-0801/MonoTAKD/blob/main/demo/bev.gif" width = "80%">

### Detection with CAMERA & BEV Side-By-Side
<img src="https://github.com/hoiliu-0801/MonoTAKD/blob/main/demo/cam_bev_demo.gif" width = "80%">