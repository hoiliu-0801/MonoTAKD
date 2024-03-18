# Visualize KITTI Objects
This tool is adapted from [Visualize-KITTI-Objects-In-Videos](https://github.com/HengLan/Visualize-KITTI-Objects-in-Videos). This tool is included to provide visualization of MonoTAKD. It can be used to visualize objects of KITTI in camera image, point cloud and bird's eye view.

## Installation
```
(base)$ mkdir VIS-KITTI
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

VIS-KITTI 

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

### Example of commands
* Visualize 3D Bounding Box of detected objects in CAMERA and save the visualization to images
```
(vis)$ python visualize.py --dataset_path=path_to_KITTI --vis_data_type='camera' --vis_box --box_type='3d' --save_img
```
* Visualize objects in BEV
```
(vis)$ python visualize.py --dataset_path=path_to_KITTI --vis_data_type='bev' --vis_box
```

## Visualization
### Visualization of detected objects in CAMERA perspective
<img src="demo/3d.gif" width = "70%">

### Visualization of detected objects in BEV perspective
<img src="demo/bev.gif" width = "70%">