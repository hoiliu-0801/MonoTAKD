# KITTI *test* Benchmark Upload Guidelines
## Overview
This guide will walk you through the process of preparing and submitting your monocular 3D object detection results to the KITTI Vision Benchmark Suite.
## Result Format Requirements
### File Organization
```
└── results/
    ├── 000000.txt
    ├── 000001.txt
    ├── 000002.txt
    ├── 000003.txt
    └── ...
```
### Detection Format
Each prediction file must be a text file with the following format for each detection

!!! THIS IS COPIED FROM [KITTI DEVKIT](https://github.com/bostondiditeam/kitti/blob/71d51b8a66c9226369797d437315c3ca2b56f312/resources/devkit_object/readme.txt#L46C1-L63C73) Please refer to it for most updated details. !!!
```
----------------------------------------------------------------------------
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
----------------------------------------------------------------------------

# Example of detection results file.
Car -1 -1 -2.9269 341.6844 177.5155 445.1921 215.7168 1.5335 1.6417 3.8825 -9.0274 1.7343 30.1617 -3.2157 0.8246
```

## Preparation Steps
1. Generate Predictions : Run your monocular 3D detection model to inference the KITTI *test* set.
2. Create a ZIP Archive : The `.txt` files are stored in a directory called `results`.
    ```
    cd results
    zip -r ../<name-of-zip-file>.zip ./*.txt

    # ZIP Example
    zip -r ../results.zip ./*.txt
    ```
3. (OPTIONAL) Check the zip file by extracting, `.txt` files should populate the folder.
    ```
    mkdir tmp
    cd tmp
    cp <name-of-zip-file>.zip tmp/
    unzip -K <name-of-zip-file>.zip
    ls
    rm -r tmp
    ```


## Submission Process
1. Register/Login: Create an account on the KITTI Vision Benchmark Suite if you don't have one. This could take days to get verified.
2. Access Submission Portal: Navigate to the 3D Object Detection Benchmark section.
3. Submit Results
    - Select "Submit Results" for the 3D Object Detection benchmark
    - Choose your ZIP file
    - Provide a method name and brief description
    - Submit your results
4. Verification Period: Wait for the automatic verification (usually takes a few hours). You will be notified through email.
5. Publication: Once verified, your results will appear on the leaderboard.