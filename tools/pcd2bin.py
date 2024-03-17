import numpy as np
import os
from pypcd import pypcd

pcd_path = '/mnt/nas_kradar/kradar_dataset/dir_all/5/os2-64/'
bin_path = '/mnt/nas_kradar/kradar_dataset/dir_all/5/kradar_LTKD/velodyne/'

# print(pcd_path)
for pcd_file in os.listdir(pcd_path):
    pcd_data = pypcd.PointCloud.from_path(pcd_path+pcd_file)
# print(pcd_data.pc_data['x'].shape)
    points = np.zeros([pcd_data.points, 4], dtype=np.float32)
    points[:, 0] = pcd_data.pc_data['x'].copy()
    points[:, 1] = pcd_data.pc_data['y'].copy()
    points[:, 2] = pcd_data.pc_data['z'].copy()
    points[:, 3] = pcd_data.pc_data['intensity'].copy().astype(np.float32)
    pcd_file = pcd_file.split("_")[-1][:-4] # os2-64_00428.pcd -> 00428
    with open(bin_path + pcd_file + '.bin', 'wb') as f:
        f.write(points.tobytes())