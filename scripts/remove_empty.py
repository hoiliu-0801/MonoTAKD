import os.path
import os
from tqdm import tqdm
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti

def get_label(idx):
    fname = '{:06d}'.format(int(idx)) + '.txt'
    label_file = os.path.join('/mnt/disk2/Data/KITTI/lpcg/training/label_2', fname)
    assert os.path.exists(label_file)
    return object3d_kitti.get_objects_from_label(label_file)

train_split_path = '/mnt/disk2/Data/KITTI/lpcg/ImageSets/train.txt'
with open(train_split_path, 'r') as f:
    sample_id_list = f.readlines()
    f.close()

non_empty_ids = []
for sample_idx in tqdm(sample_id_list):
    obj_list = get_label(sample_idx[:-1])
    if len(obj_list) <= 0:
        continue
    non_empty_ids.append(sample_idx) #31794
# Save empty IDs in a txt
with open('/mnt/disk2/Data/KITTI/lpcg/training/ImageSets/train_no_empty.txt', 'wt') as f:
    for id in non_empty_ids:
        f.write(id)
    f.close()
print('Cleaned.')