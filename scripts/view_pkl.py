import pickle
import os.path as osp
from tqdm import tqdm

## This file reads in result pkl and convert to KITTI format (for test submission)

with open('/home/ipl-pc/cmkd/output/home/ipl-pc/cmkd/tools/cfgs/kitti_models/CMKD/CMKD-scd/cmkd_kitti_R50_scd_V2/default/eval/epoch_1580/val/default/result.pkl', 'rb') as f:
    data = pickle.load(f)
    
    # Save object summary in KITTI format
    for frame in tqdm(data):
        obj_str_list = []
        for obj in range(len(frame['name'])):
            kitti_format = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (frame['name'][obj], frame['truncated'][obj], int(frame['occluded'][obj]), frame['alpha'][obj], frame['bbox'][obj][0], frame['bbox'][obj][1],
                        frame['bbox'][obj][2], frame['bbox'][obj][3], frame['dimensions'][obj][0], frame['dimensions'][obj][1], frame['dimensions'][obj][2], 
                        frame['location'][obj][0], frame['location'][obj][1], frame['location'][obj][2], frame['rotation_y'][obj])
            obj_str_list.append(kitti_format)
        # write all obj strs into a txt file
        fname = frame['frame_id'] + '.txt'
        with open(osp.join('/home/ipl-pc/cmkd/output/home/ipl-pc/cmkd/tools/cfgs/kitti_models/CMKD/CMKD-scd/cmkd_kitti_R50_scd_V2/default/eval/epoch_1580/test/kitti_format', fname), 'w') as f:
            f.write('\n'.join(obj_str_list))