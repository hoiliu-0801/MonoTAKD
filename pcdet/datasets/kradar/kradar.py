import os
import os.path as osp
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile
import random
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lib.datasets.utils import angle2class
from lib.datasets.kradar.kradar_utils import get_objects_from_label
from lib.datasets.kradar.kradar_utils import Calibration, Object3D_
from lib.datasets.kitti.kitti_utils import get_affine_transform
from lib.datasets.kitti.kitti_utils import affine_transform
from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
from lib.datasets.kitti.kitti_eval_python.eval import get_distance_eval_result
import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti
import copy
from lib.datasets.kitti.pd import PhotometricDistort


class KRadarDetection_v1_0(data.Dataset):
    def __init__(self, split, cfg):

        # basic configuration
        self.root_dir = '/mnt/nas_kradar/kradar_dataset/dir_all/' #cfg.get('root_dir')
        self.split_dir = '/home/ipl-ad/datasets/K-Radar-Dev/resources/split/'
        self.split = split
        self.num_classes = 3
        self.max_objs = 20
        # self.class_name = ['Sedan']
        # self.cls2id = {'Sedan': 1}
        self.class_name = ['Pedestrian', 'Sedan', 'Cyclist']
        # self.class_name = ['Sedan'] # modified order
        self.cls2id = {'Pedestrian':0, 'Sedan':1, 'Cyclist':2}
        self.resolution = np.array([1280, 720])  # W * H
        self.use_3d_center = cfg.get('use_3d_center', True)
        self.writelist = cfg.get('writelist', ['Sedan'])
        # self.writelist = cfg.get('writelist', ['Pedestrian', 'Sedan', 'Cyclist'])
        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT
        self.bbox2d_type = cfg.get('bbox2d_type', 'anno')
        assert self.bbox2d_type in ['anno', 'proj']
        self.meanshape = cfg.get('meanshape', False)
        self.class_merging = cfg.get('class_merging', False)
        self.use_dontcare = cfg.get('use_dontcare', False)

        if self.class_merging:
            self.writelist.extend(['Bus or Truck'])
        if self.use_dontcare:
            self.writelist.extend(['DontCare'])

        # data split loading
        # assert self.split in ['train', 'val', 'trainval', 'test']
        # if self.split == 'val':
        #     self.split = 'test'
        self.split_file = os.path.join(self.split_dir, self.split + '.txt')

        self.idx_list = [x.strip() for x in open(self.split_file).readlines()]
        print(f'Number of training images: {len(self.idx_list)}')
        # file format in .txt files: 1,00033_00001.txt --> seq,label_name.txt (str)

        # path configuration
        self.image_dir = 'cam-front'
        # self.label_dir = 'info_label'
        self.label_dir ='refined_label_v3'

        # data augmentation configuration
        self.data_augmentation = False
        # self.data_augmentation = True if split in ['train', 'trainval'] else False

        self.aug_pd = cfg.get('aug_pd', False)
        self.aug_crop = cfg.get('aug_crop', False)
        self.aug_calib = cfg.get('aug_calib', False)

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.cls_mean_size = np.array([[1.76255119, 0.66068622, 0.84422524],
                                       [1.52563191462, 1.62856739989, 3.88311640418],
                                       [1.73698127, 0.59706367, 1.76282397]])
        if not self.meanshape:
            self.cls_mean_size = np.zeros_like(self.cls_mean_size, dtype=np.float32)

        # others
        self.downsample = 32
        self.pd = PhotometricDistort()
        self.clip_2d = cfg.get('clip_2d', False)

    def get_image(self, seq, cam_idx):
        img_file = osp.join(self.root_dir, seq, self.image_dir, 'cam-front_%s.png' % cam_idx)
        assert os.path.exists(img_file)
        img = cv2.imread(img_file)
        return img[:,:1280,:]

    def get_label(self, obj_list, calib): # pass in k-radar object list
        return get_objects_from_label(obj_list, calib)

    def get_calib(self):
        return Calibration()

    def eval(self, results_dir, logger):
        logger.info("==> Loading detections and GTs...")
        img_ids = []
        for id in self.idx_list:
            img_ids.append(id)
        # KITTI
        # dt_annos = kitti.get_label_annos(results_dir, img_ids) # Saved in KITTI style, load as KITTI style
        # gt_annos = kitti.get_label_annos(self.root_dir, self.label_dir, img_ids)

        # Kradar
         # Get gt_annotations into KI
        dt_annos = kitti.get_label_annos_dt(results_dir, img_ids) # Saved in KITTI style, load as KITTI style
        gt_annos = kitti.get_kradar_gt_label_annos(self.root_dir, self.label_dir, img_ids)

        test_id = {'Sedan': 0}


        logger.info('==> Evaluating (official) ...')
        car_moderate = 0
        for category in self.writelist:
            results_str, results_dict, mAP3d_R40 = get_official_eval_result(gt_annos, dt_annos, test_id[category], dataset="KRADAR")

            if category == 'Sedan':
                car_moderate = mAP3d_R40
            logger.info(results_str)
        return car_moderate

    def __len__(self):
        return self.idx_list.__len__()

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        ### self.idx_list is list of: 1,00033_00001.txt --> seq,label_name.txt (str)
        seq, label_name = self.idx_list[item].split(',')
        # print(seq, label_name)
        cur_label_path = osp.join(self.root_dir, seq, self.label_dir, label_name)

        ### Uncover cam_index from the k-radar txt labels
        # * idx(tesseract_os2-64_cam-front_os1-128_cam-lrr)=00091_00068_00201_00068_00200, timestamp=1643184458.088149718
        # *, 0, 0, Sedan, 11.759999999999957, -3.715, -0.2, 0.17980000000000018, 2.9042000000000003, 1.3045000000000002, 1.25
        # ... (may have multiple objects in a frame, last obj has obj_idx as -1)
        ###

        # Read in the label txt
        with open(cur_label_path) as f:
            label_lines = f.readlines()
            f.close()

        ### Load Image ###
        cam_idx = label_lines[0].split(',')[0].split('=')[1].split('_')[2]
        img = self.get_image(seq, cam_idx)
        img_size = np.array((img.shape[1], img.shape[0])) # W, H
        features_size = self.resolution // self.downsample  # W * H

        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W

        info = {'img_id': self.idx_list[item],
                'img_size': img_size,
                'bbox_downsample_ratio': img_size / features_size}

        if self.split == 'test':
            # transform intrinsic into calib.P2 format
            calib = self.get_calib()
            return img, calib.P2, img, info

        #  ============================   get labels   ==============================
        calib = self.get_calib()
        objects = self.get_label(label_lines[1:], calib) # pass in k-radar label obj list

        # labels encoding
        calibs = np.zeros((self.max_objs, 3, 4), dtype=np.float32)
        indices = np.zeros((self.max_objs), dtype=np.int64)
        mask_2d = np.zeros((self.max_objs), dtype=np.bool)
        labels = np.zeros((self.max_objs), dtype=np.int8)
        depth = np.zeros((self.max_objs, 1), dtype=np.float32)
        heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
        heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
        size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
        size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        boxes = np.zeros((self.max_objs, 4), dtype=np.float32)
        boxes_3d = np.zeros((self.max_objs, 6), dtype=np.float32)

        object_num = len(objects) if len(objects) < self.max_objs else self.max_objs

        for i in range(object_num):
            # filter objects by writelist
            if objects[i].cls_type not in self.writelist:
                continue

            # filter inappropriate samples
            if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                continue

            # ignore the samples beyond the threshold [hard encoding]
            # threshold = 65
            threshold = 65
            if objects[i].pos[-1] > threshold:
                continue
            # ignore samples with truncated bbox
            if objects[i].trucation > 0:
                continue

            # process 2d bbox & get 2d center
            bbox_2d = objects[i].box2d.copy()
            x_tl, y_tl, x_rb, y_rb = bbox_2d
            if x_tl<200 or x_rb>self.resolution[0]-200:
                continue
            elif y_tl<50 or y_rb>self.resolution[1]-50:
                continue

            # process 3d center
            center_2d = objects[i].center_2d
            corner_2d = bbox_2d.copy()
            center_3d = objects[i].pix_center_3d

            # filter 3d center out of img
            proj_inside_img = True
            if center_3d[0] < 0 or center_3d[0] >= self.resolution[0]:
                proj_inside_img = False
            if center_3d[1] < 0 or center_3d[1] >= self.resolution[1]:
                proj_inside_img = False

            if proj_inside_img == False:
                continue

            # class
            cls_id = self.cls2id[objects[i].cls_type]
            labels[i] = cls_id
            # encoding 2d/3d boxes
            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            size_2d[i] = 1. * w, 1. * h
            if np.any(size_2d[i] < 0):
                print("W,H:",w,h)
            center_2d_norm = center_2d / self.resolution
            size_2d_norm = size_2d[i] / self.resolution
            corner_2d_norm = corner_2d
            corner_2d_norm[0: 2] = corner_2d[0: 2] / self.resolution
            corner_2d_norm[2: 4] = corner_2d[2: 4] / self.resolution
            center_3d_norm = center_3d / self.resolution
            # l, r = center_3d_norm[0] - corner_2d_norm[0], corner_2d_norm[2] - center_3d_norm[0]
            # t, b = center_3d_norm[1] - corner_2d_norm[1], corner_2d_norm[3] - center_3d_norm[1]
            lrtb = objects[i].lrtb
            l, r = lrtb[0: 2] / self.resolution[0]
            t, b = lrtb[2: 4] / self.resolution[1]

            if l < 0 or r < 0 or t < 0 or b < 0:
                if self.clip_2d:
                    l = np.clip(l, 0, 1)
                    r = np.clip(r, 0, 1)
                    t = np.clip(t, 0, 1)
                    b = np.clip(b, 0, 1)
                else:
                    continue
            boxes[i] = center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1]
            boxes_3d[i] = center_3d_norm[0], center_3d_norm[1], l, r, t, b

            # encoding depth
            depth[i] = objects[i].pos[-1]
            # encoding heading angle
            heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0] + objects[i].box2d[2]) / 2)
            if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
            if heading_angle < -np.pi: heading_angle += 2 * np.pi
            heading_bin[i], heading_res[i] = angle2class(heading_angle)

            # encoding size_3d
            src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
            mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
            size_3d[i] = src_size_3d[i] - mean_size
            if objects[i].trucation <= 0.5 and objects[i].occlusion <= 2:
                mask_2d[i] = 1
            # mask_2d[i] = 1
            calibs[i] = calib.P2
        # collect return data
        inputs = img
        targets = {
            'calibs': calibs,
            'indices': indices,
            'img_size': img_size,
            'labels': labels,
            'boxes': boxes,
            'boxes_3d': boxes_3d,
            'depth': depth,
            'size_2d': size_2d,
            'size_3d': size_3d,
            'src_size_3d': src_size_3d,
            'heading_bin': heading_bin,
            'heading_res': heading_res,
            'mask_2d': mask_2d}

        info = {'img_id': self.idx_list[item],
                'img_size': img_size,
                'bbox_downsample_ratio': img_size / features_size}
        return inputs, calib.P2, targets, info


