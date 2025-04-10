from turtle import shape
from cv2 import fastNlMeansDenoising

from gflags import Flag
from ordered_set import T
from .detector3d_template_TAKD import Detector3DTemplate_TAKD
from pcdet.utils import loss_utils
from pcdet.models.backbones_2d import domain_adaptation
from ..model_utils.model_nms_utils import class_agnostic_nms
from ...ops.iou3d_nms import iou3d_nms_utils
import torch
import torch.nn as nn
import copy
import numpy as np
import kornia
import kornia.augmentation as ka
from ..model_utils import model_nms_utils
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

class CMKD_TAKD(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_img = None
        self.model_lidar = None

        # whether forward teacher
        self.forward_teacher = model_cfg.get('FORWARD_TEACHER', True)

        # bev_loss cfg
        self.calculate_bev_loss = model_cfg.get('LOSS_BEV', True)
        self.bev_loss_fun = torch.nn.MSELoss(reduction='none')
        self.bev_loss_type = model_cfg.get('LOSS_BEV_TYPE', 'L2')
        self.bev_loss_weight = model_cfg.get('LOSS_BEV_WEIGHT', 32)
        self.use_nonzero_mask = model_cfg.get('use_nonzero_mask', False)

        # pass to model img
        self.calculate_depth_loss = model_cfg.get('LOSS_DEPTH', False)
        self.calculate_rpn_loss = model_cfg.get('LOSS_PRN', False)

        self.amp_training = False
        self.bev_layer = model_cfg.get('bev_layer', 'spatial_features')
        self.bev_layer_copy = model_cfg.get('bev_layer_copy', 'spatial_features_copy')
        self.bev_layer_target = model_cfg.get('bev_layer_target', 'spatial_features_target')

    def forward(self, batch_dict):
        loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

        ret_dict = {
            'loss': loss
        }

        return ret_dict, tb_dict, disp_dict
    def visual_(self, batch_dict, visual_list):
        ############ draw depth map ##############
        save_bev="/home/ipl-pc/cmkd/output/vis_result"
        data_root= "/home/ipl-pc/VirConv/data/kitti/training/image_2"
        batch_size=2
        degree=90
        # print("bev_lidar.shape:",bev_lidar.shape) # torch.Size([2, 128, 188, 140])
        # print("bev_img.shape:",bev_img.shape) # torch.Size([2, 128, 188, 140])
        for B in range(batch_size):
            #######　draw image #######
            frame_id = batch_dict["frame_id"]
            # image_path = os.path.join(data_root, frame_id[B]+".png")
            # image_=cv2.imread(image_path)
            # image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
            # Image_save_path=os.path.join(save_bev, frame_id[B]+ str(B)+".png")
            # Image.fromarray(image_).save(Image_save_path)
            ############################
            for key, bev_image in visual_list.items():
                bev_image_ = ndimage.zoom(ndimage.rotate(torch.mean(bev_image[B,:,:,:].cpu().detach(), dim=0), degree),3)
                # save_path=os.path.join(save_bev,  frame_id[B] +"_"+key+".png")
                save_path=os.path.join(save_bev,  frame_id[B] +"_"+key+".png")
                # if key =="000857":
                plt.imsave(save_path, bev_image_, cmap='inferno')
                # print(key, torch.mean(bev_image))
    def normalize_(self, bev_img):
        B, C, H, W = bev_img.shape  # [2, 128, 188, 140])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        bev_img = bev_img.cpu().detach()
        bev_img -= bev_img.min()
        bev_img /= bev_img.max()
        # bev_img[bev_img<200]=0
        return torch.tensor(bev_img).float().to(device)

    def get_training_loss(self, batch_dict):
        # forward lidar model
        bev_lidar = teacher_pred = None
        if self.training and self.forward_teacher:
            batch_dict_copy = copy.deepcopy(batch_dict)
            load_data_to_gpu(batch_dict_copy)
            self.model_lidar.eval()
            self.model_lidar.generate_predicted_boxes=False
            if self.model_lidar.dense_head is not None:
                self.model_lidar.dense_head.as_teacher=True
            with torch.no_grad():
                bev_lidar, teacher_pred = self.model_lidar(batch_dict_copy)
            if self.model_img.dense_head is not None:
                self.model_img.dense_head.teacher_pred = teacher_pred
            del batch_dict_copy

        #calculate loss
        # rpn_loss and depth loss
        # pass to model img
        self.model_img.calculate_depth_loss = self.calculate_depth_loss
        self.model_img.calculate_rpn_loss = self.calculate_rpn_loss
        ret_dict, tb_dict, disp_dict = self.model_img(batch_dict)
        # batch_dict: dict_keys(['frame_id', 'gt_boxes', 'gt_scores', 'images', 'points', 'trans_lidar_to_cam', 'trans_cam_to_img', 'use_lead_xyz', 'voxels', 'voxel_coords', 'voxel_num_points', 'image_shape', 'batch_size', 'frustum_features', 'image_features', 'voxel_features', 'spatial_features', 'spatial_features_2d'])
        loss_rpn = ret_dict.get('loss', torch.tensor(0))
        # bev_loss
        bev_img = batch_dict.get(self.bev_layer, None) # learn only camera info
        bev_img_copy = batch_dict.get(self.bev_layer_copy, None) # learn only lidar info
        gt_mask = batch_dict.get('gt_mask', None)
        loss_bev = torch.tensor(0)
        ### Get TA's bev ###
        bev_lidar_img_TA = batch_dict.get(self.bev_layer_target, None) 
        bev_lidar_img_TA = self.normalize_(bev_lidar_img_TA)
        bev_img = self.normalize_(bev_img)
        # bev_diff = bev_lidar_img_TA-bev_lidar
        bev_diff = torch.absolute(bev_lidar_img_TA-bev_lidar)
        bev_diff[bev_diff<(3/5*bev_diff.max())]=0
        # bev_diff = torch.absolute(bev_lidar-bev_lidar_img_TA)
        # exit()
        if (bev_img is not None) and (bev_lidar is not None) and (bev_lidar_img_TA is not None) is not None and self.calculate_bev_loss:

            bev_loss_mask = torch.ones((bev_lidar_img_TA.shape[0], 1, bev_lidar_img_TA.shape[2], bev_lidar_img_TA.shape[3]), device = bev_lidar_img_TA.device)
            if gt_mask:
                bev_loss_mask *= gt_mask
            if self.use_nonzero_mask:
                nonzero_mask = (bev_lidar_img_TA.sum(1,keepdim=True)!=0).float()
                nonzero_mask[nonzero_mask==0] = 0.05
                bev_loss_mask *= nonzero_mask
            noralizer = bev_loss_mask.numel() / bev_loss_mask.sum()

            if self.bev_loss_type == 'L2':
                ## TA loss ##
                # loss_bev_image_like = (self.bev_loss_fun(bev_lidar, bev_lidar_img_TA)*bev_loss_mask).mean()*noralizer
                ##　Student loss ##
                loss_bev_image_lpk = (self.bev_loss_fun(bev_diff, bev_img_copy)*bev_loss_mask).mean()*noralizer
                loss_bev_image_ipk = (self.bev_loss_fun(bev_lidar_img_TA, bev_img)*bev_loss_mask).mean()*noralizer

            # loss_bev_image*= self.bev_loss_weight
            # loss_bev_image_like*= self.bev_loss_weight
            # loss_bev_copy*= self.bev_loss_weight
            ########
            # Only IPK
            # loss_bev = loss_bev_image_like + loss_bev_image
            # Only LPK
            # loss_bev =  0.1*loss_bev_copy + loss_bev_image
            # IPK & LPK
            # loss_bev = loss_bev_image_like + 0.2*loss_bev_copy + loss_bev_image
            loss_bev =  loss_bev_image_ipk + 0.2*loss_bev_image_lpk
        ### Visualization ###
        # bev_final = bev_img + bev_img_copy
        # visual_dict=dict(bev_img=bev_img)
        # visual_dict=dict(bev_lidar=bev_lidar, bev_img=bev_img, bev_lidar_img_TA=bev_lidar_img_TA, bev_diff=bev_diff)
        # visual_dict=dict(bev_lidar=bev_lidar, bev_lidar_img_TA=bev_lidar_img_TA)
        # self.visual_(batch_dict, visual_dict)

        #all loss
        loss = loss_bev + loss_rpn

        tb_dict.update({
            "bev_loss":loss_bev.item(),
        })


        return loss, tb_dict, disp_dict
import time

class CMKD_MONO_TAKD(Detector3DTemplate_TAKD):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.rpn_loss_weight = 1

    def forward(self, batch_dict):
        time_start = time.time()
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        # batch_dict['image_features'].shape = torch.Size([2, 64, 47, 156])
        # batch_dict['voxel_features'].shape = torch.Size([2, 64, 10, 188, 140])
        if self.training:  # True
            self.calculate_rpn_loss = batch_dict.get('calculate_rpn_loss', self.calculate_rpn_loss)
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {}
            if loss != 0:
                ret_dict = {
                    'loss': loss
                }

            return ret_dict, tb_dict, disp_dict

        else:
            if 'Center' in self.model_cfg.DENSE_HEAD.NAME:
                pred_dicts, recall_dicts = self.post_processing_center_head(batch_dict)
            else:
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
            time_end = time.time()
            print(f'Time elapsed: {time_end - time_start:.4f}')
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        tb_dict = {}
        loss = 0

        if getattr(self, 'dense_head', None) is not None and self.calculate_rpn_loss:
            # rpn loss
            loss_rpn, tb_dict = self.dense_head.get_loss()
            for k,v in tb_dict.items():
                if 'rpn' in k:
                    tb_dict[k] *= self.rpn_loss_weight
            loss_rpn *= self.rpn_loss_weight
            loss += loss_rpn


        # depth loss, optional
        if self.model_cfg['VFE']['FFN'].get('LOSS',None) is not None and self.calculate_depth_loss:
            ## incomplete code here
            exit()
            loss_depth, tb_dict_depth = self.vfe.get_loss()
            tb_dict.update({
                'loss_depth': loss_depth.item(),
            })
            loss += loss_depth

            tb_dict = {
                **tb_dict,
                'loss_depth': loss_depth.item(),
            }

        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            if getattr(self, 'vis_online', False):
                from pcdet.utils import vis_utils
                path = '../data/kitti/training/image_2/' + str(batch_dict['frame_id'][index]) + '.png'
                img = cv2.imread(path)

                vis_utils.draw_box_on_img(final_boxes,img,batch_dict['trans_lidar_to_cam'][index], batch_dict['trans_cam_to_img'][index])

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels,
            }
            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict

    def post_processing_center_head(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict


class CMKD_LIDAR_TAKD(Detector3DTemplate_TAKD):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.bev_layer = model_cfg.get('bev_layer', 'spatial_features')
        self.res_layer = model_cfg.get('res_layer', 'teacher_pred')

    def forward(self, batch_dict):

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        # (['frame_id', 'gt_boxes', 'gt_scores', 'gt_boxes2d', 'images', 'points', 'depth_maps', 'trans_lidar_to_cam', 'trans_cam_to_img', 'use_lead_xyz', 'voxels', 'voxel_coords', 'voxel_num_points', 'image_shape', 'batch_size', 'voxel_features', 'encoded_spconv_tensor', 'encoded_spconv_tensor_stride', 'multi_scale_3d_features', 'spatial_features', 'spatial_features_stride', 'spatial_features_2d', 'teacher_pred'])
        # exit()
        # batch_dict['voxel_features'].shape = torch.Size([31446, 4])
        # batch_dict['depth_maps'].shape = torch.Size([2, 375, 1242])
        bev_lidar = batch_dict.get(self.bev_layer, None)
        teacher_pred = batch_dict.get(self.res_layer, None)

        return bev_lidar, teacher_pred


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib', 'index', 'cam_type']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()

