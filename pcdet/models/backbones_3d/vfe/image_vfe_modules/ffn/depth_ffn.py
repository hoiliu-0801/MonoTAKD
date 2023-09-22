import torch.nn as nn
import torch.nn.functional as F

from . import ddn, ddn_loss
from pcdet.models.model_utils.basic_block_2d import BasicBlock2D
from skimage import transform
import numpy as np
import torch

class DepthFFN(nn.Module):

    def __init__(self, model_cfg, downsample_factor):
        """
        Initialize frustum feature network via depth distribution estimation
        Args:
            model_cfg: EasyDict, Depth classification network config
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.disc_cfg = model_cfg.DISCRETIZE
        self.downsample_factor = downsample_factor

        # Create modules
        self.ddn = ddn.__all__[model_cfg.DDN.NAME](
            num_classes=self.disc_cfg["num_bins"] + 1,
            backbone_name=model_cfg.DDN.BACKBONE_NAME,
            **model_cfg.DDN.ARGS
        )

        if model_cfg.get('CHANNEL_REDUCE',None) is not None:
            self.channel_reduce = BasicBlock2D(**model_cfg.CHANNEL_REDUCE)

        # DDN_LOSS is optional
        if model_cfg.get('LOSS_',None) is not None:
            self.ddn_loss = ddn_loss.__all__[model_cfg.LOSS_.NAME](
                disc_cfg=self.disc_cfg,
                downsample_factor=downsample_factor,
                **model_cfg.LOSS_.ARGS
            )

        else:
            self.ddn_loss = None
        self.forward_ret_dict = {}

    def get_output_feature_dim(self):
        return self.channel_reduce.out_channels
    # sparse average pooling, by hoiliu
    def sparse_avg_pooling(self, feature_map, size=2):
        feature_map=feature_map.cpu().detach()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Batch = feature_map.shape[0]
        pool_out_list=[]
        for i in range(Batch):
            a = transform.downscale_local_mean(feature_map[i,:,:],(size,size))
            b = transform.downscale_local_mean(feature_map[i,:,:]!=0,(size,size))
            pool_out = a / (b+1e-10)
            pool_out_list.append(torch.from_numpy(pool_out).float().to(device))
        pool_out1 = torch.stack(pool_out_list)
        return pool_out1
    def create_depth_target(self, depth_map_target, depth_target_bin):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        B, h, w = depth_map_target.shape  # torch.Size([2, 47, 156])
        D = 120
        depth_target = torch.from_numpy(np.zeros([B, D+1, h, w])).float().to(device)
        for b in range(B):
            for i in range(h):
                for j in range(w):
                    bin_value = depth_target_bin [b,i,j]
                    if bin_value==120: # out of boundary
                        # print("depth_map_target:", depth_map_target[b,i,j])
                        depth_target [b,bin_value,i,j] = 100000
                    elif bin_value>120 or bin_value<0:
                        print("error bin")
                    else:
                        depth_target [b,bin_value,i,j] = 1
                        # depth_target [b,bin_value,i,j] = depth_map_target[b,i,j]
        # depth_target = depth_target [:, :-1, :]
        # depth_probs = F.softmax(depth_target, dim=depth_dim)  # [2, 120, 47, 156]
        # print("depth_probs:", depth_probs.shape)
        # exit()
        return depth_target

    def forward(self, batch_dict):
        """
        Predicts depths and creates image depth feature volume using depth distributions
        Args:
            batch_dict:
                images: (N, 3, H_in, W_in), Input images
        Returns:
            batch_dict:
                frustum_features: (N, C, D, H_out, W_out), Image depth features
        """
        # Pixel-wise depth classification
        images = batch_dict["images"] #([2, 3, 375, 1242])
        ddn_result = self.ddn(images)  # self.ddn is a pretrained backbone, which is used to generate pretrained depth feature and depth bin
        image_features = ddn_result["features"] #([2, 1024, 47, 156])
        depth_logits = ddn_result["logits"]  #([2, 121, 47, 156])
        # Channel reduce
        if self.channel_reduce is not None:
            image_features = self.channel_reduce(image_features) # 1024 -> 64

        # Create image feature plane-sweep volume
        frustum_features = self.create_frustum_features(image_features=image_features,
                                                        depth_logits=depth_logits)
        batch_dict["frustum_features"] = frustum_features
        batch_dict["image_features"] = image_features

        if self.training:
            # depth_maps and gt_boxes2d are optional
            self.forward_ret_dict["depth_maps"] = batch_dict.get("depth_maps",None)
            self.forward_ret_dict["gt_boxes2d"] = batch_dict.get("gt_boxes2d",None)
            self.forward_ret_dict["depth_logits"] = depth_logits # torch.Size([2, 121, 47, 156])
            #### New code ####
            #### Create Lidar-image-lije feature plane-sweep volume ###
            self.forward_ret_dict["depth_maps"] = self.sparse_avg_pooling(self.forward_ret_dict["depth_maps"], 8)
            # save_path="/home/ipl-pc/cmkd/output/vis_result"+".depth.png"
            # import matplotlib.pyplot as plt
            # print(self.forward_ret_dict["depth_maps"][0,:].shape)
            # plt.imsave(save_path, self.forward_ret_dict["depth_maps"][0,:].cpu().detach())
            # exit()
            depth_map_target = self.forward_ret_dict["depth_maps"]  ## ([47, 156])
            depth_target_bin = self.ddn_loss(**self.forward_ret_dict) # 0-120, total 121 dim
            depth_target = self.create_depth_target(depth_map_target, depth_target_bin)
            frustum_features_target = self.create_frustum_features(image_features, depth_target, target=True)
            batch_dict["frustum_features_target"] = frustum_features_target
            frustum_features, image_features = self.create_frustum_features(image_features=image_features,
                                                    depth_logits=depth_logits)
        # print(depth_map_target.max())
        # exit()


        ## 這邊也要decode 回one hot bin#
        # depth_map_target_list = []
        # print("1:",depth_map_target.shape)
        # depth_map_target_list.append(depth_map_target)
        # depth_map_target_list.append(zero_tensor)
        # depth_map_target= torch.cat(depth_map_target_list, dim=2)
        # frustum_features_target = depth_map_target * image_features_insq
        return batch_dict

    def create_frustum_features(self, image_features, depth_logits, target=False):
        """
        Create image depth feature volume by multiplying image features with depth distributions
        Args:
            image_features: (N, C, H, W), Image features
            depth_logits: (N, D+1, H, W), Depth classification logits
        Returns:
            frustum_features: (N, C, D, H, W), Image features
        """
        channel_dim = 1
        depth_dim = 2

        # Resize to match dimensions
        image_features = image_features.unsqueeze(depth_dim)  # [2, 64, 47, 156] -> [2, 64, 1, 47, 156]
        depth_logits = depth_logits.unsqueeze(channel_dim)  # [2, 120, 47, 156] -> [2, 1, 120, 47, 156]
        # Apply softmax along depth axis and remove last depth category (> Max Range)
        # print("depth_logits:", depth_logits[:,:,-1,:,:])
        if target:
            depth_probs = depth_logits[:, :, :-1]  # [2, 1, 120, 47, 156]
            # print("depth_probs:", depth_probs)
        else:
            depth_probs = F.softmax(depth_logits, dim=depth_dim)  # [2, 1, 121, 47, 156]
            depth_probs = depth_probs[:, :, :-1]  # [2, 1, 120, 47, 156]
            # print("depth_probs__:", depth_probs)
        # Multiply to form image depth feature volume
        frustum_features = depth_probs * image_features  #  [2, 64, 120, 47, 156] = [2, 1, 120, 47, 156] * [2, 64, 1, 47, 156]
        return frustum_features

    def get_loss(self):
        """
        Gets DDN loss
        Args:
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        loss, tb_dict = self.ddn_loss(**self.forward_ret_dict)
        return loss, tb_dict
