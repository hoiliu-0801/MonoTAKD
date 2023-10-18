import torch
import torch.nn as nn

from pcdet.models.model_utils.basic_block_2d import BasicBlock2D, BasicBlock2D_copy, BasicBlock2D_copy2
from pcdet.models.model_utils.GCnet import ContextBlock2d, SAM


class Conv2DCollapse(nn.Module):

    def __init__(self, model_cfg, grid_size):
        """
        Initializes 2D convolution collapse module
        Args:
            model_cfg: EasyDict, Model configuration
            grid_size: (X, Y, Z) Voxel grid size
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.num_heights = grid_size[-1]
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_in_features = self.model_cfg.get('NUM_IN_FEATURES') if self.model_cfg.get('NUM_IN_FEATURES') is not None \
                                            else self.model_cfg.NUM_BEV_FEATURES
        self.block = BasicBlock2D(in_channels=self.num_in_features * self.num_heights,
                                  out_channels=self.num_bev_features,
                                  **self.model_cfg.ARGS)
        self.blck_copy = BasicBlock2D(in_channels=self.num_in_features * self.num_heights,
                                  out_channels=self.num_bev_features,
                                  **self.model_cfg.ARGS)
        self.block_target = BasicBlock2D(in_channels=self.num_in_features * self.num_heights,
                                  out_channels=self.num_bev_features,
                                  **self.model_cfg.ARGS)
        self.GC_block = ContextBlock2d(in_channels=self.num_bev_features,
                                  out_channels=self.num_bev_features,
                                   **self.model_cfg.GC_ARGS)
        self.sam = SAM(in_channels=self.num_bev_features,
                                  out_channels=self.num_bev_features
                                  )

    def forward(self, batch_dict):
        """
        Collapses voxel features to BEV via concatenation and channel reduction
        Args:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Voxel feature representation
        Returns:
            batch_dict:
                spatial_features: (B, C, Y, X), BEV feature representation
        """
        #### Image bev ####
        voxel_features = batch_dict["voxel_features"] #  [2, 64, 10, 188, 140]
        # (B, C, Z, Y, X) -> (B, C*Z, Y, X)
        bev_features = voxel_features.flatten(start_dim=1, end_dim=2)  # [2, 640, 188, 140]
        # Channel reduction (1x1 conv): (B, C*Z, Y, X) -> (B, C, Y, X)
        bev_features_ori = self.block(bev_features) # [2, 128, 188, 140]
        batch_dict["spatial_features"] = bev_features_ori

        ## Disentagle bev-image into two copies ###
        bev_features_new = self.blck_copy(bev_features)
        bev_features_new = self.sam(bev_features_new)
        batch_dict["spatial_features_copy"] = bev_features_new

        # #### Image like bev ####
        voxel_features_target = batch_dict["voxel_features_target"]
        bev_features_target = voxel_features_target.flatten(start_dim=1, end_dim=2)  # (B, C, Z, Y, X) -> (B, C*Z, Y, X)
        bev_features_target = self.block_target(bev_features_target)
        bev_features_target = self.GC_block(bev_features_target)  # (B, C*Z, Y, X) -> (B, C, Y, X)
        batch_dict["spatial_features_target"] = bev_features_target
        # # # #### Fusion ####
        batch_dict["spatial_features_fusion"] =  batch_dict["spatial_features_copy"] + 0.2 *batch_dict["spatial_features"]
        return batch_dict
