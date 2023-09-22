import torch.nn as nn
# from pcdet.models.model_utils.basic_block_2d import BasicBlock2D, BasicBlock2D_copy

class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        # 這邊是直接壓縮得到的BEV，可以說是完全使用sparse depth轉出來的BEV plane
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape  # torch.Size([2, 64, 2, 188, 140])  -->  torch.Size([2, 128, 188, 140])
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        ### create image
        # spatial_features = spatial_features.view(N, C, D, H, W)
        # spatial_img_features = spatial_features.view(N, C , H, D * W) #　torch.Size([2, 64, 188, 280])

        return batch_dict
