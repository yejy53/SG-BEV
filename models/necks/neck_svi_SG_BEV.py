import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence, build_positional_encoding

from models.utils.linear_mlp import LinearMLP
from models.utils.jacobian import grid_sample
from models.utils.sg_bev import get_bev_features

class Neck_SVI_SG_BEV(BaseModule):
    def __init__(self, neck_input_dims=[96, 192, 384, 768], neck_output_dims=256,branch_output_dims=128):
        super().__init__()

        # LinearMLP中线进行展平和转置，再进行线性投影
        self.linear_c4 = LinearMLP(neck_input_dims[3], neck_output_dims)
        self.linear_c3 = LinearMLP(neck_input_dims[2], neck_output_dims)
        self.linear_c2 = LinearMLP(neck_input_dims[1], neck_output_dims)

        self.linear_fuse = nn.Conv2d(3 * neck_output_dims, branch_output_dims, 1)

        self.conv_transpose1 = nn.ConvTranspose2d(branch_output_dims, branch_output_dims, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv_transpose2 = nn.ConvTranspose2d(branch_output_dims, branch_output_dims, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv_transpose3 = nn.ConvTranspose2d(branch_output_dims, branch_output_dims, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.relu = nn.ReLU(inplace=True)

        self.encoder = 

    def mask_update(self,  # features,
                    proj_indices, masks_inliers, rgb_features):

        observed_masks = torch.zeros((self.bs, self.bev_h, self.bev_w), dtype=torch.bool, device=self.device)
        #observed_masks = torch.ones((self.bs, self.bev_h, self.bev_w), dtype=torch.bool, device=self.device)
        ################################################################################################################
        mask_inliers = masks_inliers[:, :, :]                  
        proj_index = proj_indices                               

        # m = (proj_index >= 0)  # -- (N, 500*500)
        threshold_index_m = torch.max(proj_index).item()
        m = (proj_index < threshold_index_m)

        if m.any():


            tmp_top_down_mask = m.view(-1)         # torch.Size([250000])
            # state_rgb[tmp_top_down_mask, :] = rgb_memory.to(self.device_mem)

            ############################################################################################################
            observed_masks += m.reshape(self.bs, self.bev_w, self.bev_h)  

        return observed_masks


    def forward(self, feature_maps,uv):
        batch_size, _, _, _ = feature_maps[0].shape
        dtype = feature_maps[0].dtype

        self.uv = uv

        c4 = feature_maps[3]
        c3 = feature_maps[2]
        c2 = feature_maps[1]

        # Apply linear layers
        c4_transformed = self.linear_c4(c4).permute(0, 2, 1).reshape(batch_size, -1, c4.shape[2], c4.shape[3]).contiguous()
        c3_transformed = self.linear_c3(c3).permute(0, 2, 1).reshape(batch_size, -1, c3.shape[2], c3.shape[3]).contiguous()
        c2_transformed = self.linear_c2(c2).permute(0, 2, 1).reshape(batch_size, -1, c2.shape[2], c2.shape[3]).contiguous()

        # Interpolate the transformed feature maps to match the size of c2
        c4_resized = F.interpolate(c4_transformed, size=c2.size()[2:], mode='bilinear', align_corners=False)
        c3_resized = F.interpolate(c3_transformed, size=c2.size()[2:], mode='bilinear', align_corners=False)
        c2_resized = F.interpolate(c2_transformed, size=c2.size()[2:], mode='bilinear', align_corners=False)

        # 以上街景特征提取部分与ST、GP相同，但是往下的indice和注意力计算不同了

        fused_feature_map = self.linear_fuse(torch.cat([c4_resized, c3_resized, c2_resized], dim=1))

        fused_feature_map = [fused_feature_map]

        bev_queries, feat_flatten, bev_h, bev_w, bev_pos, spatial_shapes, level_start_index = get_bev_features(
            feat_fpn, self.bev_queries, self.bev_h, self.bev_w, self.bev_pos)
        observed_masks = self.mask_update(
                                            proj_indices,
                                            masks_inliers,
                                            rgb_no_norm)
        
        bev_embed = self.encoder(
                    bev_queries,
                    feat_flatten,                   #####
                    feat_flatten,
                    bev_h=bev_h,
                    bev_w=bev_w,
                    bev_pos=bev_pos,
                    spatial_shapes=spatial_shapes,  ##### from feature map
                    level_start_index=level_start_index,
                    prev_bev= None,
                    shift= None,
                    #map_mask = map_mask,
                    map_mask = observed_masks,
                    map_heights = map_heights,
                    image_shape = self.image_shape,
                    indices = proj_indices
                )        

        return upsampled_feature_map