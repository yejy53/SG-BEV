import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from models.utils.linear_mlp import LinearMLP
from models.utils.jacobian import grid_sample


class Neck_SVI_ST(BaseModule):
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

        # Fuse the feature maps
        fused_feature_map = self.linear_fuse(torch.cat([c4_resized, c3_resized, c2_resized], dim=1))
        # import pdb;pdb.set_trace()
        fused_feature_map_bev,_ = grid_sample(fused_feature_map, self.uv)

        upsampled_feature_map = self.conv_transpose1(fused_feature_map_bev)
        upsampled_feature_map = self.relu(upsampled_feature_map)  # 可选的激活步骤
    
        upsampled_feature_map = self.conv_transpose2(upsampled_feature_map)
        upsampled_feature_map = self.relu(upsampled_feature_map)  # 可选的激活步骤
        
        upsampled_feature_map = self.conv_transpose3(upsampled_feature_map)
        upsampled_feature_map = self.relu(upsampled_feature_map)  # 可选的激活步骤

        return upsampled_feature_map