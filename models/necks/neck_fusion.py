import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from models.utils.linear_mlp import LinearMLP


class BaseFusionModule(BaseModule):
    def __init__(self):
        super().__init__()



    def forward(self, sate_feature_map, svi_feature_map):
        fusion_feature_map = torch.cat([sate_feature_map, svi_feature_map],dim=1)


        return fusion_feature_map

class AlignFAnew(nn.Module):
    def __init__(self, features):
        super(AlignFAnew, self).__init__()

        self.delta_gen1 = nn.Sequential(
                        nn.Conv2d(features, 128, kernel_size=1, bias=False),
                        InPlaceABNSync(128),
                        nn.Conv2d(128, 2, kernel_size=3, padding=1, bias=False)
                        )


        self.delta_gen1[2].weight.data.zero_()

    def bilinear_interpolate_torch_gridsample2(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 2.0
        norm = torch.tensor([[[[(out_w-1)/s, (out_h-1)/s]]]]).type_as(input).to(input.device) 
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

    def forward(self, low_stage, high_stage):
        h, w = low_stage.size(2), low_stage.size(3)
        
        concat = torch.cat((low_stage, high_stage), 1)
        delta1 = self.delta_gen1(concat)
        low_stage = self.bilinear_interpolate_torch_gridsample2(low_stage, (h, w), delta1)

        concat = torch.cat((low_stage, high_stage), 1)
        return concat


