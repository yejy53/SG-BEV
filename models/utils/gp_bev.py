import torch
import numpy as np
from .jacobian import grid_sample
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

def sat2grd_uv(feature, rot, shift_u, shift_v, meter_per_pixel, grd_height, device):

    B,C,H,W = feature[1].size()
    S = 32
    shift_u = shift_u * S / 4
    shift_v = shift_v * S / 4

    ii, jj = torch.meshgrid(torch.arange(0, S, dtype=torch.float32, device=device),
                            torch.arange(0, S, dtype=torch.float32, device=device), indexing='ij')  
    ii = ii.unsqueeze(dim=0).repeat(B, 1, 1)  # [B, S, S] v dimension
    jj = jj.unsqueeze(dim=0).repeat(B, 1, 1)  # [B, S, S] u dimension

    radius = torch.sqrt((ii - (S / 2 - 0.5 + shift_v.reshape(-1, 1, 1))) ** 2 + (
                jj - (S / 2 - 0.5 + shift_u.reshape(-1, 1, 1))) ** 2)

    theta = torch.atan2(ii - (S / 2 - 0.5 + shift_v.reshape(-1, 1, 1)),
                        jj - (S / 2 - 0.5 + shift_u.reshape(-1, 1, 1)))
    theta = (-np.pi / 2 + (theta) % (2 * np.pi)) % (2 * np.pi)
    theta = (theta + rot[:, None, None] * 1 / 180 * np.pi) % (2 * np.pi)

    theta = theta / 2 / np.pi * W


    phimin = torch.atan2(radius * meter_per_pixel, torch.tensor(grd_height))
    phimin = phimin / np.pi * H

    uv = torch.stack([theta, phimin], dim=-1)
    # print(uv)
    # print(uv.shape)
    return uv