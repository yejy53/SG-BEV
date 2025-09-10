import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import io
import random
import sys
import cv2
import matplotlib.pyplot as plt 
from .torch_geometry import euler_angles_to_matrix

import math


def get_BEV_projection_module(Hp,Wp, Ho, Wo, Fov=170, dty=-20, dx=0, dy=0, device = 'cpu',batch_size=1):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = device

    Fov = Fov * torch.pi / 180  # Field of View in radians
    center = torch.tensor([Wp / 2 + dx, Hp + dy]).to(device)  # Overhead view center

    anglex = torch.tensor(dx).to(device) * 2 * torch.pi / Wp
    angley = -torch.tensor(dy).to(device) * torch.pi / Hp
    anglez = torch.tensor(0).to(device)

    # Euler angles
    euler_angles = (anglex, angley, anglez)
    euler_angles = torch.stack(euler_angles, -1)

    # Calculate the rotation matrix
    R02 = euler_angles_to_matrix(euler_angles, "XYZ")
    R20 = torch.inverse(R02)

    f = Wo / 2 / torch.tan(torch.tensor(Fov / 2))
    out = torch.zeros((Wo, Ho, 2)).to(device)
    f0 = torch.zeros((Wo, Ho, 3)).to(device)
    f0[:, :, 0] = Ho / 2 - (torch.ones((Ho, Wo)).to(device) * (torch.arange(Ho)).to(device)).T
    f0[:, :, 1] = Wo / 2 - torch.ones((Ho, Wo)).to(device) * torch.arange(Wo).to(device)
    f0[:, :, 2] = -torch.ones((Wo, Ho)).to(device) * f
    f1 = R20 @ f0.reshape((-1, 3)).T  # x, y, z (3, N)
    f1_0 = torch.sqrt(torch.sum(f1**2, 0))
    f1_1 = torch.sqrt(torch.sum(f1[:2, :]**2, 0))
    theta = torch.arctan2(f1[2, :], f1_1) + torch.pi / 2  # [-pi/2, pi/2] => [0, pi]
    phi = torch.arctan2(f1[1, :], f1[0, :])  # [-pi, pi]
    phi = phi + torch.pi  # [0, 2pi]

    i_p = 1 - theta / torch.pi  # [0, 1]
    j_p = 1 - phi / (2 * torch.pi)  # [0, 1]
    out[:, :, 0] = j_p.reshape((Ho, Wo))
    out[:, :, 1] = i_p.reshape((Ho, Wo))
    out[:, :, 0] = (out[:, :, 0] - 0.5) / 0.5  # [-1, 1]
    out[:, :, 1] = (out[:, :, 1] - 0.5) / 0.5  # [-1, 1]

    # Expand to batch size dimension
    out = out.unsqueeze(0).expand(batch_size, -1, -1, -1)  # Add batch dimension


    return out