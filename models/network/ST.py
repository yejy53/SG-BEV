import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import sys
import os
import json
import mmcv 


from mmengine.config import Config
from mmseg.models import build_backbone
from models.backbones.mscan import mscan
from models.necks.neck_sate import Neck_Sate
from models.necks.neck_svi_ST import Neck_SVI_ST
from models.necks.neck_fusion import BaseFusionModule

from models.heads.head import Decoder

from models.utils.st_bev import get_BEV_projection_module
from models.utils.jacobian import grid_sample


class ST(nn.Module):
    def __init__(self, cfg, device ):
        super(ST,self).__init__()

        # 读入模型参数设置
        n_obj_classes = cfg['n_obj_classes']

        self.sate_input_dim = cfg['backbone']['embed_dims']
        self.neck_output_dim = cfg['neck_output_dim']
        self.branch_output_dim = cfg['branch_output_dim']
        self.decoder_dim = cfg['decoder_dim']

        self.encoder_backbone_config = cfg['backbone']
        
        self.device = device
        self.device_mem = device

        self.bev_h = cfg['bev_size']
        self.bev_w = cfg['bev_size']
        self.sate_size = cfg['sate_size']
        self.embed_dims = cfg['mem_feature_dim']
        self.bs = cfg['batch_size_every_processer']


        # 读入ST的参数设置
        ST_cfg = cfg['ST_config']

        self.img_height = ST_cfg['img_height']
        self.img_weight = ST_cfg['img_weight']
        self.uv_height = ST_cfg['uv_height']
        self.uv_weight = ST_cfg['uv_weight']
        self.fov = ST_cfg['fov']
        self.dty = ST_cfg['dty']
        self.dx = ST_cfg['dx']
        self.dy = ST_cfg['dy']
       
        dtype = torch.float32

        # 初始化街景backbone
        svi_backbone = build_backbone(self.encoder_backbone_config)
        svi_backbone.init_weights()
        self.svi_encoder_backbone = svi_backbone

        self.uv = get_BEV_projection_module(self.img_height,self.img_weight,self.uv_height,self.uv_weight,self.fov,self.dty,self.dx,self.dy,self.device,self.bs)
        self.svi_neck = Neck_SVI_ST(self.sate_input_dim,self.neck_output_dim, self.branch_output_dim)

        # 初始化卫星backbone
        sate_backbone = build_backbone(self.encoder_backbone_config)
        sate_backbone.init_weights()
        self.sate_encoder_backbone = sate_backbone

        self.sate_neck = Neck_Sate(self.sate_input_dim,self.neck_output_dim, self.branch_output_dim)

        # 初始化融合模块

        self.fusion_module = BaseFusionModule()

        # 初始化解码器
        self.head = Decoder(self.embed_dims, n_obj_classes)

    def svi_extract_feature(self, svi_image):
        
        # 32,32
        x = self.svi_encoder_backbone(svi_image)

        x = self.svi_neck(x,self.uv)

        return x

    def sate_extract_feature(self, sate_image):
        x = self.sate_encoder_backbone(sate_image)

        x = self.sate_neck(x)

        return x
    def forward(self, svi_image, sate_image):
        
        # 先写街景分支的，参考onebev
        svi_feature = self.svi_extract_feature(svi_image)

        sate_feature = self.sate_extract_feature(sate_image)
        
        fused_feature = self.fusion_module(sate_feature, svi_feature)

        observed_masks = torch.ones((self.bs, self.sate_size, self.sate_size), dtype=torch.bool, device=self.device)

        seg_map = self.head(fused_feature)

        return seg_map, observed_masks




    
if __name__ == '__main__':
    # 读取配置文件
    config_path = '/home/ly/code/SG_BEV/configs/ST.py'  # 确保路径正确
    cfg = Config.fromfile(config_path)

    # 准备输入数据
    svi_image = torch.randn(1, 3, 1024, 512)
    sate_image = torch.randn(1, 3, 256, 256)
    filename = ''

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ST(cfg.model, device).to(device)
    model.eval()
    # 前向传播
    with torch.no_grad():
        seg_map, observed_masks = model(svi_image.to(device), sate_image.to(device))

    