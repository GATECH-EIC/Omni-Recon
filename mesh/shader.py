import numpy as np 
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import (rearrange, reduce, repeat)
from collections import Counter

from src.utils.feature_extractor import FPN_FeatureExtractor, FPN_FeatureExtractor_Multi_Res
from src.utils.feature_network_ibrnet import ResUNet, ResUNet_Multi_Res

from src.ray_transformer import RayTransformer
from src.feature_volume import FeatureVolume, FeatureVolume_Multi_Res

from src.utils.grid_sample import grid_sample_2d

from src.ray_transformer import RayTransformer

import plyfile
import skimage.measure
from tqdm import tqdm
import os
import imageio
import time


class Gen_Shader(nn.Module):
    """
    Ray transformer
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.feat_extractor = FPN_FeatureExtractor_Multi_Res(out_ch=32)

        self.ray_transformer = RayTransformer(args = self.args, radiance_only=True)
            
        self.feature_volume = FeatureVolume_Multi_Res(self.args.volume_reso, args=self.args)

    def load_checkpoint(self, model_path): 
        ckpt = torch.load(model_path)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
            
        missing_keys, unexpected_keys = self.load_state_dict(ckpt, strict=False)
        assert len(missing_keys) == 0
            
        return
    
    def forward(self, pixel_feature, data):
        
        B, L, _, imgH, imgW = data['source_imgs'].shape  # [batch_size=1, num_src_view, 3, h, w]
        
        source_imgs = rearrange(data['source_imgs'], "B L C H W -> (B L) C H W")

        source_imgs_feat, fpn = self.feat_extractor(source_imgs)
        for i in range(len(fpn)):
            fpn[i] = rearrange(fpn[i], "(B L) C H W -> B L C H W", L=L)
        fv_input = fpn
        source_imgs_feat = rearrange(source_imgs_feat, "(B L) C H W -> B L C H W", L=L)

        if self.feature_volume is not None and not self.args.simple_appear_feat:
            feat_vol = self.feature_volume(fv_input, data)
        else:
            feat_vol = None

        point3D = pixel_feature
        point3D = torch.cat([point3D, torch.ones_like(point3D[:,:,:,:1])], axis=-1)  # [B, H, W, 4]
        point3D = rearrange(point3D, "B H W DimX -> B DimX (H W)")  # [B, H*W, 4]
        point3D = torch.bmm(torch.inverse(data['scale_mat']), torch.bmm(torch.inverse(data['trans_mat']), point3D))  # [B, H*W, 4]
        point3D = rearrange(point3D, "B DimX HW -> B HW 1 DimX")[...,:3]  # [B, H*W, 1, 3]

        radiance, srdf, points_in_pixel, ray_mask, sample_mask, clip_feat = self.ray_transformer(point3D, data, source_imgs_feat, feat_vol)  # point3D: [B RN SN DimX], Radiance: [(B RN SN) C]
        
        return radiance, srdf, points_in_pixel