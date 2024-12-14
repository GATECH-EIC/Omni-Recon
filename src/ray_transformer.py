import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import (rearrange, reduce, repeat)

from .utils.grid_sample import grid_sample_2d, grid_sample_3d
from .attention.transformer import LocalFeatureTransformer

import time

import math
PI = math.pi

class PositionEncoding(nn.Module):
    def __init__(self, L=10):
        super().__init__()
        self.L = L
        self.augmented = rearrange((PI * 2 ** torch.arange(-1, self.L - 1)), "L -> L 1 1 1")

    def forward(self, x):
        sin_term = torch.sin(self.augmented.type_as(x) * rearrange(x, "RN SN Dim -> 1 RN SN Dim")) # BUG? 
        cos_term = torch.cos(self.augmented.type_as(x) * rearrange(x, "RN SN Dim -> 1 RN SN Dim") )
        sin_cos_term = torch.stack([sin_term, cos_term])

        sin_cos_term = rearrange(sin_cos_term, "Num2 L RN SN Dim -> (RN SN) (L Num2 Dim)")

        return sin_cos_term


class RayTransformer(nn.Module):
    def __init__(self, args, img_feat_dim=32, fea_volume_dim=16, radiance_only=False):
        super().__init__()

        self.args = args
        self.offset =  [[0, 0, 0]]
        
        self.radiance_only = radiance_only

        self.volume_reso = args.volume_reso
        assert self.volume_reso > 0
        
        self.only_volume = self.args.only_volume
        if self.only_volume:
            assert self.volume_reso > 0, "if only use volume feature, must have volume"

        self.img_feat_dim = img_feat_dim
        
        self.fea_volume_dim = 32
        
        self.use_ray_renderer = args.use_ray_renderer
        
        self.predict_weight = args.predict_weight
        
        self.PE_d_hid = 8

        extra_feat_dim = 0
        
        if not self.radiance_only:
            extra_feat_dim += self.fea_volume_dim

            self.fv_upsampler = nn.Sequential(
                    nn.Linear(self.fea_volume_dim, self.img_feat_dim + extra_feat_dim),
                )
                
            self.feat_dim_total = self.img_feat_dim + extra_feat_dim
                        
            self.fv_trans = nn.ModuleList([])
            
            self.ray_trans = nn.ModuleList([])
            self.view_trans = nn.ModuleList([])
            self.q_fcs = nn.ModuleList([])
            for i in range(self.args.trans_depth):
                viewtrans = Transformer2D(
                    dim=self.feat_dim_total,
                    ff_hid_dim=int(self.feat_dim_total * 4),
                    ff_dp_rate=0.1,
                    attn_dp_rate=0.1,
                )
                self.view_trans.append(viewtrans)
                
                fvtrans = Transformer(
                    dim=self.feat_dim_total,
                    ff_hid_dim=int(self.feat_dim_total * 4),
                    n_heads=4,
                    ff_dp_rate=0.1,
                    attn_dp_rate=0.1,
                )
                self.fv_trans.append(fvtrans)
                                    
                raytrans = Transformer(
                    dim=self.feat_dim_total,
                    ff_hid_dim=int(self.feat_dim_total * 4),
                    n_heads=4,
                    ff_dp_rate=0.1,
                    attn_dp_rate=0.1,
                )
                self.ray_trans.append(raytrans)

                if i % 2 == 0:
                    q_fc = nn.Sequential(
                        nn.Linear(self.feat_dim_total + 3 + 3 * 2 * 10, self.feat_dim_total),
                        nn.ReLU(),
                        nn.Linear(self.feat_dim_total, self.feat_dim_total),
                    )
                else:
                    q_fc = nn.Identity()
                self.q_fcs.append(q_fc)
            
            if self.use_ray_renderer:
                self.ray_renderer = LocalFeatureTransformer(self.feat_dim_total, 
                                        nhead=1, layer_names=['cross'], attention='full')
            else:
                self.ray_renderer = None
            
            if self.predict_weight:
                self.norm = None
            else:
                self.norm = nn.LayerNorm(self.feat_dim_total)
                
            self.pos_enc = Embedder(
                input_dims=3,
                include_input=True,
                max_freq_log2=9,
                num_freqs=10,
                log_sampling=True,
                periodic_fns=[torch.sin, torch.cos],
            )

            
            self.DensityMLP = nn.Sequential(
                nn.Linear(self.feat_dim_total, 32), nn.ReLU(inplace=True),
                nn.Linear(32, 16), nn.ReLU(inplace=True),
                nn.Linear(16, 1))

        self.relu = nn.ReLU(inplace=True)
            
        self.softmax = nn.Softmax(dim=-2)
        
        self.simple_appear_feat = True
        self.tiny_shader = True
        
        if self.simple_appear_feat:
            radiance_dim_extra = 0
        else:
            radiance_dim_extra = self.fea_volume_dim if self.volume_reso > 0 else 0

        if self.tiny_shader:
            self.radiance_fc = nn.Sequential(nn.Linear(self.img_feat_dim + radiance_dim_extra, self.img_feat_dim + radiance_dim_extra),
                                        nn.ReLU(inplace=True))
        else:
            self.radiance_fc = nn.Sequential(nn.Linear(self.img_feat_dim + radiance_dim_extra, self.img_feat_dim + radiance_dim_extra),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.img_feat_dim + radiance_dim_extra, self.img_feat_dim + radiance_dim_extra),
                                        nn.ReLU(inplace=True))
        
        self.use_se = args.use_se
        if self.use_se:
            self.se_fc = nn.Sequential(nn.Linear(self.img_feat_dim + radiance_dim_extra, 32),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(32, 1),
                                    nn.Sigmoid())
        else:
            self.se_fc = None
            
        self.linear_radianceweight_1_softmax = nn.Sequential(
            nn.Linear(self.img_feat_dim+3+radiance_dim_extra, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 8), nn.ReLU(inplace=True),
            nn.Linear(8, 1),
        )


    def order_posenc(self, d_hid, n_samples):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table)

        return sinusoid_table


    def forward(self, point3D, batch, source_imgs_feat, fea_volume=None, appear_feat=None, clip_features=None, ray_feats=None):
        B, NV, _, H, W = batch['source_imgs'].shape  # NV: num source views
        _, RN, SN, _ = point3D.shape  # RN: num rays, SN: num samples
        FDim = source_imgs_feat.size(2) # feature dim
        CN = len(self.offset)

        # calculate relative direction
        vector_1 = (point3D - repeat(batch['ref_pose_inv'][:,:3,-1], "B DimX -> B 1 1 DimX"))
        vector_1 = repeat(vector_1, "B RN SN DimX -> B 1 RN SN DimX")
        vector_2 = (point3D.unsqueeze(1) - repeat(batch['source_poses_inv'][:,:,:3,-1], "B L DimX -> B L 1 1 DimX")) # B L RN SN DimX
        vector_1 = vector_1/torch.linalg.norm(vector_1, dim=-1, keepdim=True) # normalize to get direction
        vector_2 = vector_2/torch.linalg.norm(vector_2, dim=-1, keepdim=True)
        dir_relative = vector_1 - vector_2 
        dir_relative = dir_relative.float()
        dir_relative_dot = torch.sum(vector_1 * vector_2, dim=-1, keepdim=True)

        if fea_volume is not None:
            fea_volume_feat = grid_sample_3d(fea_volume, point3D.unsqueeze(1).float(), align_corners=self.args.align_corners_3d)
            fea_volume_feat = rearrange(fea_volume_feat, "B C RN SN -> (B RN SN) C")
        else:
            fea_volume_feat = None
        
        point3D = repeat(point3D, "B RN SN DimX -> B NV RN SN DimX", NV=NV).float()
        point3D = torch.cat([point3D, torch.ones_like(point3D[:,:,:,:,:1])], axis=4)
        
        # B NV 4 4 -> (B NV) 4 4
        points_in_pixel = torch.bmm(rearrange(batch['source_poses'], "B NV M_1 M_2 -> (B NV) M_1 M_2", M_1=4, M_2=4), 
                                rearrange(point3D, "B NV RN SN DimX -> (B NV) DimX (RN SN)"))
        
        points_in_pixel = rearrange(points_in_pixel, "(B NV) DimX (RN SN) -> B NV DimX RN SN", B=B, RN=RN)
        points_in_pixel = points_in_pixel[:,:,:3]

        proj_depth = points_in_pixel[:,:,2:3]   # [B, NV, 1, RN, SN]
        proj_depth[proj_depth <= 0] = 1e-4  

        # in 2D pixel coordinate
        mask_valid_depth = points_in_pixel[:,:,2]>0  #B NV RN SN
        mask_valid_depth = mask_valid_depth.float()
        points_in_pixel = points_in_pixel[:,:,:2] / torch.clamp(points_in_pixel[:,:,2:3], 1e-4)

        img_feat_sampled, mask = grid_sample_2d(rearrange(source_imgs_feat, "B NV C H W -> (B NV) C H W"), 
                                rearrange(points_in_pixel, "B NV Dim2 RN SN -> (B NV) RN SN Dim2"), align_corners=self.args.align_corners_2d)
        img_rgb_sampled, _ = grid_sample_2d(rearrange(batch['source_imgs'], "B NV C H W -> (B NV) C H W"), 
                                rearrange(points_in_pixel, "B NV Dim2 RN SN -> (B NV) RN SN Dim2"), align_corners=self.args.align_corners_2d)
            
            
        mask = rearrange(mask, "(B NV) RN SN -> B NV RN SN", B=B)
        mask = mask * mask_valid_depth

        img_rgb_sampled = img_rgb_sampled * rearrange(mask, "B NV RN SN -> (B NV) 1 RN SN")
        img_feat_sampled = img_feat_sampled * rearrange(mask, "B NV RN SN -> (B NV) 1 RN SN")
        
        img_feat_sampled = rearrange(img_feat_sampled, "(B NV) C RN SN -> B NV C RN SN", B=B)
        img_rgb_sampled = rearrange(img_rgb_sampled, "(B NV) C RN SN -> B NV C RN SN", B=B)

        x = rearrange(img_feat_sampled, "B NV C RN SN -> (B RN SN) NV C")

        num_valid_obs = torch.sum(rearrange(mask, "B NV RN SN -> (B RN) SN NV 1"), dim=2)  # [B_RN, SN, 1]
        sample_mask = num_valid_obs >= 1  # [B_RN, SN, 1]
            
        if appear_feat is not None:
            appear_feat_sampled, _ = grid_sample_2d(rearrange(appear_feat, "B NV C H W -> (B NV) C H W"), 
                                    rearrange(points_in_pixel, "B NV Dim2 RN SN -> (B NV) RN SN Dim2"), align_corners=self.args.align_corners_2d)
            appear_feat_sampled = appear_feat_sampled * rearrange(mask, "B NV RN SN -> (B NV) 1 RN SN")
            appear_feat_sampled = rearrange(appear_feat_sampled, "(B NV) C RN SN -> B NV C RN SN", B=B)
            x_img_fea = rearrange(appear_feat_sampled, "B NV C RN SN -> (B RN SN) NV C")
        else:
            x_img_fea = x

        if clip_features is not None:
            clip_feat_sampled, _ = grid_sample_2d(rearrange(clip_features, "B NV C H W -> (B NV) C H W"), 
                                    rearrange(points_in_pixel, "B NV Dim2 RN SN -> (B NV) RN SN Dim2"), align_corners=self.args.align_corners_2d)
            clip_feat_sampled = rearrange(clip_feat_sampled, "(B NV) C RN SN -> B NV C RN SN", B=B)
        
        if self.simple_appear_feat:
            x_fea = x_img_fea
        else:
            fea_volume_feat_repeat = repeat(fea_volume_feat, "B_RN_SN C -> B_RN_SN NV C", NV=NV)
            x_fea = torch.cat([x_img_fea, fea_volume_feat_repeat], axis=-1)  # (B RN SN) NV C

         
        if not self.radiance_only:     
            fea_volume_feat_repeat = repeat(fea_volume_feat, "B_RN_SN C -> B_RN_SN NV C", NV=NV)
            x = torch.cat([x_img_fea, fea_volume_feat_repeat], axis=-1)  # (B RN SN) NV C
            
            fea_volume_feat = self.fv_upsampler(fea_volume_feat)  # (B RN SN) C*2 
                
            input_pts = point3D[:,0,:,:,:3]  # B RN SN 3
            input_pts = rearrange(input_pts, "B RN SN Dim3 -> (B RN SN) Dim3").float()
            input_pts = self.pos_enc(input_pts)
            input_pts = rearrange(input_pts, "(B RN SN) DimX -> (B RN) SN DimX", RN=RN, SN=SN)

            ray_diff = rearrange(torch.cat([dir_relative, dir_relative_dot], dim=-1), "B NV RN SN Dim3 ->(B RN) SN NV Dim3").float()

            q = rearrange(x, "(B RN SN) NV C -> (B RN) SN NV C", RN=RN, SN=SN).max(dim=2)[0]  # (B RN) SN C

            k = rearrange(x, "(B RN SN) NV C -> (B RN) SN NV C", RN=RN, SN=SN)
                
            for i, (viewtrans, q_fc, raytrans) in enumerate(
                zip(self.view_trans, self.q_fcs, self.ray_trans)
            ):
                q = self.fv_trans[i](q, kv=rearrange(fea_volume_feat, "(B RN SN) C -> (B RN) SN C", RN=RN, SN=SN))  # (B RN) SN C
                
                q = viewtrans(q, k, ray_diff, rearrange(mask, "B NV RN SN -> (B RN) SN NV 1"))  # (B RN) SN C
                    
                # embed positional information
                if i % 2 == 0:
                    q = torch.cat((q, input_pts), dim=-1)
                    q = q_fc(q)
                # ray transformer
                if self.args.use_causal_mask:
                    causal_mask = torch.tril(torch.ones(SN, SN), diagonal=0).bool().to(q)
                    causal_mask = repeat(causal_mask, "SN1 SN2-> B_RN 1 SN1 SN2", B_RN=B*RN)
                else:   
                    causal_mask = None
                
                q = raytrans(q, mask=causal_mask)  # (B RN) SN C
            
            if self.use_ray_renderer:  ## todo: without finishing & debugging !!!
                q = self.ray_renderer(q)
                srdf = self.ray_renderer.atten_weight.squeeze()   ## this is actually alpha, i.e., the per-point weights
            elif self.predict_weight:
                q = self.DensityMLP(q)  # (B RN) SN C
                srdf = self.softmax(q)  # (B RN) SN C, this is actually the per-point weight
            else:
                q = self.norm(q)
                srdf = self.DensityMLP(q)  # (B RN) SN C

        else:
            srdf = None
            sample_mask = None

        # calculate weight using view transformers result
        view_feature = rearrange(x_fea, "(B RN SN) NV C -> B RN SN NV C", B=B, RN=RN, SN=SN)
        view_feature = self.radiance_fc(view_feature)
        
        if self.use_se:
            view_feature = self.se_fc(view_feature) * view_feature

        dir_relative = rearrange(dir_relative, "B NV RN SN Dim3 -> B RN SN NV Dim3")

        x_weight = torch.cat([view_feature, dir_relative], axis=-1)
        x_weight = self.linear_radianceweight_1_softmax(x_weight)

        mask = rearrange(mask, "B NV RN SN -> B RN SN NV 1")
        x_weight[mask==0] = -1e9
        weight = self.softmax(x_weight)

        ray_mask = mask[...,0]  # B RN SN NV
        ray_mask = torch.sum(ray_mask, -1) > 2  # B RN SN
        ray_mask = torch.sum(ray_mask, -1) > 8  # B RN
        
        radiance = (img_rgb_sampled * rearrange(weight, "B RN SN L 1 -> B L 1 RN SN", B=B, RN=RN)).sum(axis=1)
        radiance = rearrange(radiance, "B DimRGB RN SN -> (B RN SN) DimRGB")

        if clip_features is not None:
            clip_feat = (clip_feat_sampled * rearrange(weight, "B RN SN L 1 -> B L 1 RN SN", B=B, RN=RN)).sum(axis=1)
            clip_feat = rearrange(clip_feat, "B DimRGB RN SN -> (B RN SN) DimRGB")
        else:
            clip_feat = None
         
        return radiance, srdf, points_in_pixel, ray_mask, sample_mask, clip_feat


class ViewTokenNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_parameter('view_token', nn.Parameter(torch.randn([1,dim])))

    def forward(self, x):
        return torch.ones([len(x), 1]).type_as(x) * self.view_token


class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    

class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dp_rate):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.dp(self.activ(self.fc1(x)))
        x = self.dp(self.fc2(x))
        return x


# Subtraction-based efficient attention
class Attention2D(nn.Module):
    def __init__(self, dim, dp_rate):
        super(Attention2D, self).__init__()
        self.q_fc = nn.Linear(dim, dim, bias=False)
        self.k_fc = nn.Linear(dim, dim, bias=False)
        self.v_fc = nn.Linear(dim, dim, bias=False)
        self.pos_fc = nn.Sequential(
            nn.Linear(4, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.attn_fc = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)

    def forward(self, q, k, pos, mask=None):
        q = self.q_fc(q)
        k = self.k_fc(k)
        v = self.v_fc(k)

        pos = self.pos_fc(pos)
        attn = k - q[:, :, None, :] + pos
        attn = self.attn_fc(attn)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-2)
        attn = self.dp(attn)

        x = ((v + pos) * attn).sum(dim=2)
        x = self.dp(self.out_fc(x))
        return x


# View Transformer
class Transformer2D(nn.Module):
    def __init__(self, dim, ff_hid_dim, ff_dp_rate, attn_dp_rate):
        super(Transformer2D, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)
        self.attn = Attention2D(dim, attn_dp_rate)

    def forward(self, q, k, pos, mask=None):
        residue = q
        x = self.attn_norm(q)
        x = self.attn(x, k, pos, mask)
        x = x + residue

        residue = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residue

        return x


class Attention(nn.Module):
    def __init__(self, dim, n_heads, dp_rate, attn_mode="qk", pos_dim=None):
        super(Attention, self).__init__()
        self.q_fc = nn.Linear(dim, dim, bias=False)
        self.k_fc = nn.Linear(dim, dim, bias=False)

        self.v_fc = nn.Linear(dim, dim, bias=False)
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.n_heads = n_heads
        self.attn_mode = attn_mode

    def forward(self, x, kv=None, pose=None, ret_attn=False, mask=None):
        if kv is None: kv = x
        q = self.q_fc(x)
        q = q.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
        k = self.k_fc(kv)
        k = k.view(kv.shape[0], kv.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
        v = self.v_fc(kv)
        v = v.view(kv.shape[0], kv.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.shape[-1])
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dp(attn)

        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous()
        out = out.view(x.shape[0], x.shape[1], -1)
        out = self.dp(self.out_fc(out))
        if ret_attn:
            return out, attn
        else:
            return out


class Transformer(nn.Module):
    def __init__(
        self, dim, ff_hid_dim, ff_dp_rate, n_heads, attn_dp_rate, attn_mode="qk", pos_dim=None
    ):
        super(Transformer, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)
        self.attn = Attention(dim, n_heads, attn_dp_rate, attn_mode, pos_dim)

    def forward(self, x, kv=None, pos=None, ret_attn=False, mask=None):
        residue = x
        x = self.attn_norm(x)
        
        if kv is None: kv = x
        x = self.attn(x, kv, pos, ret_attn, mask=mask)
        if ret_attn:
            x, attn = x
        x = x + residue

        residue = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residue

        if ret_attn:
            return x, attn.mean(dim=1)[:, 0]
        else:
            return x