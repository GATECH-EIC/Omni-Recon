import os, piq, sys
from re import I
from stat import UF_OPAQUE

import numpy as np
import torch
from torch import optim
from PIL import Image
from tqdm import tqdm
import pytorch_lightning as pl

from einops import (rearrange, reduce, repeat)

from .utils.sampler import FixedSampler, ImportanceSampler
from .utils.feature_extractor import FPN_FeatureExtractor, FPN_FeatureExtractor_Multi_Res
from .utils.feature_network_ibrnet import ResUNet, ResUNet_Multi_Res
from .utils.single_variance_network import SingleVarianceNetwork
from .utils.renderer import VolumeRenderer, LaplaceDensity, SimpleRenderer
from .utils.loss import DepthLoss

from .feature_volume import FeatureVolume, FeatureVolume_Multi_Res
from .clip_utils import CLIP_MODEL

from .ray_transformer import RayTransformer

import plyfile
import skimage.measure
import imageio


class OmniRecon(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.train_ray_num = args.train_ray_num
        
        self.coarse_only = self.args.coarse_only

        if self.args.extract_geometry: # testing
            self.point_num = args.test_sample_coarse
            self.point_num_2 = args.test_sample_fine
        else:
            self.point_num = args.coarse_sample
            self.point_num_2 = args.fine_sample
        
        self.use_clip = args.use_clip
        self.clip_model = None

        self.feat_extractor = FPN_FeatureExtractor_Multi_Res(out_ch=32)
        
        self.appear_feat_encoder = None
            
        self.fixed_sampler = FixedSampler(point_num = self.point_num, inv_uniform=self.args.inv_uniform)
        self.importance_sampler = ImportanceSampler(point_num = self.point_num_2)
        self.deviation_network = SingleVarianceNetwork(0.3) if not self.args.disable_deviation and not self.args.vanilla_volume_rendering and not self.args.use_volsdf and not self.args.predict_weight else None # add variance network

        if self.args.use_volsdf:
            if self.args.anneal_beta:
                self.density_fun = LaplaceDensity(beta_min=0.0001)
            else:
                self.density_fun = LaplaceDensity(params_init={'beta': 0.1}, beta_min=0.0001)
        else:
            self.density_fun = None
        
        if self.args.predict_weight:
            self.renderer = SimpleRenderer(self.args)
        else:
            self.renderer = VolumeRenderer(self.args, density_fun=self.density_fun)
            
        self.ray_transformer = RayTransformer(args = self.args)
            
        if self.args.model_type == 'ibrnet':
            self.args.volume_reso = 0
            
        self.feature_volume = FeatureVolume_Multi_Res(self.args.volume_reso, args=self.args)

        self.pos_encoding = self.order_posenc(d_hid=16, n_samples=self.point_num)
        self.pos_encoding_2 = self.order_posenc(d_hid=16, n_samples=self.point_num + self.point_num_2)
        
        self.validation_step_outputs = []
        
        
    def build_clip(self):
        assert self.use_clip
        self.clip_model = CLIP_MODEL(label_src=self.args.label_src)


    def configure_optimizers(self):
        feat_extractor_params = list(self.feat_extractor.parameters())
        
        feat_extractor_params_name = [name for name, param in self.feat_extractor.named_parameters()]
        
        if self.args.ft_rgb or self.args.train_rgb_only:
            radiance_params = [param for name, param in self.named_parameters() if 'radiance' in name]
            
            other_module_params = [param for name, param in self.named_parameters() if 'radiance' not in name]
            for param in other_module_params:
                param.requires_grad = False
            
            lr_geo = 0
            lr_feat = 0
            
            optimizer = optim.Adam([
                {'params': radiance_params, 'lr': self.args.lr},
                {'params': other_module_params, 'lr': lr_geo}])
        
        else:
            other_module_params = [param for name, param in self.named_parameters() if name.replace('feat_extractor.', '') not in feat_extractor_params_name]

            optimizer = optim.Adam([
                {'params': other_module_params, 'lr': self.args.lr},
                {'params': feat_extractor_params, 'lr': self.args.lr_feature if self.args.lr_feature is not None else self.args.lr}])
        
        # optimizer = optim.Adam(self.parameters(), lr=self.args.lr)
        
        if self.args.cosine_lr:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.max_iters, eta_min=self.args.min_lr)
        
        else:
            def func(step):
                lr_min = 1e-5
                return max(self.args.lr_decay_rate**(step//self.args.lr_decay_step), lr_min)
            
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
            
        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': scheduler, 
                'interval': 'step',
                'frequency': 1
            }
        }
    

    def order_posenc(self, d_hid, n_samples):
        """
        positional encoding of the sample ordering on a ray
        """

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table)
        
        return sinusoid_table


    def build_feature_volume(self, batch, source_imgs_feat):
        return self.feature_volume(source_imgs_feat, batch)


    def sample2rgb(self, batch, points_x, z_val, ray_d, ray_idx, source_imgs_feat, feature_volume, appear_feat=None, srdf_only=False, clip_features=None, ray_feats=None):
        B, L, _, _, _ = batch['source_imgs'].shape
        _, _, imgH, imgW = batch['ref_img'].shape
        _, _, SN, _ = points_x.shape

        radiance, srdf, points_in_pixel, ray_mask, sample_mask, clip_feat = self.ray_transformer(points_x, batch, source_imgs_feat, feature_volume, appear_feat, clip_features=clip_features, ray_feats=ray_feats)
            
        if srdf_only:
            return srdf
        
        RN = ray_idx.shape[1]

        ray_d = repeat(ray_d, "RN Dim3 -> RN SN Dim3", SN=SN)

        if self.args.predict_weight:
            rgb, depth, opacity, weight, variance, clip_feat_pix = self.renderer.render(rearrange(z_val, "B RN SN -> (B RN) SN"),  
                                            rearrange(radiance, "(B RN SN) C -> (B RN) SN C", B=B, RN=RN),
                                            srdf.squeeze(dim=2),
                                            clip_feat=rearrange(clip_feat, "(B RN SN) C -> (B RN) SN C", B=B, RN=RN) if clip_feat is not None else None,
                                            )
        else:
            rgb, depth, opacity, alpha, weight, variance, clip_feat_pix = self.renderer.render(rearrange(z_val, "B RN SN -> (B RN) SN"),  
                                            rearrange(radiance, "(B RN SN) C -> (B RN) SN C", B=B, RN=RN),
                                            srdf.squeeze(dim=2),
                                            sample_mask=sample_mask,
                                            deviation_network=self.deviation_network,
                                            clip_feat=rearrange(clip_feat, "(B RN SN) C -> (B RN) SN C", B=B, RN=RN) if clip_feat is not None else None,
                                            iters=self.global_step)

        rgb = rearrange(rgb, "(B RN) C -> B RN C", B=B).float()
        depth = rearrange(depth, "(B RN) -> B RN", B=B)
        opacity = rearrange(opacity, "(B RN) -> B RN", B=B)
        weight = rearrange(weight, "(B RN) SN -> B RN SN", B=B)
        
        if clip_feat_pix is not None:
            clip_feat_pix = rearrange(clip_feat_pix, "(B RN) C -> B RN C", B=B).float()
            
        return rgb, depth, srdf, opacity, weight, points_in_pixel, variance, ray_mask, clip_feat_pix


    def infer(self, batch, ray_idx, source_imgs_feat, appear_feat=None, feature_volume=None, extract_geometry=False, clip_features=None, ray_feats=None, jitter=False):
        B, L, _, _, _ = batch['source_imgs'].shape
        _, _, imgH, imgW = batch['ref_img'].shape
        
        RN = ray_idx.shape[1]
        
        if not extract_geometry:
            # gt rgb for rays
            ref_img = rearrange(batch['ref_img'], "B DimRGB H W -> B DimRGB (H W)")

            rgb_gt = torch.gather(ref_img, 2, repeat(ray_idx, "B RN -> B DimRGB RN", DimRGB=3))
            rgb_gt = rearrange(rgb_gt, "B C RN -> B RN C")

            # gt depth for rays
            if 'depths_h' in batch:
                ref_depth = rearrange(batch['depths_h'][:,0], "B H W -> B (H W)") # use only depth of reference view 
                depth_gt = torch.gather(ref_depth, 1, ray_idx)
            else:
                depth_gt = None
            
        ray_d = torch.gather(batch['ray_d'], 2, repeat(ray_idx, "B RN -> B DimX RN", DimX=3))
        ray_d = rearrange(ray_d, "B DimX RN -> (B RN) DimX")
        ray_o = repeat(batch['ray_o'], "B DimX -> B DimX RN", RN = RN) 
        ray_o = rearrange(ray_o, "B DimX RN -> (B RN) DimX")

        # ---------------------- coarse sampling along the ray ----------------------
        if 'near_fars' in batch.keys():
            if len(batch['near_fars'].shape) == 2:
                batch['near_fars'] = batch['near_fars'].unsqueeze(1)
                
            near_z = batch['near_fars'][:,0,0]
            near_z = repeat(near_z, "B -> B RN", RN=RN)
            near_z = rearrange(near_z, "B RN -> (B RN)")
            far_z = batch['near_fars'][:,0,1]
            far_z = repeat(far_z, "B -> B RN", RN=RN)
            far_z = rearrange(far_z, "B RN -> (B RN)")

            if extract_geometry:
                camera_ray_d = torch.gather(batch['cam_ray_d'], 2, repeat(ray_idx, "B RN -> B DimX RN", DimX=3))
                camera_ray_d = rearrange(camera_ray_d, "B DimX RN -> (B RN) DimX")
                near_z = near_z / camera_ray_d[:,2]
                far_z = far_z / camera_ray_d[:,2]
            points_x, z_val, points_d = self.fixed_sampler.sample_ray(ray_o, ray_d, near_z=near_z, far_z=far_z, jitter=jitter)

        else:
            points_x, z_val, points_d = self.fixed_sampler.sample_ray(ray_o, ray_d, jitter=jitter)
            
        points_x = rearrange(points_x, "(B RN) SN DimX -> B RN SN DimX", B = B) 
        points_d = rearrange(points_d, "(B RN) SN DimX -> B RN SN DimX", B = B)

        z_val = rearrange(z_val, "(B RN) SN -> B RN SN", B = B)
        batch['z_val'] = z_val

        rgb, depth, srdf, opacity, weight, points_in_pixel, variance, ray_mask, clip_feat_pix = self.sample2rgb(batch, points_x, z_val, ray_d, ray_idx, 
                    source_imgs_feat, feature_volume=feature_volume, appear_feat=appear_feat, clip_features=clip_features, ray_feats=ray_feats)
        
        if extract_geometry and self.args.test_coarse_only:
            srdf = rearrange(srdf, "(B RN) SN Dim1 ->B RN SN Dim1", B=B)
            srdf = srdf.squeeze(-1)
            return srdf, points_x, depth, rgb

        
        if not self.coarse_only:
            # ---------------------- fine sampling along the ray ----------------------
            points_x_2, z_val_2, points_d_2 = self.importance_sampler.sample_ray(ray_o, ray_d, 
                                                            rearrange(weight, "B RN SN -> (B RN) SN", B=B).detach(), 
                                                            rearrange(z_val, "B RN SN -> (B RN) SN").detach())
            
            # SN is sample point number along the ray
            points_x_2 = rearrange(points_x_2, "(B RN) SN DimX -> B RN SN DimX", B = B) 
            points_d_2 = rearrange(points_d_2, "(B RN) SN DimX -> B RN SN DimX", B = B)
            z_val_2 = rearrange(z_val_2, "(B RN) SN -> B RN SN", B = B)

            points_x_all = torch.cat([points_x, points_x_2], axis=2)
            z_val_all = torch.cat([z_val, z_val_2], axis=2)
            sample_sort_idx = torch.sort(z_val_all,axis=2)[1]
            z_val_all = torch.gather(z_val_all, 2, sample_sort_idx)
            points_x_all = torch.gather(points_x_all, 2, repeat(sample_sort_idx, "B RN SN -> B RN SN 3"))
            batch['z_val'] = z_val_all

            rgb_2, depth_2, srdf_2, opacity_2, weight_2, points_in_pixel_2, variance, ray_mask_2, clip_feat_pix = self.sample2rgb(batch, 
            points_x_all, z_val_all, ray_d, ray_idx, source_imgs_feat, feature_volume=feature_volume, appear_feat=appear_feat, clip_features=clip_features, ray_feats=ray_feats)

            if extract_geometry:
                srdf_2 = rearrange(srdf_2, "(B RN) SN Dim1 ->B RN SN Dim1", B=B)
                srdf_2 = srdf_2.squeeze(-1)
                return srdf_2, points_x_all, depth_2, rgb_2
        
        else:
            z_val_all = z_val
            rgb_2 = rgb
            depth_2 = depth
            srdf_2 = srdf
            opacity_2 = opacity
            weight_2 = weight
            points_in_pixel_2 = points_in_pixel
            ray_mask_2 = ray_mask

        return rgb_gt, rgb, depth, depth_gt, srdf, opacity, weight, points_in_pixel,\
            rgb_2, depth_2, srdf_2, opacity_2, weight_2, points_in_pixel_2,\
            z_val, z_val_all, variance, ray_mask, ray_mask_2, clip_feat_pix


    def training_step(self, batch, batch_idx):
        B, L, _, _, _ = batch['source_imgs'].shape
        _, _, imgH, imgW = batch['ref_img'].shape
        
        # ---------------------- step 0: infer image features ----------------------
        source_imgs = rearrange(batch['source_imgs'], "B L C H W -> (B L) C H W")
        
        source_imgs_feat, fpn = self.feat_extractor(source_imgs)
        for i in range(len(fpn)):
            fpn[i] = rearrange(fpn[i], "(B L) C H W -> B L C H W", L=L)
        fv_input = fpn
        source_imgs_feat = rearrange(source_imgs_feat, "(B L) C H W -> B L C H W", L=L)
        
        appear_feat = None
            
        if self.args.volume_reso > 0:
            feature_volume = self.build_feature_volume(batch, fv_input)
        else:
            feature_volume = None
            
        ray_feats = None
            
        if self.args.use_mask_coord: # only sample rays in regions with object
            assert 'obj_masks' in batch
            ray_idx = batch['ray_idx'].long()
 
        else:
            ray_idx = torch.argsort(torch.rand(B, imgH * imgW).type_as(batch['ray_o']), dim=-1)[:,:self.train_ray_num]

        rgb_gt, rgb, depth, depth_gt, srdf, opacity, weight, points_in_pixel, \
            rgb_2, depth_2, srdf_2, opacity_2, weight_2, points_in_pixel_2, \
            z_val, z_val_all, variance, ray_mask, ray_mask_2, _ = self.infer(batch=batch, 
                                                ray_idx=ray_idx, 
                                                source_imgs_feat=source_imgs_feat,
                                                appear_feat=appear_feat,
                                                feature_volume=feature_volume,
                                                ray_feats=ray_feats,
                                                jitter=True)


        def render_loss(rgb_pr, rgb_gt, ray_mask):  # [B N C], [B N C], [B N]
            ray_mask = ray_mask.float()
            loss = torch.sum((rgb_pr-rgb_gt)**2, -1)  # [B N]
            loss = torch.sum(loss*ray_mask, 1) / (torch.sum(ray_mask, 1)+1e-6)
            loss = torch.mean(loss)
            return loss
        
        if self.args.use_orig_rgb_loss:
            loss_rgb = torch.nn.functional.mse_loss(rgb, rgb_gt)
            loss_rgb2 = torch.nn.functional.mse_loss(rgb_2, rgb_gt)
        else:
            loss_rgb = render_loss(rgb, rgb_gt, ray_mask)
            loss_rgb2 = render_loss(rgb_2, rgb_gt, ray_mask_2)
            
        if depth_gt is not None:
            B, RN = depth_gt.size()
            # Depth loss
            mask_depth = (depth_gt!=0) & (depth_gt>=batch['near_fars'][:,0,0:1]) & (depth_gt<=batch['near_fars'][:,0,1:2])

            if torch.sum(mask_depth)>0:
                # masked out where gt depth is invalid
                if self.args.neuray_depth_loss:
                    depth_loss_func = DepthLoss()
                    depth_range = repeat(batch['near_fars'][:,0,:], "B Dim -> B RN Dim", RN=RN)

                    loss_depth_ray = depth_loss_func(depth[mask_depth], depth_gt[mask_depth], depth_range=depth_range[mask_depth])
                    loss_depth_ray2 = depth_loss_func(depth_2[mask_depth], depth_gt[mask_depth], depth_range=depth_range[mask_depth])
                else:
                    loss_depth_ray =  torch.nn.functional.l1_loss(depth[mask_depth], depth_gt[mask_depth]) 
                    loss_depth_ray2 = torch.nn.functional.l1_loss(depth_2[mask_depth], depth_gt[mask_depth])
            else:
                loss_depth_ray = loss_depth_ray2 = 0.0
        else:
            loss_depth_ray = loss_depth_ray2 = 0.0

        if self.args.fine_loss_only:
            loss = self.args.weight_rgb * loss_rgb2 + \
                        self.args.weight_depth * loss_depth_ray2
        else:
            loss = self.args.weight_rgb * (loss_rgb + loss_rgb2) + \
                    self.args.weight_depth * (loss_depth_ray + loss_depth_ray2)
                                        
        self.log("train/depth_ray_coarse", loss_depth_ray)
        self.log("train/depth_ray_fine", loss_depth_ray2)
        self.log("train/rgb_coarse", loss_rgb)
        self.log("train/rgb_fine", loss_rgb2)
        self.log("train/loss_all", loss)
        self.log("train/variance", variance)
        
        return loss


    def on_validation_epoch_end(self):  #validation_epoch_end(self, batch_parts):  
        # average epoches
        
        batch_parts = self.validation_step_outputs

        psnr_coarse = [i['psnr/coarse'] for i in batch_parts]
        psnr_fine = [i['psnr/fine'] for i in batch_parts]
        loss_rgb_coarse = [i['val/loss_rgb_coarse'] for i in batch_parts]
        loss_rgb_fine = [i['val/loss_rgb_fine'] for i in batch_parts]
        loss_depth_coarse = [i['val/loss_depth_coarse'] for i in batch_parts]
        loss_depth_fine = [i['val/loss_depth_fine'] for i in batch_parts]

        psnr_coarse = sum(psnr_coarse) / len(psnr_coarse)
        psnr_fine = sum(psnr_fine) / len(psnr_fine)
        loss_rgb_coarse = sum(loss_rgb_coarse) / len(loss_rgb_coarse)
        loss_rgb_fine = sum(loss_rgb_fine) / len(loss_rgb_fine)
        loss_depth_coarse = sum(loss_depth_coarse) / len(loss_depth_coarse)
        loss_depth_fine = sum(loss_depth_fine) / len(loss_depth_fine)
        
        # logging
        self.log("psnr/coarse", psnr_coarse, sync_dist=True)
        self.log("psnr/fine", psnr_fine, sync_dist=True)
        self.log("val/rgb_coarse", loss_rgb_coarse, sync_dist=True)
        self.log("val/rgb_fine", loss_rgb_fine, sync_dist=True)
        self.log("val/loss_depth_coarse", loss_depth_coarse, sync_dist=True)
        self.log("val/loss_depth_fine", loss_depth_fine, sync_dist=True)

        loss = loss_rgb_coarse + loss_rgb_fine
        
        self.validation_step_outputs.clear()
        
        print("[val] psnr/coarse:", psnr_coarse, "[val] psnr/fine:", psnr_fine)

        return loss


    def validation_step(self, batch, batch_idx):
        if self.args.extract_geometry:
            self.extract_geometry(batch, batch_idx)
            
            # return dummy data
            loss = {"val/loss_rgb_coarse":0,
                    "val/loss_rgb_fine":0,
                    "val/loss_depth_coarse":0,
                    "val/loss_depth_fine":0,
                    "psnr/coarse":0,
                    "psnr/fine":0}
            
            self.validation_step_outputs.append(loss)

            return loss

        B, L, _, _, _ = batch['source_imgs'].shape
        _, _, imgH, imgW = batch['ref_img'].shape

        source_imgs = rearrange(batch['source_imgs'], "B L C H W -> (B L) C H W")

        source_imgs_feat, fpn = self.feat_extractor(source_imgs)
        for i in range(len(fpn)):
            fpn[i] = rearrange(fpn[i], "(B L) C H W -> B L C H W", L=L)
        fv_input = fpn
        source_imgs_feat = rearrange(source_imgs_feat, "(B L) C H W -> B L C H W", L=L)

        appear_feat = None

        ray_feats = None

        if self.use_clip:
            with torch.no_grad():
                clip_features = []
                for i in range(source_imgs.shape[0]):
                    clip_feat = self.clip_model.get_feature(source_imgs[i:i+1])[0]
                    clip_features.append(clip_feat)
                clip_features = torch.concat(clip_features, dim=0)
                clip_features = rearrange(clip_features, "(B L) C H W -> B L C H W", L=L)
        else:
            clip_features = None
            
        ray_idx_all = repeat(torch.arange(imgH * imgW), "HW -> B HW", B = B).type_as(batch['ray_o']).long() 

        rgb_list, rgb_gt_list, depth_list, rgb_list_2, depth_list_2 = [], [], [], [], []
        clip_feat_list = []

        if self.args.volume_reso > 0:
            feature_volume = self.build_feature_volume(batch, fv_input)
        else:
            feature_volume = None

        for ray_idx in tqdm(torch.split(ray_idx_all, self.train_ray_num, dim=1)):
            rgb_gt, rgb, depth, depth_gt, srdf, opacity, _, _, \
                rgb_2, depth_2, _, _, _, _, _, _, variance, ray_mask, ray_mask_2, clip_feat = \
                        self.infer(batch=batch, ray_idx=ray_idx, source_imgs_feat=source_imgs_feat, appear_feat=appear_feat, feature_volume=feature_volume, clip_features=clip_features, ray_feats=ray_feats, jitter=False)

            rgb_list.append(rgb)
            rgb_gt_list.append(rgb_gt)
            depth_list.append(depth)
            rgb_list_2.append(rgb_2)
            depth_list_2.append(depth_2)
            
            if clip_feat is not None:
                clip_feat_list.append(clip_feat)
                

        rgb_list = torch.cat(rgb_list, dim=1)
        rgb_gt_list = torch.cat(rgb_gt_list, axis=1)
        depth_list = torch.cat(depth_list, axis=1)
        rgb_list_2 = torch.cat(rgb_list_2, dim=1)
        depth_list_2 = torch.cat(depth_list_2, axis=1)
        
        # move to cpu
        to_CPU = lambda x: x.cpu().numpy()
        
        if type(variance) is not float:
            variance = to_CPU(variance)

        rgb_imgs = rearrange(rgb_list, "B (H W) DimRGB -> B DimRGB H W", H=imgH)
        rgb_gt_imgs = rearrange(rgb_gt_list, "B (H W) DimRGB -> B DimRGB H W", H=imgH)
        depths = rearrange(depth_list, "B (H W) -> B H W", H=imgH)
        rgb_imgs_2 = rearrange(rgb_list_2, "B (H W) DimRGB -> B DimRGB H W", H=imgH)
        depths_2 = rearrange(depth_list_2, "B (H W) -> B H W", H=imgH)

        if self.use_clip:
            assert len(clip_feat_list) > 0
            clip_feat_list = torch.cat(clip_feat_list, axis=1)
            clip_feat = rearrange(clip_feat_list, "B (H W) C -> B C H W", H=imgH)
            predicts = []
            with torch.no_grad():
                for i in range(clip_feat.shape[0]):
                    predict = self.clip_model.forward_feature(clip_feat[i:i+1])[0]
                    predicts.append(predict)
            self.clip_model.visualize(rgb_gt_imgs, predicts)

        # metrics
        loss_rgb = torch.nn.functional.mse_loss(rgb_list, rgb_gt_list)
        loss_rgb_2 = torch.nn.functional.mse_loss(rgb_list_2, rgb_gt_list)

        psnr_coarse = piq.psnr(torch.clamp(rgb_imgs, max=1, min=0), torch.clamp(rgb_gt_imgs, max=1, min=0)).item()
        psnr_fine = piq.psnr(torch.clamp(rgb_imgs_2, max=1, min=0), torch.clamp(rgb_gt_imgs, max=1, min=0)).item()
        
        if self.args.val_only:
            if not os.path.exists(self.args.out_dir):
                os.mkdir(self.args.out_dir)
            rgb_vis = rearrange(rgb_imgs, " B DimRGB H W ->  B H W DimRGB")
            rgb_gt_vis = rearrange(rgb_gt_imgs, " B DimRGB H W ->  B H W DimRGB")
            for i in range(rgb_vis.shape[0]):
                imageio.imwrite("/".join([self.args.out_dir, "img%d_pred.jpg" % (batch_idx*rgb_vis.shape[0] + i)]), (rgb_vis[i] * 255).detach().cpu().numpy().astype(np.uint8))
                imageio.imwrite("/".join([self.args.out_dir, "img%d_gt.jpg" % (batch_idx*rgb_vis.shape[0] + i)]), (rgb_gt_vis[i] * 255).detach().cpu().numpy().astype(np.uint8))

        # return depth loss and log it
        if 'depths_h' in batch:
            depth_gt = batch['depths_h'][:,0]
        else:
            depth_gt = None
        
        # Depth loss
        if depth_gt is not None:
            B,H,W = depth_gt.size()
            mask_depth = (depth_gt!=0) & (depth_gt>=batch['near_fars'][:,0,0:1]) & (depth_gt<=batch['near_fars'][:,0,1:2])

            if torch.sum(mask_depth)>0:
                if self.args.neuray_depth_loss:
                    depth_loss_func = DepthLoss()
                    depth_range = repeat(batch['near_fars'][:,0,:], "B Dim -> B H W Dim", H=H, W=W)

                    loss_depth_ray = depth_loss_func(depths[mask_depth], depth_gt[mask_depth], depth_range=depth_range[mask_depth])
                    loss_depth_ray2 = depth_loss_func(depths_2[mask_depth], depth_gt[mask_depth], depth_range=depth_range[mask_depth])
                else:
                    loss_depth_ray =  torch.nn.functional.l1_loss(depths[mask_depth], depth_gt[mask_depth]) 
                    loss_depth_ray2 = torch.nn.functional.l1_loss(depths_2[mask_depth], depth_gt[mask_depth])
            else:
                loss_depth_ray = loss_depth_ray2 = 0.0
        else:
            loss_depth_ray = loss_depth_ray2 = 0.0
            
        
        loss = {"val/loss_rgb_coarse":loss_rgb.item(), 
                "val/loss_rgb_fine":loss_rgb_2.item(), 
                "val/loss_depth_coarse":loss_depth_ray if type(loss_depth_ray) is float else loss_depth_ray.item(), 
                "val/loss_depth_fine":loss_depth_ray2 if type(loss_depth_ray) is float else loss_depth_ray2.item(), 
                "psnr/coarse":psnr_coarse, 
                "psnr/fine":psnr_fine,
                "val/variance": variance}
        
        self.validation_step_outputs.append(loss)

        return 


    def extract_geometry(self, batch, batch_idx):
        
        B, L, _, _, _ = batch['source_imgs'].shape
        _, _, imgH, imgW = batch['ref_img'].shape
        
        ## only exist for dtu
        if "meta" in batch:
            scan_name = batch['meta'][0].split("-")[1]
            ref_view = batch['meta'][0].split("-")[-1]
        else:
            scan_name = self.args.test_scene
            ref_view = batch['ref_view'][0]
        
        os.makedirs(os.path.join(self.args.out_dir, scan_name, "depth"), exist_ok=True)
        os.makedirs(os.path.join(self.args.out_dir, "depth", scan_name), exist_ok=True)
        os.makedirs(os.path.join(self.args.out_dir, "rgb", scan_name), exist_ok=True)

        source_imgs = rearrange(batch['source_imgs'], "B L C H W -> (B L) C H W")

        source_imgs_feat, fpn = self.feat_extractor(source_imgs)
        for i in range(len(fpn)):
            fpn[i] = rearrange(fpn[i], "(B L) C H W -> B L C H W", L=L)
        fv_input = fpn
        source_imgs_feat = rearrange(source_imgs_feat, "(B L) C H W -> B L C H W", L=L)

        ray_feats = None
            
        ray_idx_all = repeat(torch.arange(imgH * imgW), "HW -> B HW", B = B).type_as(batch['ray_o']).long()
        depth_list, rgb_list  = [], []

        if self.args.volume_reso > 0:
            feature_volume = self.build_feature_volume(batch, fv_input)
        else:
            feature_volume = None

        for ray_idx in tqdm(torch.split(ray_idx_all, self.args.test_ray_num, dim=1)):
            srdf, points_x, depth, rgb = self.infer(batch=batch, ray_idx=ray_idx, source_imgs_feat=source_imgs_feat, 
                                feature_volume=feature_volume, ray_feats=ray_feats, extract_geometry=True, jitter=False)

            ray_d = torch.gather(batch['cam_ray_d'], 2, repeat(ray_idx, "B RN -> B DimX RN", DimX=3))
            ray_d = rearrange(ray_d, "B DimX RN -> B RN DimX")

            depth = (depth.unsqueeze(-1) * ray_d)[:,:,2]
            depth_list.append(depth)
            rgb_list.append(rgb)

        depths = torch.cat(depth_list, dim=1).view(imgH, imgW) # H W
        depths = depths * batch['scale_mat'][0][0, 0]  # scale back
        rgbs = torch.cat(rgb_list, dim=1).view(imgH, imgW,-1)


        depths = depths.cpu().numpy()
        rgbs = rgbs.cpu().numpy()
        rgbs = (rgbs.astype(np.float32) * 255).astype(np.uint8)
        depth_save = ((depths / np.max(depths)).astype(np.float32) * 255).astype(np.uint8)
        Image.fromarray(depth_save).save(os.path.join(self.args.out_dir, scan_name, "depth", "%s.png"%ref_view))
        Image.fromarray(rgbs).save(os.path.join(self.args.out_dir, "rgb", scan_name, "%s.jpg"%ref_view))

        extrinsic_np = batch['extrinsic_render_view'][0].cpu().numpy()

        np.save(os.path.join(self.args.out_dir, "depth", scan_name, "%s.npy"%ref_view), 
                {"depth": depths, "rgb": rgbs, "extrinsic":extrinsic_np, "intrinsic": batch['intrinsic_render_view'][0].cpu().numpy()})


    def scene_edit(self, prompt, test_loader):
        from .utils.ip2p import InstructPix2Pix
        device = next(self.feat_extractor.parameters()).device
        self.ip2p = InstructPix2Pix(device, ip2p_use_full_precision=True)

        text_embedding = self.ip2p.pipe._encode_prompt(
            prompt, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
        ).to(device)

        text_guidance_scale = self.args.text_guidance_scale  # 7.5
        image_guidance_scale = 3
        diffusion_steps = 20
        lower_bound = upper_cound =self.args.noise_level # 0.3
        
        ### initialize source_img_pool via editted images
        source_img_pool = {}
        for _, batch in enumerate(test_loader):
            for key in batch:
                if type(batch[key]) == torch.Tensor:
                    batch[key] = batch[key].to(device)
                    
            gt_rgb = batch['ref_img']  # B C H W
            original_image = gt_rgb * batch["mask"]
            rendered_image = original_image
                        
            edited_image = self.ip2p.edit_image(
                        text_embedding.to(rendered_image),
                        rendered_image,
                        original_image,
                        guidance_scale=text_guidance_scale,
                        image_guidance_scale=image_guidance_scale,
                        diffusion_steps=diffusion_steps,
                        lower_bound=0.2,
                        upper_bound=0.98,
                    )

            render_idx = batch['idx'][0][0].item()
            source_img_pool[render_idx] = edited_image[0] * batch["mask"][0]

        ### iteratively update the source_img_pool
        for edit_iter in range(self.args.edit_iters):            
            source_img_pool_tmp = source_img_pool.copy()
            render_img_list = []
            
            for batch_id, batch in enumerate(test_loader):
                for key in batch:
                    if type(batch[key]) == torch.Tensor:
                        batch[key] = batch[key].to(device)
        
                B, _, imgH, imgW = batch['ref_img'].shape
                gt_rgb = batch['ref_img']  # B C H W
                
                assert B == 1  # only support batch size 1 for now
                
                render_idx, src_idx = batch['idx'][0][0].item(), batch['idx'][0][1:].cpu().numpy().tolist()
                L = len(src_idx)
                  
                source_imgs = []
                for i, idx in enumerate(src_idx):
                    source_imgs.append(source_img_pool_tmp[idx])
                source_imgs = torch.stack(source_imgs, dim=0)  # (B L) C H W
                source_imgs = source_imgs * batch['source_masks'][0]
                batch['source_imgs'] = rearrange(source_imgs, "(B L) C H W -> B L C H W", B=1, L=L)
                
                with torch.no_grad():
                    source_imgs_feat, fpn = self.feat_extractor(source_imgs)
                    for i in range(len(fpn)):
                        fpn[i] = rearrange(fpn[i], "(B L) C H W -> B L C H W", L=L)
                    fv_input = fpn
                    source_imgs_feat = rearrange(source_imgs_feat, "(B L) C H W -> B L C H W", L=L)

                    appear_feat = None

                    ray_feats = None

                    clip_features = None
                        
                    ray_idx_all = repeat(torch.arange(imgH * imgW), "HW -> B HW", B = B).type_as(batch['ray_o']).long() 
                    rgb_list, rgb_gt_list, depth_list, rgb_list_2, depth_list_2 = [], [], [], [], []

                    if self.args.volume_reso > 0:
                        feature_volume = self.build_feature_volume(batch, fv_input)
                    else:
                        feature_volume = None

                    for ray_idx in tqdm(torch.split(ray_idx_all, self.train_ray_num, dim=1)):
                        rgb_gt, rgb, depth, depth_gt, srdf, opacity, _, _, \
                            rgb_2, depth_2, _, _, _, _, _, _, variance, ray_mask, ray_mask_2, clip_feat = \
                                    self.infer(batch=batch, ray_idx=ray_idx, source_imgs_feat=source_imgs_feat, appear_feat=appear_feat, feature_volume=feature_volume, clip_features=clip_features, ray_feats=ray_feats, jitter=False)

                        rgb_list.append(rgb)
                        rgb_gt_list.append(rgb_gt)
                        depth_list.append(depth)
                        rgb_list_2.append(rgb_2)
                        depth_list_2.append(depth_2)
                        
                    rgb_list = torch.cat(rgb_list, dim=1)
                    rgb_gt_list = torch.cat(rgb_gt_list, axis=1)
                    depth_list = torch.cat(depth_list, axis=1)
                    rgb_list_2 = torch.cat(rgb_list_2, dim=1)
                    depth_list_2 = torch.cat(depth_list_2, axis=1)
                    
                    # move to cpu
                    to_CPU = lambda x: x.cpu().numpy()
                    
                    if type(variance) is not float:
                        variance = to_CPU(variance)

                    rgb_imgs = rearrange(rgb_list, "B (H W) DimRGB -> B DimRGB H W", H=imgH)
                    rgb_gt_imgs = rearrange(rgb_gt_list, "B (H W) DimRGB -> B DimRGB H W", H=imgH)
                    depths = rearrange(depth_list, "B (H W) -> B H W", H=imgH)
                    rgb_imgs_2 = rearrange(rgb_list_2, "B (H W) DimRGB -> B DimRGB H W", H=imgH)
                    depths_2 = rearrange(depth_list_2, "B (H W) -> B H W", H=imgH)
                    
                    rendered_image = rgb_imgs_2 * batch["mask"]
                    original_image = gt_rgb * batch["mask"]
                    
                    edited_image = self.ip2p.edit_image(
                                text_embedding.to(rendered_image),
                                rendered_image,
                                original_image,
                                guidance_scale=text_guidance_scale,  # 7.5
                                image_guidance_scale=image_guidance_scale, # 1.5
                                diffusion_steps=diffusion_steps,
                                lower_bound=lower_bound, # 0.2
                                upper_bound=upper_cound # 0.98,
                            )
                    
                    source_img_pool[render_idx] = edited_image[0] * batch["mask"][0]
                    render_img_list.append(rendered_image[0])

            if not os.path.exists(self.args.out_dir):
                os.mkdir(self.args.out_dir)
            
            for idx, img in source_img_pool.items():
                img = rearrange(img, " DimRGB H W ->  H W DimRGB")
                imageio.imwrite("/".join([self.args.out_dir, "round%d_edit_img%d.jpg" % (edit_iter, idx)]), (img * 255).detach().cpu().numpy().astype(np.uint8))

            for idx, img in enumerate(render_img_list):
                img = rearrange(img, " DimRGB H W ->  H W DimRGB")
                imageio.imwrite("/".join([self.args.out_dir, "round%d_render_img%d.jpg" % (edit_iter, idx)]), (img * 255).detach().cpu().numpy().astype(np.uint8))

        return 