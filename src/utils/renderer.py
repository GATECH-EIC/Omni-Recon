import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VolumeRenderer():
    def __init__(self, args=None, density_fun=None):
        self.args = args
        self.density_fun = density_fun
            

    def render(self, z_val, radiance, geo_value, sample_mask, cos_anneal_ratio=1.0, deviation_network=None, clip_feat=None, iters=None): 
        """
        Volume rendering pixels given srdf and radiance of samples
        Adapted from: https://github.com/xxlong0/SparseNeuS 

        z_val: z value of each sample, [RN, SN]
        radiance: radiance of each sample, [RN, SN, 3]
        geo_value: geo value of each sample, [RN, SN] (srdf)
        sample_mask: sample mask along a ray indicating whether this sample has >1 valid observation, [RN, SN, 1]
        cos_anneal_ratio: cosine annealing ratio
        deviation_network: network to predict deviation
        """
                
        if self.args.vanilla_volume_rendering:
            density = geo_value
            if self.args.act_func == 'softplus':
                alpha = 1.0 - torch.exp(-F.softplus(density))
            else:
                alpha = 1.0 - torch.exp(-torch.relu(density))
            
            batch_size, n_samples = z_val.shape
            alpha = alpha.reshape(batch_size, n_samples).clip(0.0, 1.0)

            inv_s0 = 1
        
        elif self.args.use_volsdf:
            sdf = geo_value
            if self.args.anneal_beta:
                beta_init = self.args.beta_init  # 0.1
                beta_min = self.args.beta_min  # 0.001
                beta = beta_init / (1 + (beta_init - beta_min) / beta_min * np.power(iters/self.args.max_iters, 0.8))
                density = self.density_fun(sdf, beta)
            else:
                density = self.density_fun(sdf)

            dists = z_val[:, 1:] - z_val[:, :-1]
            dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)
            density = density * dists
                    
            if self.args.act_func == 'softplus':
                alpha = 1.0 - torch.exp(-F.softplus(density))
            else:
                alpha = 1.0 - torch.exp(-torch.relu(density))
            
            batch_size, n_samples = z_val.shape
            alpha = alpha.reshape(batch_size, n_samples).clip(0.0, 1.0)

            inv_s0 = 1

        else:
            interval = z_val[:,1:]-z_val[:,:-1]
            interval = torch.cat([interval[:,0:1], interval, interval[:,-1:]], axis=1)
            interval = (interval[:,:-1] + interval[:,1:]) / 2

            batch_size, n_samples = z_val.shape 
            srdf = geo_value
            
            if self.args.disable_deviation:
                inv_s0 = inv_s = 1
            else:
                inv_s0 = deviation_network(torch.zeros([1, 3]).type_as(z_val))[:, :1].clip(1e-6, 1e6)
                inv_s = inv_s0.expand(batch_size, n_samples)

            true_cos = -1.0
            if self.args.orig_wrong_renderer:
                iter_cos = -(-true_cos * 0.5 + 0.5 * (1.0 - cos_anneal_ratio) -true_cos * cos_anneal_ratio)   # original volrecon
            else:
                iter_cos = -((-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) - true_cos * cos_anneal_ratio)   # revised volrecon

            estimated_next_srdf = srdf + iter_cos * interval * 0.5
            estimated_prev_srdf = srdf - iter_cos * interval * 0.5

            prev_cdf = torch.sigmoid(estimated_prev_srdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_srdf * inv_s)
            
            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        if self.args.use_sample_mask:
            alpha = alpha.masked_fill(~sample_mask.reshape(batch_size, n_samples), 0.)
                        
        weight = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]).type_as(z_val), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        rgb = (radiance * weight[:,:,None]).sum(axis=1)
        depth = (weight * z_val).sum(axis=1)
        opacity = weight.sum(axis=1)
        
        if clip_feat is not None:
            clip_feat_pix = (clip_feat * weight[:,:,None]).sum(axis=1)
        else:
            clip_feat_pix = None
        
        return rgb, depth, opacity, alpha, weight, 1.0 / inv_s0, clip_feat_pix



class LaplaceDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, params_init={}, beta_min=0.0001):
        super().__init__()
        
        self.learnable_beta = len(params_init) > 0

        for p in params_init:
            param = nn.Parameter(torch.tensor(params_init[p]))
            setattr(self, p, param)

        self.beta_min = torch.tensor(beta_min).cuda()

    def density_func(self, sdf, beta=None):
        if not self.learnable_beta:
            assert beta is not None
            
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)
    


class SimpleRenderer():
    def __init__(self, args=None):
        self.args = args
        self.type_ = 'log2'
        
    def entropy(self, prob):
        if self.type_ == 'log2':
            return -1*prob*torch.log2(prob+1e-10)
        elif self.type_ == '1-p':
            return prob*torch.log2(1-prob)

    def render(self, z_val, radiance, geo_value, clip_feat): 
        """
        Volume rendering pixels given srdf and radiance of samples
        Adapted from: https://github.com/xxlong0/SparseNeuS 

        z_val: z value of each sample, [RN, SN]
        radiance: radiance of each sample, [RN, SN, 3]
        geo_value: geo value of each sample, [RN, SN] (srdf)
        cos_anneal_ratio: cosine annealing ratio
        deviation_network: network to predict deviation
        """

        rgb = (radiance * geo_value[:,:,None]).sum(axis=1) 
        depth = (geo_value * z_val).sum(axis=1)
        entropy_ray = self.entropy(geo_value)
        entropy = torch.sum(entropy_ray, -1)
        opacity = entropy

        if clip_feat is not None:
            clip_feat_pix = (clip_feat * geo_value[:,:,None]).sum(axis=1)
        else:
            clip_feat_pix = None
            
        return rgb, depth, opacity, geo_value, torch.tensor([1.]).to(rgb.device), clip_feat_pix
    