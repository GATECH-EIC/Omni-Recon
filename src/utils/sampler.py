
import numpy as np
import torch
from einops import (rearrange, reduce, repeat)


class FixedSampler():
    """
    Fix-interval sampler given near and far z
    """
    def __init__(self, point_num=64, sample_radius=1.3, inv_uniform=False):
        self.sample_radius = sample_radius
        self.point_num = point_num
        self.inv_uniform = inv_uniform

    def sample_ray(self, ray_o, ray_d, jitter=False, near_z=None, far_z=None):
        """
        ray_o: ray origins, [RN, 3]
        ray_d: ray directions, [RN, 3]
        jitter: jitter the sampling points, boolean
        near_z: near z value, [RN]
        far_z: far z value, [RN]
        """

        mid_z_val = - reduce(ray_o * ray_d, "RN Dim_X -> RN", 'sum')
        mid_z_val = rearrange(mid_z_val, "RN -> 1 RN")

        if near_z is None:
            # no near and far provided
            near = mid_z_val - self.sample_radius
            far = mid_z_val + self.sample_radius
        else:
            near = near_z
            far = far_z
        
        if self.inv_uniform:
            start = 1. / near     # [RN]
            step = (1. / far - start) / (self.point_num-1)  # [RN]
            inv_z_val = torch.stack([start+i*step for i in range(self.point_num)], dim=1)  # [RN, SN]
            z_val = 1. / inv_z_val

            if jitter:
                # get intervals between samples
                mids = .5 * (z_val[:, 1:] + z_val[:, :-1])
                upper = torch.cat([mids, z_val[:, -1:]], dim=-1)
                lower = torch.cat([z_val[:, 0:1], mids], dim=-1)
                # uniform samples in those intervals
                t_rand = torch.rand_like(z_val)
                z_val = lower + (upper - lower) * t_rand   # [N_rays, N_samples]
        
            # unit_linspace = torch.from_numpy(np.linspace(0,1,self.point_num).astype("float32")).type_as(ray_o)
            # z_val2 = rearrange(unit_linspace, "SN -> SN 1") * (far-near) + near
            # print('near:', near_z[0], 'far:', far_z[0])
            # print('z_val_inv:', z_val)
            # print('z_val:', z_val2)
            # input()
    
        else:
            unit_linspace = torch.from_numpy(np.linspace(0,1,self.point_num).astype("float32")).type_as(ray_o)
            z_val = rearrange(unit_linspace, "SN -> SN 1") * (far-near) + near

            interval = 1 / (self.point_num - 1)
            
            if jitter:
                z_val_jitter = z_val + ((torch.rand(z_val.shape).type_as(ray_o)) - 0.5) * interval *  (far-near)
                z_val = z_val_jitter

            z_val = rearrange(z_val, "SN RN -> RN SN")
        
        # print('near:', near_z[0], 'far:', far_z[0], 'z_val:', z_val[0])

        points_x = rearrange(ray_o, "RN DimX -> RN 1 DimX") + rearrange(z_val, "RN SN -> RN SN 1") * rearrange(ray_d, "RN DimX -> RN 1 DimX")
        points_d = repeat(ray_d, "RN DimX -> RN SN DimX", SN=self.point_num)

        return points_x, z_val.clone(), points_d


class ImportanceSampler():
    """
    NeRF-like sampler that sample denser around the surface
    """
    def __init__(self, point_num=128):
        self.point_num = point_num


    def sort_one_dim(self, z_val, points_x):
        """
        sort by z_vals
        z_val: z value of each sample, [RN, SN]
        points_x: position of each sample, [RN, SN, 3]
        """
        sample_sort_idx = torch.sort(z_val,axis=1)[1]
        z_val_sorted = torch.gather(z_val, 1, sample_sort_idx)
        points_x_sorted = torch.gather(points_x, 1, repeat(sample_sort_idx, "RN SN -> RN SN 3"))

        return z_val_sorted, points_x_sorted


    def sample_ray(self, ray_o, ray_d, weight, z_val):    
        """
        ray_o: ray origins, [RN, 3]
        ray_d: ray directions, [RN, 3]
        weight: weight of each sample, [RN, SN]
        z_val: z value of each sample, [RN, SN]
        """

        RN, SN = z_val.shape

        cdf = torch.cumsum(weight, axis=1) / (weight.sum(axis=1)[:,None] + 1e-6)

        sample_cdf = (torch.rand(self.point_num, RN).transpose(0,1)).type_as(ray_o)

        sample_cdf = torch.clamp(sample_cdf, min=(cdf[:,0]).view(-1,1), max=(cdf[:,-1]).view(-1,1))

        right_index = torch.searchsorted(cdf, sample_cdf.contiguous())

        right_index[right_index == 0] = 1
        right_index[right_index > SN-1] = SN-1
        
        left_cdf = torch.gather(cdf, 1, right_index-1)
        right_cdf = torch.gather(cdf, 1, right_index)

        z_val_left = torch.gather(z_val, 1, right_index-1) 
        z_val_right = torch.gather(z_val, 1, right_index)

        z_val_sample = (sample_cdf - left_cdf) / (right_cdf - left_cdf + 1e-6) * (z_val_right - z_val_left) + z_val_left
        
        points_x = rearrange(ray_o, "RN DimX -> RN 1 DimX") + rearrange(z_val_sample, "RN SN -> RN SN 1") * rearrange(ray_d, "RN DimX -> RN 1 DimX")
        points_d = repeat(ray_d, "RN DimX -> RN SN DimX", SN=self.point_num)

        z_val_sample_sorted, points_x_sorted = self.sort_one_dim(z_val_sample, points_x)

        return points_x_sorted, z_val_sample_sorted, points_d
