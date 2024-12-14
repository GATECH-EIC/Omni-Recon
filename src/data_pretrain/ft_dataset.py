import torch
import cv2 as cv
from torch.utils.data import Dataset
from .asset import *
from .database import parse_database_name, get_database_split
import numpy as np
from einops import repeat

from .utils.base_utils import get_coords_mask
from .utils.dataset_utils import set_seed
from .utils.imgs_info import build_imgs_info, random_crop, random_flip, pad_imgs_info, imgs_info_slice, \
    imgs_info_to_torch
from .utils.view_select import compute_nearest_camera_indices


class FtRendererDataset(Dataset):
    default_cfg={
        'database_name': 'nerf_synthetic/lego/black_800', # llff_colmap/flower/high
        'train_database_split_type': 'train',
        
        'val_database_split_type': 'val',

        'min_wn': 8,
        # 'max_wn': 9,
        'ref_pad_interval': 32, # 16,
        'train_ray_num': 512,
        'foreground_ratio': 0.7,
        'resolution_type': 'hr',
        "use_consistent_depth_range": True,
        'use_depth_loss_for_all': False,
        "use_depth": True,
        "use_src_imgs": True, # False,
        "cost_volume_nn_num": 3,

        "use_aug": False,
        
        "aug_gso_shrink_range_prob": 0.5,
        "aug_depth_range_prob": 0.05,
        'aug_depth_range_min': 0.95,
        'aug_depth_range_max': 1.05,
        "aug_use_depth_offset": True,
        "aug_depth_offset_prob": 0.25,
        "aug_depth_offset_region_min": 0.05,
        "aug_depth_offset_region_max": 0.1,
        'aug_depth_offset_min': 0.5,
        'aug_depth_offset_max': 1.0,
        'aug_depth_offset_local': 0.1,
        "aug_use_depth_small_offset": True,
        "aug_use_global_noise": True,
        "aug_global_noise_prob": 0.5,
        "aug_depth_small_offset_prob": 0.5,
        "aug_forward_crop_size": (400,600),
        "aug_pixel_center_sample": True, # False,
        "aug_view_select_type": "easy",

        "use_consistent_min_max": False,
        "revise_depth_range": False,
    }
    def __init__(self, cfg, is_train, train_ray_num, num_src_view=4):
        if cfg is not None:
            self.cfg = {**self.default_cfg,**cfg}
        else:
            self.cfg = self.default_cfg
            
        self.cfg['min_wn'] = num_src_view
        
        self.train_ray_num = train_ray_num
            
        self.is_train = is_train
        if is_train:
            self.database = parse_database_name(self.cfg['database_name'])
            train_ids, val_ids = get_database_split(self.database, 'train')
            self.ref_ids = train_ids
            self.que_ids = train_ids
            self.num=999999
        else:
            self.database = parse_database_name(self.cfg['database_name'])
            train_ids, val_ids = get_database_split(self.database,self.cfg['val_database_split_type'])
            self.ref_ids, self.que_ids = train_ids, val_ids
            self.num=len(self.que_ids)

    def get_database_ref_que_ids(self, index):
        if self.is_train:
            database = self.database
            que_id = np.random.choice(self.ref_ids)
            ref_ids = self.ref_ids
        else:
            database = self.database
            que_id, ref_ids = self.que_ids[index], self.ref_ids
        
        return database, que_id, np.asarray(ref_ids)


    def select_working_views(self, database, que_id, ref_ids):
        database_name = database.database_name
        dist_idx = compute_nearest_camera_indices(database, [que_id], ref_ids)[0]

        if self.is_train:
            if np.random.random()>0.02: # 2% chance to include que image
                dist_idx = dist_idx[ref_ids[dist_idx]!=que_id]
                
        dist_idx = dist_idx[:self.cfg['min_wn']]
        ref_ids = ref_ids[dist_idx]
        
        return ref_ids


    def consistent_depth_range(self, ref_imgs_info, que_imgs_info):
        depth_range_all = np.concatenate([ref_imgs_info['depth_range'], que_imgs_info['depth_range']], 0)
        if self.cfg['use_consistent_min_max']:
            depth_range_all[:, 0] = np.min(depth_range_all)
            depth_range_all[:, 1] = np.max(depth_range_all)
        else:
            range_len = depth_range_all[:, 1] - depth_range_all[:, 0]
            max_len = np.max(range_len)
            range_margin = (max_len - range_len) / 2
            ref_near = depth_range_all[:, 0] - range_margin
            ref_near = np.max(np.stack([ref_near, depth_range_all[:, 0] * 0.5], -1), 1)
            depth_range_all[:, 0] = ref_near
            depth_range_all[:, 1] = ref_near + max_len
        ref_imgs_info['depth_range'] = depth_range_all[:-1]
        que_imgs_info['depth_range'] = depth_range_all[-1:]


    def __getitem__(self, index):
        set_seed(index, self.is_train)
        
        database, que_id, ref_ids_all = self.get_database_ref_que_ids(index)
        ref_ids = self.select_working_views(database, que_id, ref_ids_all)
        
        is_aligned = not database.database_name.startswith('space')
        ref_imgs_info = build_imgs_info(database, ref_ids, -1, is_aligned, has_depth=self.cfg['use_depth'])
        
        que_imgs_info = build_imgs_info(database, [que_id], has_depth=self.is_train)

        if self.cfg['use_consistent_depth_range']:
            self.consistent_depth_range(ref_imgs_info, que_imgs_info)

        ref_imgs_info = pad_imgs_info(ref_imgs_info, self.cfg['ref_pad_interval'])
        
        que_mask_cur = que_imgs_info['masks'][0,0]>0
        coords = get_coords_mask(que_mask_cur, self.train_ray_num, self.cfg['foreground_ratio']).reshape([-1,2])  # [self.train_ray_num, 2]

        # don't feed depth to gpu
        if not self.cfg['use_depth']:
            if 'depth' in ref_imgs_info: ref_imgs_info.pop('depth')
            if 'depth' in que_imgs_info: que_imgs_info.pop('depth')
            if 'true_depth' in ref_imgs_info: ref_imgs_info.pop('true_depth')
            
        ref_imgs_info = imgs_info_to_torch(ref_imgs_info)
        que_imgs_info = imgs_info_to_torch(que_imgs_info)

        # for key in ref_imgs_info:
        #     print(key, ref_imgs_info[key].shape)
        # for key in que_imgs_info:
        #     print(key, que_imgs_info[key].shape)
            
        sample = {}
        
        sample['ref_img'] = que_imgs_info['imgs'][0]  # (3, h, w)
        _, h, w = sample['ref_img'].shape
        
        sample['obj_masks'] = que_mask_cur  # (h, w)
        sample['ray_idx'] = coords[:,0] + coords[:,1] * w  # (self.train_ray_num)
        
        sample['w2cs'] = torch.eye(4).to(que_imgs_info['poses'])
        sample['w2cs'][:3,:4] = que_imgs_info['poses'][0]  # (4, 4)
        sample['c2w'] = torch.inverse(sample['w2cs'])

        sample['intrinsics'] = que_imgs_info['Ks'][0]  # (3, 3)
        
        sample['near_fars'] = torch.cat([que_imgs_info['depth_range'], ref_imgs_info['depth_range']], dim=0)

        # ## use target img as source image for debugging
        # ref_imgs_info['imgs'] = que_imgs_info['imgs'].repeat([ref_imgs_info['imgs'].shape[0], 1, 1, 1])
        # ref_imgs_info['poses'] = que_imgs_info['poses'].repeat([ref_imgs_info['poses'].shape[0], 1, 1])
        
        sample['source_imgs'] = ref_imgs_info['imgs']  # (num_src_view, 3, h, w)
        
        normalize_matrix = torch.tensor([[1/((w-1)/2), 0, -1, 0], [0, 1/((h-1)/2), -1, 0], [0,0,1,0], [0,0,0,1]]).to(sample['w2cs']) # [4, 4]
        intrinsics_pad = torch.eye(4).to(sample['w2cs']) # [4, 4]
        intrinsics_pad[:3,:3] = sample['intrinsics'] # [4, 4]
         
        sample['ref_pose'] = normalize_matrix @ intrinsics_pad @ sample['w2cs']   # 4, 4
        sample['ref_pose_inv'] = torch.inverse(sample['ref_pose'])  # [4, 4]

        w2cs_src = repeat(torch.eye(4), "X Y -> L X Y", L = ref_imgs_info['poses'].shape[0]).to(ref_imgs_info['poses']).clone() # (num_src_view, 4, 4)
        w2cs_src[:,:3,:4] = ref_imgs_info['poses'] # (num_src_view, 4, 4)
        
        _, _, h_src, w_src = sample['source_imgs'].shape
        normalize_matrix = torch.tensor([[1/((w_src-1)/2), 0, -1, 0], [0, 1/((h_src-1)/2), -1, 0], [0,0,1,0], [0,0,0,1]]).to(sample['w2cs']) # [4, 4]
        intrinsics_pad = repeat(torch.eye(4), "X Y -> L X Y", L = w2cs_src.shape[0]).to(w2cs_src).clone()  # (num_src_view, 4, 4)
        intrinsics_pad[:,:3,:3] = ref_imgs_info['Ks'] # (num_src_view, 4, 4)
        
        sample['source_poses'] = normalize_matrix @ intrinsics_pad @ w2cs_src
        sample['source_poses_inv'] = torch.inverse(sample['source_poses'])

        h_line = (np.linspace(0,h-1,h))*2/(h-1) - 1
        w_line = (np.linspace(0,w-1,w))*2/(w-1) - 1
        h_mesh, w_mesh = np.meshgrid(h_line, w_line, indexing='ij')
        w_mesh_flat = w_mesh.reshape(-1)
        h_mesh_flat = h_mesh.reshape(-1)
        homo_pixel = np.stack([w_mesh_flat, h_mesh_flat, np.ones(len(h_mesh_flat)), np.ones(len(h_mesh_flat))])  #[4,HW]

        sample['ray_o'] = sample['ref_pose_inv'][:3,-1]   # 3
        tmp_ray_d = (sample['ref_pose_inv'] @ homo_pixel)[:3] - sample['ray_o'][:,None]  # [3, h*w]
        tmp_ray_d = tmp_ray_d / torch.linalg.norm(tmp_ray_d, dim=0, keepdim=True)  # [3, h*w]
        sample['ray_d'] = tmp_ray_d  # [3, h*w]
        
        # print(que_imgs_info['depth'].shape)
        # print(ref_imgs_info['depth'].shape)

        if self.cfg['use_depth'] and self.is_train:
            # depths_h = torch.cat([que_imgs_info['depth'], ref_imgs_info['depth']], dim=0)[:,0,:,:]  # [num_src_view+1, h, w]
            depths_h = que_imgs_info['depth'][0]  # [1, h, w]

            ### calculate depth along each camera ray
            # V,H,W = depths_h.shape()      
            # cam_ray_d = (torch.inverse(normalize_matrix @ intrinsics_pad[0]) @ self.homo_pixel)[:3] # [3, h*w]
            # cam_ray_d = cam_ray_d / torch.linalg.norm(cam_ray_d, dim=0, keepdim=True)  # [3, h*w]
            # sample['cam_ray_d'] = cam_ray_d  # [3, h*w]
         
            # depths_h = depths_h.view(V,-1)
            # depths_h = depths_h/cam_ray_d[2:3,:]
            # depths_h = depths_h.view(V,H,W)
            
            sample['depths_h'] = depths_h
        
        # for key in sample:
        #     print(key, sample[key].shape)
        # input()

        ## visualize source views
        # import imageio
        # imageio.imwrite("/".join(['.', "target.jpg"]), (sample["ref_img"].permute(1,2,0) * 255).cpu().numpy().astype(np.uint8))
        # for k in range(sample["source_imgs"].shape[0]):
        #     imageio.imwrite("/".join(['.', "src_%d.jpg" % (k,)]), (sample["source_imgs"][k].permute(1,2,0) * 255).cpu().numpy().astype(np.uint8))
        # print('saved!!!')
        # input()
        
        # print(sample['ref_img'].shape, sample['source_imgs'][0].shape)
        
        return sample

    def __len__(self):
        return self.num



def select_train_ids_for_real_estate(img_ids):
    num_frames = len(img_ids)
    window_size = 32
    shift = np.random.randint(low=-1, high=2)
    id_render = np.random.randint(low=4, high=num_frames - 4 - 1)

    right_bound = min(id_render + window_size + shift, num_frames - 1)
    left_bound = max(0, right_bound - 2 * window_size)
    candidate_ids = np.arange(left_bound, right_bound)
    # remove the query frame itself with high probability
    if np.random.choice([0, 1], p=[0.01, 0.99]):
        candidate_ids = candidate_ids[candidate_ids != id_render]

    id_feat = np.random.choice(candidate_ids, size=min(8, len(candidate_ids)), replace=False)
    img_ids = np.asarray(img_ids)
    return img_ids[id_render], img_ids[id_feat]


def add_depth_offset(depth,mask,region_min,region_max,offset_min,offset_max,noise_ratio,depth_length):
    coords = np.stack(np.nonzero(mask), -1)[:, (1, 0)]
    length = np.max(coords, 0) - np.min(coords, 0)
    center = coords[np.random.randint(0, coords.shape[0])]
    lx, ly = np.random.uniform(region_min, region_max, 2) * length
    diff = coords - center[None, :]
    mask0 = np.abs(diff[:, 0]) < lx
    mask1 = np.abs(diff[:, 1]) < ly
    masked_coords = coords[mask0 & mask1]
    global_offset = np.random.uniform(offset_min, offset_max) * depth_length
    if np.random.random() < 0.5:
        global_offset = -global_offset
    local_offset = np.random.uniform(-noise_ratio, noise_ratio, masked_coords.shape[0]) * depth_length + global_offset
    depth[masked_coords[:, 1], masked_coords[:, 0]] += local_offset


def build_src_imgs_info_select(database, ref_ids, ref_ids_all, cost_volume_nn_num, pad_interval=-1):
    # ref_ids - selected ref ids for rendering
    ref_idx_exp = compute_nearest_camera_indices(database, ref_ids, ref_ids_all)
    ref_idx_exp = ref_idx_exp[:, 1:1 + cost_volume_nn_num]  # [rfn, nn], exclude the first one, i.e., the ref_id itself
    ref_ids_all = np.asarray(ref_ids_all)  # [rfn_all], e.g., [100] for Lego
    ref_ids_exp = ref_ids_all[ref_idx_exp]  # [rfn, nn], i.e., [8, 3]
    ref_ids_exp_ = ref_ids_exp.flatten()  # [rfn * nn]
    ref_ids = np.asarray(ref_ids)
    
    ref_ids_in = np.unique(np.concatenate([ref_ids_exp_, ref_ids]))  # rfn', e.g., ~10 for Lego
    
    mask0 = ref_ids_in[None, :] == ref_ids[:, None]  # [rfn, rfn']
    ref_idx_, ref_idx = np.nonzero(mask0)  # [rfn], [rfn]
    
    ref_real_idx = ref_idx[np.argsort(ref_idx_)]  # [rfn], to indicate the locations of real reference views (ref_ids) in the extract ref_imgs_info (ref_ids_in), other ref info is to calculate the cost volume for each ref view

    rfn, nn = ref_ids_exp.shape
    mask1 = ref_ids_in[None, :] == ref_ids_exp.flatten()[:, None]  # [nn*rfn, rfn']
    ref_cv_idx_, ref_cv_idx = np.nonzero(mask1)  # [nn*rfn, nn*rfn]
    ref_cv_idx = ref_cv_idx[np.argsort(ref_cv_idx_)]  # [rfn * nn]
    ref_cv_idx = ref_cv_idx.reshape([rfn, nn])  # [rfn, nn], indicate the locations of the nearest neighbors of each ref view in the extract ref_imgs_info
    
    is_aligned = not database.database_name.startswith('space')
    ref_imgs_info = build_imgs_info(database, ref_ids_in, pad_interval, is_aligned)
    
    return ref_imgs_info, ref_cv_idx, ref_real_idx
