import torch
import cv2 as cv
from torch.utils.data import Dataset
from .asset import *
from .database import parse_database_name, get_database_split
import numpy as np
from einops import repeat
import cv2

from .utils.base_utils import get_coords_mask
from .utils.dataset_utils import set_seed
from .utils.imgs_info import build_imgs_info, random_crop, random_flip, pad_imgs_info, imgs_info_slice, \
    imgs_info_to_torch
from .utils.view_select import compute_nearest_camera_indices

from .scene_transform import get_boundingbox
import collections
Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))


class GeneralRendererDataset_Scale(Dataset):
    default_cfg={
        'train_database_types':['dtu_train','space','real_iconic','real_estate','gso'],
        'type2sample_weights': {'gso':80, 'dtu_train':20, 'real_iconic':10, 'space':20, 'real_estate':40},
        # 'type2sample_weights': {'gso':20, 'dtu_train':20, 'real_iconic':20, 'space':10, 'real_estate':10},
        
        'val_database_name': 'nerf_synthetic/lego/black_800', # llff_colmap/flower/high
        'val_database_split_type': 'val',

        'min_wn': 8,
        # 'max_wn': 9,
        'ref_pad_interval': 16,
        'train_ray_num': 512,
        'foreground_ratio': 0.7,
        'resolution_type': 'hr',
        "use_consistent_depth_range": True,
        'use_depth_loss_for_all': False,
        "use_depth": True,
        "use_src_imgs": False,
        "cost_volume_nn_num": 3,

        "use_aug": False,
        
        "warp_to_ref_view": False,
        
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
        
        "use_official_dtu_src": False,
        
        "use_depth_dz": False,
        
        "fix_num_src_view": False
    }
    def __init__(self, cfg, is_train, train_ray_num, num_src_view=4, extract_geometry=False, is_finetune=False, is_test=False):
        if cfg is not None:
            self.cfg = {**self.default_cfg,**cfg}
        else:
            self.cfg = self.default_cfg
            
        self.cfg['min_wn'] = num_src_view
        self.train_ray_num = train_ray_num
        self.extract_geometry = extract_geometry
        
        self.OPENGL_CAMERA = False
            
        self.is_train = is_train
        self.is_finetune = is_finetune
        if is_train and not is_finetune:
            self.num=999999
            self.type2scene_names,self.database_types,self.database_weights = {}, [], []
            if self.cfg['resolution_type']=='hr':
                type2scene_names={'dtu_train':dtu_train_scene_names,'space':space_scene_names,
                                  'real_iconic':real_iconic_scene_names_4,
                                  'real_estate':real_estate_scene_names,'gso':gso_scene_names}
            elif self.cfg['resolution_type']=='lr':
                type2scene_names={'dtu_train':dtu_train_scene_names, 'space':space_scene_names,
                                  'real_iconic':real_iconic_scene_names_8,
                                  'real_estate':real_estate_scene_names, 'gso':gso_scene_names_400}
            else:
                raise NotImplementedError

            for database_type in self.cfg['train_database_types']:
                self.type2scene_names[database_type] = type2scene_names[database_type]
                self.database_types.append(database_type)
                self.database_weights.append(self.cfg['type2sample_weights'][database_type])
            assert(len(self.database_types)>0)
            # normalize weights
            self.database_weights=np.asarray(self.database_weights)
            self.database_weights=self.database_weights/np.sum(self.database_weights)
        elif is_finetune: 
            self.database = parse_database_name(self.cfg['val_database_name'])
            self.K = torch.from_numpy(self.database.K)
            self.ref_ids, self.que_ids = get_database_split(self.database, 'train')
            self.que_ids = self.ref_ids
            self.num=len(self.que_ids)
        else:
            self.database = parse_database_name(self.cfg['val_database_name'])
            self.K = torch.from_numpy(self.database.K)
            if is_test:
                self.ref_ids, self.que_ids = get_database_split(self.database, 'test')
            else:
                self.ref_ids, self.que_ids = get_database_split(self.database, 'val')
            if self.extract_geometry:
                self.que_ids = self.ref_ids
            self.num=len(self.que_ids)

    def get_database_ref_que_ids(self, index):
        if self.is_train:
            database_type = np.random.choice(self.database_types,1,False,p=self.database_weights)[0]
            database_scene_name = np.random.choice(self.type2scene_names[database_type])
            database = parse_database_name(database_scene_name)
            # if there is no depth for all views, we repeat random sample until find a scene with depth
            while True:
                ref_ids = database.get_img_ids(check_depth_exist=True)
                if len(ref_ids)==0:
                    database_type = np.random.choice(self.database_types, 1, False, self.database_weights)[0]
                    database_scene_name = np.random.choice(self.type2scene_names[database_type])
                    database = parse_database_name(database_scene_name)
                else: break
            que_id = np.random.choice(ref_ids)
            if database.database_name.startswith('real_estate'):
                que_id, ref_ids = select_train_ids_for_real_estate(ref_ids)
        else:
            database = self.database
            que_id, ref_ids = self.que_ids[index], self.ref_ids

        return database, que_id, np.asarray(ref_ids)

    def select_working_views_impl(self, database_name, dist_idx, ref_num):
        if self.cfg['aug_view_select_type']=='default':
            if database_name.startswith('space') or database_name.startswith('real_estate'):
                pass
            elif database_name.startswith('gso'):
                pool_ratio = np.random.randint(1, 5)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 32)]
            elif database_name.startswith('real_iconic'):
                pool_ratio = np.random.randint(1, 5)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 32)]
            elif database_name.startswith('dtu_train'):
                pool_ratio = np.random.randint(1, 3)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 12)]
            elif database_name.startswith('scannet'):
                pool_ratio = np.random.randint(1, 3)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 12)]
            elif database_name.startswith('replica'):
                pool_ratio = np.random.randint(1, 3)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 12)]
            else:
                raise NotImplementedError

        elif self.cfg['aug_view_select_type']=='hard':
            if database_name.startswith('space') or database_name.startswith('real_estate'):
                pass
            elif database_name.startswith('gso'):
                pool_ratio = np.random.randint(2, 6)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 32)]
            elif database_name.startswith('real_iconic'):
                pool_ratio = np.random.randint(2, 6)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 32)]
            elif database_name.startswith('dtu_train'):
                pool_ratio = np.random.randint(2, 4)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 24)]
            elif database_name.startswith('scannet'):
                pool_ratio = np.random.randint(2, 4)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 24)]
            elif database_name.startswith('replica'):
                pool_ratio = np.random.randint(2, 4)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 24)]
            else:
                raise NotImplementedError
            
        elif self.cfg['aug_view_select_type']=='easy':
            if database_name.startswith('space') or database_name.startswith('real_estate'):
                pass
            elif database_name.startswith('gso'):
                pool_ratio = 3
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 24)]
            elif database_name.startswith('real_iconic'):
                pool_ratio = np.random.randint(1, 4)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 20)]
            elif database_name.startswith('dtu_train'):
                pool_ratio = np.random.randint(1, 3)
                dist_idx = dist_idx[:min(ref_num * pool_ratio, 12)]
            else:
                raise NotImplementedError

        elif self.cfg['aug_view_select_type']=='no_aug':
            dist_idx = dist_idx[:ref_num]

        return dist_idx

    def select_working_views(self, database, que_id, ref_ids):
        database_name = database.database_name
        dist_idx = compute_nearest_camera_indices(database, [que_id], ref_ids)[0]

        if self.is_train or self.is_finetune:
            if np.random.random()>0.02: # 2% chance to include que image
                dist_idx = dist_idx[ref_ids[dist_idx]!=que_id]
            
            if not self.cfg['fix_num_src_view']:
                ref_num = np.random.randint(self.cfg['min_wn'], self.cfg['min_wn']+2)
            else:
                ref_num = self.cfg['min_wn']
                
            dist_idx = self.select_working_views_impl(database_name,dist_idx,ref_num)
            if not database_name.startswith('real_estate'):
                # we already select working views for real estate dataset
                np.random.shuffle(dist_idx)
                dist_idx = dist_idx[:ref_num]
                ref_ids = ref_ids[dist_idx]
            else:
                ref_ids = ref_ids[:ref_num]
        else:
            dist_idx = dist_idx[:self.cfg['min_wn']]
            ref_ids = ref_ids[dist_idx]
        return ref_ids

    def depth_range_aug_for_gso(self, depth_range, depth, mask):
        depth_range_new = depth_range.copy()
        if np.random.random() < self.cfg['aug_gso_shrink_range_prob']:
            rfn, _, h, w = depth.shape
            far_ratios, near_ratios = [], []
            for rfi in range(rfn):
                depth_val = depth[rfi][mask[rfi].astype(np.bool)]
                depth_val = depth_val[depth_val > 1e-3]
                depth_val = depth_val[depth_val < 1e4]
                depth_max = np.max(depth_val) * 1.1
                depth_min = np.min(depth_val) * 0.9
                near, far = depth_range[rfi]
                far_ratio = depth_max / far
                near_ratio = near / depth_min
                far_ratios.append(far_ratio)
                near_ratios.append(near_ratio)

            far_ratio = np.max(far_ratios)
            near_ratio = np.max(near_ratios)
            if far_ratio < 1.0: depth_range_new[:, 1] *= np.random.uniform(far_ratio, 1.0)
            if near_ratio < 1.0: depth_range_new[:, 0] /= np.random.uniform(near_ratio, 1.0)

        if np.random.random()<0.8:
            ratio0, ratio1 = np.random.uniform(0.025, 0.1, 2)
            depth_range_new[:, 0] = depth_range_new[:, 0] * (1 - ratio0)
            depth_range_new[:, 1] = depth_range_new[:, 1] * (1 + ratio1)
        return depth_range_new

    def random_change_depth_range(self, depth_range, depth, mask, database_name):
        if database_name.startswith('gso'):
            depth_range_new = self.depth_range_aug_for_gso(depth_range, depth, mask)
        else:
            depth_range_new = depth_range.copy()
            if np.random.random()<self.cfg['aug_depth_range_prob']:
                depth_range_new[:,0] *= np.random.uniform(self.cfg['aug_depth_range_min'],1.0)
                depth_range_new[:,1] *= np.random.uniform(1.0,self.cfg['aug_depth_range_max'])
        return depth_range_new


    def add_depth_noise(self,depths,masks,depth_ranges):
        rfn = depths.shape[0]
        depths_output = []
        for rfi in range(rfn):
            depth, mask, depth_range = depths[rfi,0], masks[rfi,0], depth_ranges[rfi]

            depth = depth.copy()
            near, far = depth_range
            depth_length = far - near
            if self.cfg['aug_use_depth_offset'] and np.random.random() < self.cfg['aug_depth_offset_prob']:
                add_depth_offset(depth, mask,self.cfg['aug_depth_offset_region_min'],
                                 self.cfg['aug_depth_offset_region_max'],
                                 self.cfg['aug_depth_offset_min'],
                                 self.cfg['aug_depth_offset_max'],
                                 self.cfg['aug_depth_offset_local'], depth_length)
            if self.cfg['aug_use_depth_small_offset'] and np.random.random() < self.cfg['aug_depth_small_offset_prob']:
                add_depth_offset(depth, mask, 0.1, 0.2, 0.01, 0.05, 0.005, depth_length)
            if self.cfg['aug_use_global_noise'] and np.random.random() < self.cfg['aug_global_noise_prob']:
                depth += np.random.uniform(-0.005,0.005,depth.shape).astype(np.float32)*depth_length
            depths_output.append(depth)
        return np.asarray(depths_output)[:,None,:,:]

    def generate_coords_for_training(self, database, que_imgs_info):
        if (database.database_name.startswith('real_estate') \
                or database.database_name.startswith('real_iconic') \
                or database.database_name.startswith('space')) and self.cfg['aug_pixel_center_sample']:
                que_mask_cur = np.zeros_like(que_imgs_info['masks'][0, 0]).astype(np.bool)
                h, w = que_mask_cur.shape
                center_ratio = 0.8
                begin_ratio = (1-center_ratio)/2
                hb, he = int(h*begin_ratio), int(h*(center_ratio+begin_ratio))
                wb, we = int(w*begin_ratio), int(w*(center_ratio+begin_ratio))
                que_mask_cur[hb:he,wb:we] = True
                coords = get_coords_mask(que_mask_cur, self.train_ray_num, 0.9).reshape([1, -1, 2])
        else:
            que_mask_cur = que_imgs_info['masks'][0,0]>0
            coords = get_coords_mask(que_mask_cur, self.train_ray_num, self.cfg['foreground_ratio']).reshape([1,-1,2])
        return coords

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


    def cal_scale_mat(self, img_hw, intrinsics, extrinsics, near_fars, factor=1.):
        center, radius, _ = get_boundingbox(img_hw, intrinsics, extrinsics, near_fars)

        radius = radius * factor
        scale_mat = np.diag([radius, radius, radius, 1.0])
        scale_mat[:3, 3] = center.cpu().numpy()
        scale_mat = scale_mat.astype(np.float32)

        return scale_mat, 1. / radius.cpu().numpy()


    def __getitem__(self, index):
        set_seed(index, self.is_train)
        
        database, que_id, ref_ids_all = self.get_database_ref_que_ids(index)
                
        if 'dtu' in database.database_name and self.cfg["use_official_dtu_src"]:
            ref_ids = database.get_src_views(que_id)[:self.cfg['min_wn']]
        else:
            ref_ids = self.select_working_views(database, que_id, ref_ids_all)
        
        ### This is for cost volume construction of each source view in Neuray
        # if self.cfg['use_src_imgs']:
        #     # src_imgs_info used in construction of cost volume
        #     ref_imgs_info, ref_cv_idx, ref_real_idx = build_src_imgs_info_select(database,ref_ids,ref_ids_all,self.cfg['cost_volume_nn_num'])
        # else:
        #     ref_idx = compute_nearest_camera_indices(database, ref_ids)[:,1:4] # used in cost volume construction
        #     is_aligned = not database.database_name.startswith('space')
        #     ref_imgs_info = build_imgs_info(database, ref_ids, -1, is_aligned)
        
        is_aligned = not database.database_name.startswith('space')
        ref_imgs_info = build_imgs_info(database, ref_ids, -1, is_aligned, has_depth=self.cfg['use_depth'])
        
        que_imgs_info = build_imgs_info(database, [que_id], has_depth=self.is_train or self.is_finetune)

        if self.is_train and self.cfg['use_aug']:
            # data augmentation
            depth_range_all = np.concatenate([ref_imgs_info['depth_range'],que_imgs_info['depth_range']],0)
            if database.database_name.startswith('gso'): # only used in gso currently
                depth_all = np.concatenate([ref_imgs_info['depth'],que_imgs_info['depth']],0)
                mask_all = np.concatenate([ref_imgs_info['masks'],que_imgs_info['masks']],0)
            else:
                depth_all, mask_all = None, None
            depth_range_all = self.random_change_depth_range(depth_range_all, depth_all, mask_all, database.database_name)
            ref_imgs_info['depth_range'] = depth_range_all[:-1]
            que_imgs_info['depth_range'] = depth_range_all[-1:]

            if database.database_name.startswith('gso') and self.cfg['use_depth']:
                depth_aug = self.add_depth_noise(ref_imgs_info['depth'], ref_imgs_info['masks'], ref_imgs_info['depth_range'])
                ref_imgs_info['true_depth'] = ref_imgs_info['depth']
                ref_imgs_info['depth'] = depth_aug

            if database.database_name.startswith('real_estate') \
                or database.database_name.startswith('real_iconic') \
                or database.database_name.startswith('space'):
                # crop all datasets
                ref_imgs_info, que_imgs_info = random_crop(ref_imgs_info, que_imgs_info, self.cfg['aug_forward_crop_size'])
                if np.random.random()<0.5:
                    ref_imgs_info, que_imgs_info = random_flip(ref_imgs_info, que_imgs_info)

            if self.cfg['use_depth_loss_for_all'] and self.cfg['use_depth']:
                if not database.database_name.startswith('gso'):
                    ref_imgs_info['true_depth'] = ref_imgs_info['depth']
        
        if self.cfg['use_consistent_depth_range']:
            self.consistent_depth_range(ref_imgs_info, que_imgs_info)
        
        ref_imgs_info = pad_imgs_info(ref_imgs_info, self.cfg['ref_pad_interval'])
        
        ### generate coords
        # if self.is_train:
        #     coords = self.generate_coords_for_training(database, que_imgs_info)
        # else:
        #     qn, _, hn, wn = que_imgs_info['imgs'].shape
        #     coords = np.stack(np.meshgrid(np.arange(wn),np.arange(hn)),-1)
        #     coords = coords.reshape([1,-1,2]).astype(np.float32)
        # que_imgs_info['coords'] = coords

        if (database.database_name.startswith('real_estate') \
                or database.database_name.startswith('real_iconic') \
                or database.database_name.startswith('space')) and self.cfg['aug_pixel_center_sample']:
                que_mask_cur = np.zeros_like(que_imgs_info['masks'][0, 0]).astype(np.bool)
                h, w = que_mask_cur.shape
                center_ratio = 0.8
                begin_ratio = (1-center_ratio)/2
                hb, he = int(h*begin_ratio), int(h*(center_ratio+begin_ratio))
                wb, we = int(w*begin_ratio), int(w*(center_ratio+begin_ratio))
                que_mask_cur[hb:he,wb:we] = True
                coords = get_coords_mask(que_mask_cur, self.train_ray_num, 0.9).reshape([-1, 2])  # [self.train_ray_num, 2]
        else:
            que_mask_cur = que_imgs_info['masks'][0,0]>0
            coords = get_coords_mask(que_mask_cur, self.train_ray_num, self.cfg['foreground_ratio']).reshape([-1,2])  # [self.train_ray_num, 2]

        # don't feed depth to gpu
        if not self.cfg['use_depth']:
            if 'depth' in ref_imgs_info: ref_imgs_info.pop('depth')
            if 'depth' in que_imgs_info: que_imgs_info.pop('depth')
            if 'true_depth' in ref_imgs_info: ref_imgs_info.pop('true_depth')
            
        ### This is for cost volume construction of each source view in Neuray
        # if self.cfg['use_src_imgs']:
        #     src_imgs_info = ref_imgs_info.copy()
        #     ref_imgs_info = imgs_info_slice(ref_imgs_info, ref_real_idx)
        #     ref_imgs_info['nn_ids'] = ref_cv_idx
        # else:
        #     # 'nn_ids' used in constructing cost volume (specify source image ids)
        #     ref_imgs_info['nn_ids'] = ref_idx.astype(np.int64)


        # for key in ref_imgs_info:
        #     print(key, ref_imgs_info[key].shape)
        # for key in que_imgs_info:
        #     print(key, que_imgs_info[key].shape)
        
        
        ### add the scale matrix
        _, h, w = que_imgs_info['imgs'][0].shape
        
        w2cs = np.eye(4)
        w2cs[:3,:4] = que_imgs_info['poses'][0]  # (4, 4)
        w2cs = w2cs.reshape(-1,4,4) # (1, 4, 4)
        intrinsics = que_imgs_info['Ks']  # (1, 3, 3)

        w2cs_src = repeat(np.eye(4), "X Y -> L X Y", L = ref_imgs_info['poses'].shape[0]) # (num_src_view, 4, 4)
        w2cs_src[:,:3,:4] = ref_imgs_info['poses'] # (num_src_view, 4, 4)
        intrinsics_src = ref_imgs_info['Ks'] # (num_src_view, 3, 3)

        sample = {}
        sample['extrinsic_render_view'] = torch.from_numpy(w2cs[0])
        sample['intrinsic_render_view'] = torch.from_numpy(intrinsics[0])
        
        w2cs_all = np.concatenate([w2cs, w2cs_src], axis=0).astype(np.float32)  # (1+num_src_view, 4, 4)
        
        if self.cfg['warp_to_ref_view']:
            w2c_ref_inv = np.linalg.inv(w2cs_all[0])
            w2cs_all = w2cs_all @ w2c_ref_inv
        
        intrinsics_all = repeat(np.eye(4), "X Y -> L X Y", L = w2cs_all.shape[0])  # (num_src_view, 4, 4)
        intrinsics_all[:,:3,:3] = np.concatenate([intrinsics, intrinsics_src], axis=0).astype(np.float32)
        
        near_fars = np.concatenate([que_imgs_info['depth_range'], ref_imgs_info['depth_range']], axis=0).astype(np.float32)

        scale_mat, scale_factor = self.cal_scale_mat(img_hw=[h, w],  ## warning: here we use the hw of query images, while the hw of source iamges may be slightly larger
                                                     intrinsics=intrinsics_all, extrinsics=w2cs_all,
                                                     near_fars=near_fars, factor=1)
                
        new_near_fars = []
        new_w2cs = []
        new_c2ws = []
                
        for intrinsic, extrinsic, near_far in zip(intrinsics_all, w2cs_all, near_fars):

            P = intrinsic @ extrinsic @ scale_mat
            P = P[:3, :4]
            
            c2w = load_K_Rt_from_P(None, P)[1]

            w2c = np.linalg.inv(c2w)
            new_w2cs.append(w2c)
            new_c2ws.append(c2w)

            # camera_o = c2w[:3, 3]
            # dist = np.sqrt(np.sum(camera_o ** 2))
            # near = dist - 1
            # far = dist + 1
            # # new_near_fars.append([0.95 * near, 1.05 * far])
            # print([near, far])
            
            near = near_far[0] * scale_factor
            far = near_far[1] * scale_factor
            new_near_fars.append([near, far])
            # print([near, far])
            # input()
        
        w2cs_all, c2ws_all, near_fars = np.stack(new_w2cs), np.stack(new_c2ws), np.stack(new_near_fars)
        
        w2cs_all = torch.from_numpy(w2cs_all)
        c2ws_all = torch.from_numpy(c2ws_all)
        near_fars = torch.from_numpy(near_fars)
        w2cs = w2cs_all[0]  # (4, 4)
        c2ws = c2ws_all[0]  # (4, 4)
        w2cs_src = w2cs_all[1:] # (num_src_view, 4, 4)

        ref_imgs_info = imgs_info_to_torch(ref_imgs_info)
        que_imgs_info = imgs_info_to_torch(que_imgs_info)
        
        sample["scale_mat"] = scale_mat
        sample["trans_mat"] = w2c_ref_inv
        
        sample['database'] = database.database_name
        
        sample['ref_img'] = que_imgs_info['imgs'][0]  # (3, h, w)
        
        sample['obj_masks'] = que_mask_cur  # (h, w)
        sample['ray_idx'] = coords[:,0] + coords[:,1] * w  # (self.train_ray_num)
        
        sample['w2cs'] = w2cs  # (4, 4)
        sample['c2w'] = c2ws  # (4, 4)
        sample['intrinsics'] = que_imgs_info['Ks'][0]  # (3, 3)
        sample['near_fars'] = near_fars
        
        # ## use target img as source image for debugging
        # ref_imgs_info['imgs'] = que_imgs_info['imgs'].repeat([ref_imgs_info['imgs'].shape[0], 1, 1, 1])
        # ref_imgs_info['poses'] = que_imgs_info['poses'].repeat([ref_imgs_info['poses'].shape[0], 1, 1])
        
        sample['source_imgs'] = ref_imgs_info['imgs']  # (num_src_view, 3, h, w)
        
        normalize_matrix = torch.tensor([[1/((w-1)/2), 0, -1, 0], [0, 1/((h-1)/2), -1, 0], [0,0,1,0], [0,0,0,1]]).to(sample['w2cs']) # [4, 4]
        intrinsics_pad = torch.eye(4).to(sample['w2cs']) # [4, 4]
        intrinsics_pad[:3,:3] = sample['intrinsics'] # [4, 4]
         
        sample['ref_pose'] = normalize_matrix @ intrinsics_pad @ sample['w2cs']   # 4, 4
        sample['ref_pose_inv'] = torch.inverse(sample['ref_pose'])  # [4, 4]

        # w2cs_src = repeat(torch.eye(4), "X Y -> L X Y", L = ref_imgs_info['poses'].shape[0]).to(ref_imgs_info['poses']).clone() # (num_src_view, 4, 4)
        # w2cs_src[:,:3,:4] = ref_imgs_info['poses'] # (num_src_view, 4, 4)
        
        _, _, h_src, w_src = sample['source_imgs'].shape
        normalize_matrix = torch.tensor([[1/((w_src-1)/2), 0, -1, 0], [0, 1/((h_src-1)/2), -1, 0], [0,0,1,0], [0,0,0,1]]).to(sample['w2cs']) # [4, 4]
        # normalize_matrix = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0,0,1,0], [0,0,0,1]]).to(sample['w2cs']) # [4, 4]
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

        cam_ray_d = (torch.inverse(normalize_matrix @ intrinsics_pad[0]) @ homo_pixel)[:3] # [3, h*w]
        cam_ray_d = cam_ray_d / torch.linalg.norm(cam_ray_d, dim=0, keepdim=True)  # [3, h*w]
        sample['cam_ray_d'] = cam_ray_d  # [3, h*w]
        
        sample['ref_view'] = que_id

        sample['rays'] = Rays(origins=sample["ray_o"], viewdirs=sample["ray_d"])
        sample['pixels'] = torch.permute(sample["ref_img"], (1,2,0))
        
        # print(que_imgs_info['depth'].shape)
        # print(ref_imgs_info['depth'].shape)

        if self.cfg['use_depth'] and (self.is_train or self.is_finetune):            
            # depths_h = torch.cat([que_imgs_info['depth'], ref_imgs_info['depth']], dim=0)[:,0,:,:]  # [num_src_view+1, h, w]
            # depths_h = que_imgs_info['depth'][0]  # [1, h, w]
            
            ### visualize depth before scaling
            # from imgviz import depth2rgb
            # import imageio
            # depth = que_imgs_info['depth'][0][0].cpu().numpy()
            # depth_mm = (depth*1000).astype(np.uint8)
            # depth_vis = depth2rgb(depth, min_value=que_imgs_info['depth_range'][0,0], max_value=que_imgs_info['depth_range'][0,1])
            
            # imageio.imwrite('depth_mm_before.png', depth_mm)
            # imageio.imwrite('depth_vis_before.png', depth_vis)
            # imageio.imwrite("/".join(['.', "rgb_gt.jpg"]), (sample["ref_img"].permute(1,2,0) * 255).cpu().numpy().astype(np.uint8))
            
            depths_h = que_imgs_info['depth'][0] * scale_factor # [1, h, w]

            ### calculate depth along each camera ray
            # V,H,W = depths_h.shape()      
            # cam_ray_d = (torch.inverse(normalize_matrix @ intrinsics_pad[0]) @ self.homo_pixel)[:3] # [3, h*w]
            # cam_ray_d = cam_ray_d / torch.linalg.norm(cam_ray_d, dim=0, keepdim=True)  # [3, h*w]
            # sample['cam_ray_d'] = cam_ray_d  # [3, h*w]
         
            # depths_h = depths_h.view(V,-1)
            # depths_h = depths_h/cam_ray_d[2:3,:]
            # depths_h = depths_h.view(V,H,W)

            sample['depths_h'] = depths_h
            
            if self.cfg['use_depth_dz'] and 'dtu' in database.database_name:
                cam_ray_d = (torch.inverse(normalize_matrix @ intrinsics_pad[0]) @ homo_pixel)[:3]
                cam_ray_d = cam_ray_d / torch.linalg.norm(cam_ray_d, dim=0, keepdim=True)
                V,H,W = depths_h.size()       
                depths_h = depths_h.view(V,-1)
                depths_h = depths_h/cam_ray_d[2:3,:]
                sample['depths_h'] = depths_h.view(V,H,W)
                        
            else:
                sample['depths_h'] = depths_h
        
            # ### visualize depth
            # from imgviz import depth2rgb
            # import imageio
            # depth = sample['depths_h'][0].cpu().numpy()
            # depth_mm = (depth*1000).astype(np.uint8)
            # depth_vis = depth2rgb(depth, min_value=near_fars[0,0], max_value=near_fars[0,1])
            
            # imageio.imwrite('depth_mm.png', depth_mm)
            # imageio.imwrite('depth_vis.png', depth_vis)
            # imageio.imwrite("/".join(['.', "rgb_gt.jpg"]), (sample["ref_img"].permute(1,2,0) * 255).cpu().numpy().astype(np.uint8))
            # input()
            
        
        # for key in sample:
        #     print(key, sample[key].shape)
        # input()


        # ## visualize source views
        # import imageio
        # print(database.database_name)
        # print(que_id, ref_ids)
        # imageio.imwrite("/".join(['.', "target.jpg"]), (sample["ref_img"].permute(1,2,0) * 255).cpu().numpy().astype(np.uint8))
        # for k in range(sample["source_imgs"].shape[0]):
        #     imageio.imwrite("/".join(['.', "src_%d.jpg" % (k,)]), (sample["source_imgs"][k].permute(1,2,0) * 255).cpu().numpy().astype(np.uint8))
        # print(f'{database.database_name}: Saved!!!')
        # input()
        

        # ### visualize the camera distribution
        # ray_o = c2ws_all[:,:3,-1] # (4, 3)
        # ray_z = c2ws_all[:,:3,2]  # (4, 3)
        # ray_z = ray_z / torch.linalg.norm(ray_z, dim=1, keepdim=True)  # (4, 3)
        
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D

        # # Create a new figure for 3D plotting
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # # Plot each camera in the space
        # for i in range(ray_o.shape[0]):
        #     pos = ray_o[i]
        #     direc = ray_z[i]

        #     # Plot the camera position
        #     ax.scatter(pos[0], pos[1], pos[2], marker='o')

        #     # Plot a line indicating the camera's direction
        #     ax.quiver(pos[0], pos[1], pos[2], direc[0], direc[1], direc[2], length=0.1, normalize=True)

        # # Set labels for axes
        # ax.set_xlabel('X Axis')
        # ax.set_ylabel('Y Axis')
        # ax.set_zlabel('Z Axis')
        
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
        # ax.set_zlim(-1, 1)

        # plt.savefig('camera_distrib.png')
        # print('save camera distrib!!')
        # input()
                
        return sample

    def __len__(self):
        return self.num


class FinetuningRendererDataset(Dataset):
    default_cfg={
        "database_name": "nerf_synthetic/lego/black_800",
        "database_split": "val_all"
    }
    def __init__(self,cfg, is_train):
        self.cfg={**self.default_cfg,**cfg}
        self.is_train=is_train
        self.train_ids, self.val_ids = get_database_split(parse_database_name(self.cfg['database_name']),self.cfg['database_split'])

    def __getitem__(self, index):
        output={'index': index}
        return output

    def __len__(self):
        if self.is_train:
            return 99999999
        else:
            return len(self.val_ids)


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


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose