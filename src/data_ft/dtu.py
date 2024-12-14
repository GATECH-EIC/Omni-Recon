import torch
import cv2 as cv
import numpy as np
import os
import sys
import logging
from einops import repeat
from torch.utils.data import Dataset

from .scene_transform import get_boundingbox
from .data_utils import get_nearest_pose_ids

import collections
Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
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


class DTU_Finetune(Dataset):
    def __init__(self, root_dir, scan_id, split, n_src_views=3, src_via_dist=False, supersampling=1,
                 hold_every=8, img_wh=[800, 600], clip_wh=[0, 0], original_img_wh=[1600, 1200],
                 N_rays=512, near=425, far=900, set_id=0, color_bkgd_aug="white", pair_filepath=None, use_dataloader=False,args=None):
        super(DTU_Finetune, self).__init__()
        logging.info('Load data: Begin')
        
        self.args = args

        self.root_dir = root_dir
        self.split = split
        self.scan_id = scan_id
        self.hold_every = hold_every
        
        self.supersampling = supersampling
        
        self.use_dataloader = use_dataloader
        
        if pair_filepath is not None:
            self.pair_filepath = pair_filepath
        else:
            self.pair_filepath = "src/data_pretrain/dtu/dtu_pairs.txt"
        self.ref_src_pairs = self.build_metas()
            
        if 'DTU_TRAIN' in self.root_dir:
            img_wh = [640, 512]
            original_img_wh = [640, 512]
        elif 'DTU_TEST' in self.root_dir:
            img_wh = [800, 600]
            original_img_wh = [1600, 1200]
        else:
            print("No such root_dir:", self.root_dir)
            sys.exit()
        
        self.OPENGL_CAMERA = False
        self.NUM_IMAGE = 49

        self.offset_dist = 0 # 25mm
                
        assert color_bkgd_aug in ["white", "black", "random"]
        self.color_bkgd_aug = color_bkgd_aug
        
        self.src_via_dist = src_via_dist
        self.n_src_views = n_src_views

        # if set_id==0:
        #     self.view_list = [23, 24, 33, 22, 15, 34, 14, 32, 16, 35, 25]
        # else:
        #     self.view_list = [43, 42, 44, 33, 34, 32, 45, 23, 41, 24, 31]
        
        ## Align the coordinate system to the one used for extracting the mesh
        # REF_VIEW_WORLD = self.view_list[0]
        # REF_VIEW_WORLD_LIST = self.view_list[:num_ref_world_view]
        
        # if self.src_via_dist: # and not self.baking_only:
        #     self.src_view_idx = np.array([])
        # else:
        #     self.src_view_idx = np.array(self.view_list[:self.n_src_views])

        self.total_idx = list(np.arange(self.NUM_IMAGE))
        # self.i_test = np.array(list(set(np.arange(0, self.NUM_IMAGE, self.hold_every)) - set(self.src_view_idx)))  # [np.argmin(dists)]
        self.i_test = np.array(list(set(np.arange(0, self.NUM_IMAGE, self.hold_every))))  # [np.argmin(dists)]
        self.i_train = np.array(list(set(self.total_idx) - set(self.i_test)))

        if self.split == 'all':
            self.render_idx = self.total_idx
            self.i_train = np.array(self.total_idx)
        else:
            self.render_idx = self.i_test if self.split != 'train' else self.i_train
        self.render_idx = np.array(self.render_idx)
        
        ### this is for debug
        # self.render_idx = np.array([23, 24, 33])
        # self.offset_dist = 25
        
        self.near = near
        self.far = far
        
        if self.scan_id is not None:
            self.data_dir = os.path.join(self.root_dir, self.scan_id)
        else:
            self.data_dir = self.root_dir

        self.img_wh = np.array(img_wh) * self.supersampling

        if len(clip_wh) == 2:
            clip_wh = clip_wh + clip_wh
        self.clip_wh = np.array(clip_wh) * self.supersampling

        self.original_img_wh = original_img_wh
        self.N_rays = N_rays

        self.world_mats_np = []
        self.images_list = []
        self.masks_list = []
        
        if 'DTU_TEST' in self.root_dir:
            for vid in self.total_idx:
                proj_mat_filename = os.path.join(self.root_dir, 'cameras/{:0>8}_cam.txt'.format(vid))
                P = self.read_cam_file(proj_mat_filename)
                self.world_mats_np.append(P)
                img_filename = os.path.join(self.data_dir, 'image/{:0>6}.png'.format(vid))
                self.images_list.append(img_filename)
                
                img_filename = os.path.join(self.data_dir, 'mask/{:0>3}.png'.format(vid))
                self.masks_list.append(img_filename)

        else:
            light_idx = 3
            for vid in self.total_idx:
                proj_mat_filename = os.path.join(self.root_dir, 'Cameras/train/{:0>8}_cam.txt'.format(vid))
                P = self.read_cam_file(proj_mat_filename)
                self.world_mats_np.append(P)
                img_filename = os.path.join(self.root_dir,
                                            f'Rectified/{scan_id}_train/rect_{vid + 1:03d}_{light_idx}_r5000.png')
                self.images_list.append(img_filename)
                
                # img_filename = os.path.join(self.data_dir, 'mask/{:0>3}.png'.format(vid))
                # self.masks_list.append(img_filename)

        self.raw_near_fars = np.stack([np.array([self.near, self.far]) for i in range(len(self.images_list))])
        
        # ref_world_mat = self.world_mats_np[REF_VIEW_WORLD]
        # if no_warp_to_ref_view:
        #     self.ref_w2c = np.eye(4)
        # else:
        #     self.ref_w2c = np.linalg.inv(load_K_Rt_from_P(None, ref_world_mat[:3, :4])[1])
        
        self.images = []
        self.masks = []
        self.all_intrinsics = []
        self.all_w2cs = []
        self.all_w2cs_original = []
        self.all_render_w2cs = []
        self.all_render_w2cs_original = []

        self.load_scene()  # load the scene

        # ! estimate scale_mat
        # self.scale_mat, self.scale_factor = self.cal_scale_mat(
        #     img_hw=[self.img_wh[1]//self.supersampling, self.img_wh[0]//self.supersampling],
        #     intrinsics=self.all_intrinsics[REF_VIEW_WORLD_LIST],
        #     extrinsics=self.all_w2cs[REF_VIEW_WORLD_LIST],
        #     near_fars=self.raw_near_fars[REF_VIEW_WORLD_LIST],
        #     factor=1.1)
    
        # self.cuda_tensors = []

        # * after scaling and translation, unit bounding box
        # self.scaled_intrinsics, self.scaled_w2cs, self.scaled_c2ws, \
        # self.scaled_near_fars, self.scaled_render_w2cs,  \
        # self.scaled_render_c2ws = self.scale_cam_info()
        
        # self.cuda_tensors.extend(["scaled_intrinsics", "scaled_w2cs",  "scaled_c2ws", "scaled_near_fars", "scaled_render_w2cs", "scaled_render_c2ws"])

        self.bbox_min = np.array([-1.0, -1.0, -1.0])
        self.bbox_max = np.array([1.0, 1.0, 1.0])
        self.partial_vol_origin = torch.Tensor([-1., -1., -1.])

        self.img_W, self.img_H = self.img_wh
        h_line = torch.linspace(0,self.img_H-1,self.img_H)*2/(self.img_H-1) - 1
        w_line = torch.linspace(0,self.img_W-1,self.img_W)*2/(self.img_W-1) - 1
        h_mesh, w_mesh = torch.meshgrid(h_line, w_line, indexing='ij')
        self.w_mesh_flat = w_mesh.flatten()
        self.h_mesh_flat = h_mesh.flatten()
        self.homo_pixel = torch.stack([self.w_mesh_flat, self.h_mesh_flat, torch.ones(self.h_mesh_flat.shape[0]).to(self.h_mesh_flat), torch.ones(self.h_mesh_flat.shape[0]).to(self.h_mesh_flat)])  # [4, h*w]

        self.K = self.all_intrinsics[0]
        
        # self.scale_mat = torch.from_numpy(self.scale_mat)
        # self.K = self.scaled_intrinsics[0]
        # self.trans_mat = torch.from_numpy(np.linalg.inv(self.ref_w2c))
        # self.all_render_w2cs_original = torch.from_numpy(np.stack(self.all_render_w2cs_original))
        # self.normalize_matrix = torch.tensor([[1/((self.img_W-1)/2), 0, -1, 0], [0, 1/((self.img_H-1)/2), -1, 0], [0,0,1,0], [0,0,0,1]]).to(self.scaled_w2cs)
        
        # self.intrinsics_pad = repeat(torch.eye(4), "X Y -> L X Y", L = self.scaled_w2cs.shape[0]).clone()
        
        # self.intrinsics_pad[:,:3,:3] = self.scaled_intrinsics[:, :3, :3]

        # self.normalize_matrix_source = torch.tensor([[1/((self.img_W//self.supersampling-1)/2), 0, -1, 0], [0, 1/((self.img_H//self.supersampling-1)/2), -1, 0], [0,0,1,0], [0,0,0,1]]).to(self.scaled_w2cs)
        # # self.all_source_poses = self.normalize_matrix_source @ self.intrinsics_pad[self.i_train] @ self.scaled_render_w2cs[self.i_train]
        # self.all_source_poses = self.normalize_matrix_source @ self.intrinsics_pad[self.i_train] @ self.scaled_w2cs[self.i_train]
        # self.all_source_poses_inv = torch.inverse(self.all_source_poses)

        # self.cuda_tensors.extend(["scale_mat", "K", "trans_mat", "all_render_w2cs_original", "normalize_matrix", "images", "masks", "intrinsics_pad", "homo_pixel", "all_source_poses", "all_source_poses_inv"])

        logging.info('Load data: End')


    def read_cam_file(self, filename):
        """
        Load camera file e.g., 00000000_cam.txt
        """
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        intrinsics_ = np.float32(np.diag([1, 1, 1, 1]))
        intrinsics_[:3, :3] = intrinsics

        if 'DTU_TRAIN' in self.root_dir:
            intrinsics_[:2] *= 4
        
        P = intrinsics_ @ extrinsics

        return P


    def build_metas(self):
        ref_src_pairs = {}

        with open(self.pair_filepath) as f:
            num_viewpoint = int(f.readline())
            # viewpoints (49)
            for _ in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]

                ref_src_pairs[ref_view] = src_views

        return ref_src_pairs
    

    def load_scene(self):

        scale_x = self.img_wh[0] / self.original_img_wh[0]
        scale_y = self.img_wh[1] / self.original_img_wh[1]

        for idx in range(len(self.images_list)):
            image = cv.imread(self.images_list[idx])
            
            ## still load gt images at the target resolution
            w = self.img_wh[0]//self.supersampling
            h = self.img_wh[1]//self.supersampling
            image = cv.resize(image, (w, h)) / 255.
            clip_wh = self.clip_wh // self.supersampling

            image = image[clip_wh[1]:h - clip_wh[3],
                    clip_wh[0]:w - clip_wh[2]]
            
            self.images.append(np.transpose(image[:, :, ::-1], (2, 0, 1)))

            mask = cv.imread(self.masks_list[idx])
            mask = cv.resize(mask, (w, h)) / 255.
            mask = mask[clip_wh[1]:h - clip_wh[3],
                    clip_wh[0]:w - clip_wh[2]]
            
            self.masks.append(np.transpose(mask[:, :, ::-1], (2, 0, 1)))

            P = self.world_mats_np[idx]
            P = P[:3, :4]
            intrinsics, c2w = load_K_Rt_from_P(None, P)
            w2c = np.linalg.inv(c2w)

            render_c2w = c2w.copy()
            render_c2w[:3,3] += render_c2w[:3,0]*self.offset_dist

            render_w2c = np.linalg.inv(render_c2w)

            intrinsics[:1] *= scale_x
            intrinsics[1:2] *= scale_y

            intrinsics[0, 2] -= self.clip_wh[0]
            intrinsics[1, 2] -= self.clip_wh[1]

            self.all_intrinsics.append(intrinsics)
            ### - transform from world system to ref-camera system
            # self.all_w2cs.append(w2c @ np.linalg.inv(self.ref_w2c))   # the new "world" space is the camera space of the reference view
            # self.all_render_w2cs.append(render_w2c @ np.linalg.inv(self.ref_w2c))
            self.all_w2cs.append(w2c)
            self.all_render_w2cs.append(render_w2c)
            self.all_w2cs_original.append(w2c)
            self.all_render_w2cs_original.append(render_w2c)

        self.images = torch.from_numpy(np.stack(self.images)).to(torch.float32)
        self.masks = torch.from_numpy(np.stack(self.masks)).to(torch.float32)
        self.all_intrinsics = torch.from_numpy(np.stack(self.all_intrinsics)).to(torch.float32)
        self.all_w2cs = torch.from_numpy(np.stack(self.all_w2cs)).to(torch.float32)
        self.all_render_w2cs = torch.from_numpy(np.stack(self.all_render_w2cs)).to(torch.float32)
        self.all_render_c2ws = torch.inverse(self.all_render_w2cs)
        self.img_wh = [self.img_wh[0] - self.clip_wh[0] - self.clip_wh[2],
                       self.img_wh[1] - self.clip_wh[1] - self.clip_wh[3]]


    def cal_scale_mat(self, img_hw, intrinsics, extrinsics, near_fars, factor=1.):
        center, radius, _ = get_boundingbox(img_hw, intrinsics, extrinsics, near_fars)
        radius = radius * factor
        scale_mat = np.diag([radius, radius, radius, 1.0])
        scale_mat[:3, 3] = center.cpu().numpy()
        scale_mat = scale_mat.astype(np.float32)

        return scale_mat, 1. / radius.cpu().numpy()


    def scale_cam_info(self):
        new_intrinsics = []
        new_near_fars = []
        new_w2cs = []
        new_c2ws = []
        new_render_w2cs = []
        new_render_c2ws = []
        for idx in range(self.images.shape[0]):
            intrinsics = self.all_intrinsics[idx]
            P = intrinsics @ self.all_w2cs[idx] @ self.scale_mat
            P = P.cpu().numpy()[:3, :4]

            c2w = load_K_Rt_from_P(None, P)[1]
            w2c = np.linalg.inv(c2w)
            new_w2cs.append(w2c)
            new_c2ws.append(c2w)
            new_intrinsics.append(intrinsics)

            camera_o = c2w[:3, 3]
            dist = np.sqrt(np.sum(camera_o ** 2))
            near = dist - 1
            far = dist + 1

            new_near_fars.append([0.95 * near, 1.05 * far])

            P = intrinsics @ self.all_render_w2cs[idx] @ self.scale_mat
            P = P.cpu().numpy()[:3, :4]

            c2w = load_K_Rt_from_P(None, P)[1]
            w2c = np.linalg.inv(c2w)
            new_render_w2cs.append(w2c)
            new_render_c2ws.append(c2w)

        new_intrinsics, new_w2cs, new_c2ws, new_near_fars = \
            np.stack(new_intrinsics), np.stack(new_w2cs), np.stack(new_c2ws), \
            np.stack(new_near_fars)
        new_render_w2cs, new_render_c2ws = np.stack(new_render_w2cs), np.stack(new_render_c2ws)

        new_intrinsics = torch.from_numpy(np.float32(new_intrinsics))
        new_w2cs = torch.from_numpy(np.float32(new_w2cs))
        new_c2ws = torch.from_numpy(np.float32(new_c2ws))
        new_near_fars = torch.from_numpy(np.float32(new_near_fars))
        new_render_w2cs = torch.from_numpy(np.float32(new_render_w2cs))
        new_render_c2ws = torch.from_numpy(np.float32(new_render_c2ws))

        return new_intrinsics, new_w2cs, new_c2ws, new_near_fars, new_render_w2cs, new_render_c2ws


    def __len__(self):
        return self.render_idx.shape[0]

    def fetch_data(self, index):
        sample = {}

        ref_view = self.render_idx[index % len(self.render_idx)]  # [B]
        
        sample['extrinsic_render_view'] = self.all_render_w2cs_original[ref_view]
        sample['intrinsic_render_view'] = self.all_intrinsics[ref_view][:3, :3]
        sample['meta'] = "%s-%s-%08d"%(self.root_dir.split("/")[-1], self.scan_id, np.where(self.render_idx==ref_view)[0][0])
        
        if self.src_via_dist or ref_view not in self.ref_src_pairs.keys():
            src_idx_in_train = get_nearest_pose_ids(self.all_render_c2ws[ref_view].cpu().numpy(),
                                            self.all_render_c2ws[self.i_train].cpu().numpy(),
                                            self.n_src_views,
                                            tar_id=-1 if self.split != 'train' else np.where(self.i_train==ref_view)[0][0],
                                            angular_dist_method='dist')
            src_views = self.i_train[src_idx_in_train]            
            # print(ref_view, src_views)
        else:
            src_views = self.ref_src_pairs[ref_view][:self.n_src_views]
            if self.split == 'all':
                src_views = [ref_view] + src_views[:-1]
        
        view_ids = [ref_view] + list(src_views)
        sample['obj_masks'] = self.masks[ref_view].permute(1,2,0)  # [h, w, 3]
        
        w2c_ref = self.all_w2cs[ref_view]
        w2c_ref_inv = np.linalg.inv(w2c_ref)

        imgs, depths_h = [], []
        intrinsics, w2cs, render_w2cs, near_fars = [], [], [], []  # record proj mats between views
        
        for i, vid in enumerate(view_ids):
            img = self.images[vid]
            imgs += [img]
            
            near_fars.append(self.raw_near_fars[vid])
            intrinsics.append(self.all_intrinsics[vid])

            w2cs.append(self.all_w2cs[vid] @ w2c_ref_inv)   # the new "world" space is the camera space of the reference view
            render_w2cs.append(self.all_render_w2cs[vid] @ w2c_ref_inv)   # the new "world" space is the camera space of the reference view
            
            # depth_filename = os.path.join(self.root_dir,
            #                     f'Depths_raw/{scan}/depth_map_{vid:04d}.pfm')
            # if os.path.exists(depth_filename):  # and i == 0
            #     depth_h = self.read_depth(depth_filename)
            #     depths_h.append(depth_h)
        
        scale_mat, scale_factor = self.cal_scale_mat(img_hw=[self.img_wh[1]//self.supersampling, self.img_wh[0]//self.supersampling],
                                                     intrinsics=intrinsics, extrinsics=w2cs,
                                                     near_fars=near_fars, factor=1.1)
        new_near_fars = []
        new_w2cs = []
        new_render_w2cs = []
        new_c2ws = []
        # new_depths_h = []
                
        # for intrinsic, extrinsic, depth in zip(intrinsics, w2cs, depths_h):
        for intrinsic, extrinsic, render_w2c in zip(intrinsics, w2cs, render_w2cs):

            P = intrinsic @ extrinsic @ scale_mat
            # P = P[:3, :4]
            P = P.cpu().numpy()[:3, :4]
            c2w = load_K_Rt_from_P(None, P)[1]

            w2c = np.linalg.inv(c2w)
            new_w2cs.append(w2c)
            new_c2ws.append(c2w)

            camera_o = c2w[:3, 3]
            dist = np.sqrt(np.sum(camera_o ** 2))
            near = dist - 1
            far = dist + 1
            new_near_fars.append([0.95 * near, 1.05 * far])
            # new_depths_h.append(depth * scale_factor)

            P = intrinsic @ render_w2c @ scale_mat
            P = P.cpu().numpy()[:3, :4]
            c2w = load_K_Rt_from_P(None, P)[1]
            w2c = np.linalg.inv(c2w)
            new_render_w2cs.append(w2c)

        imgs = torch.stack(imgs).float()
        # depths_h = np.stack(new_depths_h)

        intrinsics, w2cs, render_w2cs, c2ws, near_fars = np.stack(intrinsics), np.stack(new_w2cs), np.stack(new_render_w2cs), np.stack(new_c2ws), np.stack(new_near_fars)

        start_idx = 0

        sample['images'] = imgs[start_idx:]  # (V, 3, H, W)
        sample['w2cs'] = torch.from_numpy(w2cs.astype(np.float32))[start_idx:]  # (V, 4, 4)
        sample['render_w2cs'] = torch.from_numpy(render_w2cs.astype(np.float32))[start_idx:]  # (V, 4, 4)
        sample['c2ws'] = torch.from_numpy(c2ws.astype(np.float32))[start_idx:]  # (V, 4, 4)
        # sample['near_fars'] = torch.from_numpy(near_fars.astype(np.float32))[start_idx:]  # (V, 2)
        sample['intrinsics'] = torch.from_numpy(intrinsics.astype(np.float32))[start_idx:, :3, :3]  # (V, 3, 3)
        sample['K'] = torch.from_numpy(intrinsics.astype(np.float32))[0]  # (4, 4)

        sample['scale_mat'] = torch.from_numpy(scale_mat)
        sample['trans_mat'] = torch.from_numpy(w2c_ref_inv)

        sample['ref_img'] = sample['images'][0] # 3, 512, 640
        sample['source_imgs'] = sample['images'][1:] # 3, 3, 512, 640

        intrinsics_pad = repeat(torch.eye(4), "X Y -> L X Y", L = len(sample['w2cs'])).clone()
        intrinsics_pad[:,:3,:3] = sample['intrinsics']

        sample['ref_pose'] = (intrinsics_pad @ sample['render_w2cs'])[0]     # 4, 4
        sample['source_poses'] = (intrinsics_pad @ sample['w2cs'])[1:] 
                
        # from 0~W to NDC's -1~1
        normalize_matrix = torch.tensor([[1/((self.img_W-1)/2), 0, -1, 0], [0, 1/((self.img_H-1)/2), -1, 0], [0,0,1,0], [0,0,0,1]]).to(sample['ref_pose'])    
        sample['ref_pose'] = normalize_matrix @ sample['ref_pose']
        sample['ref_pose_inv'] = torch.inverse(sample['ref_pose'])
        
        normalize_matrix_src= torch.tensor([[1/((self.img_W//self.supersampling-1)/2), 0, -1, 0], [0, 1/((self.img_H//self.supersampling-1)/2), -1, 0], [0,0,1,0], [0,0,0,1]]).to(sample['ref_pose']) 
        sample['source_poses'] = normalize_matrix_src @ sample['source_poses']
        sample['source_poses_inv'] = torch.inverse(sample['source_poses'])
        
        sample['ray_o'] = sample['ref_pose_inv'][:3,-1]  # 3

        tmp_ray_d = (sample['ref_pose_inv'] @ self.homo_pixel)[:3] - sample['ray_o'][:,None]
        tmp_ray_d = tmp_ray_d / torch.linalg.norm(tmp_ray_d, dim=0, keepdim=True)
        sample['ray_d'] = tmp_ray_d

        cam_ray_d = (torch.inverse(normalize_matrix @ intrinsics_pad[0]) @ self.homo_pixel)[:3]
        cam_ray_d = cam_ray_d / torch.linalg.norm(cam_ray_d, dim=0, keepdim=True)
        sample['cam_ray_d'] = cam_ray_d
        
        sample['cam2ndc'] = normalize_matrix @ intrinsics_pad[0]
        
        ### visualize source views
        # import imageio
        # imageio.imwrite("/".join(['.', "target.jpg"]), (sample["ref_img"].permute(1,2,0) * 255).cpu().numpy().astype(np.uint8))
        # for k in range(sample["source_imgs"].shape[0]):
        #     imageio.imwrite("/".join(['.', "src_%d.jpg" % (k,)]), (sample["source_imgs"][k].permute(1,2,0) * 255).cpu().numpy().astype(np.uint8))
        # print(f'Saved!!!')
        # input()


        # print("extrinsic_render_view:", sample['extrinsic_render_view'])
        # print("intrinsics:", sample['intrinsics'])
        # print("ref_w2c:", w2c_ref)
        # print("scale_mat:", sample['scale_mat'])
        # print("trans_mat:", sample['trans_mat'])
        # print("ref_pose:", sample['ref_pose'])
        # print("source_poses:", sample['source_poses'])
        # print("ray_o:", sample['ray_o'])
        # print("ray_d:", sample['ray_d'])
        # print("cam_ray_d:", sample['cam_ray_d'])
        # input()
        
        return sample
    

    @torch.no_grad()
    def __getitem__(self, index):
        
        # sample = {'rays': self.all_rays[idx],
        #           'rgbs': self.all_rgbs[idx]}
        data = self.fetch_data(index)

        if self.color_bkgd_aug == "random":
            color_bkgd = torch.rand(3, device=self.images.device)
        elif self.color_bkgd_aug == "white":
            color_bkgd = torch.ones(3, device=self.images.device)
        elif self.color_bkgd_aug == "black":
            color_bkgd = torch.zeros(3, device=self.images.device)

        data['color_bkgd'] = color_bkgd
        data['rays'] = Rays(origins=data["ray_o"], viewdirs=data["ray_d"])
        data['pixels'] = torch.permute(data["ref_img"], (1,2,0))

                    
        # print('intrinsics:', data['intrinsics'])
        # print('w2cs:', data['w2cs'])
        # print('scale_mat:', data['scale_mat'])
        # print('trans_mat:', data['trans_mat'])
        # print('near_fars:', data['near_fars'])
        # print('ref_pose:', data['ref_pose'])
        # print('ray_o:', data['ray_o'])
        # print('ray_d:', data['ray_d'])
        # input()
        
        return data