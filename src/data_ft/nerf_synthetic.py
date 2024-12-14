"""
Adapted from Ruilong Li, UC Berkeley.
"""

import collections
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat

import collections
from .data_utils import get_nearest_pose_ids
Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))

from .utils import get_spiral_fn

def _load_renderings(root_fp: str, subject_id: str, split: str):
    """Load images from disk."""
    if not root_fp.startswith("/"):
        # allow relative path. e.g., "./data/nerf_synthetic/"
        root_fp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            root_fp,
        )

    data_dir = os.path.join(root_fp, subject_id)
    with open(
        os.path.join(data_dir, "transforms_{}.json".format(split)), "r"
    ) as fp:
        meta = json.load(fp)
    images = []
    camtoworlds = []

    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        fname = os.path.join(data_dir, frame["file_path"] + ".png")
        rgba = imageio.imread(fname)
        c2w = frame["transform_matrix"]
        
        ## blender to opencv
        w2c_blender = np.linalg.inv(c2w)
        w2c_opencv = w2c_blender
        w2c_opencv[1:3] *= -1
        c2w = np.linalg.inv(w2c_opencv)
        
        camtoworlds.append(c2w)
        
        images.append(rgba)

    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)

    h, w = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    return images, camtoworlds, focal


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "val", "trainval", "test"]
    SUBJECT_IDS = [
        "chair",
        "drums",
        "ficus",
        "hotdog",
        "lego",
        "materials",
        "mic",
        "ship",
    ]

    WIDTH, HEIGHT = 800, 800
    NEAR, FAR = 2.0, 6.0
    OPENGL_CAMERA = False

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        n_src_views = 3,
        color_bkgd_aug: str = "white",
        batch_size: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        supersampling = 1,
        get_spiral: bool = False,
        baking_only: bool = False,
        use_dataloader: bool = False
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.num_images = batch_size
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        self.supersampling = supersampling
        
        self.n_src_views = n_src_views
        
        self.baking_only = baking_only
        
        self.use_dataloader = use_dataloader
        
        if self.supersampling != 1:
            self.HEIGHT = self.HEIGHT*self.supersampling
            self.WIDTH = self.WIDTH*self.supersampling
            
        if split == "trainval":
            _images_train, _camtoworlds_train, _focal_train = _load_renderings(
                root_fp, subject_id, "train"
            )
            _images_val, _camtoworlds_val, _focal_val = _load_renderings(
                root_fp, subject_id, "val"
            )
            self.images = np.concatenate([_images_train, _images_val])
            self.camtoworlds = np.concatenate(
                [_camtoworlds_train, _camtoworlds_val]
            )
            self.focal = _focal_train
        else:
            self.images, self.camtoworlds, self.focal = _load_renderings(
                root_fp, subject_id, split
            )
        if get_spiral:
            self.camtoworlds = get_spiral_fn(self.camtoworlds[:,:3])
            # self.camtoworlds[:2,3] += 3
            self.images = np.repeat(self.images[0:1], self.camtoworlds.shape[0], axis=0)
        self.images = torch.from_numpy(self.images).to(torch.uint8)
        
        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)  # [N, 3, 4]
        self.K = torch.tensor(
            [
                [self.focal, 0, self.WIDTH / 2.0],
                [0, self.focal, self.HEIGHT / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )  # (3, 3)
        if self.supersampling:
            assert self.images.shape[1:3] == (self.HEIGHT/self.supersampling, self.WIDTH/self.supersampling)

        ## source views
        self.images_source, self.camtoworlds_source, focal_source = _load_renderings(root_fp, subject_id, "test")  # assume the same focul across all splits
        self.images_source = torch.from_numpy(self.images_source).to(torch.uint8)
        self.camtoworlds_source = torch.from_numpy(self.camtoworlds_source).to(torch.float32)

        self.normalize_matrix = torch.tensor([[1/((self.WIDTH-1)/2), 0, -1, 0], [0, 1/((self.HEIGHT-1)/2), -1, 0], [0,0,1,0], [0,0,0,1]]).to(self.camtoworlds_source)
        self.normalize_matrix_source = torch.tensor([[1/((self.WIDTH//self.supersampling-1)/2), 0, -1, 0], [0, 1/((self.HEIGHT//self.supersampling-1)/2), -1, 0], [0,0,1,0], [0,0,0,1]]).to(self.camtoworlds_source)

        self.intrinsics_pad = repeat(torch.eye(4), "X Y -> L X Y", L = self.camtoworlds_source.shape[0]).clone()
        self.intrinsics_pad[:,:3,:3] = self.K[:3, :3]     
        self.intrinsics_pad = self.intrinsics_pad.to(self.camtoworlds_source)

        # self.img_W, self.img_H = self.WIDTH, self.HEIGHT
        # h_line = torch.linspace(0,self.img_H-1,self.img_H, device=self.camtoworlds_source.device)*2/(self.img_H-1) - 1
        # w_line = torch.linspace(0,self.img_W-1,self.img_W, device=self.camtoworlds_source.device)*2/(self.img_W-1) - 1
        # h_mesh, w_mesh = torch.meshgrid(h_line, w_line, indexing='ij')
        # self.w_mesh_flat = w_mesh.flatten()
        # self.h_mesh_flat = h_mesh.flatten()
        # self.homo_pixel = torch.stack([self.w_mesh_flat, self.h_mesh_flat, torch.ones(self.h_mesh_flat.shape[0]).to(self.h_mesh_flat), torch.ones(self.h_mesh_flat.shape[0]).to(self.h_mesh_flat)])  # [4, h*w]

        self.cuda_tensors = ["images", "camtoworlds", "K", "images_source", "camtoworlds_source"]
        

    def __len__(self):
        return len(self.images) if not self.baking_only else 1

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        pixels, alpha = torch.split(rgba, [3, 1], dim=-1)

        if self.color_bkgd_aug == "random":
            color_bkgd = torch.rand(3, device=self.images.device)
        elif self.color_bkgd_aug == "white":
            color_bkgd = torch.ones(3, device=self.images.device)
        elif self.color_bkgd_aug == "black":
            color_bkgd = torch.zeros(3, device=self.images.device)

        pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        
        data["source_imgs"] = data["source_imgs"][:,:,:3,:,:] * data["source_imgs"][:,:,3:4,:,:] + color_bkgd[0] * (1-data["source_imgs"][:,:,3:4,:,:])
        
        data["pixels"] = pixels  # [n, h, w, 3]
        data['ref_img'] = torch.permute(pixels, (0,3,1,2)) # [n, 3, h, w]
        data["color_bkgd"] = color_bkgd  # [3,]

        # ## visualize source views
        # import imageio
        # imageio.imwrite("/".join(['.', "target.jpg"]), (data["pixels"][0] * 255).cpu().numpy().astype(np.uint8))
        # src_imgs = data["source_imgs"][0]
        # for k in range(src_imgs.shape[0]):
        #     imageio.imwrite("/".join(['.', "src_%d.jpg" % (k,)]), (src_imgs[k].permute(1,2,0) * 255).cpu().numpy().astype(np.uint8))
        # print('saved!!!')
        # input()
        
        if self.use_dataloader:
            assert data['ref_img'].shape[0] == 1
            
            for key in data:
                if key == "rays":
                    data['rays'] = Rays(origins=data["ray_o"][0], viewdirs=data["ray_d"][0])
                elif data[key].shape[0] == 1:
                    data[key] = data[key][0]
        
        return data

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        if type(index) is int:
            index = [index]
            
        image_id = index
        num_images = len(image_id)
        
        x, y = torch.meshgrid(
            torch.arange(self.WIDTH, device=self.images.device),
            torch.arange(self.HEIGHT, device=self.images.device),
            indexing="xy",
        )
        x = x.flatten()
        y = y.flatten()

        # generate rays
        rgba = self.images[image_id] / 255.0  # (num_images, H, W, 4)
        c2w : torch.Tensor = self.camtoworlds[image_id]  # (num_images, 3, 4)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [H*W, 3]


        directions = (camera_dirs[None, :, None, :] * c2w[:, None, :3, :3]).sum(dim=-1) # (N, H*W, 3)
        origins = torch.broadcast_to(c2w[:, None, :3, -1], directions.shape) # (N, H*W, 3)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )
        
        data = {}
        
        w2c = torch.linalg.inv(c2w)
        
        data['ref_pose'] = self.normalize_matrix @ self.intrinsics_pad[image_id] @ w2c   # B, 4, 4
        data['ref_pose_inv'] = torch.inverse(data['ref_pose'])
        
        # print('ref_pose_inv:', data['ref_pose_inv'][:, :3, -1])
        # print('c2w:', c2w[:, :3, -1])

        data["ray_o"] = data['ref_pose_inv'][:, :3, -1] # (N, 3)
        data["ray_d"] = viewdirs.permute(0,2,1) # (N, 3, H*W)

        # homo_pixel = self.homo_pixel[None, ...].repeat(len(index), 1, 1)  # (B, 4, HW)
        # tmp_ray_d = (data['ref_pose_inv'] @ homo_pixel)[:, :3] - data['ray_o'][:,:, None]  # [B, 3, h*w]
        # data['ray_d'] = tmp_ray_d / torch.norm(tmp_ray_d, dim=1, keepdim=True) # [B, 3, h*w]
        # data['ray_d'] = data['ray_d'].float()  # [B, 3, h*w]
        
        origins = torch.reshape(origins, (num_images, self.HEIGHT, self.WIDTH, 3))
        viewdirs = torch.reshape(viewdirs, (num_images, self.HEIGHT, self.WIDTH, 3))
        rgba = torch.reshape(rgba, (num_images, self.HEIGHT//self.supersampling, self.WIDTH//self.supersampling, 4))

        rays = Rays(origins=origins, viewdirs=viewdirs)
        
        data["rgba"] = rgba # [n, h, w, 4]
        data["rays"] = rays # [n, h, w, 3]
        data["w2cs"] = w2c  # [n, 4, 4]
        
        data["near_fars"] = np.array([[self.near, self.far] for _ in index])

        ## source view selection
        source_imgs = []
        source_poses = []
        source_poses_inv = []
        
        if not self.baking_only:
            # ## use target img as source image for debugging
            # src_img = torch.permute(data["rgba"], (0,3,1,2))
            # source_imgs.append(src_img)
            
            # source_pose = data['ref_pose']
            # source_pose_inv = data['ref_pose_inv']

            # source_poses.append(source_pose)
            # source_poses_inv.append(source_pose_inv)
                
            for i, idx in enumerate(index):
                src_idx = get_nearest_pose_ids(c2w.cpu().numpy(),
                                                    self.camtoworlds_source.cpu().numpy(),
                                                    self.n_src_views,
                                                    tar_id=-1 if 'train' not in self.split else idx,
                                                    angular_dist_method='vector')

                src_img = torch.permute(self.images_source[src_idx], (0,3,1,2)) / 255.0
                source_imgs.append(src_img)
                
                source_pose = self.normalize_matrix_source @ self.intrinsics_pad[src_idx] @ torch.inverse(self.camtoworlds_source[src_idx])  # convert to w2c matrix and multiple with the intrinsic & normalization matrices
                source_pose_inv = torch.inverse(source_pose)

                source_poses.append(source_pose)
                source_poses_inv.append(source_pose_inv)
                # source_poses_inv.append(self.camtoworlds_source[src_idx])  # here source_poses_inv denotes the c2w matrix, which is self.camtoworlds_source
                
                # print('source_pose_inv', source_pose_inv[:, :3, -1])
                # print('c2w:', self.camtoworlds_source[src_idx][:, :3, -1])
                    
            data['source_imgs'] = torch.stack(source_imgs)
            data['source_poses'] = torch.stack(source_poses)
            data['source_poses_inv'] = torch.stack(source_poses_inv)
            
        else:
            data['source_imgs'] = torch.permute(self.images_source, (0,3,1,2)) / 255.0
            data['source_poses'] = self.normalize_matrix_source @ self.intrinsics_pad @ torch.inverse(self.camtoworlds_source)
            # data['source_poses_inv'] = self.camtoworlds_source
            
            data['source_poses_inv'] = torch.inverse(self.normalize_matrix_source @ self.intrinsics_pad @ torch.inverse(self.camtoworlds_source))
            
            data['source_imgs'] = data['source_imgs'].unsqueeze(0)
            data['source_poses'] = data['source_poses'].unsqueeze(0)
            data['source_poses_inv'] = data['source_poses_inv'].unsqueeze(0)
        
        # print('w2cs:', data['w2cs'])
        # print('near_fars:', data['near_fars'])
        # print('ref_pose:', data['ref_pose'])
        # print('ray_o:', data['ray_o'])
        # print('ray_d:', data['ray_d'])
        # input()
        
        return data


    def fetch_rays(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        image_id = index
        num_images = len(image_id)
        x, y = torch.meshgrid(
            torch.arange(self.WIDTH, device=self.images.device),
            torch.arange(self.HEIGHT, device=self.images.device),
            indexing="xy",
        )
        x = x.flatten()
        y = y.flatten()

        # generate rays
        c2w : torch.Tensor = self.camtoworlds[image_id]  # (num_images, 3, 4)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [H*W, 3]

        directions = (camera_dirs[None, :, None, :] * c2w[:, None, :3, :3]).sum(dim=-1) # (N, H*W, 3)
        origins = torch.broadcast_to(c2w[:, None, :3, -1], directions.shape) # (N, H*W, 3)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        origins = torch.reshape(origins, (num_images, self.HEIGHT, self.WIDTH, 3))
        viewdirs = torch.reshape(viewdirs, (num_images, self.HEIGHT, self.WIDTH, 3))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return rays,   # [n, h, w, 3]