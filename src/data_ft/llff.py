"""
Adapted from Anpei Chen, https://github.com/apchenstu/TensoRF/blob/main/dataLoader/llff.py
"""

import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import math
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *
import collections
Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))



def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


class SubjectLoader(Dataset):

    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "test"]
    SUBJECT_IDS = [
        "fern",
        "flower",
        "fortress",
        "horns",
        "leaves",
        "orchids",
        "room",
        "trex",
    ]

    WIDTH, HEIGHT = 1008, 756
    NEAR, FAR = 1.0, -1.0
    OPENGL_CAMERA = True

    def __init__(
        self, 
        subject_id: str,
        root_fp: str,
        color_bkgd_aug: str = "white",
        split: str = "train", 
        batch_size: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        downsample=4,
        is_stack=True, 
        hold_every=8,
        supersampling=1,
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        self.root_dir = '/'.join([root_fp, subject_id])
        self.split = split
        self.num_images = batch_size
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        self.hold_every = hold_every
        self.is_stack = is_stack
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        self.downsample = downsample
        self.supersampling = supersampling

        self.define_transforms()

        self.blender2opencv = np.eye(4)#np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()

        self.scene_bbox = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])
        # self.scene_bbox = torch.tensor([[-1.67, -1.5, -1.0], [1.67, 1.5, 1.0]])
        # self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        # self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.avg_camera = torch.eye(4)[None, ...]

        self.images = self.all_rgbs
        # self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        self.K = torch.tensor(
            [
                [self.focal[0], 0, self.WIDTH / 2.0],
                [0, self.focal[1], self.HEIGHT / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )  # (3, 3)
        assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)

    def reverse_ndc(self, vtx_3dpos: torch. Tensor):
        ax = -(2*self.focal[0]) / (self.WIDTH)
        ay = -(2*self.focal[1]) / (self.HEIGHT)
        az = 1
        bz = 2*self.near

        zs = bz / (vtx_3dpos[...,2:3] - az)
        xs = vtx_3dpos[...,0:1]*zs/ax
        ys = vtx_3dpos[...,1:2]*zs/ay

        return torch.concat([xs,ys,zs], -1)
    
    def ndc_y_rescale(self, vtx):
        vtx[...,1] *= (self.HEIGHT*self.supersampling)/(math.ceil(self.HEIGHT*self.supersampling/8.)*8)
        return vtx

    def read_meta(self):
        poses_bounds = np.load(os.path.join(self.root_dir, 'poses_bounds.npy'))  # (N_images, 17)
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images_4/*')))
        # load full resolution image then resize
        if self.split in ['train', 'test']:
            assert len(poses_bounds) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        self.near_fars = poses_bounds[:, -2:]  # (N_images, 2)

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1]  # original intrinsics, same for all images
        self.img_wh = np.array([int(W / self.downsample), int(H / self.downsample)])
        self.focal = [self.focal * self.img_wh[0] / W, self.focal * self.img_wh[1] / H]

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        self.poses, self.pose_avg = center_poses(poses, self.blender2opencv)

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.near_fars.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        self.near_fars /= scale_factor
        self.poses[..., 3] /= scale_factor

        # ray directions for all pixels, same for all images (same H, W, focal)
        W, H = self.img_wh
        self.directions = get_ray_directions_blender(H, W, self.focal, supersampling=int(self.supersampling))  # (H, W, 3)

        i_test = np.arange(0, self.poses.shape[0], self.hold_every)  # [np.argmin(dists)]
        img_list = i_test if self.split != 'train' else list(set(np.arange(len(self.poses))) - set(i_test))

        # use first N_images-1 to train, the LAST is val
        self.all_rays = []
        self.all_rgbs = []
        self.camtoworlds = []
        for i in img_list:
            image_path = self.image_paths[i]
            c2w = torch.FloatTensor(self.poses[i])
            self.camtoworlds.append(c2w)

            img = Image.open(image_path).convert('RGB')
            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)

            img = self.transform(img)  # (3, h, w)

            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
            self.all_rgbs += [img]
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            rays_o, rays_d = ndc_rays_blender(H, W, self.focal[0], 1.0, rays_o, rays_d)
            # viewdir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w,3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0).reshape(-1,int(self.HEIGHT*self.supersampling), int(self.WIDTH*self.supersampling), 6)   # (len(self.meta['frames]),h,w, 6)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)

        self.camtoworlds = torch.stack(self.camtoworlds, 0)


    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.images)

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        image_id = index
        num_images = len(index)
        x, y = torch.meshgrid(
            torch.arange(self.WIDTH*int(self.supersampling), device=self.images.device),
            torch.arange(self.HEIGHT*int(self.supersampling), device=self.images.device),
            indexing="xy",
        )
        x = x.flatten()
        y = y.flatten()

        # generate rays
        rgb = self.images[image_id]  # (num_images, H, W, 4)
        c2w = self.camtoworlds[image_id]  # (num_images, 3, 4)
        c2w = torch.concat([c2w, c2w.new_zeros([c2w.shape[0],1,4])], dim=1)
        c2w[:,-1,-1] = 1
        rays = self.all_rays[image_id]
    
        origins = torch.reshape(rays[...,:3], (num_images, self.HEIGHT*int(self.supersampling), self.WIDTH*int(self.supersampling), 3))
        viewdirs = torch.reshape(rays[...,3:], (num_images, self.HEIGHT*int(self.supersampling), self.WIDTH*int(self.supersampling), 3))
        rgb = torch.reshape(rgb, (num_images, int(self.HEIGHT), int(self.WIDTH), 3))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        w2c = torch.linalg.inv(c2w)

        return {
            "rgb": rgb,     # [n, h, w, 4] 
            "rays": rays,   # [n, h, w, 3]
            "w2cs": w2c,    # [n, 4, 4]
        }

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
        data['pixels'] = data['rgb']
        data['rays'] = Rays(data['rays'][0].to(self.images.device), data['rays'][1].to(self.images.device))
        return data