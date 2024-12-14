"""
Adapted from Ruilong Li, UC Berkeley.
"""

import collections
import json
import os
import math

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

import collections
Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))

def _load_K(kname: str):
    lines = [[float(w) for w in line.strip().split()] for line in open(kname)]
    return np.array(lines).astype(np.float32)[:3,:3]

def _parse_pose(cname: str):
    lines = [[float(w) for w in line.strip().split()] for line in open(cname)]
    if len(lines[0]) == 2:
        lines = lines[1:]
    if len(lines[-1]) == 2:
        lines = lines[:-1]
    return np.array(lines).astype(np.float32)


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
    # with open(
    #     os.path.join(data_dir, "transforms_{}.json".format(split)), "r"
    # ) as fp:
    #     meta = json.load(fp)
    raw_rgb_path = os.listdir('/'.join([data_dir,'rgb']))
    prefix_split_table = {'train': '0',
                           'test': '1',}
    prefix = prefix_split_table[split]
    img_names = []
    for p in raw_rgb_path:
        if p[0] == prefix:
            img_names.append(p[:-4])

    images = []
    camtoworlds = []

    for name in img_names:
        fname = '/'.join([data_dir, 'rgb', name + ".png"])
        rgba = imageio.imread(fname)
        if rgba.shape[-1] == 3:
            # deal with Steamtrain which do not have alpha channel
            rgba = np.pad(rgba, ((0,0),(0,0),(0,1)), constant_values=255)
        cname = '/'.join([data_dir, 'pose', name + ".txt"])
        c2w = _parse_pose(cname)
        images.append(rgba)
        camtoworlds.append(c2w)

    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)

    return images, camtoworlds


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "test"]
    SUBJECT_IDS = [
        "Jade",
        "Fountain",
        "Statues",
        "Character",
    ]

    WIDTH, HEIGHT = 768, 576
    OPENGL_CAMERA = False

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "white",
        batch_size: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        supersampling = 1,
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.num_images = batch_size
        # self.near = self.NEAR if near is None else near
        # self.far = self.FAR if far is None else far
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        self.supersampling = supersampling
        if self.supersampling != 1:
            self.HEIGHT = self.HEIGHT*self.supersampling
            self.WIDTH = self.WIDTH*self.supersampling
        self.images, self.camtoworlds = _load_renderings(
            root_fp, subject_id, split
        )
        self.images = torch.from_numpy(self.images).to(torch.uint8)
        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        self.K = torch.tensor(
            _load_K('/'.join([root_fp, subject_id, 'intrinsics.txt'])),
            dtype=torch.float32,
        )  # (3, 3)
        if self.supersampling:
            assert self.images.shape[1:3] == (self.HEIGHT/self.supersampling, self.WIDTH/self.supersampling)

        if subject_id == 'Character':
            self.HAS_CLOSE = True

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def ndc_y_rescale(self, vtx):
        vtx[...,1] *= (self.HEIGHT*self.supersampling)/(math.ceil(self.HEIGHT*self.supersampling/8.)*8)
        return vtx

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
        return {
            "pixels": pixels,  # [n, h, w, 4]
            "rays": rays,  # [n, h, w, 3]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},  # [n, 4, 4]
        }

    def fetch_data(self, index):
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

        origins = torch.reshape(origins, (num_images, self.HEIGHT, self.WIDTH, 3))
        viewdirs = torch.reshape(viewdirs, (num_images, self.HEIGHT, self.WIDTH, 3))
        rgba = torch.reshape(rgba, (num_images, self.HEIGHT//self.supersampling, self.WIDTH//self.supersampling, 4))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        w2c = torch.linalg.inv(c2w)

        return {
            "rgba": rgba,   # [n, h, w, 4]
            "rays": rays,   # [n, h, w, 3]
            "w2cs": w2c,     # [n, 4, 4]
        }

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