import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses, SyncBatchNorm
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import test_batchify_fn 
from encoding.models.sseg import BaseNet

if __name__ == '__main__':
    from lseg_encoder.additional_utils.models import LSeg_MultiEvalModule
    from lseg_encoder.additional_utils.encoding_models import MultiEvalModule
    from lseg_encoder.modules.lseg_module import LSegModule
else:
    from src.lseg_encoder.additional_utils.models import LSeg_MultiEvalModule
    from src.lseg_encoder.additional_utils.encoding_models import MultiEvalModule
    from src.lseg_encoder.modules.lseg_module import LSegModule
    
import math
import types
import functools
import torchvision.transforms as torch_transforms
import copy
import itertools
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import clip
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
# from data import get_dataset
import torchvision.transforms as transforms



def get_new_pallete(num_cls):
    n = num_cls
    pallete = [0]*(n*3)
    for j in range(0,n):
            lab = j
            pallete[j*3+0] = 0
            pallete[j*3+1] = 0
            pallete[j*3+2] = 0
            i = 0
            while (lab > 0):
                    pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                    pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                    pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                    i = i + 1
                    lab >>= 3
    return pallete


def get_new_mask_pallete(npimg, new_palette, out_label_flag=False, labels=None):
    """Get image color pallete for visualizing masks"""
    # put colormap
    out_img = Image.fromarray(npimg.squeeze().astype('uint8'))
    out_img.putpalette(new_palette)

    if out_label_flag:
        assert labels is not None
        u_index = np.unique(npimg)
        patches = []
        for i, index in enumerate(u_index):
            label = labels[index]
            cur_color = [new_palette[index * 3] / 255.0, new_palette[index * 3 + 1] / 255.0, new_palette[index * 3 + 2] / 255.0]
            red_patch = mpatches.Patch(color=cur_color, label=label)
            patches.append(red_patch)
    return out_img, patches


class CLIP_MODEL(nn.Module):
    """
    Ray transformer
    """
    def __init__(self, label_src=None):
        super().__init__()

        self.alpha = 0.5
            
        scale_inv = False
        widehead = True
        dataset = 'ignore' # 'ade20k'
        backbone = 'clip_vitl16_384'
        weights = '/data/yfu314/checkpoints/demo_e200.ckpt'
        ignore_index = 255
        data_path = None # '../datasets/'

        module = LSegModule.load_from_checkpoint(
            checkpoint_path=weights,
            data_path=data_path,
            dataset=dataset,
            backbone=backbone,
            aux=False,
            num_features=256,
            aux_weight=0,
            se_loss=False,
            se_weight=0,
            base_lr=0,
            batch_size=1,
            max_epochs=0,
            ignore_index=ignore_index,
            dropout=0.0,
            scale_inv=scale_inv,
            augment=False,
            no_batchnorm=False,
            widehead=widehead,
            widehead_hr=False,
            map_locatin="cpu",
            arch_option=0,
            block_depth=0,
            activation='lrelu',
        )

        if isinstance(module.net, BaseNet):
            model = module.net
        else:
            model = module
    
        model = model.eval()
        model = model.cpu()
        
        # scales = (
        #     [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
        #     if dataset == "citys"
        #     else [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        # )  
        
        scales = [0.75, 1.0, 1.25, 1.75]

        model.mean = [0.5, 0.5, 0.5]
        model.std = [0.5, 0.5, 0.5]

        self.evaluator = LSeg_MultiEvalModule(
            model, scales=scales, flip=True
        ).cuda()

        # self.evaluator = MultiEvalModule(
        #     model, scales=scales, flip=True
        # ).cuda()

        self.evaluator.eval()
        
        self.mean = torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1).cuda()
        self.std = torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1).cuda()
        
        self.label_src = label_src if label_src is not None else 'plant,grass,cat,stone,other'

        self.labels = []
        lines = self.label_src.split(',')
        for line in lines:
            label = line
            self.labels.append(label)
        

    def get_feature(self, images):   
        images = (images - self.mean) / self.std
             
        with torch.no_grad():
            image_features = self.evaluator.parallel_forward(images, self.labels, return_feature=True)  #evaluator.forward(image, labels) #parallel_forward
            
        return image_features
    
    
    def forward_feature(self, image_features):        
        with torch.no_grad():
            outputs = self.evaluator.parallel_forward(image_features, self.labels, input_feature=True)  #evaluator.forward(image, labels) #parallel_forward
            #outputs = model(image,labels)
            predicts = [
                torch.max(output, 1)[1].cpu().numpy() 
                for output in outputs
            ]
            
        return predicts
    
    
    def forward(self, images):
        images = (images - self.mean) / self.std
        
        with torch.no_grad():
            outputs = self.evaluator.parallel_forward(images, self.labels)  #evaluator.forward(image, labels) #parallel_forward
            #outputs = model(image,labels)
            predicts = [
                torch.max(output, 1)[1].cpu().numpy() 
                for output in outputs
            ]
            
        return predicts
    
    
    def visualize(self, images, predicts):
        for i, img in enumerate(images):
            predict = predicts[i]
            
            img = img.permute(1,2,0).cpu().numpy()
            
            new_palette = get_new_pallete(len(self.labels))
            mask, patches = get_new_mask_pallete(predict, new_palette, out_label_flag=True, labels=self.labels)

            img = Image.fromarray(np.uint8(255*img)).convert("RGBA")
            seg = mask.convert("RGBA")
            out = Image.blend(img, seg, self.alpha)
            
            plt.axis('off')
            plt.imsave(f'./rgb-{i}.png', np.array(img))
            
            plt.figure()
            plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.5, 1), prop={'size': 20})
            plt.axis('off')
            plt.imsave(f'./seg-{i}.png', np.array(seg))


if __name__ == '__main__':
    # img_path = "/home/yfu314/VolRecon/src/lseg_encoder/cat1.jpeg"
    # label_src="plant,grass,cat,stone,other"

    # img_path = "/data/yfu314/dataset_nerf/nerf_llff_data/room/images_4/DJI_20200226_143850_006.png"
    # label_src="desk,tv,wall,ground,roof,chairs,bin,light,other"
    
    img_path = "/data/yfu314/dataset_nerf/nerf_llff_data/flower/images_4/image000.png"
    label_src="other,red flowers"
    
    # img_path = "/data/yfu314/dataset_nerf/nerf_synthetic/lego/val/r_39.png"
    # label_src="other,excavator"

    
    clip_model = CLIP_MODEL(label_src=label_src)
    
    image = Image.open(img_path)
    image = np.array(image)/255
    image = torch.tensor(image).unsqueeze(0).permute(0,3,1,2).float().cuda()
    image = image[:,:3,:,:]
        
    # predicts = clip_model(image)
    
    image_features = clip_model.get_feature(image)[0]
    predicts = clip_model.forward_feature(image_features)
    
    clip_model.visualize(image, predicts)
    