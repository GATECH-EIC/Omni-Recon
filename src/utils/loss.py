import torch
import torch.nn as nn


class DepthLoss():
    default_cfg={
        'depth_correct_thresh': 0.02,
        'depth_loss_type': 'l2',
        'depth_loss_l1_beta': 0.05,
    }
    def __init__(self):
        self.cfg = self.default_cfg
        if self.cfg['depth_loss_type']=='smooth_l1':
            self.loss_op=nn.SmoothL1Loss(reduction='none',beta=self.cfg['depth_loss_l1_beta'])
            
    def process(self, depth, near, far):
            depth = torch.clamp(depth, min=1e-5)
            depth = -1 / depth
            depth = (depth - near) / (far - near)
            depth = torch.clamp(depth, min=0, max=1.0)
            return depth
        

    def __call__(self, depth_pred, depth_gt, depth_range):
        near, far = -1/depth_range[:,0], -1/depth_range[:,1]
        
        depth_pred = self.process(depth_pred, near, far)
        depth_gt = self.process(depth_gt, near, far)

        if self.cfg['depth_loss_type']=='l2':
            loss = (depth_gt - depth_pred)**2
        elif self.cfg['depth_loss_type']=='smooth_l1':
            loss = self.loss_op(depth_gt, depth_pred)

        loss = torch.mean(loss)
        
        return loss

