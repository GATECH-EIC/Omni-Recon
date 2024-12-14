import argparse
from re import I
import os
import sys
import glob
from stat import UF_OPAQUE
from tqdm import tqdm
import math

import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import ProgressBar
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint

from src.model import OmniRecon
from src.data_pretrain.dtu_train import MVSDataset
from src.data_pretrain.dtu_test_sparse import DtuFitSparse
from src.data_ft.dtu import DTU_Finetune
from src.data_pretrain.general_fit import GeneralFit

from src.data_pretrain.train_dataset_scale import GeneralRendererDataset_Scale
from src.data_pretrain.ft_dataset import FtRendererDataset

from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ProgressBar

from pytorch_lightning.utilities.types import STEP_OUTPUT


PI = math.pi
device = "cuda" if torch.cuda.is_available() else "cpu"


def find_latest_checkpoint(checkpoint_dir):
    # List all checkpoint files in the directory
    # Assuming the checkpoint files have '.ckpt' extension
    list_of_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
    
    if not list_of_files:  # Check if list is empty
        return None
    
    # Find the checkpoint file with the latest modification time
    latest_file = max(list_of_files, key=os.path.getmtime)
    
    return latest_file


if __name__ == "__main__":

    seed_everything(0, workers=True)

    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', dest='root_dir', type=str,
        help='directory of training dataset')
    parser.add_argument('--load_ckpt', dest='load_ckpt', type=str, default=None,
        help='load pretrained lightning ckpt')
    parser.add_argument('--train_ray_num', dest='train_ray_num', type=int, default=1024,
        help='ray number in one image')
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002,
        help='learning rate')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=2,
        help='batch size')
    parser.add_argument('--max_epochs', dest='max_epochs', type=int, default=16,
        help='max num of epochs')
    parser.add_argument('--max_iters', dest='max_iters', type=int, default=200000,
        help='max num of iters')
    parser.add_argument('--val_only', dest='val_only', action="store_true",
        help='only validate')

    parser.add_argument('--volume_reso', dest='volume_reso', type=int, default=144, 
        help="3D feature volume resolution") # set as 0 to disable

    parser.add_argument('--coarse_sample', dest='coarse_sample', type=int, default=64,
        help='number of coarse samples during training')
    parser.add_argument('--fine_sample', dest='fine_sample', type=int, default=64,
        help='number of fine samples during training')
    # loss weights
    parser.add_argument('--weight_rgb', dest='weight_rgb', type=float, default=1.0)
    parser.add_argument('--weight_depth', dest='weight_depth', type=float, default=1.0)
    parser.add_argument('--logdir', default='./checkpoints', help='the directory to save checkpoints/logs')

    parser.add_argument('--test_dir', dest='test_dir', type=str,
        help='directory of test dataset')
    parser.add_argument('--out_dir', dest='out_dir', type=str,
        help='directory of to save test result')
    parser.add_argument('--extract_geometry', dest='extract_geometry', action='store_true', 
        help='if you only want to extract geometry')
    
    parser.add_argument('--test_general', dest='test_general', action='store_true', 
        help='test on custom dataset')
    
    parser.add_argument('--test_ray_num', dest='test_ray_num', type=int, default=1200)
    parser.add_argument('--test_sample_coarse', dest='test_sample_coarse', type=int, default=64)
    parser.add_argument('--test_sample_fine', dest='test_sample_fine', type=int, default=64)
    parser.add_argument('--test_coarse_only', dest='test_coarse_only', action="store_true",
        help='only use coarse samples during testing')

    parser.add_argument('--num_src_view', dest='num_src_view', type=int, default=4)

    parser.add_argument('--test_n_view', dest='test_n_view', type=int, default=3)
    parser.add_argument('--src_via_dist', dest='src_via_dist', action="store_true",
        help='calculate source views based on camera distances')
    
    parser.add_argument('--set', dest='set', type=int, default=0,
        help='two sets are provided by SparseNeuS')

    parser.add_argument('--extract_mesh', dest='extract_mesh', action='store_true', 
        help='if you only want to extract mesh')

    parser.add_argument('--grid_size', dest='grid_size', type=int, default=256, 
            help='grid size for performing marching cube')

    parser.add_argument('--only_volume', dest='only_volume', action='store_true', 
        help='reconstruct the geometry using cost volume only')

    parser.add_argument('--model_type', type=str, default='default', 
        help='specify the model structure')

    parser.add_argument('--feature_net', type=str, default='default', 
        help='specify the feature extraction network choice')

    parser.add_argument('--use_se', action='store_true', 
        help='use se module in the simplergb model')

    parser.add_argument('--use_clip', action='store_true', 
        help='whether to learn semantic features using CLIP')

    parser.add_argument('--label_src', type=str, default='flower,other', 
        help='labels of the text inputs to CLIP')

    parser.add_argument('--test_scene', type=str, default='scan65', 
        help='test scene')

    parser.add_argument('--coarse_only', action='store_true', 
        help='only enable coarse sampling')

    parser.add_argument('--use_mask_coord', action='store_true', 
        help='generate coordinates based on foreground masks')
    
    parser.add_argument(
        "--mesh_level", type=float, default=0
    )

    parser.add_argument('--debugging', action='store_true', 
        help='debugging mode')

    parser.add_argument('--vanilla_volume_rendering', action='store_true', 
        help='use vanilla volume rendering instead SDF')
    
    parser.add_argument('--use_sample_mask', action='store_true', 
        help='use sample-wise mask when volume rendering')
    
    parser.add_argument('--act_func', type=str, default='relu', 
        help='activation function')

    parser.add_argument(
        "--lr_decay_rate", type=float, default=1
    )

    parser.add_argument(
        "--lr_decay_step", type=float, default=50000
    )

    parser.add_argument(
        "--lr_feature", type=float, default=None
    )
    
    parser.add_argument('--inv_uniform', action='store_true', 
        help='use inv uniform z_val sampling')

    parser.add_argument('--disable_deviation', action='store_true', 
        help='disable the deviation network in sdf renderer')

    parser.add_argument('--use_aug', action='store_true', 
        help='enable data augmentation')
    
    parser.add_argument('--mean_var_feat', action='store_true', 
        help='concat mean var features when feature volumes are disabled')

    parser.add_argument('--neuray_depth_loss', action='store_true', 
        help='use neuray-style depth loss')

    parser.add_argument('--ft_rgb', action='store_true', 
        help='finetuning rgb branch from depth-only pretrained models, which will use small lr for the density branch')

    parser.add_argument('--train_rgb_only', action='store_true', 
        help='only train the rgb branch from depth-only pretrained models')

    parser.add_argument('--correct_dtu_coord_trans', action='store_true', 
        help='correct the opengl to opencv translation in DTU') 

    parser.add_argument('--aug_view_select_type', type=str, default='easy', 
        help='the intervals between source views during training')

    parser.add_argument('--use_scale_matrix', action='store_true', 
        help='use scale matrixs for projecting all scenes to the range of [-1, 1]') 

    parser.add_argument('--warp_to_ref_view', action='store_true', 
        help='warp all source views to the ref view') 

    parser.add_argument('--no_warp_to_ref_view', action='store_true') 

    parser.add_argument('--align_corners_fv', action='store_true', 
        help='whether align_corners=True when performing grid sample') 

    parser.add_argument('--align_corners_2d', action='store_true', 
        help='whether align_corners=True when performing grid sample') 

    parser.add_argument('--align_corners_3d', action='store_true', 
        help='whether align_corners=True when performing grid sample') 

    parser.add_argument('--use_volsdf', action='store_true', 
        help='use the sdf to density convertion in volsdf') 

    parser.add_argument('--anneal_beta', action='store_true', 
        help='use annealing beta instead of learnable beta in volsdf') 
    
    parser.add_argument('--use_official_dtu_src', action='store_true', 
        help='use the officially predefined DTU source views') 
    
    parser.add_argument('--fine_loss_only', action='store_true', 
        help='only use the fine loss') 

    parser.add_argument('--use_depth_dz', action='store_true', 
        help='use the z-normalized depth in DTU') 

    parser.add_argument('--use_orig_rgb_loss', action='store_true', 
        help='use the original rgb loss') 

    parser.add_argument(
        "--beta_init", type=float, default=0.1,
         help='init beta value in VolSDF renderer'
    )

    parser.add_argument(
        "--beta_min", type=float, default=0.001,
         help='min beta value in VolSDF renderer'
    )

    parser.add_argument('--init_net_type', type=str, default='cost_volume', 
        help='the default init net')

    parser.add_argument('--trans_depth', type=int, default=2, 
        help='the transformer depth')

    parser.add_argument('--scan', type=int, default=None, 
        help='the dtu scan id for mesh extraction')

    parser.add_argument('--cosine_lr', action='store_true', 
        help='use cosine lr scheduler') 

    parser.add_argument('--min_lr', type=float, default=1e-6, 
        help='the minial learing rate in cosine lr scheduler')

    parser.add_argument('--use_causal_mask', action='store_true', 
        help='use causal mask') 

    parser.add_argument('--auto_ckpt', action='store_true', 
        help='automatically load ckpt') 
    
    parser.add_argument('--edit_mode', action='store_true', 
        help='3D scene editting mode') 

    parser.add_argument('--prompt', type=str, default=None, 
        help='the prompt that instructs the editting process') 

    parser.add_argument('--edit_iters', type=int, default=20, 
        help='number of iterations to update all images')

    parser.add_argument('--text_guidance_scale', type=float, default=7.5, 
        help='text guidance scale in editting')
    
    parser.add_argument('--noise_level', type=float, default=0.3, 
        help='noise level in editting')
    
    parser.add_argument('--orig_wrong_renderer', action='store_true', 
        help='use the original wrong renderer')

    parser.add_argument('--use_ray_renderer', action='store_true', 
        help='use the use_ray_renderer instead of SDF-based renderer')

    parser.add_argument('--predict_weight', action='store_true', 
        help='directly predict the weight instead of sdf')

    parser.add_argument('--extract_geometry_all_views', action='store_true', 
        help='extract geometry from all views')

    parser.add_argument('--extract_other_dataset', action='store_true', 
        help='extract geometry from datasets other than DTU')

    parser.add_argument('--camera_correction', action='store_true', 
        help='correct the camera perspective projection')

    parser.add_argument('--simple_appear_feat', action='store_true', 
        help='do not use feature volume in the appearance branch')

    parser.add_argument('--tiny_shader', action='store_true', 
        help='use small shader')

    parser.add_argument('--no_viewtrans', action='store_true', 
        help='no view transformer')

    parser.add_argument('--no_raytrans', action='store_true', 
        help='no ray transformer')

    args = parser.parse_args()

    batch_size = args.batch_size
    num_workers = 4

    devices = int(torch.cuda.device_count())
    
    args.white_bkgd = False

    if args.out_dir is None:
        args.out_dir = args.logdir
            
    
    if args.edit_mode:
        if args.scan is not None:
            scan = args.scan
        else:
            scan = 65
        
        test_data_class = DtuFitSparse
                            
        dataset_tmp = test_data_class(root_dir=args.test_dir, 
                            split="test", 
                            scan_id='scan%d'%scan, 
                            n_views=10,
                            n_src_views=4,
                            src_via_dist=True,
                            set=args.set,
                            novel_vs=True,
                            no_offset=True,
                            args=args)
        dataloader_test = DataLoader(dataset_tmp,
                                    batch_size=1, 
                                    num_workers=1 if not args.debugging else 0, 
                                    shuffle=False)
    
    elif not args.extract_geometry:
        train_data_class = MVSDataset
            
        dataset_train = train_data_class(            
                root_dir="/data/yfu314/dtu/DTU_TRAIN",
                split="train",
                split_filepath="src/data_pretrain/dtu/lists/train.txt",
                pair_filepath="src/data_pretrain/dtu/dtu_pairs.txt",
                n_views=args.num_src_view + 1,
                no_warp_to_ref_view=args.no_warp_to_ref_view,
                args=args
                )

        dataset_val = DTU_Finetune(root_dir="/data/yfu314/dtu/DTU_TEST", 
                                    split="test", 
                                    scan_id=args.test_scene, 
                                    n_src_views=args.num_src_view,
                                    src_via_dist=args.src_via_dist,
                                    args=args)
        
        print("dataset_train:", len(dataset_train))
        print("dataset_val:", len(dataset_val))

        dataloader_train = DataLoader(dataset_train,
                                        batch_size=batch_size, 
                                        num_workers=num_workers if not args.debugging else 0, 
                                        pin_memory=True,
                                        shuffle=True)  
        dataloader_val = DataLoader(dataset_val,
                                    batch_size=batch_size, 
                                    num_workers=4 if not args.debugging else 0,
                                    pin_memory=True,
                                    shuffle=False)
        
    else:
        dataloader_test = []

        if args.extract_other_dataset:
            assert args.extract_geometry_all_views
            assert args.use_scale_matrix
            
            if args.test_scene in ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']:
                cfg = {'val_database_name': 'nerf_synthetic/%s/black_800'%args.test_scene}
            else:
                cfg = {'val_database_name': args.test_scene}
                
            cfg['warp_to_ref_view'] = True

            dataset_class = GeneralRendererDataset_Scale
        
            dataset_tmp = dataset_class(cfg=cfg, is_train=False, train_ray_num=args.train_ray_num, num_src_view=args.num_src_view, extract_geometry=True)
            dataloader_tmp = DataLoader(dataset_tmp,
                                        batch_size=1, 
                                        num_workers=1 if not args.debugging else 0, 
                                        shuffle=False)
            dataloader_test.append(dataloader_tmp)
            

        elif not args.test_general:
            if args.scan is not None:
                scans = [args.scan]
            else:
                scans = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
                                
            for scan in scans:
                if args.extract_geometry_all_views:
                    dataset_tmp = DTU_Finetune(root_dir=args.test_dir, 
                                        split="all", 
                                        scan_id='scan%d'%scan, 
                                        n_src_views=args.num_src_view,
                                        src_via_dist=args.src_via_dist,
                                        args=args)
                    
                else:
                    test_data_class = DtuFitSparse
                        
                    dataset_tmp = test_data_class(root_dir=args.test_dir, 
                                        split="test", 
                                        scan_id='scan%d'%scan, 
                                        n_views=args.test_n_view,
                                        n_src_views=args.num_src_view,
                                        src_via_dist=args.src_via_dist,
                                        set=args.set,
                                        args=args)
                    
                dataloader_tmp = DataLoader(dataset_tmp,
                                            batch_size=1, 
                                            num_workers=1 if not args.debugging else 0, 
                                            shuffle=False)  
                dataloader_test.append(dataloader_tmp)
        else:
            for scan in ["general"]:
                
                dataset_tmp = GeneralFit(root_dir=args.test_dir, 
                                    scan_id=scan, 
                                    n_views=args.test_n_view)
                dataloader_tmp = DataLoader(dataset_tmp,
                                                batch_size=1, 
                                                num_workers=1, 
                                                shuffle=False)  
                dataloader_test.append(dataloader_tmp)


    if args.load_ckpt:
        if args.auto_ckpt:
            args.load_ckpt = find_latest_checkpoint(args.load_ckpt)

        model = OmniRecon.load_from_checkpoint(checkpoint_path=args.load_ckpt, strict=True if not args.ft_rgb else False, args=args, load_only_params=False if not args.ft_rgb else True)
        print("Model loaded:", args.load_ckpt)
    else:
        model = OmniRecon(args)
    
    if args.use_clip:
        model.build_clip()
    
    if args.edit_mode:
        model.scene_edit(prompt=args.prompt, test_loader=dataloader_test)
    
    else:
        logger = WandbLogger(
            name = "model-"+args.logdir.rsplit('/')[-1],
            save_dir = args.logdir,
            offline=True,
        )        

        class IterationProgressBar(ProgressBar):
            def __init__(self):
                super().__init__()
                self.enable = True
                self.steps = 0
                self.total_steps = 0
                self.train_batch_idx = 0
                self.pbar = None

            def on_train_start(self, trainer, pl_module):
                super().on_train_start(trainer, pl_module)
                self.total_steps = trainer.max_steps
                self.pbar = tqdm(
                    desc='Training',
                    initial=self.steps,
                    total=self.total_steps,
                    dynamic_ncols=True,
                    file=sys.stdout
                )

            def on_train_batch_end(
                self,
                trainer: "pl.Trainer",
                pl_module: "pl.LightningModule",
                outputs: STEP_OUTPUT,
                batch,
                batch_idx: int
            ) -> None:
                super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
                self.steps += 1
                if self.pbar is not None:
                    self.pbar.n = self.steps
                    self.pbar.refresh()
                    
                    # Update postfix with loss if available
                    if isinstance(outputs, dict) and 'loss' in outputs:
                        self.pbar.set_postfix({'loss': f"{outputs['loss']:.4f}"})

            def on_train_end(self, trainer, pl_module):
                if self.pbar is not None:
                    self.pbar.close()
                    self.pbar = None

            def disable(self):
                self.enable = False
                if self.pbar is not None:
                    self.pbar.disable = True

            def enable(self):
                self.enable = True
                if self.pbar is not None:
                    self.pbar.disable = False


        checkpoint_callback = ModelCheckpoint(
            dirpath=args.logdir,
            save_top_k=-1,
            verbose=True,
            every_n_train_steps=10000,
        )


        trainer = pl.Trainer(
                accelerator="gpu" if device=="cuda" else "cpu", 
                devices=devices,
                strategy = "ddp" if not args.ft_rgb else DDPStrategy(find_unused_parameters=True),
                max_steps=args.max_iters,
                check_val_every_n_epoch=1, 
                logger=logger,
                num_sanity_val_steps=0,
                callbacks=[IterationProgressBar(), checkpoint_callback],
                )

        ModelSummary(model, max_depth=1)


        if not args.extract_geometry:
            if args.val_only:
                print("[only validation]")
                trainer.validate(model, dataloader_val)
            else:
                print("[start training]")
                trainer.fit(model, dataloader_train, dataloader_val)
        else:
            for dataloader_test1 in tqdm(dataloader_test):
                trainer.validate(model, dataloader_test1)
            
        print("end")


