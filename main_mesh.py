import argparse
import os

from mesh.finetune import finetune
from mesh.test import test
from mesh.extract_mesh import extract_mesh

def main():
    parser = argparse.ArgumentParser(description='InstantNGP network fiunting with exported mesh')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--scene', type=str, default=None)
    parser.add_argument('--color_bkgd_aug', type=str, default="white")
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--uvmap_path', type=str, default=None)
    parser.add_argument('--uv_path', type=str, default=None)

    parser.add_argument('--antialias', action='store_true', default=False)
    parser.add_argument('--filter_mode', type=str, default='linear')

    parser.add_argument('--train_uvmap', action='store_true', default=False)
    parser.add_argument('--train_mesh', action='store_true', default=False)
    parser.add_argument('--train_mesh_epoch', type=int, default=0)
    parser.add_argument('--training_ratio', type=float, default=1.)
    parser.add_argument('--prune_triangle', action='store_true', default=False)
    parser.add_argument('--prune_triangle_keep_static', action='store_true', default=False)
    parser.add_argument('--grow_triangle', action='store_true', default=False)
    parser.add_argument('--add_mesh_group', type=int, default=0)
    parser.add_argument('--mesh_noise_scale', type=float, default=0.)


    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr_base', type=float, default=1e-2)
    parser.add_argument('--lr_mesh', type=float, default=1e-2)
    parser.add_argument('--lr_uvmap', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--inherite_head_ranking', action='store_true')

    parser.add_argument('--out_dir', type=str, default="ret")
    parser.add_argument('--opengl', help='enable OpenGL rendering', action='store_true', default=False)
    parser.add_argument('--mesh_path', type=str, default=None)
    parser.add_argument('--loss_type', type=str, default="l2")
    parser.add_argument('--optim_type', type=str, default="Adam")
    parser.add_argument('--export_image', action='store_true', default=False)
    parser.add_argument('--export_mesh', action='store_true', default=False)
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--test_spiral', action='store_true', default=False)
    parser.add_argument('--eval_every', type=int, default=0)
    parser.add_argument('--save_on_eval', action='store_true')
    parser.add_argument('--train_split', type=str, default='trainval')

    parser.add_argument('--post_net', type=str, default=None)
    parser.add_argument('--post_use_depth', action='store_true', default=False)
    parser.add_argument('--post_net_path', type=str, default=None)

    parser.add_argument('--supersampling', type=str, default='none')
    parser.add_argument('--co_train', action='store_true')

    ### args for model definition
    parser.add_argument('--volume_reso', type=int, default=144, 
        help="3D feature volume resolution") # set as 0 to disable
    parser.add_argument('--only_volume', action='store_true', 
        help='reconstruct the geometry using cost volume only')
    parser.add_argument('--num_src_view', type=int, default=3, 
        help="number of source views") 
    parser.add_argument('--src_via_dist', action="store_true",
        help='calculate source views based on camera distances')
    parser.add_argument('--num_ref_world_view', type=int, default=5, 
        help="number of views for calculating the scale matrix in DTU's coordinate transformation")

    parser.add_argument('--model_type', type=str, default='default', 
        help='specify the model structure')

    parser.add_argument('--feature_net', type=str, default='default', 
        help='specify the feature extraction network choice')
        
    parser.add_argument('--grid_size', dest='grid_size', type=int, default=256, 
            help='grid size for performing marching cube')

    parser.add_argument('--extract_mesh', dest='extract_mesh', action='store_true', 
        help='if you only want to extract mesh')

    parser.add_argument('--strat_src_view_select', action='store_true', 
        help='use the proposed source view selection scheme for baking')

    parser.add_argument(
        "--mesh_level", type=float, default=0
    )

    parser.add_argument(
        "--para_factor", type=int, default=32, 
        help='parallel factor when extracting mesh / voxels using selected source views'
    )

    parser.add_argument(
        "--src_color_thres", type=float, default=0.1,
        help='color threshold for determing the cube size when applying the proposed src view selection'
    )
    
    parser.add_argument(
        "--src_sample_num", type=int, default=1,
        help='number of samples within a local cube when applying the proposed src view selection'
    )

    parser.add_argument('--coarse_only', action='store_true', 
        help='only enable coarse sampling')
    
    parser.add_argument('--use_se', action='store_true', 
        help='use se module in the simplergb model')

    parser.add_argument('--use_clip', action='store_true', 
        help='whether to learn semantic features using CLIP')

    parser.add_argument('--mean_var_feat', action='store_true', 
        help='concat mean var features when feature volumes are disabled')
    
    parser.add_argument('--act_func', type=str, default='relu', 
        help='activation function')

    parser.add_argument('--use_sample_mask', action='store_true', 
        help='use sample-wise mask when volume rendering')
    
    parser.add_argument('--align_corners_fv', action='store_true', 
        help='whether align_corners=True when performing grid sample')

    parser.add_argument('--align_corners_2d', action='store_true', 
        help='whether align_corners=True when performing grid sample') 

    parser.add_argument('--align_corners_3d', action='store_true', 
        help='whether align_corners=True when performing grid sample') 

    parser.add_argument('--no_warp_to_ref_view', action='store_true', 
        help='only apply on volrecon-dtu dataloader') 

    parser.add_argument('--init_net_type', type=str, default='cost_volume', 
        help='the default init net')
    parser.add_argument('--trans_depth', type=int, default=2, 
        help='the transformer depth in GNT')

    parser.add_argument('--large_volume', action='store_true', 
        help='use large feature volume')

    parser.add_argument('--cosine_lr', action='store_true', 
        help='use cosine lr scheduler') 

    parser.add_argument('--min_lr', type=float, default=1e-6, 
        help='the minial learing rate in cosine lr scheduler')

    parser.add_argument('--use_causal_mask', action='store_true', 
        help='use causal mask in GNT design') 

    parser.add_argument('--auto_ckpt', action='store_true', 
        help='use full attention in GNT') 

    parser.add_argument('--orig_wrong_renderer', action='store_true', 
        help='use the original wrong renderer')

    parser.add_argument('--use_ray_renderer', action='store_true', 
        help='use the use_ray_renderer instead of SDF-based renderer')

    parser.add_argument('--predict_weight', action='store_true', 
        help='directly predict the weight instead of sdf')
    
    parser.add_argument('--camera_correction', action='store_true', 
        help='correct the camera perspective projection')

    parser.add_argument('--simple_appear_feat', action='store_true', 
        help='do not use feature volume in the appearance branch')

    parser.add_argument('--tiny_shader', action='store_true', 
        help='use small shader')

    parser.add_argument('--min_lr_scale', type=float, default=0.1, 
        help='the scale factor for deciding minial learing rate in cosine lr scheduler')

    parser.add_argument('--max_epoch', type=int, default=None, 
        help='the maximal epoch')
    
    args = parser.parse_args()

    assert not args.train_mesh or args.antialias, "Must enable antialias before enabling mesh training"

    # args.out_dir = os.path.join("ckpt", args.out_dir)
    print(f'Saving results under {args.out_dir}')
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Run.

    if args.extract_mesh:
        extract_mesh(
            data_path = args.data_path,
            scene = args.scene,
            model_path = args.model_path,
            batch_size = 1,
            out_dir = args.out_dir,
            args=args
        )

    else:
        if args.test_only:
            test(
                test_spiral = args.test_spiral,
                data_path = args.data_path,
                scene = args.scene,
                model_path = args.model_path,
                uvmap_path = args.uvmap_path,
                uv_path = args.uv_path,
                batch_size = args.batch_size,
                out_dir = args.out_dir,
                use_opengl = args.opengl,
                mesh_path = args.mesh_path,
                export_image = args.export_image,
                antialias = args.antialias,
                post_net_path = args.post_net_path,
                post_use_depth = args.post_use_depth,
                supersampling=args.supersampling,
                filter_mode=args.filter_mode,
                args=args
            )

        else:
            finetune(
                data_path = args.data_path,
                train_split = args.train_split,
                scene = args.scene,
                color_bkgd_aug = args.color_bkgd_aug,
                model_path = args.model_path,
                epochs = args.epochs,
                batch_size = args.batch_size,
                lr_base = args.lr_base,
                lr_mesh = args.lr_mesh,
                lr_uvmap = args.lr_uvmap,
                weight_decay = args.weight_decay,
                out_dir = args.out_dir,
                use_opengl = args.opengl,
                mesh_path = args.mesh_path,
                uvmap_path = args.uvmap_path,
                uv_path = args.uv_path,
                loss_type = args.loss_type,
                optim_type = args.optim_type,
                export_image = args.export_image,
                antialias = args.antialias,
                train_uvmap = args.train_uvmap,
                train_mesh = args.train_mesh,
                train_mesh_epoch = args.train_mesh_epoch,
                prune_triangle = args.prune_triangle,
                prune_triangle_keep_static = args.prune_triangle_keep_static,
                grow_triangle = args.grow_triangle,
                export_mesh = args.export_mesh,
                post_net = args.post_net,
                post_use_depth = args.post_use_depth,
                supersampling=args.supersampling,
                co_train=args.co_train,
                filter_mode=args.filter_mode,
                eval_every=args.eval_every,
                save_on_eval=args.save_on_eval,
                training_ratio=args.training_ratio,
                args=args
            )

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------