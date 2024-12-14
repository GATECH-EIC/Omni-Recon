import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
import trimesh

from mesh.post_net import *
from mesh.eval import evaluate

from mesh.shader import Gen_Shader
from torch.utils.data import DataLoader
from src.data_pretrain.train_dataset_scale import GeneralRendererDataset_Scale

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def test(
    test_spiral = False,
    data_path = None,
    scene = None,
    renderer="foundation-nerf",
    model_path = None,
    uvmap_path = None,
    uv_path = None,
    batch_size=4,
    res = (800, 800),   # (H, W)
    use_opengl = False,
    out_dir = None,
    mesh_path = None,
    export_image = False,
    antialias = False,
    post_net_path = None,
    post_use_depth = False,
    supersampling = 'none',
    filter_mode = 'linear',
    args=None
):
    logger.addHandler(logging.FileHandler('/'.join([out_dir, 'test.log'])))
    logger.addHandler(logging.StreamHandler())

    # Load mesh with unwrapped altas texture map
    mesh = trimesh.load(mesh_path, process=False, maintain_order=True)
    pos_idx = mesh.faces
    pos = mesh.vertices

    # Create position/triangle index tensors
    '''
    # pos_idx: (#faces, 3)
    # vtx_pos: (#vertices, 3)
    '''
    pos_idx = torch.from_numpy(pos_idx.astype(np.int32)).cuda()
    vtx_pos = torch.from_numpy(pos.astype(np.float32)).cuda()
    logger.info("vertix count: %d" % vtx_pos.shape[0])
    logger.info("surfaces count: %d" % pos_idx.shape[0])

    # Load model
    if renderer == "foundation-nerf":
        model = Gen_Shader(args=args)

        model.load_checkpoint(model_path)
        print("Model loaded:", model_path)
        
        model.cuda()

        uv_map = None
        uvs = None
    elif renderer == "UV_map":
        model = torch.load(model_path)
        uv_map = torch.load(uvmap_path, map_location="cuda:0")
        uvs = torch.load(uv_path, map_location="cuda:0")
    else:
        raise NotImplementedError

    if post_net_path is not None:
        post_model = torch.load(post_net_path)
        post_model.eval()
    else:
        post_model = None

    glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()

    model.eval()
    if uv_map is not None:
        uv_map = uv_map.detach()

    if scene in ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']:
        dataset = 'synthetic'
        # from code.data_ft.nerf_synthetic import SubjectLoader
        # test_dataset = SubjectLoader(subject_id=scene, root_fp=data_path, split="test", n_src_views=args.num_src_view, batch_size=batch_size,
        #                         supersampling=2 if supersampling == 'simple' else 1, get_spiral=test_spiral)

        cfg = {'val_database_name': 'nerf_synthetic/%s/black_800'%args.scene}
        cfg['fix_num_src_view'] = True
        cfg['warp_to_ref_view'] = True
        cfg['aug_view_select_type'] = 'no_aug'
        # test_dataset = GeneralRendererDataset_Scale(cfg=cfg, is_train=False, train_ray_num=512, num_src_view=args.num_src_view, extract_geometry=False, is_test=True)
        test_dataset = GeneralRendererDataset_Scale(cfg=cfg, is_train=False, train_ray_num=512, num_src_view=args.num_src_view, extract_geometry=False)
        test_dataset.color_bkgd_aug = 'black'
        res = [800, 800]
    elif scene in ["Wineholder", "Steamtrain", "Spaceship", "Palace", "Bike", "Robot", "Lifestyle", "Toad",]:
        dataset = "nsvf"
        from src.data_ft.nsvf_synthetic import SubjectLoader
        test_dataset = SubjectLoader(subject_id=scene, root_fp=data_path, split="test", batch_size=batch_size,
                                supersampling=2 if supersampling == 'simple' else 1, get_spiral=test_spiral)

        res = [800, 800]
    elif scene in ["Barn", "Caterpillar", "Family", "Ignatius", "Truck",]:
        dataset = "tanks"
        from src.data_ft.tanksandtemples import SubjectLoader
        test_dataset = SubjectLoader(subject_id=scene, root_fp=data_path, split="test", batch_size=batch_size,
                                supersampling=2 if supersampling == 'simple' else 1, get_spiral=test_spiral)

        res = [1080, 1920]
    elif scene in ["Jade", "Fountain", "Statues", "Character",]:
        dataset = "mvs"
        from src.data_ft.blendedMVS import SubjectLoader
        test_dataset = SubjectLoader(subject_id=scene, root_fp=data_path, split="test", batch_size=batch_size,
                                supersampling=2 if supersampling == 'simple' else 1)

        if scene in ['Jade', 'Fountain']:
            print('BKGD is set to black by MVS dataset')
            test_dataset.color_bkgd_aug = 'black'
        elif scene in ['Character', 'Statues']:
            print('BKGD is set to white by MVS dataset')
            test_dataset.color_bkgd_aug = 'white'

        res = [576, 768]
    elif scene in ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex",]:
        dataset = 'llff'
        from src.data_ft.llff import SubjectLoader
        test_dataset = SubjectLoader(subject_id=scene, root_fp=data_path, split="test", batch_size=batch_size,
                                supersampling=2 if supersampling == 'simple' else 1)

        res = [756, 1008]

    elif "scan" in scene:
        dataset = 'dtu'
        from src.data_ft.dtu import DTU_Finetune
        test_dataset = DTU_Finetune(            
                root_dir=data_path,
                scan_id=scene,
                split="test",
                n_src_views=args.num_src_view,
                src_via_dist=args.src_via_dist,
                supersampling=2 if supersampling == 'simple' else 1,
                )

        res = [600, 800] # [512, 640]

    test_dataloader = DataLoader(test_dataset,
                                batch_size=batch_size, 
                                num_workers=1, 
                                shuffle=False)

    with torch.no_grad():
        evaluate(
            dataset,
            test_dataset,
            test_dataloader,
            batch_size,
            vtx_pos,
            res,
            glctx,
            pos_idx,
            uv_map,
            uvs,
            renderer,
            antialias,
            supersampling,
            filter_mode,
            model,
            post_model,
            post_use_depth,
            out_dir,
            export_image=export_image,
            gen_log=True,
            requires_time=False,
            test_spiral=test_spiral,
            logger=logger,
            args=args
        )