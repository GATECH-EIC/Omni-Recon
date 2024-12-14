import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
import trimesh
import os

from mesh.post_net import *
from mesh.eval import evaluate

from mesh.shader import Gen_Shader

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def extract_mesh(
    test_spiral = False,
    data_path = None,
    scene = None,
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
    logger.addHandler(logging.FileHandler('/'.join([out_dir, 'extract_mesh.log'])))
    
    # Load model
    model = Gen_Shader(args=args)

    model.load_checkpoint(model_path)
    print("Model loaded:", model_path)
    
    model.cuda()

    uv_map = None
    uvs = None

    model.eval()
    if uv_map is not None:
        uv_map = uv_map.detach()

    if scene in ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']:
        dataset = 'synthetic'
        from src.data_ft.nerf_synthetic import SubjectLoader
        test_dataset = SubjectLoader(subject_id=scene, root_fp=data_path, split="train", n_src_views=args.num_src_view, batch_size=batch_size,
                                supersampling=2 if supersampling == 'simple' else 1, get_spiral=test_spiral, baking_only=True)

        for attr_name in test_dataset.cuda_tensors:
            attr = getattr(test_dataset, attr_name)
            setattr(test_dataset, attr_name, attr.cuda())

        res = [800, 800]
    elif scene in ["Wineholder", "Steamtrain", "Spaceship", "Palace", "Bike", "Robot", "Lifestyle", "Toad",]:
        dataset = "nsvf"
        from src.data_ft.nsvf_synthetic import SubjectLoader
        test_dataset = SubjectLoader(subject_id=scene, root_fp=data_path, split="train", batch_size=batch_size,
                                supersampling=2 if supersampling == 'simple' else 1, get_spiral=test_spiral)

        res = [800, 800]
    elif scene in ["Barn", "Caterpillar", "Family", "Ignatius", "Truck",]:
        dataset = "tanks"
        from src.data_ft.tanksandtemples import SubjectLoader
        test_dataset = SubjectLoader(subject_id=scene, root_fp=data_path, split="train", batch_size=batch_size,
                                supersampling=2 if supersampling == 'simple' else 1, get_spiral=test_spiral)

        res = [1080, 1920]
    elif scene in ["Jade", "Fountain", "Statues", "Character",]:
        dataset = "mvs"
        from src.data_ft.blendedMVS import SubjectLoader
        test_dataset = SubjectLoader(subject_id=scene, root_fp=data_path, split="train", batch_size=batch_size,
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
        test_dataset = SubjectLoader(subject_id=scene, root_fp=data_path, split="train", batch_size=batch_size,
                                supersampling=2 if supersampling == 'simple' else 1)

        res = [756, 1008]

    elif "scan" in scene:
        dataset = 'dtu'
        from src.data_ft.dtu import SubjectLoader
        test_dataset = SubjectLoader(            
                root_dir=data_path,
                scan_id=scene,
                split="test",
                n_src_views=args.num_src_view,
                src_via_dist=args.src_via_dist,
                supersampling=2 if supersampling == 'simple' else 1,
                num_ref_world_view=args.num_ref_world_view,
                baking_only=True,
                no_warp_to_ref_view=args.no_warp_to_ref_view
                )

        for attr_name in test_dataset.cuda_tensors:
            attr = getattr(test_dataset, attr_name)
            setattr(test_dataset, attr_name, attr.cuda())

        ## save the trans2w_mat used for further finetuning
        scale_mat, trans_mat = test_dataset.scale_mat.cpu().numpy(), test_dataset.trans_mat.cpu().numpy()
        trans2w_mat = np.matmul(trans_mat, scale_mat)
        
        save_dir = os.path.join(args.out_dir, "mesh")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, 'trans2w_mat.npy'), trans2w_mat)

        res = [600, 800] # [512, 640]


    for i in range(len(test_dataset)):
        index = [i]
        data = test_dataset[index]
        with torch.no_grad():
            model.extract_mesh(data)
            

