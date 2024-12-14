import time
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import nvdiffrast.torch as dr
import trimesh
from tqdm import tqdm
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader

from mesh.post_net import *
from mesh.render import render
from mesh.utils import w2clip, w2clip2, writemesh2ply, inherite_model_rank, inherite_head_rank, add_noisy_mesh, vtx_to_ndc, subdivide_large_triangles

from mesh.eval import evaluate
from mesh.shader import Gen_Shader
from src.data_pretrain.train_dataset_scale import GeneralRendererDataset_Scale
import time

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def finetune(
    data_path = None,
    train_split = "trainval",
    scene = None,
    color_bkgd_aug = "white",
    renderer="foundation-nerf",
    model_path = None,
    uvmap_path = None,
    uv_path = None,
    epochs = 60,
    batch_size=4,
    lr_base = 1e-2,
    lr_mesh = 1e-2,
    lr_uvmap = 1e-2,
    lr_ramp = 0.1,
    weight_decay = 1e-5,
    out_dir = None,
    use_opengl = False,
    mesh_path = None,
    loss_type = "l2",
    optim_type = "Adam",
    export_image = True,
    export_mesh = False,
    antialias = False,
    add_mesh_group = 0,
    mesh_noise_scale = 0.,
    train_mesh = False,
    train_mesh_epoch = 0,
    train_uvmap = False,
    prune_triangle = False,
    prune_triangle_keep_static = False,
    grow_triangle = False,
    post_net = None,
    post_use_depth = False,
    supersampling = 'none',
    co_train = False,
    filter_mode = 'linear',
    eval_every = 0,
    save_on_eval = False,
    training_ratio=1.,
    args=None
):

    assert ~(co_train and supersampling != 'none'), "must enable supersampling when using co-training"
    writer = SummaryWriter('/'.join([out_dir, 'log']))

    if export_image:
        img_dir = '/'.join([out_dir, 'img_dir'])
        os.makedirs(img_dir, exist_ok=True)

    logger.addHandler(logging.FileHandler('/'.join([out_dir, 'train.log'])))
    logger.addHandler(logging.StreamHandler())

    # Load mesh with unwrapped altas texture map
    mesh = trimesh.load(mesh_path, process=False, maintain_order=True)
    pos_idx = mesh.faces
    pos = mesh.vertices
    if add_mesh_group > 0:
        pos_idx, pos = add_noisy_mesh(mesh, add_mesh_group, mesh_noise_scale)

    # Create position/triangle index tensors
    '''
    # pos_idx: (#faces, 3)
    # vtx_pos: (#vertices, 3)
    '''
    pos_idx = torch.from_numpy(pos_idx.astype(np.int32)).cuda()
    vtx_pos = torch.from_numpy(pos.astype(np.float32)).cuda()

    print("vertix count: %d" % vtx_pos.shape[0])
    print("surfaces count: %d" % pos_idx.shape[0])

    # Intialize Nerf Synthetic dataset
    if scene in ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']:
        dataset = 'synthetic'
        # from code.data_ft.nerf_synthetic import SubjectLoader
        # train_dataset = SubjectLoader(subject_id=scene, root_fp=data_path, split=train_split, batch_size=batch_size, color_bkgd_aug=color_bkgd_aug,
        #                                 supersampling=2 if supersampling == 'simple' else 1)

        # # for attr_name in train_dataset.cuda_tensors:
        # #     attr = getattr(train_dataset, attr_name)
        # #     setattr(train_dataset, attr_name, attr.cuda())

        # test_dataset = SubjectLoader(subject_id=scene, root_fp=data_path, split="test", batch_size=batch_size,
        #                         supersampling=2 if supersampling == 'simple' else 1)

        cfg = {'val_database_name': 'nerf_synthetic/%s/black_800'%args.scene}
        cfg['fix_num_src_view'] = True
        cfg['warp_to_ref_view'] = True
        cfg['aug_view_select_type'] = 'no_aug'
        train_dataset = GeneralRendererDataset_Scale(cfg=cfg, is_train=False, train_ray_num=512, num_src_view=args.num_src_view, is_finetune=True)
        train_dataset.color_bkgd_aug = 'black'
        
        test_dataset = GeneralRendererDataset_Scale(cfg=cfg, is_train=False, train_ray_num=512, num_src_view=args.num_src_view, extract_geometry=False)
        test_dataset.color_bkgd_aug = 'black'
        res = [800, 800]

    elif scene in ["Wineholder", "Steamtrain", "Spaceship", "Palace", "Bike", "Robot", "Lifestyle", "Toad",]:
        dataset = "nsvf"
        from src.data_ft.nsvf_synthetic import SubjectLoader
        train_dataset = SubjectLoader(subject_id=scene, root_fp=data_path, split=train_split, batch_size=batch_size, color_bkgd_aug=color_bkgd_aug,
                                        supersampling=2 if supersampling == 'simple' else 1)
        # train_dataset.images = train_dataset.images.cuda()
        # train_dataset.camtoworlds = train_dataset.camtoworlds.cuda()
        # train_dataset.K = train_dataset.K.cuda()

        test_dataset = SubjectLoader(subject_id=scene, root_fp=data_path, split="test", batch_size=batch_size,
                                supersampling=2 if supersampling == 'simple' else 1)

        res = [800, 800]
    elif scene in ["Barn", "Caterpillar", "Family", "Ignatius", "Truck",]:
        dataset = "tanks"
        from src.data_ft.tanksandtemples import SubjectLoader
        train_dataset = SubjectLoader(subject_id=scene, root_fp=data_path, split=train_split, batch_size=batch_size, color_bkgd_aug=color_bkgd_aug,
                                        supersampling=2 if supersampling == 'simple' else 1)
        # train_dataset.images = train_dataset.images.cuda()
        # train_dataset.camtoworlds = train_dataset.camtoworlds.cuda()
        # train_dataset.K = train_dataset.K.cuda()

        test_dataset = SubjectLoader(subject_id=scene, root_fp=data_path, split="test", batch_size=batch_size,
                                supersampling=2 if supersampling == 'simple' else 1)

        res = [1080, 1920]
    elif scene in ["Jade", "Fountain", "Statues", "Character",]:
        dataset = "mvs"
        from src.data_ft.blendedMVS import SubjectLoader
        train_dataset = SubjectLoader(subject_id=scene, root_fp=data_path, split=train_split, batch_size=batch_size, color_bkgd_aug=color_bkgd_aug,
                                        supersampling=2 if supersampling == 'simple' else 1)
        # train_dataset.images = train_dataset.images.cuda()
        # train_dataset.camtoworlds = train_dataset.camtoworlds.cuda()
        # train_dataset.K = train_dataset.K.cuda()


        test_dataset = SubjectLoader(subject_id=scene, root_fp=data_path, split="test", batch_size=batch_size,
                                supersampling=2 if supersampling == 'simple' else 1)

        if scene in ['Jade', 'Fountain']:
            logger.info('BKGD is set to black by MVS dataset')
            train_dataset.color_bkgd_aug = 'black'
            test_dataset.color_bkgd_aug = 'black'
        elif scene in ['Character', 'Statues']:
            logger.info('BKGD is set to white by MVS dataset')
            train_dataset.color_bkgd_aug = 'white'
            test_dataset.color_bkgd_aug = 'white'

        res = [576, 768]
    elif scene in ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex",]:
        dataset = 'llff'
        from src.data_ft.llff import SubjectLoader
        train_dataset = SubjectLoader(subject_id=scene, root_fp=data_path, split=train_split, batch_size=batch_size, color_bkgd_aug=color_bkgd_aug,
                                        supersampling=2 if supersampling == 'simple' else 1)
        # train_dataset.images = train_dataset.images.cuda()
        # train_dataset.camtoworlds = train_dataset.camtoworlds.cuda()
        # train_dataset.K = train_dataset.K.cuda()
        test_dataset = SubjectLoader(subject_id=scene, root_fp=data_path, split="test", batch_size=batch_size,
                                supersampling=2 if supersampling == 'simple' else 1)

        res = [756, 1008]
        # vtx_pos = train_dataset.reverse_ndc(vtx_pos)
        # if export_mesh:
        #     mesh = trimesh.Trimesh(vtx_pos.cpu().detach().numpy(), pos_idx.detach().cpu().numpy(), process=False)
        #     print('exporting!'W)
        #     mesh.export("/".join([out_dir, "exported_mesh.ply"]))
        #     exit()

    elif "scan" in scene:
        dataset = 'dtu'
        from src.data_ft.dtu import DTU_Finetune
        train_dataset = DTU_Finetune(            
                root_dir=data_path,
                scan_id=scene,
                split="train",
                n_src_views=args.num_src_view,
                src_via_dist=args.src_via_dist,
                supersampling=2 if supersampling == 'simple' else 1,
                )

        # for attr_name in train_dataset.cuda_tensors:
        #     attr = getattr(train_dataset, attr_name)
        #     setattr(train_dataset, attr_name, attr.cuda())
        
        test_dataset = DTU_Finetune(            
                root_dir=data_path,
                scan_id=scene,
                split="test",
                n_src_views=args.num_src_view,
                src_via_dist=args.src_via_dist,
                supersampling=1,
                )

        res = [600, 800] # [512, 640]
        
    else:
        raise NotImplementedError("Invalid scene name: %s" % scene)

    train_dataloader = DataLoader(train_dataset,
                                batch_size=batch_size, 
                                num_workers=2, 
                                shuffle=False) 

    test_dataloader = DataLoader(test_dataset,
                                batch_size=batch_size, 
                                num_workers=1, 
                                shuffle=False) 

    # precompute projection matrix
    if 'scan' in args.scene: ## this is DTU
        offset = 0.031
    else:
        offset = 0
        
    proj = train_dataset.K.new_zeros([1, 4, 4]) # (num_images, 4, 4)
    proj[:, 0, 0] = train_dataset.K[0, 0] / res[1] *2# change mapping from (-400, +400) to (-1, +1)
    proj[:, 0, 2] = offset
    proj[:, 1, 1] = train_dataset.K[1, 1] / res[0] *2 * (-1 if train_dataset.OPENGL_CAMERA else 1) # change mapping from (-400, +400) to (-1, +1)
    proj[:, 1, 2] = offset
    proj[:, 2, 3] = -0.1 if hasattr(train_dataset, 'HAS_CLOSE') else -0.5
    proj[:, 3, 2] = -1 if train_dataset.OPENGL_CAMERA else 1 # w should never be negative
    proj = proj.cuda()

    # Load InstantNGP Network
    if renderer == "foundation-nerf":
        model = Gen_Shader(args=args)

        model.load_checkpoint(model_path)
        print("Model loaded:", model_path)
        
        model.cuda()

        uv_map = None
        uvs = None
        
        # for name, param in model.named_parameters():
        #     print(name, param.requires_grad)
        #     input()
        
    elif renderer == "UV_map":
        model = Gen_Shader(args=args)

        model.load_checkpoint(model_path)
        print("Model loaded:", model_path)
        
        model.cuda()
        
        uvs = torch.load(uv_path).detach()
        if uvmap_path is None:
            uv_map = torch.zeros([4096,4096,15]).cuda()
        else:
            uv_map = torch.load(uvmap_path, map_location="cuda:0")
    else:
        raise NotImplementedError

    # Initialize the post processing network
    # post_dim = 3
    # if post_use_depth:
    #     post_dim += 1

    # if post_net == "VDSR":
    #     post_model = VDSR(post_dim)
    # elif post_net == "SRCNN":
    #     post_model = SRCNN(post_dim)
    # elif post_net == "SRCNN_K3":
    #     post_model = SRCNN_K3(post_dim)
    # elif post_net == "SRCNN_K7":
    #     post_model = SRCNN_K7(post_dim)
    # elif post_net == "SRCNN_K9":
    #     post_model = SRCNN_K9(post_dim)
    # elif post_net == "SRCNN_L":
    #     post_model = SRCNN_L(post_dim)
    # elif post_net == "SRCNN_S":
    #     post_model = SRCNN_C8(post_dim)
    # elif post_net == "SRCNN_C8":
    #     post_model = SRCNN_C16(post_dim)
    # elif post_net == "SRCNN_C16":
    #     post_model = SRCNN_S(post_dim)
    # elif post_net == "SRCNN_GATE":
    #     assert post_use_depth
    #     post_model = Gated_CNN()
    # elif post_net != None:
    #     raise NotImplementedError
    # else:
    #     post_model = None

    # if post_model is not None:
    #     post_model.cuda()
    post_model = None

    # Adam optimizer for texture with a learning rate ramp.
    params_list = []
    # print(model)
    params_list.append(
        {'params': model.parameters(), 'lr': lr_base}
    )
    
    if post_net is not None:
        params_list.append(
            {'params': post_model.parameters(), 'lr': lr_base}
        )

    if optim_type == "Adam":
        optimizer = torch.optim.Adam(params_list)
    elif optim_type == "AdamW":
        optimizer = torch.optim.AdamW(params_list)

    if train_mesh:
        vtx_pos.requires_grad_()
        params_vtx = [{'params': vtx_pos, 'lr': lr_mesh},]
        if optim_type == "Adam":
            optimizer_vtx = torch.optim.Adam(params_vtx)
        elif optim_type == "AdamW":
            optimizer_vtx = torch.optim.AdamW(params_vtx)

    if uv_map is not None and train_uvmap:
        uv_map.requires_grad_()
        params_uv = [{'params': uv_map, 'lr': lr_uvmap},]
        if optim_type == "Adam":
            optimizer_uv = torch.optim.Adam(params_uv)
        elif optim_type == "AdamW":
            optimizer_uv = torch.optim.AdamW(params_uv)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_base*args.min_lr_scale)
    scheduler_vtx = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_vtx, T_max=epochs, eta_min=lr_mesh*args.min_lr_scale)

    # Set up loss function
    if loss_type == "l2":
        loss_func = torch.nn.MSELoss()
    elif loss_type == "l1":
        loss_func = torch.nn.L1Loss()
    else:
        logger.info(f"loss_type: {loss_type} is not supported")
    logger.info(f"loss_type: {loss_type} is used for training")

    glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()
    # Training

    epoch_train_psnr = []
    epoch_post_psnr = []
    epoch_train_masked_psnr = []
    epoch_train_loss = []


    pbar = tqdm(range(epochs))

    if prune_triangle:
        triangle_count = [pos_idx.shape[0],]


    logger.info("Start training")

    time_stamp0 = time.time()

    for epoch in pbar:
        train_psnr = []
        post_psnr = []
        train_masked_psnr = []
        train_loss = []
        
        if args.max_epoch is not None:
            if epoch >= args.max_epoch:
                break

        perm = torch.randperm(int(training_ratio*len(train_dataset)))

        if prune_triangle and epoch % 10 == 0:
            triangle_mask = torch.zeros(pos_idx.shape[0], device=pos_idx.device, dtype=bool, requires_grad=False)
            do_prune = True
        else:
            do_prune = False

        if grow_triangle and epoch % 20 == 0 and epoch > 0 and epoch <= epochs - 10:
            do_grow = True
        else:
            do_grow = False
        
        for i, data in enumerate(train_dataloader):
            model.train()
            
            # t1 = time.time()
            
            for key in data:
                if type(data[key]) == torch.Tensor:
                    data[key] = data[key].cuda()
                
            rays = data["rays"]
            images = data["pixels"]
            if "obj_masks" in data:
                obj_masks = data["obj_masks"]  # [B, H, W, 3]
                if len(obj_masks.shape) == 3:
                    obj_masks = obj_masks.unsqueeze(-1)
            else:
                obj_masks = None
            
            data["w2cs"] = data['extrinsic_render_view'].float()
            w2c = data["w2cs"]  # [B, 4, 4]
            if dataset == 'llff':
                world_pos = train_dataset.reverse_ndc(vtx_pos)
                proj_pos = w2clip2(world_pos, w2c, proj)
                proj_pos = train_dataset.ndc_y_rescale(proj_pos)
                # uvs = vtx_to_ndc(train_dataset.HEIGHT, train_dataset.WIDTH, train_dataset.focal[0], 1.0, vtx_pos)
                uvs = vtx_pos.detach()
            # elif dataset == 'mvs':
            #     proj_pos = w2clip2(vtx_pos, w2c, proj)
            else:
                proj_pos = w2clip2(vtx_pos, w2c, proj)
            # writemesh2ply(verts=proj_pos[0], faces=pos_idx, ply_filename_out="temp.ply")
            # if export_mesh:
            #     mesh = trimesh.Trimesh((proj_pos[0,:,:3]/proj_pos[0,:,3,None]).cpu().detach().numpy(), pos_idx.detach().cpu().numpy(), process=False)
            #     print('exporting!')
            #     mesh.export("/".join([out_dir, "exported_mesh.ply"]))
            #     imageio.imwrite("/".join([img_dir, "orig_img%d_epoch_%d.jpg" % (i,epoch)]), (images[0] * 255).cpu().numpy().astype(np.uint8))
            # exit()
            
            # t2 = time.time()
            
            pred_color, extra_output, pixel_feature, depth = render(
                data,
                glctx,
                vtx_pos,
                proj_pos,
                pos_idx,
                uv_map,
                uvs,
                rays[0],
                rays[1],
                res,
                renderer,
                model,
                train_dataset.color_bkgd_aug,
                antialias=antialias,
                requires_triangle_reference=True,
                gen_depth_graph=False,
                supersampling=supersampling,
                filter_mode=filter_mode,
            )
            occupy_graph = extra_output.get('occupy_graph', None)
            depth_graph = extra_output.get('depth_graph', None)
            
            if do_prune:
                if obj_masks is not None:
                    ind_masks = obj_masks[..., 0].view(-1) > 0
                    extra_output['triangle_ref'] = extra_output['triangle_ref'][ind_masks]
                triangle_mask[extra_output['triangle_ref']] = True
            
            if post_net is not None:
                with torch.no_grad():
                    post_embedding = pred_color
                    if post_use_depth:
                        post_embedding = torch.cat([post_embedding, extra_output['depth_graph']], dim=-1)
                post_color = post_model(post_embedding.detach())
                
            t3 = time.time()
            
            # Save images
            if i < 4 and epoch % 10 == 0 and export_image:
            # if epoch % 10 == 0 and export_image:
                if depth_graph is not None:
                    imageio.imwrite("/".join([img_dir, "depth_img%d_epoch_%d.jpg" % (i, epoch)]), (depth_graph[0] * 255).detach().cpu().numpy().astype(np.uint8))
                imageio.imwrite("/".join([img_dir, "pred_image%d_epoch_%d.jpg" % (i, epoch)]), (pred_color[0] * 255).detach().cpu().numpy().astype(np.uint8))
                imageio.imwrite("/".join([img_dir, "orig_img%d_epoch_%d.jpg" % (i,epoch)]), (images[0] * 255).cpu().numpy().astype(np.uint8))
                if obj_masks is not None:
                    imageio.imwrite("/".join([img_dir, "error_img%d_epoch_%d.jpg" % (i,epoch)]), (abs(pred_color[0]-images[0]) * obj_masks[0] * 255).cpu().detach().numpy().astype(np.uint8))
                else:
                    imageio.imwrite("/".join([img_dir, "error_img%d_epoch_%d.jpg" % (i,epoch)]), (abs(pred_color[0]-images[0]) * 255).cpu().detach().numpy().astype(np.uint8))
            # Compute loss and perform a training step
            loss = 0.0
            if obj_masks is not None:
                loss_pred_color = loss_func(pred_color * obj_masks, images * obj_masks)
            else:
                loss_pred_color = loss_func(pred_color, images)
                
            if post_net is not None:
                if obj_masks is not None:
                    loss_post_color = loss_func(post_color * obj_masks, images * obj_masks)
                else:
                    loss_post_color = loss_func(post_color, images)
                loss += loss_post_color*1000

            loss += loss_pred_color * 1000
            optimizer.zero_grad()
            if train_mesh:
                optimizer_vtx.zero_grad()
            if train_uvmap:
                optimizer_uv.zero_grad()

            loss.backward()
            optimizer.step()
            
            if train_mesh and epoch >= train_mesh_epoch and not (prune_triangle_keep_static and do_prune):
                optimizer_vtx.step()
            if train_uvmap:
                optimizer_uv.step()

            train_loss.append(loss.item())

            with torch.no_grad():
                if obj_masks is not None:
                    full_mes_loss = F.mse_loss(pred_color * obj_masks, images * obj_masks)
                else:
                    full_mes_loss = F.mse_loss(pred_color, images)
                train_psnr.append((-10.0 * torch.log(full_mes_loss) / np.log(10.0)).item())
                if supersampling == None:
                    masked_mse_loss = F.mse_loss(pred_color*occupy_graph.detach(), images*occupy_graph.detach())
                    train_masked_psnr.append((-10.0 * torch.log(masked_mse_loss) / np.log(10.0)).item())
                else:
                    train_masked_psnr.append(0)

                if post_net is not None:
                    if obj_masks is not None:
                        full_post_loss = F.mse_loss(post_color * obj_masks, images * obj_masks)
                    else:
                        full_post_loss = F.mse_loss(post_color, images)
                    post_psnr.append((-10.0 * torch.log(full_post_loss) / np.log(10.0)).item())
            
            # t4 = time.time()
            # print(t2-t1,t3-t2,t4-t3)
            
        scheduler.step()
        scheduler_vtx.step()

        if do_prune:
            pos_idx = pos_idx[triangle_mask]
            triangle_count.append(triangle_mask.sum())
            logger.info("Pruned %d triangles from %d triangles" % (len(triangle_mask) - triangle_mask.sum().cpu().item(), len(triangle_mask)))

        if do_grow:
            with torch.no_grad():
                vtx_edges = vtx_pos[pos_idx.long()]
                vtx_distances = torch.cat([vtx_edges[:,0:1] - vtx_edges[:,1:2], vtx_edges[:,1:2] - vtx_edges[:,2:3], vtx_edges[:,2:3] - vtx_edges[:,0:1]], dim=1)
                vtx_distances = torch.linalg.norm(vtx_distances, axis=-1).amax(dim=-1)
                large_triangle_mask = vtx_distances > vtx_distances.mean() * 2
                pos_idx, vtx_pos = subdivide_large_triangles(pos_idx, vtx_pos, large_triangle_mask)
                logger.info("Subdivided %d large triangles from %d triangles" % (large_triangle_mask.int().sum().cpu().item(), len(large_triangle_mask)))
                
            vtx_pos.requires_grad_()
            params_vtx = [{'params': vtx_pos, 'lr': lr_mesh},]
            if optim_type == "Adam":
                optimizer_vtx = torch.optim.Adam(params_vtx)
            elif optim_type == "AdamW":
                optimizer_vtx = torch.optim.AdamW(params_vtx)

        # TODO: do we need to empty cache?
        torch.cuda.empty_cache()

        # Logging
        train_loss = sum(train_loss)/len(train_loss)
        train_psnr = sum(train_psnr)/len(train_psnr)
        train_masked_psnr = sum(train_masked_psnr)/len(train_masked_psnr)

        epoch_train_psnr.append(train_psnr)
        epoch_train_loss.append(train_loss)
        epoch_train_masked_psnr.append(train_masked_psnr)

        if post_net is not None:
            post_psnr = sum(post_psnr)/len(post_psnr)
            epoch_post_psnr.append(post_psnr)
        else:
            post_psnr = 0

        writer.add_scalar('train_psnr', train_psnr, epoch)

        logger.info(f"train_loss: {train_loss}, train_psnr: {train_psnr}, post_psnr: {post_psnr}, train_masked_psnr: {train_masked_psnr}")
        # scheduler.step()

        if (epoch+1) == epochs or (epoch+1) % eval_every == 0:
            model.eval()
            if post_net is not None:
                post_model.eval()
            
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
                requires_time=False,
                export_image=export_image,
                gen_log=True,
                writer=writer,
                test_epoch=epoch,
                save_on_eval=save_on_eval,
                logger=logger,
                args=args
            )


            torch.save(
                model.state_dict(), 
                "/".join([out_dir, f"{epoch}.pt"])
           )
            if export_mesh:
                mesh = trimesh.Trimesh(vtx_pos.detach().cpu().numpy(), pos_idx.detach().cpu().numpy(), process=False)
                mesh.export("/".join([out_dir, f"mesh_{epoch}.ply"]))

            model.train()
            if post_net is not None:
                post_model.train()

        writer.add_scalar('time', time.time() - time_stamp0, epoch)

    if prune_triangle:
        logger.info(f"triangle_count: {str(triangle_count)}")

    del train_dataset
    
    # Drawing Results
    plt.figure()
    plt.title("train_psnr&masked_train_psnr")
    plt.plot(np.arange(epochs), epoch_train_psnr)
    plt.plot(np.arange(epochs), epoch_train_masked_psnr)
    plt.ylim(min(min(epoch_train_psnr), min(epoch_train_masked_psnr)) - 0.5, 
                max(max(epoch_train_psnr), max(epoch_train_masked_psnr)) + 0.5)

    plt.legend(["train_psnr", "masked_train_psnr"], loc="lower right")
    plt.savefig("/".join([out_dir, "train_psnr.png"]))

    # Testing
    model.eval()

    if post_net is not None:
        post_model.eval()
    if uv_map is not None:
        uv_map = uv_map.detach()
    vtx_pos = vtx_pos.detach()
    

    # Saving
    torch.save(model.state_dict(), "/".join([out_dir, f"final.pt"]))

    if post_net is not None:
        torch.save(post_model.state_dict(), "/".join([out_dir, "post_model.pt"]))

    if export_mesh:
        mesh = trimesh.Trimesh(vtx_pos.cpu().numpy(), pos_idx.cpu().numpy(), process=False)
        mesh.export("/".join([out_dir, "final_mesh.ply"]))

    if uv_map is not None:
        torch.save(uv_map, "/".join([out_dir, "uv_map.pt"]))
    if uvs is not None:
        torch.save(uvs, "/".join([out_dir, "uvs.pt"]))