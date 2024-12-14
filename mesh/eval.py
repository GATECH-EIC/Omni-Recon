import time
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
import trimesh
import pickle

from mesh.post_net import *
from mesh.render import render
from mesh.utils import w2clip2, colorize_np
import tqdm

@torch.no_grad()
def evaluate(
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
    export_image=False,
    gen_log=False,
    writer=None,
    writer_prefix='',
    test_epoch=0,
    save_on_eval=False,
    test_spiral=False,
    logger=None,
    args=None
):
    if export_image:
        img_dir = '/'.join([out_dir, 'eval'])
        os.makedirs(img_dir, exist_ok=True)

    # for attr_name in test_dataset.cuda_tensors:
    #     attr = getattr(test_dataset, attr_name)
    #     setattr(test_dataset, attr_name, attr.cuda())
    
    # near, far = test_dataset.near, test_dataset.far
    
    if 'scan' in args.scene: ## this is DTU
        offset = 0.031
    else:
        offset = 0

    proj = test_dataset.K.new_zeros([1, 4, 4]) # (num_images, 4, 4)
    proj[:, 0, 0] = test_dataset.K[0, 0] / res[1] * 2 # change mapping from (-400, +400) to (-1, +1)
    proj[:, 0, 2] = offset
    proj[:, 1, 1] = test_dataset.K[1, 1] / res[0] * 2 * (-1 if test_dataset.OPENGL_CAMERA else 1) # change mapping from (-400, +400) to (-1, +1)
    proj[:, 1, 2] = offset
    proj[:, 2, 2] = 0 # (far + near) / (far - near)
    proj[:, 2, 3] = -0.1 if hasattr(test_dataset, 'HAS_CLOSE') else -0.5 # -2 * far * near / (far - near) 
    proj[:, 3, 2] = -1 if test_dataset.OPENGL_CAMERA else 1 # w should never be negative
    proj = proj.cuda()

    test_order = np.arange(0, len(test_dataset))

    freams = []
    gt_freams = []

    test_psnr = []
    post_psnr = []
    test_masked_psnr = []
    test_batch_size = batch_size
    total_time_rast, total_time_ngp, total_time_post = 0.0, 0.0, 0.0
    cnt = 0
    for i, data in enumerate(test_dataloader):
        ### visualize source views
        # imageio.imwrite("/".join([out_dir, "target.jpg"]), (data["pixels"][0] * 255).cpu().numpy().astype(np.uint8))
        # src_imgs = data["source_imgs"][0]
        # for k in range(src_imgs.shape[0]):
        #     imageio.imwrite("/".join([out_dir, "src_%d.jpg" % (k,)]), (src_imgs[k].permute(1,2,0) * 255).cpu().numpy().astype(np.uint8))
        # print('saved!!!')
        # input()
        
        for key in data:
            if type(data[key]) == torch.Tensor:
                data[key] = data[key].cuda()

        # K = data["K"][0]
        # proj = K.new_zeros([1, 4, 4]) # (num_images, 4, 4)
        # proj[:, 0, 0] = K[0, 0] / res[1] * 2 # change mapping from (-400, +400) to (-1, +1)
        # proj[:, 0, 2] = 0
        # proj[:, 1, 1] = K[1, 1] / res[0] * 2 * (-1 if test_dataset.OPENGL_CAMERA else 1) # change mapping from (-400, +400) to (-1, +1)
        # proj[:, 1, 2] = 0
        # proj[:, 2, 2] = 0 # (far + near) / (far - near)
        # proj[:, 2, 3] = -0.1 if hasattr(test_dataset, 'HAS_CLOSE') else -0.5 # -2 * far * near / (far - near) 
        # proj[:, 3, 2] = -1 if test_dataset.OPENGL_CAMERA else 1 # w should never be negative
        # proj = proj.cuda()

        rays = data["rays"]
        images = data["pixels"]  # [B, H, W, 3]
        if "obj_masks" in data:
            obj_masks = data["obj_masks"]  # [B, H, W, 3]
            if len(obj_masks.shape) == 3:
                obj_masks = obj_masks.unsqueeze(-1)
        else:
            obj_masks = None
        data["w2cs"] = data['extrinsic_render_view'].float()
        w2c = data["w2cs"]  # [B, 4, 4]
        if dataset == 'llff':
            world_pos = test_dataset.reverse_ndc(vtx_pos)
            proj_pos = w2clip2(vtx_pos, w2c, proj)
            proj_pos = test_dataset.ndc_y_rescale(proj_pos)
            # uvs = vtx_to_ndc(train_dataset.HEIGHT, train_dataset.WIDTH, train_dataset.focal[0], 1.0, vtx_pos)
            uvs = vtx_pos
        else:
            proj_pos = w2clip2(vtx_pos, w2c, proj) # [B, num_vertices, 4]

        # pred_color, occupy_graph, depth_graph, rast_time, ngp_time = render(
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
            test_dataset.color_bkgd_aug,
            requires_time=requires_time,
            gen_depth_graph=True,
            antialias = antialias,
            supersampling=supersampling,
            filter_mode=filter_mode,
        )

        if requires_time:
            torch.cuda.current_stream().synchronize()
            time1 = time.time()

        if post_model is not None:
            if post_use_depth:
                post_embedding = torch.cat([pred_color, extra_output['depth_graph']], dim=-1)
            post_color = post_model(post_embedding)
            post_color = torch.clip(post_color, 0, 1)

        if requires_time:
            torch.cuda.current_stream().synchronize()
            time2 = time.time()
            post_time = time2 - time1

        if obj_masks is not None: 
            pred_color = pred_color * obj_masks
            images = images * obj_masks
            
        if test_spiral:
            imageio.imwrite("/".join([out_dir, "img%d_pred.jpg" % (i,)]), (pred_color[0] * 255).detach().cpu().numpy().astype(np.uint8))
        elif export_image:
            # imageio.imwrite("/".join([out_dir, "img%d_depth.jpg" % (i,)]), (extra_output['depth_graph'][0] * 255).detach().cpu().numpy().astype(np.uint8))
            imageio.imwrite("/".join([out_dir, "img%d_orig.jpg" % (i,)]), (images[0] * 255).cpu().numpy().astype(np.uint8))
            if post_model is None:
                imageio.imwrite("/".join([out_dir, "img%d_pred.jpg" % (i,)]), (pred_color[0] * 255).detach().cpu().numpy().astype(np.uint8))
                if obj_masks is not None:
                    imageio.imwrite("/".join([out_dir, "img%d_error.jpg" % (i,)]), (abs(pred_color[0]-images[0]) * obj_masks[0] * 255).detach().cpu().numpy().astype(np.uint8))
                else:
                    imageio.imwrite("/".join([out_dir, "img%d_error.jpg" % (i,)]), (abs(pred_color[0]-images[0]) * 255).detach().cpu().numpy().astype(np.uint8))
            else:
                imageio.imwrite("/".join([out_dir, "img%d_pred.jpg" % (i,)]), (pred_color[0] * 255).detach().cpu().numpy().astype(np.uint8))
                if obj_masks is not None:
                    imageio.imwrite("/".join([out_dir, "img%d_pred_error.jpg" % (i,)]), (abs(pred_color[0]-images[0]) * obj_masks[0] * 255).detach().cpu().numpy().astype(np.uint8))
                else:
                    imageio.imwrite("/".join([out_dir, "img%d_pred_error.jpg" % (i,)]), (abs(pred_color[0]-images[0]) * 255).detach().cpu().numpy().astype(np.uint8))

            ### save depth
            # print(depth.max(), depth.min())
            imageio.imwrite("/".join([out_dir, "depth_%d.png" % (i,)]), (depth[0] * 1000).detach().cpu().numpy().astype(np.uint16))
            depth_colored = colorize_np(depth[0].detach().cpu().numpy(), range=(depth.min().item(), depth.max().item()))
            imageio.imwrite("/".join([out_dir, "depth_vis_%d.png" % (i,)]), (depth_colored * 255).astype(np.uint8))
            
            for pred_i, gt_i in zip(pred_color, images):
                freams.append(pred_i.cpu().numpy())
                gt_freams.append(gt_i.cpu().numpy())

        if requires_time:
            if i >= len(test_dataloader)//2:
                total_time_rast += extra_output['rast_time']
                total_time_ngp += extra_output['ngp_time']
                total_time_post += post_time
                cnt += 1
        
        if obj_masks is not None:
            full_mse_loss = F.mse_loss(pred_color*obj_masks, images*obj_masks)
        else:
            full_mse_loss = F.mse_loss(pred_color, images)
        test_psnr.append((-10.0 * torch.log(full_mse_loss) / np.log(10.0)).item())
        if supersampling == None:
            masked_mse_loss = F.mse_loss(pred_color*extra_output['occupy_graph'].detach(), images*extra_output['occupy_graph'].detach())
            test_masked_psnr.append((-10.0 * torch.log(masked_mse_loss) / np.log(10.0)).item())
        else:
            test_masked_psnr.append(0)

        if post_model is not None:
            if obj_masks is not None:
                post_mse_loss = F.mse_loss(post_color*obj_masks, images*obj_masks)
            else:
                post_mse_loss = F.mse_loss(post_color, images)
            post_psnr.append((-10.0 * torch.log(post_mse_loss) / np.log(10.0)).item())

    # Logging
    test_psnr = sum(test_psnr)/len(test_psnr)
    test_masked_psnr = sum(test_masked_psnr)/len(test_masked_psnr)

    if post_model is not None:
        post_psnr = sum(post_psnr)/len(post_psnr)
    else:
        post_psnr = 0

    if gen_log:
        if requires_time:
            logger.info(f"test_psnr: {test_psnr}, post_psnr: {post_psnr}, test_masked_psnr: {test_masked_psnr}, ave_rast_t: {total_time_rast/cnt}s, avg_ngp_t: {total_time_ngp/cnt}s\n"
                    f"avg_post_t: {total_time_post/cnt}")
        else:
            logger.info(f"test_psnr: {test_psnr}, post_psnr: {post_psnr}, test_masked_psnr: {test_masked_psnr}")

    if writer is not None:
        if writer_prefix is not None:
            prefix = writer_prefix
        else:
            prefix = ''
        writer.add_scalar(prefix + 'test_psnr', test_psnr, test_epoch)

    if save_on_eval:
        torch.save(model, "/".join([out_dir, prefix + "model_finetuned.pt"]))
        if post_model is not None:
            torch.save(post_model, "/".join([out_dir, prefix + "post_model.pt"]))

        if True:
            mesh = trimesh.Trimesh(vtx_pos.cpu().numpy(), pos_idx.cpu().numpy(), process=False)
            mesh.export("/".join([out_dir, prefix + "exported_mesh.ply"]))

        if uv_map is not None:
            torch.save(uv_map, "/".join([out_dir, prefix + "uv_map.pt"]))
        if uvs is not None:
            torch.save(uvs, "/".join([out_dir, prefix + "uvs.pt"]))

    # for attr_name in test_dataset.cuda_tensors:
    #     attr = getattr(test_dataset, attr_name)
    #     setattr(test_dataset, attr_name, attr.cpu())

    if export_image:
        freams = np.array(freams, np.float32)
        gt_freams = np.array(gt_freams, np.float32)

        pickle.dump(freams, open(out_dir + "/pred_frames.pkl", "wb"))
        pickle.dump(gt_freams, open(out_dir + "/gt_frames.pkl", "wb"))
