import time
import math

import torch
import nvdiffrast.torch as dr

from einops import rearrange

def render(
    data,
    glctx,
    vtx_pos,
    proj_pos,
    pos_idx,
    uv_map,
    uvs,
    rays_origin, 
    rays_direction,
    res,
    renderer,
    model,
    color_bkgd_aug,
    antialias=False,
    gen_occupy_graph=True,
    gen_depth_graph=False,
    requires_time=False,
    requires_triangle_reference=False,
    supersampling='none',
    filter_mode='linear',
    ):

    if supersampling == 'simple' or supersampling == 'defer':
        res = [r*2 for r in res]
    elif supersampling != 'none':
        raise NotImplementedError

    r_res = [math.ceil(res[0]/8.)*8, math.ceil(res[1]/8.)*8]
    crop = [(r_res[0]-res[0])//2, (r_res[1]-res[1])//2]
    def crop_func(x: torch.Tensor):
        if crop[0] != 0:
            x = x[:,crop[0]:-crop[0]]
        if crop[1] != 0:
            x = x[:,:,crop[1]:-crop[1]]
        return x.contiguous()

    if requires_time:
        time_1 = time.time()

    pixel_feature, rast_out, valid_idx = rast(
        glctx,
        proj_pos,
        pos_idx,
        r_res,
        renderer,
        vtx_feature = vtx_pos if uvs is None else uvs,
        uv_map = uv_map,
        filter_mode = filter_mode,
    )

    pixel_feature = crop_func(pixel_feature)
    rast_out = crop_func(rast_out)
    valid_idx = crop_func(valid_idx)

    if requires_time:
        torch.cuda.current_stream().synchronize()
        time_2 = time.time()

    if supersampling == 'defer':
        pixel_feature[~valid_idx] = 0

        if antialias == True:
            pixel_feature = dr.antialias(pixel_feature, rast_out, proj_pos, pos_idx)

        pixel_feature = torch.mean(pixel_feature.view(pixel_feature.shape[0],res[0]//2,2, res[1]//2,2,pixel_feature.shape[-1]), (2,4))
        rast_out = torch.mean(rast_out.view(rast_out.shape[0],res[0]//2,2, res[1]//2,2,rast_out.shape[-1]), (2,4))
        valid_idx = torch.sum(valid_idx.view(valid_idx.shape[0],res[0]//2,2, res[1]//2,2).int(), (2,4)) > 0
        res = [r//2 for r in res]
    
    pred_color = render_color(
        data,
        pixel_feature,
        valid_idx,
        rays_direction,
        res,
        renderer,
        model,
    )

    if color_bkgd_aug == "white":
        pred_color[~valid_idx] = 1
    elif color_bkgd_aug == "black":
        pred_color[~valid_idx] = 0
    
    if antialias == True and supersampling!='defer':
        pred_color = dr.antialias(pred_color, rast_out, proj_pos, pos_idx)

    if supersampling == 'simple':
        pred_color = torch.mean(pred_color.view(pred_color.shape[0],res[0]//2,2, res[1]//2,2,pred_color.shape[-1]), (2,4))

    if requires_time:
        torch.cuda.current_stream().synchronize()
        time_3 = time.time()
    
    H, W = pixel_feature.shape[1], pixel_feature.shape[2]
    xyzw = torch.cat([pixel_feature, torch.ones_like(pixel_feature[..., :1])], dim=-1)  # [B, H, W, 4]
    xyzw = rearrange(xyzw, "B H W DimX -> B DimX (H W)")
    xyzw_cam = torch.bmm(data["w2cs"], xyzw)  # [B, 4, H*W]
    xyzw_cam = rearrange(xyzw_cam, "B DimX (H W) -> B H W DimX", H=H, W=W)
    depth = xyzw_cam[..., 2]
    # depth = rast_out[..., 2] ## comparable with the above with different scales
    
    extra_output = {}
    if gen_occupy_graph:
        occupy_graph = valid_idx.unsqueeze(-1).expand([*valid_idx.shape, 3]).float()
        extra_output['occupy_graph'] = occupy_graph
    if gen_depth_graph:
        depth_graph = rast_out[...,-2].detach().unsqueeze(-1).expand([*valid_idx.shape, 1]).float()
        extra_output['depth_graph'] = depth_graph
    if requires_time:
        extra_output['rast_time'] = time_2 - time_1
        extra_output['ngp_time'] = time_3 - time_2
    if requires_triangle_reference:
        extra_output['triangle_ref'] = torch.clone(rast_out[...,-1]).detach().long().flatten() -1
        
    return pred_color, extra_output, pixel_feature, depth

def rast(
    glctx, 
    proj_pos,
    pos_idx, 
    r_res,
    renderer,
    vtx_feature,
    uv_map=None,
    filter_mode='linear',
):
    with dr.DepthPeeler(glctx, proj_pos, pos_idx, resolution=r_res) as peeler:
        # rasterization for each layer
        rast_out, rast_out_db = peeler.rasterize_next_layer()

        if renderer == "foundation-nerf":
            pixel_feature, _  = dr.interpolate(vtx_feature, rast_out, pos_idx)  # [B, height, width, 3]
        elif renderer == "UV_map":
            pixel_uv, pixel_uv_db = dr.interpolate(vtx_feature, rast_out, pos_idx, rast_out_db, diff_attrs="all")
            pixel_feature = dr.texture(uv_map[None, ...].contiguous(), pixel_uv, uv_da=pixel_uv_db, filter_mode=filter_mode, max_mip_level=0)
        else:
            raise NotImplementedError

    # TODO: Support Multi-depth Rendering
    valid_idx = rast_out[...,-1] > 0

    return pixel_feature, rast_out, valid_idx


def render_color(
    data,
    pixel_feature,
    valid_idx,
    rays_direction,
    res,
    renderer,
    model,
):     

    # valid_rays_d = rays_direction[valid_idx]
    # valid_feature = pixel_feature[valid_idx]
    
    # print(pixel_feature.shape) # [B, H, W, 3]
    # print(valid_idx.shape)  # [B, H, W]
    # print(valid_feature.shape)  # [num_valid, 3]
      
    pred_color = torch.empty([int(rays_direction.shape[0]), res[0], res[1], 3]).to(pixel_feature.device)

    if renderer == "foundation-nerf":
        radiance, _, _ = model(pixel_feature, data) # model(valid_feature, data)
        pred_color[valid_idx] = radiance.view(pred_color.shape[0], res[0], res[1], 3)[valid_idx]
        
    elif renderer == "UV_map":
        valid_rays_d = rays_direction[valid_idx]
        valid_feature = pixel_feature[valid_idx]
        rgb = model.query_rgb(valid_rays_d, valid_feature)
        pred_color[valid_idx] = rgb
    else:
        raise NotImplementedError

    return pred_color

