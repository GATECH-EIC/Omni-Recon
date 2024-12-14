import numpy as np
import torch
import plyfile
import trimesh
from copy import deepcopy

import cv2

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib import cm

HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6     # float32 only has 7 decimal digits precision

    
def w2clip(vtx_3dpos: torch.Tensor, w2c, camera_intrinsic, height, width):
    """Project Input 3D Positions into Clip Space of given Camera"""
    assert w2c is None or w2c.dim() == 3, "w2c must in shape (num_images, 4, 4)"
    if w2c is None:
        w2c = torch.eye(4).to(vtx_3dpos.device)[None, None, ...]
    else:
        w2c = w2c.unsqueeze(1) # (N, 1, 4, 4)

    proj = camera_intrinsic.new_zeros([w2c.shape[0], 4, 4]) # (num_images, 4, 4)
    proj[:, 0, 0] = camera_intrinsic[0, 0] / width *2# change mapping from (-400, +400) to (-1, +1)
    proj[:, 0, 2] = 0
    proj[:, 1, 1] = -camera_intrinsic[1, 1] / height *2# change mapping from (-400, +400) to (-1, +1)
    proj[:, 1, 2] = 0
    proj[:, 2, 3] = -1
    proj[:, 3, 2] = -1 # w should never be negative

    if vtx_3dpos.dim() == 2:
        vtx_3dpos = vtx_3dpos.unsqueeze(-2).unsqueeze(0) # (1, V, 1, 3)
    if vtx_3dpos.size(-1) == 3:
        vtx_3dpos = torch.concat([vtx_3dpos, vtx_3dpos.new_ones((*vtx_3dpos.shape[:-1],1))], dim=-1)

    # World Coordinate ==> Camera Coordinate
    vtx_camera3dpos = (vtx_3dpos * w2c).sum(-1) # (N, V, 4)
    # Camera Coordinate ==> Clip Coordinate
    vtx_camera3dpos = vtx_camera3dpos.unsqueeze(-2) # (N, V, 1, 4)
    proj = proj.unsqueeze(1) # (N, 1, 4, 4)

    vtx_clip3dpos = (vtx_camera3dpos * proj).sum(-1) # (N, V, 4)

    return vtx_clip3dpos


def w2clip2(vtx_pos3d: torch.Tensor, w2c, proj):
    """Project Input 3D Positions into Clip Space of given Camera"""
    assert w2c.dim() == 3, "w2c must in shape (num_images, 4, 4)"
    assert proj.dim() == 3, "proj matrix must in shape (1, 4, 4)"

    if vtx_pos3d.dim() == 2:
        vtx_pos3d = vtx_pos3d.unsqueeze(-2).unsqueeze(0) # (1, V, 1, 3)
    if vtx_pos3d.size(-1) == 3:
        vtx_pos3d = torch.concat([vtx_pos3d, vtx_pos3d.new_ones((*vtx_pos3d.shape[:-1],1))], dim=-1)

    w2i = torch.einsum("nki,jk->nji", [w2c, proj[0]]).unsqueeze(1) # (num_images, 4, 4)
    vtx_clip3dpos = (vtx_pos3d * w2i).sum(-1) # (N, V, 4)

    return vtx_clip3dpos

@torch.no_grad()
def vtx_to_ndc(H, W, focal, near, vtx_3dpos):
    # Shift ray origins to near plane
    # Projection
    o0 = -1. / (W / (2. * focal)) * vtx_3dpos[..., 0] / vtx_3dpos[..., 2]
    o1 = -1. / (H / (2. * focal)) * vtx_3dpos[..., 1] / vtx_3dpos[..., 2]
    o2 = 1. + 2. * near / vtx_3dpos[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)

    return rays_o

def writemesh2ply(verts: torch.Tensor, faces: torch.Tensor, ply_filename_out):
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        temp0 = verts[i, :3]/verts[i,3]
        temp = tuple(temp0)
        verts_tuple[i] = temp 

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

@torch.no_grad()
def inherite_model_rank(model: torch.nn.Module):
    for _, param in model.named_parameters():

        if param.numel() == 0:
            continue

        mean = param.mean()
        std_var = param.var().sqrt()
        _, rank_old = param.sort()

        new_param = param.new_empty(param.shape)
        new_param.normal_(mean, std_var)
        new_param, _ = new_param.sort()
        new_param = new_param[rank_old]

        param.copy_(new_param)
    
    return model

@torch.no_grad()
def inherite_head_rank(model: torch.nn.Module):
    # for _, param in model.mlp_head.named_parameters():

    #     if param.numel() == 0:
    #         continue

    #     mean = param.mean()
    #     std_var = param.var().sqrt()
    #     _, rank_old = param.sort()

    #     new_param = param.new_empty(param.shape)
    #     new_param.normal_(mean, std_var)
    #     new_param, _ = new_param.sort()
    #     new_param = new_param[rank_old]

    #     param.copy_(new_param)

    for _, param in model.mlp_head.named_parameters():

        if param.numel() == 0:
            continue

        mean = param.mean()
        std_var = param.var().sqrt()
        _, rank_old = param.sort()

        new_param = param.new_empty(param.shape)
        new_param.normal_(mean, std_var)
        new_param, _ = new_param.sort()
        new_param = new_param[rank_old]

        param.add_(new_param*0.1)


    return model

def add_noisy_mesh(mesh: trimesh.Trimesh, add_mesh_group, mesh_noise_scale):
    if mesh_noise_scale == 0.:
        mesh_noise_scale = 1./512 *0.5

    pos_idx = mesh.faces
    pos = mesh.vertices

    offset = pos.shape[0]

    for _ in range(add_mesh_group):
        new_mesh = mesh.permutate.noise(mesh.scale, mesh_noise_scale)
        new_pos_idx = new_mesh.faces + offset
        new_pos = new_mesh.vertices

        pos_idx = np.concatenate([pos_idx, new_pos_idx], axis=0)
        pos = np.concatenate([pos, new_pos], axis=0)
        offset = pos.shape[0]

    return pos, pos_idx


@torch.no_grad()
def subdivide_large_triangles(pos_idx, vtx_pos, large_triangle_mask):
    large_triangle_idx = pos_idx[large_triangle_mask] # (K,3)
    large_triangle_vtx_pos = vtx_pos[large_triangle_idx.long()] # (K,3,3)

    large_triangle_edge_mean = torch.cat([
        large_triangle_vtx_pos[:,[0,1],:].mean(dim=1, keepdim=True), 
        large_triangle_vtx_pos[:,[1,2],:].mean(dim=1, keepdim=True),
        large_triangle_vtx_pos[:,[2,0],:].mean(dim=1, keepdim=True)],
        dim=1) # (K,3,3)
    
    new_vtx_pos = torch.cat([large_triangle_vtx_pos, large_triangle_edge_mean], dim=1) # (K,6,3)
    new_vtx_pos = new_vtx_pos[:, [0,3,5,4,3,1,5,3,4,5,2,4], :].reshape(-1,3) # (K*12,3)
    new_vtx_pos = torch.cat([vtx_pos, new_vtx_pos], dim=0) # (N+K*12,3)

    pos_idx = pos_idx[~large_triangle_mask] # (N',3)
    new_pos_idx = torch.arange(large_triangle_edge_mean.shape[0]*12, dtype=torch.int32).reshape(-1,3).to(large_triangle_idx.device) # (K*4,3)
    new_pos_idx = new_pos_idx + vtx_pos.shape[0] # (K*4,3)
    new_pos_idx = torch.cat([pos_idx, new_pos_idx], dim=0) # (N'+K*4,3)

    return new_pos_idx, new_vtx_pos



def colorize_np(x, cmap_name='jet', mask=None, range=None, append_cbar=False, cbar_in_image=False, cbar_precision=2):
    '''
    turn a grayscale image into a color image
    :param x: input grayscale, [H, W]
    :param cmap_name: the colorization method
    :param mask: the mask image, [H, W]
    :param range: the range for scaling, automatic if None, [min, max]
    :param append_cbar: if append the color bar
    :param cbar_in_image: put the color bar inside the image to keep the output image the same size as the input image
    :return: colorized image, [H, W]
    '''
    if range is not None:
        vmin, vmax = range
    elif mask is not None:
        # vmin, vmax = np.percentile(x[mask], (2, 100))
        vmin = np.min(x[mask][np.nonzero(x[mask])])
        vmax = np.max(x[mask])
        # vmin = vmin - np.abs(vmin) * 0.01
        x[np.logical_not(mask)] = vmin
        # print(vmin, vmax)
    else:
        vmin, vmax = np.percentile(x, (1, 100))
        vmax += TINY_NUMBER

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin)
    # x = np.clip(x, 0., 1.)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.ones_like(x_new) * (1. - mask)

    cbar = get_vertical_colorbar(h=x.shape[0], vmin=vmin, vmax=vmax, cmap_name=cmap_name, cbar_precision=cbar_precision)

    if append_cbar:
        if cbar_in_image:
            x_new[:, -cbar.shape[1]:, :] = cbar
        else:
            x_new = np.concatenate((x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1)
        return x_new
    else:
        return x_new
    

def get_vertical_colorbar(h, vmin, vmax, cmap_name='jet', label=None, cbar_precision=2):
    '''
    :param w: pixels
    :param h: pixels
    :param vmin: min value
    :param vmax: max value
    :param cmap_name:
    :param label
    :return:
    '''
    fig = Figure(figsize=(2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting.
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    ticks=tick_loc,
                                    orientation='vertical')

    tick_label = [str(np.round(x, cbar_precision)) for x in tick_loc]
    if cbar_precision == 0:
        tick_label = [x[:-2] for x in tick_label]

    cb1.set_ticklabels(tick_label)

    cb1.ax.tick_params(labelsize=18, rotation=0)

    if label is not None:
        cb1.set_label(label)

    fig.tight_layout()

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    im = im[:, :, :3].astype(np.float32) / 255.
    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im