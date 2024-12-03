import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from plyfile import PlyData, PlyElement
# from utils.sh_utils import RGB2SH, SH2RGB
import trimesh 
from errno import EEXIST
from os import makedirs, path
import os


def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

def load_ply_file(path, max_sh_degree):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    return xyz, features_dc, features_extra, opacities, scales, rots


def construct_list_of_attributes(_features_dc, _features_rest, _scaling, _rotation, self_features_dc, self__features_rest):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(self_features_dc.shape[1]*self_features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(self__features_rest.shape[1]*self__features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(_scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(_rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l


def save_ply( path, downsample=1,
    _xyz=None, _features_dc=None, _features_rest=None, _opacity=None, _scaling=None, _rotation=None):
    mkdir_p(os.path.dirname(path))

    xyz = _xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = _features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = _features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = _opacity.detach().cpu().numpy()
    scale = _scaling.detach().cpu().numpy()
    rotation = _rotation.detach().cpu().numpy()

    print("xyz", xyz.shape)
    print("f_dc", f_dc.shape)
    print("f_rest", f_rest.shape)
    print("opacities", opacities.shape)
    print("scale", scale.shape)
    print("rotation", rotation.shape)


    attributes = construct_list_of_attributes(f_dc, f_rest, scale, rotation, _features_dc, _features_rest)
    dtype_full = [(attribute, 'f4') for attribute in attributes]

    if downsample > 1:
        downsample = int(downsample)
        xyz = xyz[::downsample]
        normals = normals[::downsample]
        f_dc = f_dc[::downsample]
        f_rest = f_rest[::downsample]
        opacities = opacities[::downsample]
        scale = scale[::downsample]
        rotation = rotation[::downsample]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

    # save also a normal ply file for meshlab to visualize
    import trimesh

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # color = SH2RGB(f_dc[:, 0, :])
    # color = np.clip(color, 0, 1)
    # pcd.colors = o3d.utility.Vector3dVector(color)
    # o3d.io.write_point_cloud(path.replace(".ply", "_normal.ply"), pcd)
    print("f_dc", f_dc.shape)
    pcd = trimesh.points.PointCloud(xyz, colors=SH2RGB(f_dc))
    pcd.export(path.replace(".ply", "_normal.ply"))


sky_num = 0

ply_path = '/cluster/work/cvl/qimaqi/cvpr_2025/datasets/MatrixCity/colmap_aerial/output/scaffold/point_cloud/final/point_cloud.ply'
degree = 1
xyz, features_dc, features_extra, opacities, scales, rots = load_ply_file(ply_path, degree)



xyz = torch.from_numpy(xyz).float()
features_dc = torch.from_numpy(features_dc).permute(0, 2, 1).float()
features_rest = torch.from_numpy(features_extra).permute(0, 2, 1).float()
opacity = torch.from_numpy(opacities).float()
scaling = torch.from_numpy(scales).float()
rotation = torch.from_numpy(rots).float()

_xyz = xyz[sky_num:]
_features_dc = features_dc[sky_num:]
_features_rest = features_rest[sky_num:]
_opacity = opacity[sky_num:]
_scaling = scaling[sky_num:]
_rotation = rotation[sky_num:]

print(_xyz.shape)
print(_features_dc.shape)
print(_features_rest.shape)
print(_opacity.shape)
print(_scaling.shape)
print(_rotation.shape)

save_ply( ply_path.replace(".ply", "_no_sky.ply"), _xyz=_xyz, _features_dc=_features_dc, _features_rest=_features_rest, _opacity=_opacity, _scaling=_scaling, _rotation=_rotation)