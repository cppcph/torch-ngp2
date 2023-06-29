import numpy as np

import torch
from torch.utils.data import Dataset

import trimesh
import pysdf

def map_color(value, cmap_name='viridis', vmin=None, vmax=None):
    # value: [N], float
    # return: RGB, [N, 3], float in [0, 1]
    import matplotlib.cm as cm
    if vmin is None: vmin = value.min()
    if vmax is None: vmax = value.max()
    value = (value - vmin) / (vmax - vmin) # range in [0, 1]
    cmap = cm.get_cmap(cmap_name) 
    rgb = cmap(value)[:, :3]  # will return rgba, we take only first 3 so we get rgb
    return rgb

def plot_pointcloud(pc, sdfs):
    # pc: [N, 3]
    # sdfs: [N, 1]
    color = map_color(sdfs.squeeze(1))
    pc = trimesh.PointCloud(pc, color)
    trimesh.Scene([pc]).show()    

# SDF dataset
class SDFDataset(Dataset):
    def __init__(self, path, size=100, num_samples=2**18, clip_sdf=None):
        super().__init__()
        self.path = path

        # load obj 
        self.mesh = trimesh.load(path, force='mesh')

        # normalize to [-1, 1] (different from instant-sdf where is [0, 1])
        vs = self.mesh.vertices
        vmin = vs.min(0)
        vmax = vs.max(0)
        v_center = (vmin + vmax) / 2
        v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
        vs = (vs - v_center[None, :]) * v_scale
        self.mesh.vertices = vs

        print(f"[INFO] mesh: {self.mesh.vertices.shape} {self.mesh.faces.shape}")

        if not self.mesh.is_watertight:
            print(f"[WARN] mesh is not watertight! SDF maybe incorrect.")
        #trimesh.Scene([self.mesh]).show()

        self.sdf_fn = pysdf.SDF(self.mesh.vertices, self.mesh.faces)
        
        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."
        self.clip_sdf = clip_sdf

        self.size = size

    
    def __len__(self):
        return self.size

    def __getitem__(self, _):

        # online sampling
        sdfs = np.zeros((self.num_samples, 1))
        # surface
        points_surface = self.mesh.sample(self.num_samples * 7 // 8)
        # perturb surface
        points_surface[self.num_samples // 2:] += 0.01 * np.random.randn(self.num_samples * 3 // 8, 3)
        # random
        points_uniform = np.random.rand(self.num_samples // 8, 3) * 2 - 1
        points = np.concatenate([points_surface, points_uniform], axis=0).astype(np.float32)

        sdfs[self.num_samples // 2:] = -self.sdf_fn(points[self.num_samples // 2:])[:,None].astype(np.float32)
 
        # clip sdf
        if self.clip_sdf is not None:
            sdfs = sdfs.clip(-self.clip_sdf, self.clip_sdf)

        results = {
            'sdfs': sdfs,
            'points': points,
        }

        #plot_pointcloud(points, sdfs)

        return results

class SurfaceDataset(Dataset):
    def __init__(self, path, size=100, num_samples=2**18, max_num_sample=2**22, online_sampling=True, normalize_mesh=True, device: str = "cuda"):
        super().__init__()
        self.path = path
        self.device ="cuda"
        # load obj 
        self.mesh = trimesh.load(path, force='mesh')

        # normalize to [-1, 1] (different from instant-sdf where is [0, 1])
        if normalize_mesh:
            vs = self.mesh.vertices
            vmin = vs.min(0)
            vmax = vs.max(0)
            v_center = (vmin + vmax) / 2
            v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
            vs = (vs - v_center[None, :]) * v_scale
            self.mesh.vertices = vs

        print(f"[INFO] mesh: {self.mesh.vertices.shape} {self.mesh.faces.shape}")

        if not self.mesh.is_watertight:
            print(f"[WARN] mesh is not watertight! SDF maybe incorrect.")

        self.sdf_fn = pysdf.SDF(self.mesh.vertices, self.mesh.faces)
        self.num_samples = num_samples
        self.max_num_samples = max_num_sample
        self.size = size
        self.online_sampling= online_sampling
        if not online_sampling:
            self._offline_sampling_()

    def _sampling_(self):
        # online sampling
        sdfs = np.zeros((self.num_samples, 1))
        # sample surface
        points,triangle_id = trimesh.sample.sample_surface_even(self.mesh, self.num_samples)

        # compute the barycentric coordinates of each sample
        bary = trimesh.triangles.points_to_barycentric(
        triangles=self.mesh.triangles[triangle_id], points=points)
        # interpolate vertex normals from barycentric coordinates
        gradients = trimesh.unitize((self.mesh.vertex_normals[self.mesh.faces[triangle_id]] *
                                trimesh.unitize(bary).reshape(
                                    (-1, 3, 1))).sum(axis=1))

        # gradients = self.mesh.face_normals[triangle_id]

        self.results = {
            'sdfs': sdfs,
            'points': points,
            'gradients':gradients,
        }

        for key, value in self.results.items():
            value = value.astype(np.float32)
            self.results[key] = torch.from_numpy(value).to(self.device)

    def _offline_sampling_(self):
        # online sampling
        sdfs = np.zeros((self.max_num_samples, 1))
        # sample surface
        points,triangle_id = trimesh.sample.sample_surface_even(self.mesh, self.max_num_samples)
        #gradients = self.mesh.face_normals[triangle_id]
        
        # compute the barycentric coordinates of each sample
        bary = trimesh.triangles.points_to_barycentric(
        triangles=self.mesh.triangles[triangle_id], points=points)
        # interpolate vertex normals from barycentric coordinates
        gradients = trimesh.unitize((self.mesh.vertex_normals[self.mesh.faces[triangle_id]] *
                                trimesh.unitize(bary).reshape((-1, 3, 1))).sum(axis=1))

        self.max_results = {
            'sdfs': sdfs,
            'points': points,
            'gradients':gradients,
        }

    def _random_select_results_(self):
        if self.max_results is None:
            self._offline_sampling_()
        idx = np.random.randint(0,self.max_num_samples,self.num_samples)
        self.results = {
            'sdfs': self.max_results['sdfs'][idx],
            'points': self.max_results['points'][idx],
            'gradients':self.max_results['gradients'][idx],
        }

        for key, value in self.results.items():
            value = value.astype(np.float32)
            self.results[key] = torch.from_numpy(value).to(self.device)


    def __len__(self):
        return self.size

    def __getitem__(self, _):
        if self.online_sampling:
            self._sampling_()
        else:
            self._random_select_results_()

        return self.results
