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
    
class SDFDatasetNoNormalization(Dataset):
    def __init__(self, path, size=100, num_samples=2**18, clip_sdf=None):
        super().__init__()
        self.path = path

        # load obj 
        self.mesh = trimesh.load(path, force='mesh')


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
    
class SDFDatasetSubspace(Dataset):
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
        
        #make a tensor with size 4 as subsapce vector
        subspaces= np.zeros((self.num_samples, 4)).astype(np.float32)

        results = {
            'sdfs': sdfs,
            'points': points,
            'subspace':subspaces,
        }

        #plot_pointcloud(points, sdfs)

        return results
class DeformationMappingDataset(Dataset):
    def __init__(self, path, size=100, num_samples=2**15, clip_sdf=None,subspace_size=6):
        super().__init__()
        self.path = path
        self.subspace_size=subspace_size
        # suffix=["x","y","z","xplus","yplus","zplus",]
        suffix= ["o","o","o","o","o","o"]
        vs= None
        for i in range(6):
            # load obj 
            self.mesh = trimesh.load(path+"_"+suffix[i]+".obj", force='mesh')
            if not self.mesh.is_watertight:
                print(f"[WARN] mesh is not watertight! SDF maybe incorrect.")
            #trimesh.Scene([self.mesh]).show()

            # normalize to [-1, 1] (different from instant-sdf where is [0, 1])
            originalvs = self.mesh.vertices

            #create 6 sets of vertices add perturbation to the vertices
            if i==0:
                vs = np.zeros((6,originalvs.shape[0],originalvs.shape[1])) 
            vs[i] = originalvs
            vmin = vs[i].min(0)
            vmax = vs[i].max(0)
            v_center = (vmin + vmax) / 2
            v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
            vs[i] = (vs[i] - v_center[None, :]) * v_scale

        #set a list of 100 meshes with vertices randomly interpolated from the 4
        self.vmesh=[]
        self.vweights=[]
        for i in range(100):
            newmesh=self.mesh.copy()
            #generate 4 positive float number with sum of 1
            weights = np.random.dirichlet(np.ones(6),size=1)
            #make weights float
            weights= weights.astype(np.float32)
            #weigh the vertices with the weights
            x = weights[0][0]*vs[0]+weights[0][1]*vs[1]+weights[0][2]*vs[2]+weights[0][3]*vs[3]+weights[0][4]*vs[4]+weights[0][5]*vs[5]
            newmesh.vertices = x   
            self.vweights.append(weights)         
            self.vmesh.append(newmesh)

        
        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."
        self.clip_sdf = clip_sdf

        self.size = size
    def __len__(self):
        return self.size
        
    def __getitem__(self, _):
        points_surface = self.mesh.sample(self.num_samples).astype(np.float32)
        points_mapped= self.mapping_func(points_surface).astype(np.float32)
        results = {
            "original":points_surface,
            "mapped":points_mapped,
            # 'subspace':subspaces_all,
        }
        return results



        


class SDFDatasetModes(Dataset):
    def __init__(self, path, size=100, num_samples=2**15, clip_sdf=None,subspace_size=6):
        super().__init__()
        self.path = path
        self.subspace_size=subspace_size
        # suffix=["x","y","z","xplus","yplus","zplus",]
        suffix= ["o","o","o","x","x","x"]
        vs= None
        self.original_mesh = trimesh.load(path+"_o"+".obj", force='mesh')
        v= self.original_mesh.vertices
        vmin = v.min(0)
        vmax = v.max(0)
        v_center = (vmin + vmax) / 2
        v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
        self.original_mesh.vertices = (v - v_center[None, :]) * v_scale

        for i in range(6):
            # load obj 
            self.mesh = trimesh.load(path+"_"+suffix[i]+".obj", force='mesh')
            if not self.mesh.is_watertight:
                print(f"[WARN] mesh is not watertight! SDF maybe incorrect.")
            #trimesh.Scene([self.mesh]).show()

            # normalize to [-1, 1] (different from instant-sdf where is [0, 1])
            originalvs = self.mesh.vertices

            #create 6 sets of vertices add perturbation to the vertices
            if i==0:
                vs = np.zeros((6,originalvs.shape[0],originalvs.shape[1])) 
            vs[i] = originalvs
            vmin = vs[i].min(0)
            vmax = vs[i].max(0)
            v_center = (vmin + vmax) / 2
            v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
            vs[i] = (vs[i] - v_center[None, :]) * v_scale

        #set a list of 100 meshes with vertices randomly interpolated from the 4
        self.vmesh=[]
        self.vweights=[]
        for i in range(100):
            newmesh=self.mesh.copy()
            #generate 4 positive float number with sum of 1
            weights = np.random.dirichlet(np.ones(6),size=1)
            #make weights float
            weights= weights.astype(np.float32)
            #weigh the vertices with the weights
            x = weights[0][0]*vs[0]+weights[0][1]*vs[1]+weights[0][2]*vs[2]+weights[0][3]*vs[3]+weights[0][4]*vs[4]+weights[0][5]*vs[5]
            newmesh.vertices = x   
            self.vweights.append(weights)         
            self.vmesh.append(newmesh)

        
        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."
        self.clip_sdf = clip_sdf

        self.size = size

    
    def __len__(self):
        return self.size

    def __getitem__(self, _):

        #randomly choose 16 meshes to sample
        meshidx = np.random.randint(0,100,16)
        #create empty numpy array for the 16 sdfs to concatenate
        sdfs_all = np.empty((0, 1))
        #create empty numpy array for the 16 points to concatenate
        points_all = np.empty((0, 3))
        #create empty numpy array for the 16 subspaces to concatenate
        subspaces_all = np.empty((0, self.subspace_size))   
        surfaceflag_all= np.empty((0, 1))
        reference_points_on_original_mesh_all = np.empty((0, 3))


        
        for i in range(16):
            idx=meshidx[i]
            sdf_fn = pysdf.SDF(self.vmesh[idx].vertices, self.vmesh[idx].faces)
            # online sampling
            sdfs = np.zeros((self.num_samples, 1))
            surfaceflag=np.zeros((self.num_samples, 1))
            surfaceflag[0:self.num_samples//2]=1
            reference_points_on_original_mesh=np.zeros((self.num_samples, 3))
            # surface
            points_surface,triangle_id = trimesh.sample.sample_surface_even(self.vmesh[idx], self.num_samples* 7 // 8)

            # compute the barycentric coordinates of each sample
            bary = trimesh.triangles.points_to_barycentric(
            triangles=self.vmesh[idx].triangles[triangle_id], points=points_surface)
            # interpolate the position on the original mesh from barycentric coordinates
            reference_points_on_original_mesh[0:self.num_samples* 7 // 8,:] = (self.original_mesh.vertices[self.original_mesh.faces[triangle_id]] *
                                bary.reshape((-1, 3, 1))).sum(axis=1)

            # perturb surface
            points_surface[self.num_samples // 2:] += 0.01 * np.random.randn(self.num_samples * 3 // 8, 3)
            # random
            points_uniform = np.random.rand(self.num_samples // 8, 3) * 2 - 1
            points = np.concatenate([points_surface, points_uniform], axis=0).astype(np.float32)
            sdfs[self.num_samples // 2:] = -sdf_fn(points[self.num_samples // 2:])[:,None].astype(np.float32)

            #repeat the weights according to the numebr of samples to fill the subspace vector

            subspaces = np.repeat(self.vweights[idx],self.num_samples,axis=0).astype(np.float32)

            #concatenate the sdfs,points and the subspaces for all meshes
            sdfs_all=np.concatenate((sdfs_all,sdfs),axis=0).astype(np.float32)
            subspaces_all=np.concatenate((subspaces_all,subspaces),axis=0).astype(np.float32)
            points_all=np.concatenate((points_all,points),axis=0).astype(np.float32)
            surfaceflag_all=np.concatenate((surfaceflag_all,surfaceflag),axis=0).astype(np.float32)
            reference_points_on_original_mesh_all=np.concatenate((reference_points_on_original_mesh_all,reference_points_on_original_mesh),
                                                                 axis=0).astype(np.float32)
 
        

        results = {
            'sdfs': sdfs_all,
            'points': points_all,
            'subspace':subspaces_all,
            'surface_flag':surfaceflag_all,
            'reference_points_on_original_mesh':reference_points_on_original_mesh_all,
        }

        #plot_pointcloud(points, sdfs)

        return results

class SDFDatasetTestPreencoder(Dataset):
    def __init__(self, path, size=100, num_samples=2**15, clip_sdf=None,subspace_size=7,subspace_suffix=["o","x","y","z","xplus","yplus","zplus"]):
        super().__init__()
        self.path = path
        self.subspace_size=subspace_size
        # suffix=["x","y","z","xplus","yplus","zplus",]
        suffix= subspace_suffix
        assert(len(suffix)==subspace_size)
        vs= None
        self.original_mesh = trimesh.load(path+"_o"+".obj", force='mesh')
        v= self.original_mesh.vertices
        vmin = v.min(0)
        vmax = v.max(0)
        v_center = (vmin + vmax) / 2
        v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
        self.original_mesh.vertices = (v - v_center[None, :]) * v_scale

        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."
        self.clip_sdf = clip_sdf

        self.size = size

        for i in range(subspace_size):
            # load obj 
            self.mesh = trimesh.load(path+"_"+suffix[i]+".obj", force='mesh')
            if not self.mesh.is_watertight:
                print(f"[WARN] mesh is not watertight! SDF maybe incorrect.")
            #trimesh.Scene([self.mesh]).show()

            # normalize to [-1, 1] (different from instant-sdf where is [0, 1])
            originalvs = self.mesh.vertices

            if i==0:
                vs = np.zeros((subspace_size,originalvs.shape[0],originalvs.shape[1])) 
            vs[i] = originalvs
            vmin = vs[i].min(0)
            vmax = vs[i].max(0)
            v_center = (vmin + vmax) / 2
            v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
            vs[i] = (vs[i] - v_center[None, :]) * v_scale
            self.vs=vs

        #set a list of 100 meshes with vertices randomly interpolated from the 4
        self.vmesh=[]
        self.vweights=[]
        #create empty numpy array for the 16 sdfs to concatenate
        sdfs_all = np.empty((0, 1))
        #create empty numpy array for the 16 points to concatenate
        points_all = np.empty((0, 3))
        #create empty numpy array for the 16 subspaces to concatenate
        subspaces_all = np.empty((0, self.subspace_size))   
        surfaceflag_all= np.empty((0, 1))
        reference_points_on_original_mesh_all = np.empty((0, 3))
        for i in range(64):
            newmesh=self.mesh.copy()
            #generate positive float number with sum of 1
            weights = np.random.dirichlet(np.ones(subspace_size),size=1)
            #make weights float
            weights= weights.astype(np.float32)
            #weigh the vertices with the weights
            x = sum(weights[0][j] * vs[j] for j in range(len(vs)))

            newmesh.vertices = x   
            self.vweights.append(weights)         
            self.vmesh.append(newmesh)
            sdf_fn = pysdf.SDF(self.vmesh[i].vertices, self.vmesh[i].faces)
            # offline sampling
            offline_sample_size=2**15
            sdfs = np.zeros((offline_sample_size, 1))
            surfaceflag=np.zeros((offline_sample_size, 1))
            surfaceflag[:]=1
            reference_points_on_original_mesh=np.zeros((offline_sample_size, 3))
            # surface
            points_surface,triangle_id = trimesh.sample.sample_surface_even(self.vmesh[i], offline_sample_size)

            # compute the barycentric coordinates of each sample
            bary = trimesh.triangles.points_to_barycentric(
            triangles=self.vmesh[i].triangles[triangle_id], points=points_surface)
            # interpolate the position on the original mesh from barycentric coordinates
            reference_points_on_original_mesh[0:offline_sample_size,:] = (self.original_mesh.vertices[self.original_mesh.faces[triangle_id]] *
                                bary.reshape((-1, 3, 1))).sum(axis=1)
            
            subspaces = np.repeat(self.vweights[i],offline_sample_size,axis=0).astype(np.float32)

            #concatenate the sdfs,points and the subspaces for all meshes
            sdfs_all=np.concatenate((sdfs_all,sdfs),axis=0).astype(np.float32)
            subspaces_all=np.concatenate((subspaces_all,subspaces),axis=0).astype(np.float32)
            points_all=np.concatenate((points_all,points_surface),axis=0).astype(np.float32)
            surfaceflag_all=np.concatenate((surfaceflag_all,surfaceflag),axis=0).astype(np.float32)
            reference_points_on_original_mesh_all=np.concatenate((reference_points_on_original_mesh_all,reference_points_on_original_mesh),
                                                                 axis=0).astype(np.float32)
            self.results = {
            'sdfs': sdfs_all,
            'points': points_all,
            'subspace':subspaces_all,
            'surface_flag':surfaceflag_all,
            'reference_points_on_original_mesh':reference_points_on_original_mesh_all,
        }
        
       

    
    def __len__(self):
        return self.size

    def __getitem__(self, _):

        #randomly choose samples from self.result
        idx = np.random.randint(0,self.results['points'].shape[0],self.num_samples)
        sdfs=self.results['sdfs'][idx]
        points=self.results['points'][idx]
        subspaces=self.results['subspace'][idx]
        surfaceflag=self.results['surface_flag'][idx]
        reference_points_on_original_mesh=self.results['reference_points_on_original_mesh'][idx]
        results = {
            'sdfs': sdfs,
            'points': points,
            'subspace':subspaces,
            'surface_flag':surfaceflag,
            'reference_points_on_original_mesh':reference_points_on_original_mesh,
        }

        return results
    
class SDFDatasetSMPL(Dataset):
    def __init__(self, size=100, num_samples=2**15, clip_sdf=None,subspace_size=10):
        super().__init__()
        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."

        self.size = size

        self.num_samples = num_samples
        self.subspace_size=subspace_size

        self.clip_sdf = clip_sdf

        self.vmesh=[]
        self.vweights=[]
        #create empty numpy array for the 16 sdfs to concatenate
        sdfs_all = np.empty((0, 1))
        #create empty numpy array for the 16 points to concatenate
        points_all = np.empty((0, 3))
        #create empty numpy array for the 16 subspaces to concatenate
        subspaces_all = np.empty((0, self.subspace_size))   
        surfaceflag_all= np.empty((0, 1))
        reference_points_on_original_mesh_all = np.empty((0, 3))
        #load 'SMPLTemplateNormalized.obj'
        self.original_mesh = trimesh.load("SMPLTemplateNormalized.obj", force='mesh')
        #load blendshapes
        blendshapes = np.load("blendshape.npy")

        for i in range(64):
            newmesh=self.original_mesh.copy()
            #generate a serie of random number of gaussian distribution of std 1
            weights = np.random.randn(subspace_size)
            weights=weights.reshape(1,-1)
            #make weights float
            weights= weights.astype(np.float32)
            #weigh the vertices with the weights
            x = sum(weights[0,j] * blendshapes[:,:,j] for j in range(subspace_size))
            newmesh.vertices += x
            self.vweights.append(weights)         
            self.vmesh.append(newmesh)

            sdf_fn = pysdf.SDF(newmesh.vertices, newmesh.faces)
            # offline sampling
            offline_sample_size=2**15
            sdfs = np.zeros((offline_sample_size, 1))
            surfaceflag=np.zeros((offline_sample_size, 1))
            surfaceflag[:]=1
            reference_points_on_original_mesh=np.zeros((offline_sample_size, 3))
            # surface
            points_surface,triangle_id = trimesh.sample.sample_surface_even(self.vmesh[i], offline_sample_size)

            # compute the barycentric coordinates of each sample
            bary = trimesh.triangles.points_to_barycentric(
            triangles=self.vmesh[i].triangles[triangle_id], points=points_surface)
            # interpolate the position on the original mesh from barycentric coordinates
            reference_points_on_original_mesh[0:offline_sample_size,:] = (self.original_mesh.vertices[self.original_mesh.faces[triangle_id]] *
                                bary.reshape((-1, 3, 1))).sum(axis=1)
            
            #add purturbed points
            points_perturbed,triangle_id = trimesh.sample.sample_surface_even(self.vmesh[i], offline_sample_size)
            points_perturbed+=0.01*np.random.randn(offline_sample_size,3)
            #concatenate the purturbed points to the surface points
            points_surface=np.concatenate((points_surface,points_perturbed),axis=0).astype(np.float32)

            #calculate the sdf of the perturbed points
            sdfs_perturbed = -sdf_fn(points_perturbed)[:,None].astype(np.float32)
            #add the perturbed points to the sdfs
            sdfs=np.concatenate((sdfs,sdfs_perturbed),axis=0).astype(np.float32)
            #set the surface flag of the perturbed points to 0
            surfaceflag_perturbed=np.zeros((offline_sample_size, 1))
            surfaceflag=np.concatenate((surfaceflag,surfaceflag_perturbed),axis=0).astype(np.float32)
            #set the reference points of the perturbed points to the original mesh as 0
            reference_points_on_original_mesh_perturbed=np.zeros((offline_sample_size, 3))
            reference_points_on_original_mesh=np.concatenate((reference_points_on_original_mesh,reference_points_on_original_mesh_perturbed),
                                                                    axis=0).astype(np.float32)
            
            #add random points
            points_uniform = np.random.rand(offline_sample_size, 3) * 2 - 1
            #concatenate the random points to the surface points
            points_surface=np.concatenate((points_surface,points_uniform),axis=0).astype(np.float32)
            #calculate the sdf of the random points
            sdfs_random = -sdf_fn(points_uniform)[:,None].astype(np.float32)
            #add the random points to the sdfs
            sdfs=np.concatenate((sdfs,sdfs_random),axis=0).astype(np.float32)
            #set the surface flag of the random points to 0
            surfaceflag_random=np.zeros((offline_sample_size, 1))
            surfaceflag=np.concatenate((surfaceflag,surfaceflag_random),axis=0).astype(np.float32)
            #set the reference points of the random points to the original mesh as 0
            reference_points_on_original_mesh_random=np.zeros((offline_sample_size, 3))
            reference_points_on_original_mesh=np.concatenate((reference_points_on_original_mesh,reference_points_on_original_mesh_random),
                                                                    axis=0).astype(np.float32)

            
            subspaces = np.repeat(self.vweights[i],3*offline_sample_size,axis=0).astype(np.float32)

            #concatenate the sdfs,points and the subspaces for all meshes
            sdfs_all=np.concatenate((sdfs_all,sdfs),axis=0).astype(np.float32)
            subspaces_all=np.concatenate((subspaces_all,subspaces),axis=0).astype(np.float32)
            points_all=np.concatenate((points_all,points_surface),axis=0).astype(np.float32)
            surfaceflag_all=np.concatenate((surfaceflag_all,surfaceflag),axis=0).astype(np.float32)
            reference_points_on_original_mesh_all=np.concatenate((reference_points_on_original_mesh_all,reference_points_on_original_mesh),
                                                                 axis=0).astype(np.float32)
            self.results ={
            'sdfs': sdfs_all,
            'points': points_all,
            'subspace':subspaces_all,
            'surface_flag':surfaceflag_all,
            'reference_points_on_original_mesh':reference_points_on_original_mesh_all,
        }
        
       

    
    def __len__(self):
        return self.size

    def __getitem__(self, _):

        #randomly choose samples from self.result
        idx = np.random.randint(0,self.results['points'].shape[0],self.num_samples)
        sdfs=self.results['sdfs'][idx]
        points=self.results['points'][idx]
        subspaces=self.results['subspace'][idx]
        surfaceflag=self.results['surface_flag'][idx]
        reference_points_on_original_mesh=self.results['reference_points_on_original_mesh'][idx]
        results = {
            'sdfs': sdfs,
            'points': points,
            'subspace':subspaces,
            'surface_flag':surfaceflag,
            'reference_points_on_original_mesh':reference_points_on_original_mesh,
        }

        return results


class SDFDatasetSMPLMini(Dataset):
    def __init__(self, size=100, num_samples=2**15, clip_sdf=None, subspace_size=1):
        super().__init__()
        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."

        self.size = size
        self.subspace_size = subspace_size
        self.clip_sdf = clip_sdf

        self.vmesh = []
        self.vweights = []

        # Initializing arrays for surface, perturbed, and volume points
        sdfs_surface, sdfs_perturbed, sdfs_volume = [], [], []
        points_surface, points_perturbed, points_volume = [], [], []
        subspace_surface, subspace_perturbed, subspace_volume = [], [], []
        surfaceflag_surface, surfaceflag_perturbed, surfaceflag_volume = [], [], []
        ref_points_surface, ref_points_perturbed, ref_points_volume = [], [], []

        # Load mesh and blendshapes
        self.original_mesh = trimesh.load("SMPLTemplateNormalized.obj", force='mesh')
        self.blendshapes = np.load("blendshape.npy")

        for i in range(64):
            newmesh = self.original_mesh.copy()
            weights = np.random.uniform(-20, 20, subspace_size).reshape(1, -1).astype(np.float32)
            deformation = sum(weights[0, j] * self.blendshapes[:, :, j] for j in range(self.subspace_size))
            newmesh.vertices += deformation
            newmesh.vertices *= 0.6
            self.vweights.append(weights)
            self.vmesh.append(newmesh)

            sdf_fn = pysdf.SDF(newmesh.vertices, newmesh.faces)

            # Sampling surface points
            surface_points, triangle_ids = trimesh.sample.sample_surface(self.vmesh[i], self.num_samples)
            surface_sdfs = np.zeros((self.num_samples, 1))
            barycentric_coords = trimesh.triangles.points_to_barycentric(triangles=self.vmesh[i].triangles[triangle_ids], points=surface_points)
            ref_points = (self.original_mesh.vertices[self.original_mesh.faces[triangle_ids]] * barycentric_coords.reshape((-1, 3, 1))).sum(axis=1)
            
            # Sampling perturbed points
            perturbed_points = surface_points + 0.01 * np.random.randn(self.num_samples, 3)
            perturbed_sdfs = -sdf_fn(perturbed_points)[:, None]
            
            # Sampling volume points
            volume_points = np.random.rand(self.num_samples, 3) * 2 - 1
            volume_sdfs = -sdf_fn(volume_points)[:, None]
            
            # Storing data by type
            sdfs_surface.append(surface_sdfs); points_surface.append(surface_points); subspace_surface.append(np.repeat(weights, self.num_samples, axis=0)); surfaceflag_surface.append(np.ones((self.num_samples, 1))); ref_points_surface.append(ref_points)
            sdfs_perturbed.append(perturbed_sdfs); points_perturbed.append(perturbed_points); subspace_perturbed.append(np.repeat(weights, self.num_samples, axis=0)); surfaceflag_perturbed.append(np.zeros((self.num_samples, 1))); ref_points_perturbed.append(np.zeros((self.num_samples, 3)))
            sdfs_volume.append(volume_sdfs); points_volume.append(volume_points); subspace_volume.append(np.repeat(weights, self.num_samples, axis=0)); surfaceflag_volume.append(np.zeros((self.num_samples, 1))); ref_points_volume.append(np.zeros((self.num_samples, 3)))

        # Concatenating arrays for each type
        self.results = {
    'sdfs': np.concatenate([np.concatenate(sdfs_surface, axis=0),
                            np.concatenate(sdfs_perturbed, axis=0),
                            np.concatenate(sdfs_volume, axis=0)], axis=0).astype(np.float32),
    'points': np.concatenate([np.concatenate(points_surface, axis=0),
                              np.concatenate(points_perturbed, axis=0),
                              np.concatenate(points_volume, axis=0)], axis=0).astype(np.float32),
    'subspace': np.concatenate([np.concatenate(subspace_surface, axis=0),
                                np.concatenate(subspace_perturbed, axis=0),
                                np.concatenate(subspace_volume, axis=0)], axis=0).astype(np.float32),
    'surface_flag': np.concatenate([np.concatenate(surfaceflag_surface, axis=0),
                                    np.concatenate(surfaceflag_perturbed, axis=0),
                                    np.concatenate(surfaceflag_volume, axis=0)], axis=0).astype(np.float32),
    'reference_points': np.concatenate([np.concatenate(ref_points_surface, axis=0),
                                        np.concatenate(ref_points_perturbed, axis=0),
                                        np.concatenate(ref_points_volume, axis=0)], axis=0).astype(np.float32),
}
    
    def __len__(self):
        return self.size
    def one_single_case(self):
        newmesh = self.original_mesh.copy()
        weights = np.random.uniform(-20, 20, self.subspace_size).reshape(1, -1).astype(np.float32)
        deformation = sum(weights[0, j] * self.blendshapes[:, :, j] for j in range(self.subspace_size))
        newmesh.vertices += deformation
        newmesh.vertices *= 0.6
        # sdf_fn = pysdf.SDF(newmesh.vertices, newmesh.faces)

        # Sampling surface points
        surface_points, triangle_ids = trimesh.sample.sample_surface(newmesh, self.num_samples)
        surface_sdfs = np.zeros((self.num_samples, 1))
        barycentric_coords = trimesh.triangles.points_to_barycentric(triangles= newmesh.triangles[triangle_ids], points=surface_points)
        ref_points = (self.original_mesh.vertices[self.original_mesh.faces[triangle_ids]] * barycentric_coords.reshape((-1, 3, 1))).sum(axis=1)
        # add the weights to the end of each line of the surface points
        weights_repeated = np.tile(weights, (surface_points.shape[0], 1))
        surface_points_with_weights = np.hstack((surface_points, weights_repeated))
        result={
            "sdfs":surface_sdfs.astype(np.float32),
            "points_with_weights":surface_points_with_weights.astype(np.float32),
            "subspace":weights_repeated.astype(np.float32),
            "reference_points":ref_points.astype(np.float32)
        }
        return result

    def __getitem__(self, _):

        #randomly choose samples from self.result
        idx = np.random.randint(0,self.results['points'].shape[0],self.num_samples)
        sdfs=self.results['sdfs'][idx]
        points=self.results['points'][idx]
        subspaces=self.results['subspace'][idx]
        surfaceflag=self.results['surface_flag'][idx]
        reference_points_on_original_mesh=self.results['reference_points'][idx]
        results = {
            'sdfs': sdfs,
            'points': points,
            'subspace':subspaces,
            'surface_flag':surfaceflag,
            'reference_points_on_original_mesh':reference_points_on_original_mesh,
        }

        return results



    
class SDFDatasetTestPreencoderEndToEnd(Dataset):
    def __init__(self, path, size=100, num_samples=2**15, clip_sdf=None,subspace_size=7,subspace_suffix=["o","x","y","z","xplus","yplus","zplus"]):
        super().__init__()
        self.path = path
        self.subspace_size=subspace_size
        # suffix=["x","y","z","xplus","yplus","zplus",]
        suffix= subspace_suffix
        assert(len(suffix)==subspace_size)
        vs= None
        self.original_mesh = trimesh.load(path+"_o"+".obj", force='mesh')
        v= self.original_mesh.vertices
        vmin = v.min(0)
        vmax = v.max(0)
        v_center = (vmin + vmax) / 2
        v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
        self.original_mesh.vertices = (v - v_center[None, :]) * v_scale

        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."
        self.clip_sdf = clip_sdf

        self.size = size

        for i in range(subspace_size):
            # load obj 
            self.mesh = trimesh.load(path+"_"+suffix[i]+".obj", force='mesh')
            if not self.mesh.is_watertight:
                print(f"[WARN] mesh is not watertight! SDF maybe incorrect.")
            #trimesh.Scene([self.mesh]).show()

            # normalize to [-1, 1] (different from instant-sdf where is [0, 1])
            originalvs = self.mesh.vertices

            if i==0:
                vs = np.zeros((subspace_size,originalvs.shape[0],originalvs.shape[1])) 
            vs[i] = originalvs
            vmin = vs[i].min(0)
            vmax = vs[i].max(0)
            v_center = (vmin + vmax) / 2
            v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
            vs[i] = (vs[i] - v_center[None, :]) * v_scale
            self.vs=vs

        #set a list of 100 meshes with vertices randomly interpolated from the 4
        self.vmesh=[]
        self.vweights=[]
        #create empty numpy array for the 16 sdfs to concatenate
        sdfs_all = np.empty((0, 1))
        #create empty numpy array for the 16 points to concatenate
        points_all = np.empty((0, 3))
        #create empty numpy array for the 16 subspaces to concatenate
        subspaces_all = np.empty((0, self.subspace_size))   
        surfaceflag_all= np.empty((0, 1))
        reference_points_on_original_mesh_all = np.empty((0, 3))
        for i in range(64):

            newmesh=self.mesh.copy()
            #generate positive float number with sum of 1
            weights = np.random.dirichlet(np.ones(subspace_size),size=1)
            #make weights float
            weights= weights.astype(np.float32)
            #weigh the vertices with the weights
            x = sum(weights[0][j] * vs[j] for j in range(len(vs)))

            newmesh.vertices = x   
            self.vweights.append(weights)         
            self.vmesh.append(newmesh)
            sdf_fn = pysdf.SDF(self.vmesh[i].vertices, self.vmesh[i].faces)
            # offline sampling
            offline_sample_size=2**15
            sdfs = np.zeros((offline_sample_size, 1))
            surfaceflag=np.zeros((offline_sample_size, 1))
            surfaceflag[:]=1
            reference_points_on_original_mesh=np.zeros((offline_sample_size, 3))
            # surface
            points_surface,triangle_id = trimesh.sample.sample_surface_even(self.vmesh[i], offline_sample_size)

            # compute the barycentric coordinates of each sample
            bary = trimesh.triangles.points_to_barycentric(
            triangles=self.vmesh[i].triangles[triangle_id], points=points_surface)
            # interpolate the position on the original mesh from barycentric coordinates
            reference_points_on_original_mesh[0:offline_sample_size,:] = (self.original_mesh.vertices[self.original_mesh.faces[triangle_id]] *
                                bary.reshape((-1, 3, 1))).sum(axis=1)
            
            #add purturbed points
            points_perturbed,triangle_id = trimesh.sample.sample_surface_even(self.vmesh[i], offline_sample_size)
            points_perturbed+=0.01*np.random.randn(offline_sample_size,3)
            #concatenate the purturbed points to the surface points
            points_surface=np.concatenate((points_surface,points_perturbed),axis=0).astype(np.float32)

            #calculate the sdf of the perturbed points
            sdfs_perturbed = -sdf_fn(points_perturbed)[:,None].astype(np.float32)
            #add the perturbed points to the sdfs
            sdfs=np.concatenate((sdfs,sdfs_perturbed),axis=0).astype(np.float32)
            #set the surface flag of the perturbed points to 0
            surfaceflag_perturbed=np.zeros((offline_sample_size, 1))
            surfaceflag=np.concatenate((surfaceflag,surfaceflag_perturbed),axis=0).astype(np.float32)
            #set the reference points of the perturbed points to the original mesh as 0
            reference_points_on_original_mesh_perturbed=np.zeros((offline_sample_size, 3))
            reference_points_on_original_mesh=np.concatenate((reference_points_on_original_mesh,reference_points_on_original_mesh_perturbed),
                                                                    axis=0).astype(np.float32)

            #add random points
            points_uniform = np.random.rand(offline_sample_size, 3) * 2 - 1
            #concatenate the random points to the surface points
            points_surface=np.concatenate((points_surface,points_uniform),axis=0).astype(np.float32)
            #calculate the sdf of the random points
            sdfs_random = -sdf_fn(points_uniform)[:,None].astype(np.float32)
            #add the random points to the sdfs
            sdfs=np.concatenate((sdfs,sdfs_random),axis=0).astype(np.float32)
            #set the surface flag of the random points to 0
            surfaceflag_random=np.zeros((offline_sample_size, 1))
            surfaceflag=np.concatenate((surfaceflag,surfaceflag_random),axis=0).astype(np.float32)
            #set the reference points of the random points to the original mesh as 0
            reference_points_on_original_mesh_random=np.zeros((offline_sample_size, 3))
            reference_points_on_original_mesh=np.concatenate((reference_points_on_original_mesh,reference_points_on_original_mesh_random),
                                                                    axis=0).astype(np.float32)
            


            #repeat the weights according to the numebr of surface flag to fill the subspace vector
            subspaces = np.repeat(self.vweights[i],np.shape(surfaceflag)[0],axis=0).astype(np.float32)
            

            


            #concatenate the sdfs,points and the subspaces for all meshes
            sdfs_all=np.concatenate((sdfs_all,sdfs),axis=0).astype(np.float32)
            subspaces_all=np.concatenate((subspaces_all,subspaces),axis=0).astype(np.float32)
            points_all=np.concatenate((points_all,points_surface),axis=0).astype(np.float32)
            surfaceflag_all=np.concatenate((surfaceflag_all,surfaceflag),axis=0).astype(np.float32)
            reference_points_on_original_mesh_all=np.concatenate((reference_points_on_original_mesh_all,reference_points_on_original_mesh),
                                                                 axis=0).astype(np.float32)
            self.results = {
            'sdfs': sdfs_all,
            'points': points_all,
            'subspace':subspaces_all,
            'surface_flag':surfaceflag_all,
            'reference_points_on_original_mesh':reference_points_on_original_mesh_all,
        }
        
       

    
    def __len__(self):
        return self.size

    def __getitem__(self, _):

        #randomly choose samples from self.result
        idx = np.random.randint(0,self.results['points'].shape[0],self.num_samples)
        sdfs=self.results['sdfs'][idx]
        points=self.results['points'][idx]
        subspaces=self.results['subspace'][idx]
        surfaceflag=self.results['surface_flag'][idx]
        reference_points_on_original_mesh=self.results['reference_points_on_original_mesh'][idx]
        results = {
            'sdfs': sdfs,
            'points': points,
            'subspace':subspaces,
            'surface_flag':surfaceflag,
            'reference_points_on_original_mesh':reference_points_on_original_mesh,
        }

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
    
class SurfacePerturbedDataset(Dataset):
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
        # perturb surface
        points[self.num_samples // 2:] += 0.01 * np.random.randn(self.num_samples // 2, 3)
        #compute sdf for perturbed surface
        sdfs[self.num_samples // 2:] = self.sdf_fn(points[self.num_samples // 2:])

        #find the closest face for each perturbed point
        closest_face = trimesh.proximity.closest_point(self.mesh, points[self.num_samples // 2:])[1]
        triangle_id[self.num_samples // 2:] = closest_face
        # compute the gradient of each sample
        gradients = self.mesh.face_normals[triangle_id]

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
