import os
import glob
import tqdm
import random
import warnings
import tensorboardX

import numpy as np
import pandas as pd

import time
from datetime import datetime

# import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

import packaging
def grad_fd(model,X,epsilon,device='cuda'):
    pred = model(X)
    ex= torch.tensor([epsilon,0,0],device= device)
    ey= torch.tensor([0,epsilon,0],device= device)
    ez= torch.tensor([0,0,epsilon],device= device)
    diffx= model(X+ex)-pred
    diffy= model(X+ey)-pred
    diffz= model(X+ez)-pred
    grad_fd= torch.cat([diffx,diffy,diffz],1)/epsilon
    return grad_fd

def gen_img_grad(model):
    size_x = 300
    size_y = 300

    pz = 0.0
    factor = 1.0

    # Create the grid
    grid_x, grid_y = torch.meshgrid(torch.linspace(-factor, factor, size_x), torch.linspace(factor, -factor, size_y))
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).to('cuda')

    # Expand the grid to have a batch dimension of size 1
    grid = grid.expand(1, -1, -1, -1)

    # Set the z-coordinate to pz
    grid = torch.cat([grid, torch.full((1, size_x, size_y, 1), pz, dtype=torch.float32, device='cuda')], dim=-1)
    grid = torch.squeeze(grid, 0)
    grid = grid.reshape(size_x * size_y, 3)
    # Compute the gradients using grid_sample
    grad_model_fd = grad_fd(model, grid, 1e-2)

    # Reshape and normalize the gradients
    grad_model_fd = grad_model_fd.reshape(size_x, size_y, 3)
    # grad_stacked_fd = grad_model_fd.detach().cpu().numpy()
    grad_stacked_fd = (grad_model_fd + 1) / 2.0 
    grad_stacked_fd = torch.clamp(grad_stacked_fd, 0, 1)
    return grad_stacked_fd

def gen_img_dist(model):
    #generate distance field image 
    size_x = 300
    size_y = 300

    pz = 0.0
    factor = 1.0
    # Create the grid
    grid_x, grid_y = torch.meshgrid(torch.linspace(-factor, factor, size_x), torch.linspace(factor, -factor, size_y))
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).to('cuda')

    # Expand the grid to have a batch dimension of size 1
    grid = grid.expand(1, -1, -1, -1)

    # Set the z-coordinate to pz
    grid = torch.cat([grid, torch.full((1, size_x, size_y, 1), pz, dtype=torch.float32, device='cuda')], dim=-1)
    grid = torch.squeeze(grid, 0)
    grid = grid.reshape(size_x * size_y, 3)

    # Compute the distance field using grid_sample
    dist_model = model(grid)

    # Reshape and normalize the distance field
    dist_model = dist_model.reshape(size_x, size_y)

    #take the absolute value of the distance field
    dist_model = torch.abs(dist_model)
    return dist_model


def mse(image1, image2):
    # ensure the two images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")
    # calculate the squared error for each channel
    se = np.square(image1 - image2)
    # calculate the mean squared error across all channels
    mse = np.mean(se)
    return mse

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if packaging.version.parse(torch.__version__) < packaging.version.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3]
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [N, 1] --> [x, y, z]
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    #print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    #print(u.shape, u.max(), u.min(), np.percentile(u, 50))
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles

def compute_sd_and_grad(mesh, samples: np.ndarray):
        closest_point, _, triangle_id = trimesh.proximity.closest_point(mesh, samples)
        triangle_normals = mesh.face_normals[triangle_id]

        # EXERCISE 1a: compute the ground-truth signed distance to the surface
        diff = samples - closest_point
        distance = np.linalg.norm(diff, axis=-1)
        sign = np.sign(np.sum(diff * triangle_normals, axis=-1))
        sd = sign * distance
        # -----------------------------------------------------

        # EXERCISE 2a: compute the gradient of the sign distance field
        sd_grad = diff / sd[:, None]
        # -----------------------------------------------------
        return sd, sd_grad

class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 mesh=None, # mesh for evaluation
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metirc
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        
        self.name = name
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.mesh = mesh

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth.tar"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

        if self.use_tensorboardX:
            self._reference_grid_(17)
            self._reference_surface_(2**10)

    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	
    def _grad_fd(self,X):
        pred = self.model(X)
        epsilon= 1e-2
        ex= torch.tensor([epsilon,0,0],device= self.device)
        ey= torch.tensor([0,epsilon,0],device= self.device)
        ez= torch.tensor([0,0,epsilon],device= self.device)
        diffx= self.model(X+ex)-pred
        diffy= self.model(X+ey)-pred
        diffz= self.model(X+ez)-pred
        
        grad_fd= torch.cat([diffx,diffy,diffz],1)/epsilon
        return grad_fd
    
    def train_step(self, data):
        # assert batch_size == 1
        X = data["points"][0] # [B, 3]
        y = data["sdfs"][0] # [B]
        
        pred = self.model(X)
        loss = self.criterion(pred, y)

        if self.use_tensorboardX:
            pred_ref_grid= self.model(self.reference_grid_points)
            #remove dimension of pred_ref_grid
            pred_ref_grid= pred_ref_grid.squeeze()
            self.criterionRef= nn.L1Loss()
            loss_sd_ref_grid= self.criterionRef(pred_ref_grid,self.reference_grid_sdf)
            grad_fd_ref_grid= self._grad_fd(self.reference_grid_points)
            loss_grad_ref_grid= self.criterionRef(grad_fd_ref_grid,self.reference_grid_gradients)

            pred_ref_surface= self.model(self.reference_surface_points)
            pred_ref_surface= pred_ref_surface.squeeze()
            loss_sd_ref_surface= self.criterionRef(pred_ref_surface,self.reference_surface_sdf)
            grad_fd_ref_surface= self._grad_fd(self.reference_surface_points)
            loss_grad_ref_surface= self.criterionRef(grad_fd_ref_surface,self.reference_surface_gradients)
        else :
            loss_sd_ref_grid= 0
            loss_grad_ref_grid= 0
            loss_sd_ref_surface= 0
            loss_grad_ref_surface= 0

        return pred, y, loss, loss_sd_ref_grid,loss_grad_ref_grid,loss_sd_ref_surface,loss_grad_ref_surface

    def eval_step(self, data):
        return self.train_step(data)

    def test_step(self, data):  
        X = data["points"][0]
        pred = self.model(X)
        return pred        

    def save_mesh(self, save_path=None, resolution=256):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            pts = pts.to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sdfs = self.model(pts)
            return sdfs

        bounds_min = torch.FloatTensor([-1, -1, -1])
        bounds_max = torch.FloatTensor([1, 1, 1])

        vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=0, query_func=query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def _reference_grid_ (self, num_samples_per_dim: int): 
        #sample points on a grid
        x = np.linspace(-0.97, 0.97, num_samples_per_dim)
        y = np.linspace(-0.97, 0.97, num_samples_per_dim)
        z = np.linspace(-0.97, 0.97, num_samples_per_dim)
        xv, yv, zv = np.meshgrid(x, y, z)
        points= np.stack([xv,yv,zv],axis=3).reshape(-1,3)
        #calculate the sdf and gradient of grid points from mesh
        grid_sdf,grid_gradients= compute_sd_and_grad(self.mesh,points)
        # make points torch tensor
        self.reference_grid_points=  torch.tensor(points,device= self.device).float()
        self.reference_grid_sdf=torch.tensor(grid_sdf,device= self.device)
        #convert grid sdf to float
        self.reference_grid_sdf= self.reference_grid_sdf.float()

        self.reference_grid_gradients=torch.tensor(grid_gradients,device= self.device)
        #convert grid gradient to float
        self.reference_grid_gradients= self.reference_grid_gradients.float()
    
    def _reference_surface_ (self, num_samples: int):
        sdfs = np.zeros((num_samples, 1))
        # sample surface
        points,triangle_id = trimesh.sample.sample_surface_even(self.mesh,num_samples)

        # compute the barycentric coordinates of each sample
        bary = trimesh.triangles.points_to_barycentric(
        triangles=self.mesh.triangles[triangle_id], points=points)
        # interpolate vertex normals from barycentric coordinates
        gradients = trimesh.unitize((self.mesh.vertex_normals[self.mesh.faces[triangle_id]] *
                                trimesh.unitize(bary).reshape(
                                    (-1, 3, 1))).sum(axis=1))
        self.reference_surface_sdf=torch.tensor(sdfs,device= self.device)
        self.reference_surface_sdf= self.reference_surface_sdf.float()
        self.reference_surface_points=torch.tensor(points,device= self.device).float()
        self.reference_surface_gradients=torch.tensor(gradients,device= self.device)
        self.reference_surface_sdf= self.reference_surface_sdf.float()

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))
        
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                # self.save_mesh()
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader):
        #if os.path.exists(self.best_path):
        #    self.load_checkpoint(self.best_path)
        #else:
        #    self.load_checkpoint()
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader)
        self.use_tensorboardX = use_tensorboardX



    def prepare_data(self, data):
        if isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, np.ndarray):
                    data[i] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[i] = v.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[k] = v.to(self.device, non_blocking=True)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device, non_blocking=True)
        else: # is_tensor, or other similar objects that has `to`
            data = data.to(self.device, non_blocking=True)

        return data

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        total_loss_sd_ref_grid = 0
        total_loss_grad_ref_grid = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            self.local_step += 1
            self.global_step += 1
            
            data = self.prepare_data(data)

            self.optimizer.zero_grad()

            #measure time here
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss, loss_sd_ref_grid,loss_grad_ref_grid, loss_sd_ref_surface,loss_grad_ref_surface= self.train_step(data)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()
            print(start.elapsed_time(end))
            
            if self.ema is not None:
                self.ema.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.use_tensorboardX:
                loss_sd_ref_grid_val = loss_sd_ref_grid.item()
                loss_grad_ref_grid_val = loss_grad_ref_grid.item()
                loss_sd_ref_surface_val = loss_sd_ref_surface.item()
                loss_grad_ref_surface_val = loss_grad_ref_surface.item()
                total_loss_sd_ref_grid += loss_sd_ref_grid_val
                total_loss_grad_ref_grid += loss_grad_ref_grid_val
                self.writer.add_scalar("train/loss_sd_ref_grid", loss_sd_ref_grid_val, self.global_step)
                self.writer.add_scalar("train/loss_grad_ref_grid", loss_grad_ref_grid_val, self.global_step)
                self.writer.add_scalar("train/loss_sd_ref_surface", loss_sd_ref_surface_val, self.global_step)    
                self.writer.add_scalar("train/loss_grad_ref_surface", loss_grad_ref_surface_val, self.global_step)

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")


    def evaluate_one_epoch(self, loader):
        # self.log(f"++> Evaluate at epoch {self.epoch} ...")

        # total_loss = 0
        # if self.local_rank == 0:
        #     for metric in self.metrics:
        #         metric.clear()

        # self.model.eval()

        # if self.local_rank == 0:
        #     pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        # with torch.no_grad():
        #     self.local_step = 0
        #     for data in loader:    
        #         self.local_step += 1
                
        #         data = self.prepare_data(data)

        #         if self.ema is not None:
        #             self.ema.store()
        #             self.ema.copy_to()
            
        #         with torch.cuda.amp.autocast(enabled=self.fp16):
        #             preds, truths, loss, _, _= self.eval_step(data)

        #         if self.ema is not None:
        #             self.ema.restore()
                
        #         # all_gather/reduce the statistics (NCCL only support all_*)
        #         if self.world_size > 1:
        #             dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        #             loss = loss / self.world_size
                    
        #             preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
        #             dist.all_gather(preds_list, preds)
        #             preds = torch.cat(preds_list, dim=0)

        #             truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
        #             dist.all_gather(truths_list, truths)
        #             truths = torch.cat(truths_list, dim=0)

        #         loss_val = loss.item()
        #         total_loss += loss_val

        #         # only rank = 0 will perform evaluation.
        #         if self.local_rank == 0:

        #             for metric in self.metrics:
        #                 metric.update(preds, truths)

        #             pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
        #             pbar.update(loader.batch_size)

        # average_loss = total_loss / self.local_step
        # self.stats["valid_loss"].append(average_loss)

        # if self.local_rank == 0:
        #     pbar.close()
        #     if not self.use_loss_as_metric and len(self.metrics) > 0:
        #         result = self.metrics[0].measure()
        #         self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
        #     else:
        #         self.stats["results"].append(average_loss) # if no metric, choose best by min loss

        #     for metric in self.metrics:
        #         self.log(metric.report(), style="blue")
        #         if self.use_tensorboardX:
        #             metric.write(self.writer, self.epoch, prefix="evaluate")
        #         metric.clear()

        # self.log(f"++> Evaluate epoch {self.epoch} Finished.")
        return 0

    def save_checkpoint(self, full=False, best=False):

        state = {
            'epoch': self.epoch,
            'stats': self.stats,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}.pth.tar"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = self.stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")            

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        
        if self.optimizer and  'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer, use default.")
        
        # strange bug: keyerror 'lr_lambdas'
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler, use default.")

        if 'scaler' in checkpoint_dict:
            self.scaler.load_state_dict(checkpoint_dict['scaler'])     

class TrainerSurfaceMapping(object):
    def __init__(self, 
                 name, # name of this experiment
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 mesh=None, # mesh for evaluation
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metirc
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        
        self.name = name
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.mesh = mesh

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth.tar"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	
    
    def train_step(self, data):
        # assert batch_size == 1
        X = data["original"][0] # [B, 3]
        y = data["mappped"][0] # [B,3]
        
        pred = self.model(X)
        loss = self.criterion(pred, y)

        return pred, y, loss

    def eval_step(self, data):
        return self.train_step(data)

    def test_step(self, data):  
        X = data["points"][0]
        pred = self.model(X)
        return pred        

    def save_mesh(self, save_path=None, resolution=256):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            pts = pts.to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sdfs = self.model(pts)
            return sdfs

        bounds_min = torch.FloatTensor([-1, -1, -1])
        bounds_max = torch.FloatTensor([1, 1, 1])

        vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=0, query_func=query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))
        
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                # self.save_mesh()
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader):
        #if os.path.exists(self.best_path):
        #    self.load_checkpoint(self.best_path)
        #else:
        #    self.load_checkpoint()
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader)
        self.use_tensorboardX = use_tensorboardX



    def prepare_data(self, data):
        if isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, np.ndarray):
                    data[i] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[i] = v.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[k] = v.to(self.device, non_blocking=True)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device, non_blocking=True)
        else: # is_tensor, or other similar objects that has `to`
            data = data.to(self.device, non_blocking=True)

        return data

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        total_loss_sd_ref_grid = 0
        total_loss_grad_ref_grid = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            self.local_step += 1
            self.global_step += 1
            
            data = self.prepare_data(data)

            self.optimizer.zero_grad()

            #measure time here
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss= self.train_step(data)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()
            print(start.elapsed_time(end))
            
            if self.ema is not None:
                self.ema.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")


    def evaluate_one_epoch(self, loader):
        return 0

    def save_checkpoint(self, full=False, best=False):

        state = {
            'epoch': self.epoch,
            'stats': self.stats,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}.pth.tar"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = self.stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")            

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        
        if self.optimizer and  'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer, use default.")
        
        # strange bug: keyerror 'lr_lambdas'
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler, use default.")

        if 'scaler' in checkpoint_dict:
            self.scaler.load_state_dict(checkpoint_dict['scaler'])    

class TrainerSubspace(object):
    def __init__(self, 
                 name, # name of this experiment
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 mesh=None, # mesh for evaluation
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metirc
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        
        self.name = name
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.mesh = mesh

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth.tar"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

        if self.use_tensorboardX:
            self._reference_grid_(17)
            self._reference_surface_(2**10)

    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	
    def _grad_fd(self,X,subspace):
        pred = self.model(X)
        epsilon= 1e-2
        ex= torch.tensor([epsilon,0,0],device= self.device)
        ey= torch.tensor([0,epsilon,0],device= self.device)
        ez= torch.tensor([0,0,epsilon],device= self.device)
        diffx= self.model(X+ex,subspace)-pred
        diffy= self.model(X+ey,subspace)-pred
        diffz= self.model(X+ez,subspace)-pred
        
        grad_fd= torch.cat([diffx,diffy,diffz],1)/epsilon
        return grad_fd
    
    def train_step(self, data):
        # assert batch_size == 1
        X = data["points"][0] # [B, 3]
        subspaces= data["subspace"][0] # [B, 4]
        y = data["sdfs"][0] # [B]
        
        pred = self.model(X,subspaces)
        loss = self.criterion(pred, y)

        if self.use_tensorboardX:
            pred_ref_grid= self.model(self.reference_grid_points, subspaces)
            #remove dimension of pred_ref_grid
            pred_ref_grid= pred_ref_grid.squeeze()
            self.criterionRef= nn.L1Loss()
            loss_sd_ref_grid= self.criterionRef(pred_ref_grid,self.reference_grid_sdf)
            grad_fd_ref_grid= self._grad_fd(self.reference_grid_points, subspaces)
            loss_grad_ref_grid= self.criterionRef(grad_fd_ref_grid,self.reference_grid_gradients)

            pred_ref_surface= self.model(self.reference_surface_points, subspaces)
            pred_ref_surface= pred_ref_surface.squeeze()
            loss_sd_ref_surface= self.criterionRef(pred_ref_surface,self.reference_surface_sdf)
            grad_fd_ref_surface= self._grad_fd(self.reference_surface_points, subspaces)
            loss_grad_ref_surface= self.criterionRef(grad_fd_ref_surface,self.reference_surface_gradients)
        else :
            loss_sd_ref_grid= 0
            loss_grad_ref_grid= 0
            loss_sd_ref_surface= 0
            loss_grad_ref_surface= 0

        return pred, y, loss, loss_sd_ref_grid,loss_grad_ref_grid,loss_sd_ref_surface,loss_grad_ref_surface

    def eval_step(self, data):
        return self.train_step(data)

    def test_step(self, data):  
        X = data["points"][0]
        subspace= data["subspace"][0]
        pred = self.model(X,subspace)
        return pred        

    def save_mesh(self, save_path=None, resolution=256):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            pts = pts.to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    #make a subspace of zeros with the same number of rows as pts
                    subspace= torch.zeros(pts.shape[0],6,device= self.device)
                    #make subspace the first column to be one
                    subspace[:,0]= 1
                    sdfs = self.model(pts,subspace)
            return sdfs

        bounds_min = torch.FloatTensor([-1, -1, -1])
        bounds_max = torch.FloatTensor([1, 1, 1])

        vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=0, query_func=query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def _reference_grid_ (self, num_samples_per_dim: int): 
        #sample points on a grid
        x = np.linspace(-0.97, 0.97, num_samples_per_dim)
        y = np.linspace(-0.97, 0.97, num_samples_per_dim)
        z = np.linspace(-0.97, 0.97, num_samples_per_dim)
        xv, yv, zv = np.meshgrid(x, y, z)
        points= np.stack([xv,yv,zv],axis=3).reshape(-1,3)
        #calculate the sdf and gradient of grid points from mesh
        grid_sdf,grid_gradients= compute_sd_and_grad(self.mesh,points)
        # make points torch tensor
        self.reference_grid_points=  torch.tensor(points,device= self.device).float()
        self.reference_grid_sdf=torch.tensor(grid_sdf,device= self.device)
        #convert grid sdf to float
        self.reference_grid_sdf= self.reference_grid_sdf.float()

        self.reference_grid_gradients=torch.tensor(grid_gradients,device= self.device)
        #convert grid gradient to float
        self.reference_grid_gradients= self.reference_grid_gradients.float()
    
    def _reference_surface_ (self, num_samples: int):
        sdfs = np.zeros((num_samples, 1))
        # sample surface
        points,triangle_id = trimesh.sample.sample_surface_even(self.mesh,num_samples)

        # compute the barycentric coordinates of each sample
        bary = trimesh.triangles.points_to_barycentric(
        triangles=self.mesh.triangles[triangle_id], points=points)
        # interpolate vertex normals from barycentric coordinates
        gradients = trimesh.unitize((self.mesh.vertex_normals[self.mesh.faces[triangle_id]] *
                                trimesh.unitize(bary).reshape(
                                    (-1, 3, 1))).sum(axis=1))
        self.reference_surface_sdf=torch.tensor(sdfs,device= self.device)
        self.reference_surface_sdf= self.reference_surface_sdf.float()
        self.reference_surface_points=torch.tensor(points,device= self.device).float()
        self.reference_surface_gradients=torch.tensor(gradients,device= self.device)
        self.reference_surface_sdf= self.reference_surface_sdf.float()

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))
        
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                # self.save_mesh()
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader):
        #if os.path.exists(self.best_path):
        #    self.load_checkpoint(self.best_path)
        #else:
        #    self.load_checkpoint()
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader)
        self.use_tensorboardX = use_tensorboardX



    def prepare_data(self, data):
        if isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, np.ndarray):
                    data[i] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[i] = v.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[k] = v.to(self.device, non_blocking=True)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device, non_blocking=True)
        else: # is_tensor, or other similar objects that has `to`
            data = data.to(self.device, non_blocking=True)

        return data

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        total_loss_sd_ref_grid = 0
        total_loss_grad_ref_grid = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            self.local_step += 1
            self.global_step += 1
            
            data = self.prepare_data(data)

            self.optimizer.zero_grad()

            #measure time here
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss, loss_sd_ref_grid,loss_grad_ref_grid, loss_sd_ref_surface,loss_grad_ref_surface= self.train_step(data)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()
            print(start.elapsed_time(end))
            
            if self.ema is not None:
                self.ema.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.use_tensorboardX:
                loss_sd_ref_grid_val = loss_sd_ref_grid.item()
                loss_grad_ref_grid_val = loss_grad_ref_grid.item()
                loss_sd_ref_surface_val = loss_sd_ref_surface.item()
                loss_grad_ref_surface_val = loss_grad_ref_surface.item()
                total_loss_sd_ref_grid += loss_sd_ref_grid_val
                total_loss_grad_ref_grid += loss_grad_ref_grid_val
                self.writer.add_scalar("train/loss_sd_ref_grid", loss_sd_ref_grid_val, self.global_step)
                self.writer.add_scalar("train/loss_grad_ref_grid", loss_grad_ref_grid_val, self.global_step)
                self.writer.add_scalar("train/loss_sd_ref_surface", loss_sd_ref_surface_val, self.global_step)    
                self.writer.add_scalar("train/loss_grad_ref_surface", loss_grad_ref_surface_val, self.global_step)

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")


    def evaluate_one_epoch(self, loader):
        # self.log(f"++> Evaluate at epoch {self.epoch} ...")

        # total_loss = 0
        # if self.local_rank == 0:
        #     for metric in self.metrics:
        #         metric.clear()

        # self.model.eval()

        # if self.local_rank == 0:
        #     pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        # with torch.no_grad():
        #     self.local_step = 0
        #     for data in loader:    
        #         self.local_step += 1
                
        #         data = self.prepare_data(data)

        #         if self.ema is not None:
        #             self.ema.store()
        #             self.ema.copy_to()
            
        #         with torch.cuda.amp.autocast(enabled=self.fp16):
        #             preds, truths, loss, _, _= self.eval_step(data)

        #         if self.ema is not None:
        #             self.ema.restore()
                
        #         # all_gather/reduce the statistics (NCCL only support all_*)
        #         if self.world_size > 1:
        #             dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        #             loss = loss / self.world_size
                    
        #             preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
        #             dist.all_gather(preds_list, preds)
        #             preds = torch.cat(preds_list, dim=0)

        #             truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
        #             dist.all_gather(truths_list, truths)
        #             truths = torch.cat(truths_list, dim=0)

        #         loss_val = loss.item()
        #         total_loss += loss_val

        #         # only rank = 0 will perform evaluation.
        #         if self.local_rank == 0:

        #             for metric in self.metrics:
        #                 metric.update(preds, truths)

        #             pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
        #             pbar.update(loader.batch_size)

        # average_loss = total_loss / self.local_step
        # self.stats["valid_loss"].append(average_loss)

        # if self.local_rank == 0:
        #     pbar.close()
        #     if not self.use_loss_as_metric and len(self.metrics) > 0:
        #         result = self.metrics[0].measure()
        #         self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
        #     else:
        #         self.stats["results"].append(average_loss) # if no metric, choose best by min loss

        #     for metric in self.metrics:
        #         self.log(metric.report(), style="blue")
        #         if self.use_tensorboardX:
        #             metric.write(self.writer, self.epoch, prefix="evaluate")
        #         metric.clear()

        # self.log(f"++> Evaluate epoch {self.epoch} Finished.")
        return 0

    def save_checkpoint(self, full=False, best=False):

        state = {
            'epoch': self.epoch,
            'stats': self.stats,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}.pth.tar"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = self.stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")            

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        
        if self.optimizer and  'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer, use default.")
        
        # strange bug: keyerror 'lr_lambdas'
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler, use default.")

        if 'scaler' in checkpoint_dict:
            self.scaler.load_state_dict(checkpoint_dict['scaler'])                



class TrainerOurs(object):
    def __init__(self, 
                 name, # name of this experiment
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 mesh=None, # mesh for evaluation
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metirc
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 num_volume_samples=2**15, # num of samples to use for volume 
                 ):
        
        self.name = name
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.mesh = mesh
        self.number_random_sample_every_step = num_volume_samples
        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = nn.L1Loss() 
        self.criterion_grad = nn.L1Loss() 
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth.tar"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

        self._sample_grid_(16)
        if self.use_tensorboardX:
            self._reference_grid_(17)
            self._reference_surface_(2**10)

    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	
    def _grad_fd(self,X):
        #concatenate X and the perturbation
        epsilon= 1e-2
        ex= torch.tensor([epsilon,0,0],device= self.device,requires_grad=False)
        ey= torch.tensor([0,epsilon,0],device= self.device,requires_grad=False)
        ez= torch.tensor([0,0,epsilon],device= self.device,requires_grad=False)
        pred = self.model(X)
        diffx= -self.model(X-ex)+pred
        diffy= -self.model(X-ey)+pred
        diffz= -self.model(X-ez)+pred
        grad_fd= torch.cat([diffx,diffy,diffz],1)/epsilon
        return grad_fd
    
    def train_step(self, data):

        pred_grid= self.model(self.grid_points)
        pred_grid= pred_grid.squeeze()
        loss_sd_grid= self.criterion(pred_grid,self.grid_sdf)

        grad_fd_grid= self._grad_fd(self.grid_points)
        loss_grad_grid= self.criterion(grad_fd_grid,self.grid_gradients)

        # assert batch_size == 1
        X = data["points"][0] # [B, 3]
        y = data["sdfs"][0] # [B]
        grad =data["gradients"][0]
        pred = self.model(X)
        loss_sd = self.criterion(pred, y)

        grad_fd= self._grad_fd(X)
        loss_grad=self.criterion_grad(grad_fd,grad)

        volume_samples = self._sample_volume(self.number_random_sample_every_step)
        grad_unsupervised = self._grad_fd(volume_samples)
        
        eikonal_l1 = torch.sum(torch.abs(torch.linalg.norm(grad_unsupervised, dim=1) - 1))
        eikonal_l1_avg = 100*eikonal_l1/self.number_random_sample_every_step


        if self.use_tensorboardX:
            pred_ref_grid= self.model(self.reference_grid_points)
            #remove dimension of pred_ref_grid
            pred_ref_grid= pred_ref_grid.squeeze()
            self.criterionRef= nn.L1Loss()
            loss_sd_ref_grid= self.criterionRef(pred_ref_grid,self.reference_grid_sdf)
            grad_fd_ref_grid= self._grad_fd(self.reference_grid_points)
            loss_grad_ref_grid= self.criterionRef(grad_fd_ref_grid,self.reference_grid_gradients)

            pred_ref_surface= self.model(self.reference_surface_points)
            pred_ref_surface= pred_ref_surface.squeeze()
            loss_sd_ref_surface= self.criterionRef(pred_ref_surface,self.reference_surface_sdf)
            grad_fd_ref_surface= self._grad_fd(self.reference_surface_points)
            loss_grad_ref_surface= self.criterionRef(grad_fd_ref_surface,self.reference_surface_gradients)
        else :
            loss_sd_ref_grid= 0
            loss_grad_ref_grid= 0
            loss_sd_ref_surface= 0
            loss_grad_ref_surface= 0

        return pred, y, 10000*loss_sd+ 10*loss_grad + eikonal_l1_avg+100*loss_sd_grid+10*loss_grad_grid,100*loss_sd,10*loss_grad,eikonal_l1_avg,100*loss_sd_grid,10*loss_grad_grid,loss_sd_ref_grid,loss_grad_ref_grid,loss_sd_ref_surface,loss_grad_ref_surface

    def eval_step(self, data):
        return self.train_step(data)

    def test_step(self, data):  
        X = data["points"][0]
        pred = self.model(X)
        return pred        

    def save_mesh(self, save_path=None, resolution=256):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            pts = pts.to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sdfs = self.model(pts)
            return sdfs

        bounds_min = torch.FloatTensor([-1, -1, -1])
        bounds_max = torch.FloatTensor([1, 1, 1])

        vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=0, query_func=query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------
    def _sample_volume(self, num_samples: int):
        # sample points in the volume between -1 and 1 randomly
        points = torch.rand(num_samples, 3, device=self.device) * 2 - 1
        points = points * 0.98
        return points
    
    def _sample_grid_ (self, num_samples_per_dim: int): 
        #sample points on a grid
        x = np.linspace(-0.98, 0.98, num_samples_per_dim)
        y = np.linspace(-0.98, 0.98, num_samples_per_dim)
        z = np.linspace(-0.98, 0.98, num_samples_per_dim)
        xv, yv, zv = np.meshgrid(x, y, z)
        points= np.stack([xv,yv,zv],axis=3).reshape(-1,3)
        #calculate the sdf and gradient of grid points from mesh
        grid_sdf,grid_gradients= compute_sd_and_grad(self.mesh,points)
        # make points torch tensor
        self.grid_points=  torch.tensor(points,device= self.device).float()
        self.grid_sdf=torch.tensor(grid_sdf,device= self.device)
        #convert grid sdf to float
        self.grid_sdf= self.grid_sdf.float()

        self.grid_gradients=torch.tensor(grid_gradients,device= self.device)
        #convert grid gradient to float
        self.grid_gradients= self.grid_gradients.float()

    def _reference_grid_ (self, num_samples_per_dim: int): 
        #sample points on a grid
        x = np.linspace(-0.97, 0.97, num_samples_per_dim)
        y = np.linspace(-0.97, 0.97, num_samples_per_dim)
        z = np.linspace(-0.97, 0.97, num_samples_per_dim)
        xv, yv, zv = np.meshgrid(x, y, z)
        points= np.stack([xv,yv,zv],axis=3).reshape(-1,3)
        #calculate the sdf and gradient of grid points from mesh
        grid_sdf,grid_gradients= compute_sd_and_grad(self.mesh,points)
        # make points torch tensor
        self.reference_grid_points=  torch.tensor(points,device= self.device).float()
        self.reference_grid_sdf=torch.tensor(grid_sdf,device= self.device)
        #convert grid sdf to float
        self.reference_grid_sdf= self.reference_grid_sdf.float()

        self.reference_grid_gradients=torch.tensor(grid_gradients,device= self.device)
        #convert grid gradient to float
        self.reference_grid_gradients= self.reference_grid_gradients.float()

    def _reference_surface_ (self, num_samples: int):
        sdfs = np.zeros((num_samples, 1))
        # sample surface
        points,triangle_id = trimesh.sample.sample_surface_even(self.mesh,num_samples)

        # compute the barycentric coordinates of each sample
        bary = trimesh.triangles.points_to_barycentric(
        triangles=self.mesh.triangles[triangle_id], points=points)
        # interpolate vertex normals from barycentric coordinates
        gradients = trimesh.unitize((self.mesh.vertex_normals[self.mesh.faces[triangle_id]] *
                                trimesh.unitize(bary).reshape(
                                    (-1, 3, 1))).sum(axis=1))
        self.reference_surface_sdf=torch.tensor(sdfs,device= self.device)
        self.reference_surface_sdf= self.reference_surface_sdf.float()
        self.reference_surface_points=torch.tensor(points,device= self.device).float()
        self.reference_surface_gradients=torch.tensor(gradients,device= self.device)
        self.reference_surface_sdf= self.reference_surface_sdf.float()
        

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))
        
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                # self.save_mesh()
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader):
        #if os.path.exists(self.best_path):
        #    self.load_checkpoint(self.best_path)
        #else:
        #    self.load_checkpoint()
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader)
        self.use_tensorboardX = use_tensorboardX



    def prepare_data(self, data):
        if isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, np.ndarray):
                    data[i] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[i] = v.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[k] = v.to(self.device, non_blocking=True)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device, non_blocking=True)
        else: # is_tensor, or other similar objects that has `to`
            data = data.to(self.device, non_blocking=True)

        return data

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        total_loss_sd_ref_grid = 0
        total_loss_grad_ref_grid = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            self.local_step += 1
            self.global_step += 1
            
            data = self.prepare_data(data)

            self.optimizer.zero_grad()

            #measure time here
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss,loss_sd,loss_grad,loss_eikonal,loss_sd_grid,loss_grad_grid,loss_sd_ref_grid,loss_grad_ref_grid,loss_sd_ref_surface,loss_grad_ref_surface = self.train_step(data)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()
            print(start.elapsed_time(end))

            if self.ema is not None:
                self.ema.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            loss_val = loss.item()
            total_loss += loss_val
            loss_sd_val=loss_sd.item()
            loss_grad_val= loss_grad.item()
            loss_eikonal_val= loss_eikonal.item()
            loss_sd_grid_val= loss_sd_grid.item()
            loss_grad_grid_val= loss_grad_grid.item()

            if self.use_tensorboardX:
                loss_sd_ref_grid_val = loss_sd_ref_grid.item()
                loss_grad_ref_grid_val = loss_grad_ref_grid.item()
                loss_sd_ref_surface_val = loss_sd_ref_surface.item()
                loss_grad_ref_surface_val = loss_grad_ref_surface.item()
                total_loss_sd_ref_grid += loss_sd_ref_grid_val
                total_loss_grad_ref_grid += loss_grad_ref_grid_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/loss_sd", loss_sd_val, self.global_step)
                    self.writer.add_scalar("train/loss_grad", loss_grad_val, self.global_step)
                    self.writer.add_scalar("train/loss_eikonal", loss_eikonal_val, self.global_step)
                    self.writer.add_scalar("train/loss_sd_grid", loss_sd_grid_val, self.global_step)
                    self.writer.add_scalar("train/loss_grad_grid", loss_grad_grid_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)
                    self.writer.add_scalar("train/loss_sd_ref_grid", loss_sd_ref_grid_val, self.global_step)    
                    self.writer.add_scalar("train/loss_grad_ref_grid", loss_grad_ref_grid_val, self.global_step)
                    self.writer.add_scalar("train/loss_sd_ref_surface", loss_sd_ref_surface_val, self.global_step)    
                    self.writer.add_scalar("train/loss_grad_ref_surface", loss_grad_ref_surface_val, self.global_step)
                    # a= gen_img_grad(self.model)
                    # a= a.permute([2,0,1])
                    # self.writer.add_image("train/grad_image", a, self.global_step)

                    # b= gen_img_dist(self.model)

                    #make matrix b to be a heatmap for better visualization in Tensorboard
                    # heatmap = plt.get_cmap('viridis')(b.detach().cpu().numpy())
                    # rgb = heatmap[:, :, :3]
                    # b=torch.from_numpy(rgb).to("cuda")
                    # b= b.permute([2,0,1])
                    # self.writer.add_image("train/sdf_image", b, self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")


    def evaluate_one_epoch(self, loader):
        # self.log(f"++> Evaluate at epoch {self.epoch} ...")

        # total_loss = 0
        # if self.local_rank == 0:
        #     for metric in self.metrics:
        #         metric.clear()

        # self.model.eval()

        # if self.local_rank == 0:
        #     pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        # with torch.no_grad():
        #     self.local_step = 0
        #     for data in loader:    
        #         self.local_step += 1
                
        #         data = self.prepare_data(data)

        #         if self.ema is not None:
        #             self.ema.store()
        #             self.ema.copy_to()
            
        #         with torch.cuda.amp.autocast(enabled=self.fp16):
        #             preds, truths, loss,loss_sd,loss_grad,loss_eikonal,loss_sd_grid,loss_grad_grid,loss_sd_ref_grid,loss_grad_ref_grid= self.eval_step(data)

        #         if self.ema is not None:
        #             self.ema.restore()
                
        #         # all_gather/reduce the statistics (NCCL only support all_*)
        #         if self.world_size > 1:
        #             dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        #             loss = loss / self.world_size
                    
        #             preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
        #             dist.all_gather(preds_list, preds)
        #             preds = torch.cat(preds_list, dim=0)

        #             truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
        #             dist.all_gather(truths_list, truths)
        #             truths = torch.cat(truths_list, dim=0)

        #         loss_val = loss.item()
        #         total_loss += loss_val

        #         # only rank = 0 will perform evaluation.
        #         if self.local_rank == 0:

        #             for metric in self.metrics:
        #                 metric.update(preds, truths)

        #             pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
        #             pbar.update(loader.batch_size)

        # average_loss = total_loss / self.local_step
        # self.stats["valid_loss"].append(average_loss)

        # if self.local_rank == 0:
        #     pbar.close()
        #     if not self.use_loss_as_metric and len(self.metrics) > 0:
        #         result = self.metrics[0].measure()
        #         self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
        #     else:
        #         self.stats["results"].append(average_loss) # if no metric, choose best by min loss

        #     for metric in self.metrics:
        #         self.log(metric.report(), style="blue")
        #         if self.use_tensorboardX:
        #             metric.write(self.writer, self.epoch, prefix="evaluate")
        #         metric.clear()

        # self.log(f"++> Evaluate epoch {self.epoch} Finished.")
        return 0

    def save_checkpoint(self, full=False, best=False):

        state = {
            'epoch': self.epoch,
            'stats': self.stats,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}.pth.tar"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = self.stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")            

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        
        if self.optimizer and  'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer, use default.")
        
        # strange bug: keyerror 'lr_lambdas'
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler, use default.")

        if 'scaler' in checkpoint_dict:
            self.scaler.load_state_dict(checkpoint_dict['scaler'])       