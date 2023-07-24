import torch
import argparse

from sdf.utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    opt = parser.parse_args()
    print(opt)

    seed_everything(opt.seed)
    opt.tcnn=True
    opt.fp16=True

    if opt.ff:
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        from sdf.netowrk_ff import SDFNetwork
    elif opt.tcnn:
        assert opt.fp16, "tcnn mode must be used with fp16 mode"
        from sdf.network_tcnn import SDFNetwork        
    else:
        from sdf.netowrk import SDFNetwork

    model = SDFNetwork(encoding="hashgrid",activation="Softplus")
    print(model)

    if opt.test:
        trainer = Trainer('ngp', model, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint='best', eval_interval=1)
        trainer.save_mesh(os.path.join(opt.workspace, 'results', 'output.ply'), 1024)

    else:
        from sdf.provider import SurfaceDataset
        from loss import mape_loss

        opt.path="data/sinus_cow.obj"
        opt.path_new="data/sinus_cow.obj"

        mesh = trimesh.load(opt.path, force='mesh')
        vs = mesh.vertices
        vmin = vs.min(0)
        vmax = vs.max(0)
        v_center = (vmin + vmax) / 2
        v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
        vs = (vs - v_center[None, :]) * v_scale
        mesh.vertices = vs


        mesh_deformed = trimesh.load(opt.path_new, force='mesh')
        vs = mesh_deformed.vertices
        vmin = vs.min(0)
        vmax = vs.max(0)
        v_center = (vmin + vmax) / 2
        v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
        vs = (vs - v_center[None, :]) * v_scale
        mesh_deformed.vertices = vs

        train_dataset = SurfaceDataset(mesh,num_samples=2**15,online_sampling=False,normalize_mesh=False) 
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

        # valid_dataset = SurfaceDataset(mesh,num_samples=2**15,online_sampling=False,normalize_mesh=False) 
        # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

        criterion = torch.nn.L1Loss()

        train_dataset_new = SurfaceDataset(mesh_deformed,num_samples=2**15,online_sampling=False,normalize_mesh=False) 
        train_dataset_new.size=100
        train_loader_new = torch.utils.data.DataLoader(train_dataset_new, batch_size=1, shuffle=True)

        # valid_dataset_new = SurfaceDataset(mesh_deformed,num_samples=2**15,online_sampling=False,normalize_mesh=False) 
        # valid_loader_new = torch.utils.data.DataLoader(valid_dataset_new, batch_size=1)

        optimizer = lambda model: torch.optim.Adam([
            {'name': 'encoding', 'params': model.encoder.parameters()},
            {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
        ], lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)

        opt.lr_new=1e-3
        optimizer_new = lambda model: torch.optim.Adam([
            {'name': 'encoding', 'params': model.encoder.parameters()},
            {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
        ], lr=opt.lr_new, betas=(0.9, 0.99), eps=1e-15)

        scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        scheduler_new = lambda optimizer_new: optim.lr_scheduler.StepLR(optimizer_new, step_size=10, gamma=0.1)
        time_stamp=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        name="coldstart/"+"coldstart_sinuscow_OURS_1time_"+model.activation+time_stamp+"_lr_"+str(opt.lr_new)+opt.path_new.split("/")[-1].split(".")[0]

        trainer = TrainerOurs('ngp', model, workspace=name+"Pre", optimizer=optimizer, criterion=criterion, ema_decay=0.95,
                           fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint='latest', eval_interval=1,
                           use_tensorboardX=True,mesh=train_dataset.mesh)

        trainer.train(train_loader, train_loader, 50)
        trainer.save_mesh(os.path.join("coldstart", 'results', 'output1time.ply'), 1024)
        
        # trainer_new = TrainerOurs('ngp', model, workspace=name, optimizer=optimizer_new, criterion=criterion, ema_decay=0.95,
        #                     fp16=opt.fp16, lr_scheduler=scheduler_new, use_checkpoint='latest', eval_interval=1,
        #                     use_tensorboardX=True,mesh=train_dataset_new.mesh)
        
        # trainer_new.train(train_loader_new, train_loader_new, 1)

        # # also test
        # trainer_new.save_mesh(os.path.join(opt.workspace, 'results', 'output_10_iteration.ply'), 1024)
