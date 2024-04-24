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
    opt.tcnn=False
    opt.fp16=False

    if opt.ff:
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        from sdf.netowrk_ff import SDFNetwork
    elif opt.tcnn:
        assert opt.fp16, "tcnn mode must be used with fp16 mode"
        from sdf.network_tcnn import SDFNetwork        
    else:
        from sdf.netowrk import SDFNetwork,SDFNetworkWithSubspaceInput,SDFNetworkWithSubspaceInputOnlyForPreencoder

    from sdf.provider import SDFDatasetTestPreencoder
    from loss import mape_loss

    model = SDFNetworkWithSubspaceInputOnlyForPreencoder(encoding="hashgrid",num_layers=5,num_layers_pre=5, subspace_size=7)
    #load model from"trainedENDTOEND.pth"
    model.load_state_dict(torch.load("WorkSpaceFolder/OInputOnlyForPreencoderTrainedOnlyWithPreencoder_lr_0.01_2023-11-14-17-04-59cow.obj.pth"))

    from sdf.provider import SDFDatasetTestPreencoder,SDFDatasetTestPreencoderEndToEnd
    

    train_dataset = SDFDatasetTestPreencoderEndToEnd("mode/deformed_cow", size=1, num_samples=2**10)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    valid_dataset = SDFDatasetTestPreencoderEndToEnd("mode/deformed_cow", size=1, num_samples=2**10) # just a dummy
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

    criterion = torch.nn.L1Loss()

    optimizer = lambda model: torch.optim.Adam([
        {'name': 'net0', 'params': model.preencoder.parameters(), 'weight_decay': 1e-6},
        {'name': 'encoding', 'params': model.encoder.parameters()},
        {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
        
    ], lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)

    # weights= np.array([[1,1,1,1,1,1,1]])/7
    weights= np.array([[1,0,0,0,0,0,0]])
    dataset = SDFDatasetTestPreencoder("mode/deformed_cow", size=100, num_samples=8)
    newmesh=dataset.mesh.copy()
    x = sum(weights[0][j] * dataset.vs[j] for j in range(len(dataset.vs)))
    newmesh.vertices = x

    scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    time_stamp=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    name="WorkSpaceFolder/"+"RetrainWithFullParameter"+"lr_"+str(opt.lr)+"_"+time_stamp+opt.path.split("/")[-1]

    trainer = TrainerTestPreencoder('ngp', model, workspace=name, optimizer=optimizer, criterion=criterion, ema_decay=0.95,
                        fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint='latest',
                        eval_interval=1,use_tensorboardX=True,mesh=newmesh)

    trainer.train(train_loader, valid_loader, 2000)

    # also test
    # trainer.save_mesh(os.path.join(opt.workspace, 'results', 'sinus_original_output.ply'), 1024)


    #save model
    layer_to_save= {'preencoder':model.preencoder.state_dict()}
    torch.save(model.state_dict(), name+".pth")
    torch.save(layer_to_save, name+"pre.pth")
   




    