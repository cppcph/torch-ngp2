import torch
import argparse

from sdf.utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
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
    #train a perfect sdf model with 5 layers without subspace and preencoder
    model_sdf=SDFNetwork(encoding="hashgrid",num_layers=5)
    #randomly initialize the model
    for layer in model_sdf.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)

    #train from scratch
    
    from sdf.provider import SDFDatasetNoNormalization,SDFDatasetTestPreencoder
    train_dataset0=SDFDatasetNoNormalization("SMPLTemplateNormalized.obj", size=100, num_samples=2**14)
    train_loader0 = torch.utils.data.DataLoader(train_dataset0, batch_size=1, shuffle=True)
    valid_dataset0 = SDFDatasetNoNormalization("SMPLTemplateNormalized.obj", size=1, num_samples=2**14) # just a dummy
    valid_loader0 = torch.utils.data.DataLoader(valid_dataset0, batch_size=1)
    from loss import mape_loss
    # criterion = mape_loss
    #use L1 loss
    criterion = torch.nn.L1Loss()
    
    optimizer = lambda model: torch.optim.Adam([
            {'name': 'encoding', 'params': model.encoder.parameters()},
            {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
        ], lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)
    scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    time_stamp=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    name="SMPLmini/"+"OOsdf_Reference"+"lr_"+str(opt.lr)+"_"+time_stamp+opt.path.split("/")[-1]

    trainer0 = Trainer('ngp', model_sdf, workspace=name, optimizer=optimizer, criterion=criterion, ema_decay=0.95,
                           fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint='latest',
                            eval_interval=1,use_tensorboardX=True,mesh=train_dataset0.mesh)

    # trainer0.train(train_loader0, valid_loader0, 100)
    #save model
    # torch.save(model_sdf.state_dict(), name+".pth")
    # trainer0.save_mesh(os.path.join('SMPL', 'canonicalSMPL.ply'), 1024)
    #load model 'canonicalSMPL.ply'
    model_sdf.load_state_dict(torch.load("SMPL/OOsdf_Reference.pth"))

    
    from sdf.provider import SDFDatasetSMPLMini,SDFDatasetTestPreencoderEndToEnd
    ##second part
    # newmesh = trimesh.load("medio_cow_random_deformed_norm.obj", force='mesh')
    newmesh = trimesh.load("SMPLOneDefNormalized.obj", force='mesh')
    dataset = SDFDatasetSMPLMini( size=100, num_samples=8)
    #load the weights
    weights=np.load("weights.npy")
    weights[0]=10
    #leave the weights only one 
    weights=weights[0]
    #Add extra dimension to weights
    weights=weights.reshape(1,-1)


    model = SDFNetworkWithSubspaceInputOnlyForPreencoder(encoding="hashgrid",num_layers=5,num_layers_pre=5, subspace_size=1)

    #match the corresponding parameters of the parts of model and model sdf
    transfer_encoder_weights(model_sdf,model)
    transfer_backbone_weights(model_sdf,model)


    #load model
    # model.load_state_dict(torch.load("WorkSpaceFolder/original.pth"))
    # print(model)
    

    train_dataset = SDFDatasetSMPLMini(size=10, num_samples=2**10)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    valid_dataset = SDFDatasetSMPLMini(size=1, num_samples=2**10) # just a dummy
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

    criterion = torch.nn.L1Loss()

    optimizer = lambda model: torch.optim.Adam([
        {'name': 'net0', 'params': model.preencoder.parameters(), 'weight_decay': 1e-6},
        # {'name': 'encoding', 'params': model.encoder.parameters()},
        # {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
        
    ], lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)

    scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    time_stamp=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    name="SMPLmini/"+"BInputOnlyForPreencoderTrainedOnlyWithPreencoder_"+"lr_"+str(opt.lr)+"_"+time_stamp+opt.path.split("/")[-1]

    

    trainer = TrainerTestPreencoder('ngp', model, workspace=name, optimizer=optimizer, criterion=criterion, ema_decay=0.95,
                        fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint='latest',
                        eval_interval=1,use_tensorboardX=True,mesh=newmesh,subspace=weights)

    trainer.train(train_loader, valid_loader, 2000)


    layer_to_save= {'preencoder':model.preencoder.state_dict()}
    torch.save(model.state_dict(), name+".pth")
    torch.save(layer_to_save, name+"pre.pth")

    optimizer = lambda model: torch.optim.Adam([
        {'name': 'net0', 'params': model.preencoder.parameters(), 'weight_decay': 1e-6},
        {'name': 'encoding', 'params': model.encoder.parameters()},
        {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
        
    ], lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)

    scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    time_stamp=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    name="SMPLmini/"+"BRetrainForFullParameter_"+"lr_"+str(opt.lr)+"_"+time_stamp+opt.path.split("/")[-1]

    trainer = TrainerTestPreencoder('ngp', model, workspace=name, optimizer=optimizer, criterion=criterion, ema_decay=0.95,
                        fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint='latest',
                        eval_interval=1,use_tensorboardX=True,mesh=newmesh,subspace=weights)

    # trainer.train(train_loader, valid_loader, 2000)

    layer_to_save= {'preencoder':model.preencoder.state_dict()}
    torch.save(model.state_dict(), name+".pth")
    torch.save(layer_to_save, name+"pre.pth")

    original_mesh = trimesh.load("SMPLTemplateNormalized.obj", force='mesh')
    #load blendshapes
    blendshapes = np.load("blendshape.npy")

    #generate a serie of random number of uniform distribution of std 1
    weights= weights.astype(np.float32)
    #weigh the vertices with the weights
    x = sum(weights[0,j] * blendshapes[:,:,j] for j in range(1))
    original_mesh.vertices += x
    original_mesh.vertices *= 0.6
    original_mesh.export("SMPLOneDefNormalized.obj")

    from sdf.provider import SDFDatasetNoNormalization
    train_dataset0=SDFDatasetNoNormalization("SMPLOneDefNormalized.obj", size=1, num_samples=2**14)
    train_loader0 = torch.utils.data.DataLoader(train_dataset0, batch_size=1, shuffle=True)
    valid_dataset0 = SDFDatasetNoNormalization("SMPLOneDefNormalized.obj", size=1, num_samples=2**14) # just a dummy
    valid_loader0 = torch.utils.data.DataLoader(valid_dataset0, batch_size=1)
    from loss import mape_loss
    # criterion = mape_loss
    #use L1 loss
    criterion = torch.nn.L1Loss()
    opt.lr=1e-2
    
    optimizer = lambda model: torch.optim.Adam([
            # {'name': 'net0', 'params': model.preencoder.parameters(), 'weight_decay': 1e-6},
            {'name': 'encoding', 'params': model.encoder.parameters()},
            {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
        ], lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)
    scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    time_stamp=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    name="SMPLminiTesis/"+"SingleInstanceTrainingDrop2B"+"lr_"+str(opt.lr)+"_"+time_stamp+opt.path.split("/")[-1]

    trainer0 = TrainerWithFixedSubspace('ngp', model, workspace=name, optimizer=optimizer, criterion=criterion, ema_decay=0.95,
                           fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint='latest',
                            eval_interval=1,use_tensorboardX=True,mesh=newmesh,subspace=weights)

    trainer0.train(train_loader0, valid_loader0, 400)
    trainer0.save_mesh(os.path.join( 'SMPLminiTesis', 'reconstructedSMPLDeformed400Drop2B.ply'), 1024)