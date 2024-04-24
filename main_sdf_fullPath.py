import torch
import argparse

from sdf.utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4, help="initial learning rate")
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
    
    from sdf.provider import SDFDataset,SDFDatasetTestPreencoder
    train_dataset0=SDFDataset("data/cow.obj", size=100, num_samples=2**14)
    train_loader0 = torch.utils.data.DataLoader(train_dataset0, batch_size=1, shuffle=True)
    valid_dataset0 = SDFDataset("data/cow.obj", size=1, num_samples=2**14) # just a dummy
    valid_loader0 = torch.utils.data.DataLoader(valid_dataset0, batch_size=1)
    from loss import mape_loss
    # criterion = mape_loss
    #use L1 loss
    criterion = torch.nn.L1Loss()
    
    optimizer = lambda model: torch.optim.Adam([
            {'name': 'encoding', 'params': model.encoder.parameters()},
            {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
        ], lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)
    scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    time_stamp=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    name="WorkSpaceFolder/"+"OOsdf_Reference"+"lr_"+str(opt.lr)+"_"+time_stamp+opt.path.split("/")[-1]

    trainer0 = Trainer('ngp', model_sdf, workspace=name, optimizer=optimizer, criterion=criterion, ema_decay=0.95,
                           fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint='latest',
                            eval_interval=1,use_tensorboardX=True,mesh=train_dataset0.mesh)

    trainer0.train(train_loader0, valid_loader0, 100)
    #save model
    torch.save(model_sdf.state_dict(), name+".pth")

    
    from sdf.provider import SDFDatasetTestPreencoder,SDFDatasetTestPreencoderEndToEnd
    ##second part
    # newmesh = trimesh.load("medio_cow_random_deformed_norm.obj", force='mesh')
    newmesh = trimesh.load("medio_cow_sinus_deformed_norm_high_freq.obj", force='mesh')
    dataset = SDFDatasetTestPreencoder("mode/deformed_cow", size=100, num_samples=8)
    v=newmesh.vertices.reshape(-1,1)
    #calculate the weights based on the mode
    #make each vertex matrix a vector and concatenate them together
    for i in range(len(dataset.vs)):
        vertices_vector= dataset.vs[i].reshape(-1,1)
        if i==0:
            vertices_matrix=vertices_vector
        else:
            vertices_matrix=np.concatenate((vertices_matrix,vertices_vector),axis=1)
    
    from scipy.optimize import minimize
    def objective_function(weights, B, v):
        projected_vector = np.dot(B,weights)
        #make the projected vector a column vector
        projected_vector=projected_vector.reshape(-1,1)
        return np.linalg.norm(projected_vector - v)
    
    constraint = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
              {'type': 'ineq', 'fun': lambda weights: weights},
              {'type': 'ineq', 'fun': lambda weights: 1 - weights})
    initial_weights = np.array([[1,1,1,1,1,1,1]])/7
    result = minimize(objective_function, initial_weights, args=(vertices_matrix, v), constraints=constraint)

    # Extract the optimized weights
    optimized_weights = result.x

    # Calculate the projected vector
    projected_vector = np.dot(vertices_matrix,optimized_weights)

    #calculate error    
    error = np.linalg.norm(projected_vector.reshape(-1,1) - v)
    print(error)

    model = SDFNetworkWithSubspaceInputOnlyForPreencoder(encoding="hashgrid",num_layers=5,num_layers_pre=5, subspace_size=7)

    #match the corresponding parameters of the parts of model and model sdf
    transfer_encoder_weights(model_sdf,model)
    transfer_backbone_weights(model_sdf,model)


    #load model
    # model.load_state_dict(torch.load("WorkSpaceFolder/original.pth"))
    # print(model)
    

    train_dataset = SDFDatasetTestPreencoderEndToEnd("mode/deformed_cow", size=10, num_samples=2**10)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    valid_dataset = SDFDatasetTestPreencoderEndToEnd("mode/deformed_cow", size=1, num_samples=2**10) # just a dummy
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

    criterion = torch.nn.L1Loss()

    optimizer = lambda model: torch.optim.Adam([
        {'name': 'net0', 'params': model.preencoder.parameters(), 'weight_decay': 1e-6},
        # {'name': 'encoding', 'params': model.encoder.parameters()},
        # {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
        
    ], lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)

    scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    time_stamp=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    name="WorkSpaceFolder/"+"BInputOnlyForPreencoderTrainedOnlyWithPreencoder_"+"lr_"+str(opt.lr)+"_"+time_stamp+opt.path.split("/")[-1]

    

    trainer = TrainerTestPreencoder('ngp', model, workspace=name, optimizer=optimizer, criterion=criterion, ema_decay=0.95,
                        fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint='latest',
                        eval_interval=1,use_tensorboardX=True,mesh=newmesh,subspace=optimized_weights)

    trainer.train(train_loader, valid_loader, 200)


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
    name="WorkSpaceFolder/"+"BRetrainForFullParameter_"+"lr_"+str(opt.lr)+"_"+time_stamp+opt.path.split("/")[-1]

    trainer = TrainerTestPreencoder('ngp', model, workspace=name, optimizer=optimizer, criterion=criterion, ema_decay=0.95,
                        fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint='latest',
                        eval_interval=1,use_tensorboardX=True,mesh=newmesh,subspace=optimized_weights)

    trainer.train(train_loader, valid_loader, 200)

    layer_to_save= {'preencoder':model.preencoder.state_dict()}
    torch.save(model.state_dict(), name+".pth")
    torch.save(layer_to_save, name+"pre.pth")

    from sdf.provider import SDFDatasetNoNormalization
    train_dataset0=SDFDatasetNoNormalization("medio_cow_sinus_deformed_norm_high_freq.obj", size=1, num_samples=2**14)
    train_loader0 = torch.utils.data.DataLoader(train_dataset0, batch_size=1, shuffle=True)
    valid_dataset0 = SDFDatasetNoNormalization("medio_cow_sinus_deformed_norm_high_freq.obj", size=1, num_samples=2**14) # just a dummy
    valid_loader0 = torch.utils.data.DataLoader(valid_dataset0, batch_size=1)
    from loss import mape_loss
    # criterion = mape_loss
    #use L1 loss
    criterion = torch.nn.L1Loss()
    opt.lr=1e-4
    
    optimizer = lambda model: torch.optim.Adam([
            # {'name': 'net0', 'params': model.preencoder.parameters(), 'weight_decay': 1e-6},
            {'name': 'encoding', 'params': model.encoder.parameters()},
            {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
        ], lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)
    scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    time_stamp=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    name="ForTensorboard5/"+"warmstart_OnlyNGP_"+"lr_"+str(opt.lr)+"_"+time_stamp+opt.path.split("/")[-1]

    trainer0 = TrainerWithFixedSubspace('ngp', model, workspace=name, optimizer=optimizer, criterion=criterion, ema_decay=0.95,
                           fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint='latest',
                            eval_interval=1,use_tensorboardX=True,mesh=newmesh,subspace=optimized_weights)

    trainer0.train(train_loader0, valid_loader0, 100)