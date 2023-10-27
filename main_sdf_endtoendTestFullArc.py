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

    model = SDFNetworkWithSubspaceInput(encoding="hashgrid",num_layers=5,num_layers_pre=5, subspace_size=7)
    #load model from"trainedENDTOEND.pth"
    model.load_state_dict(torch.load("WorkSpaceFolder/trainedENDTOENDFullArcSDLOSS.pth"))
    model.cuda()

    from sdf.provider import SDFDatasetTestPreencoder,SDFDatasetTestPreencoderEndToEnd
    size_x = 300
    size_y = 300
    pz= 0.0
    factor=1.0
    grid_x, grid_y = torch.meshgrid(torch.linspace(-factor, factor, size_x), torch.linspace(factor, -factor, size_y))
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).to('cuda')

    # Expand the grid to have a batch dimension of size 1
    grid = grid.expand(1, -1, -1, -1)

    # Set the z-coordinate to pz
    grid = torch.cat([grid, torch.full((1, size_x, size_y, 1), pz, dtype=torch.float32, device='cuda')], dim=-1)
    grid = torch.squeeze(grid, 0)
    grid = grid.reshape(size_x * size_y, 3)
    grid_numpy = grid.detach().cpu().numpy()
    
    #calculate the sdf ground truth for the grid
    weights= np.array([[1,1,1,1,1,1,1]])/7
    dataset = SDFDatasetTestPreencoder("mode/deformed_cow", size=100, num_samples=8)
    newmesh=dataset.mesh.copy()
    x = sum(weights[0][j] * dataset.vs[j] for j in range(len(dataset.vs)))
    newmesh.vertices = x
    import pysdf
    sdf_fn = pysdf.SDF(newmesh.vertices, newmesh.faces)
    sdf_gt = -sdf_fn(grid_numpy)
    sdf_gt= sdf_gt.reshape(size_x, size_y)
    sdf_gt = sdf_gt.transpose()
    sdf_gt = sdf_gt.reshape(-1)

    #plot the sdf ground truth
    import matplotlib.pyplot as plt
    plt.imshow(sdf_gt.reshape(size_x, size_y))
    #add color bar
    plt.colorbar()
    plt.show()

    #calculate the sdf from the model
    subspace_vector = torch.from_numpy(weights.astype(np.float32)).cuda()
    subspace_vector = subspace_vector.repeat(grid.shape[0],1).cuda()
    grid_c = torch.from_numpy(grid_numpy.astype(np.float32)).cuda()
    sdf_m,points_canonical = model(grid_c,subspace_vector)
    sdf_m = sdf_m.cpu().detach().numpy()
    sdf_m = sdf_m.reshape(size_x, size_y)
    sdf_m = sdf_m.transpose()
    sdf_m = sdf_m.reshape(-1)

    #plot the sdf from the model
    plt.imshow(sdf_m.reshape(size_x, size_y))
    #add color bar
    plt.colorbar()
    plt.show()


    #plot the difference between the sdf from the model and the sdf ground truth
    plt.imshow((sdf_m-sdf_gt).reshape(size_x, size_y))
    #add color bar
    plt.colorbar()
    plt.show()


   
   




    