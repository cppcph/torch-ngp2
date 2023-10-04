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
        from sdf.netowrk import SDFNetwork,SDFNetworkWithSubspaceInput

    model = SDFNetworkWithSubspaceInput(encoding="hashgrid",num_layers=5,num_layers_pre=6, subspace_size=7)
    #load model
    model.load_state_dict(torch.load("WorkSpaceFolder/oct37.pth"))
    print(model)
    model.to("cuda")
    #Generate point cloud of deformed cow
    

    from sdf.provider import SDFDatasetTestPreencoder

    dataset = SDFDatasetTestPreencoder("mode/deformed_cow", size=100, num_samples=2**14)
    
    vs=dataset.vs

    newmesh=dataset.mesh.copy()
    #generate 4 positive float number with sum of 1
    # weights = np.random.dirichlet(np.ones(6),size=1)
    # weights= np.array([[0,0,0,0,0,1]])
    weights= np.array([[0,0,0,0,1,0,0]])
    #make weights float
    weights= weights.astype(np.float32)
    #weigh the vertices with the weights
    x = sum([weights[0][i]*vs[i] for i in range(len(vs))])
    newmesh.vertices = x 
   

    #points with a fixed distance to the surface
    # Get the vertex normals
    vertex_normals = newmesh.vertex_normals

    # Offset each vertex along its normal
    distance = 0.0
    newmesh.vertices += vertex_normals * distance
    points_surface,triangle_id = trimesh.sample.sample_surface_even(newmesh, 5000)

    #create the subspace vector with the weights
    subspace_vector = torch.from_numpy(weights.astype(np.float32)).cuda()
    #make the vector as the size of the points
    subspace_vector = subspace_vector.repeat(points_surface.shape[0],1).cuda()
    #convert the points to tensor
    points_surface_c = torch.from_numpy(points_surface.astype(np.float32)).cuda()
    #map the points according to the preencoder
    a,points_canonical = model(points_surface_c,subspace_vector)
    #convert the points back to numpy
    points_canonical = points_canonical.cpu().detach().numpy()


    #plot points on the surface
    import matplotlib.pyplot as plt
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axes.set_xlim3d(left=-1, right=1) 
    ax.axes.set_ylim3d(bottom=-1, top=1) 
    ax.axes.set_zlim3d(bottom=-1, top=1) 

    #fix the perspective of the plot
    ax.view_init(elev=135, azim=-95)

    # Plot the points
    # ax.scatter(points_canonical[:, 0], points_canonical[:, 1], points_canonical[:, 2],marker='.')
    ax.scatter(points_surface[:, 0], points_surface[:, 1], points_surface[:, 2],marker='.')
    # Set labels for axes
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Show the plot
    plt.show()

     # Create a 3D plot
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.axes.set_xlim3d(left=-1, right=1) 
    ax1.axes.set_ylim3d(bottom=-1, top=1) 
    ax1.axes.set_zlim3d(bottom=-1, top=1) 

    #fix the perspective of the plot
    ax1.view_init(elev=135, azim=-95)

    # Plot the points
    ax1.scatter(points_canonical[:, 0], points_canonical[:, 1], points_canonical[:, 2],marker='.')
    # ax1.scatter(points_surface[:, 0], points_surface[:, 1], points_surface[:, 2],marker='.')
    # Set labels for axes
    ax1.set_xlabel('X Label')
    ax1.set_ylabel('Y Label')
    ax1.set_zlabel('Z Label')

    # Show the plot
    plt.show()

