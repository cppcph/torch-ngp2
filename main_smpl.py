import pickle
import trimesh
import numpy as np

with open('f.pkl', 'rb') as file:
    loaded_object = pickle.load(file,encoding='latin1')
print(loaded_object)

f= loaded_object['f']
vertex_o= loaded_object['v_template']
shapedirs=loaded_object['shapedirs']
weights = np.random.randn(10)
weights1 = np.zeros(10)
weights1[0]=20
d = np.array(shapedirs)
# vertex_o-=5*d
x = sum(weights[j] * d[:,:,j] for j in range(10))
x1 = sum(weights1[j] * d[:,:,j] for j in range(10))

v=vertex_o
vmin = v.min(0)
print(vmin)
vmax = v.max(0)
print(vmax)
v_center = (vmin + vmax) / 2
v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
v_normalized= (v - v_center[None, :]) * v_scale
v_deformed_normalized= v_normalized+x * v_scale
print(v_scale)

v1= vertex_o+x1
vmin = v1.min(0)
print(vmin)
vmax = v1.max(0)
print(vmax)
v_center = (vmin + vmax) / 2
v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2))
v_normalized1= (v1 - v_center[None, :]) * v_scale
print(v_scale)

#write vertex and face into obj file
def write_obj_file(vertices, faces, filename):
    with open(filename, 'w') as file:
        for vertex in vertices:
            file.write(f"v {' '.join(map(str, vertex))}\n")

        for face in faces:
            file.write(f"f {' '.join(map(str, face + 1))}\n")
# write_obj_file(v_normalized, f, 'SMPLTemplateNormalized.obj')
# write_obj_file(vertex_o, f, 'SMPLTemplate.obj')
# write_obj_file(vertex_o+x1, f, 'SMPLOneDef.obj')
# write_obj_file(v_deformed_normalized, f, 'SMPLDeformedNormalized.obj')
# write_obj_file(v_normalized1, f, 'SMPLTwoDefNormalized1.obj')
#write blendshape into numpy file
# np.save("blendshape.npy",d* v_scale)
# np.save("blendshapeOriginal.npy",d)

#write weights into numpy file
# np.save("weights.npy",weights)
# np.save("scale.npy",v_scale)    




