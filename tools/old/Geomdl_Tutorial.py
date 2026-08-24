
import sys
sys.path.append("../")



from tools.HullParameterization import Hull_Parameterization as HP

import numpy as np
from geomdl import NURBS
#from geomdl.visualization import VisMPL
from geomdl import exchange
from scipy.interpolate import griddata
from geomdl import fitting, NURBS, utilities
import csv
import json

# Load a hull design and generate a point cloud
Vectors = np.loadtxt('data/Sample_Hull/Input_Vectors_SampleHulls.csv', delimiter=",", dtype=np.float64)

#Create one hull: 
Hull = HP(Vectors[0])

#Check Constriants:
constraints = Hull.input_Constraints()
cons = constraints > 0
print(sum(cons)) # should be zero

#make the .stl file of the hull:
strpath =  'data/Sample_Hull/Sample_Hull_Mesh_LowQuality' 

mesh = Hull.gen_stl(NUM_WL=19, PointsPerWL=82, bit_AddTransom = 0, bit_AddDeckLid = 0, bit_RefineBowAndStern = 1, namepath = strpath)
#mesh = Hull.gen_stl(NUM_WL=52, PointsPerWL=400, bit_AddTransom = 1, bit_AddDeckLid = 0, bit_RefineBowAndStern = 1, namepath = strpath)

nwl = 49
n = 201
Z = np.linspace(0,Hull.Dd,nwl-2)
Z = np.append(Z, 0.0001*Hull.Dd)
Z = np.append(Z, 0.001*Hull.Dd)
Z = np.sort(Z)
PC = Hull.gen_pointCloud(NUM_WL = nwl, PointsPerWL = n, bit_GridOrList = 0, Z = Z)

print(PC.shape)

save_path = 'data/Sample_Hull/Sample_Hull_Point_Cloud.csv'
np.savetxt(save_path, PC, delimiter=",")



'''
size_u = nwl
size_v = n
degree_u = 5
degree_v = 9

# Do global surface interpolation
surf = fitting.interpolate_surface(PC, size_u, size_v, degree_u, degree_v, centripetal=True)

Surf_Dict = {
    "poles": surf.ctrlpts2d,
    "weights": surf.weights,
    "umults": [surf.knotvector_u.count(k) for k in set(surf.knotvector_u)],
    "vmults": [surf.knotvector_v.count(k) for k in set(surf.knotvector_v)],
    "uknots": list(set(surf.knotvector_u)),
    "vknots": list(set(surf.knotvector_v)),
    "udegree": surf.degree_u,
    "vdegree": surf.degree_v,
    "uperiodic": False,
    "vperiodic": False
}

save_path = 'data/Sample_Hull/Sample_Hull_NURBS_'+str(degree_u) + "x" + str(degree_v)+'.json'

with open(save_path, 'w') as f:
    json.dump(Surf_Dict, f)

surf.delta = 0.005

exchange.export_obj(surf, "data/Sample_Hull/Sample_Hull_NURBS_"+str(degree_u) + "x" + str(degree_v)+".obj", vertex_normal=True, parametric_vertices=True,vertex_spacing = 1)

'''