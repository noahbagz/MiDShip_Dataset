#! python3

# env: /opt/anaconda3/bin/python

# r: numpy, pandas


import numpy as np
import pandas as pd
import importlib
import os

import rhino_StructGen as rhino_SG
import rhinoscriptsyntax as rs

import Parametric_Structure_V2
#from Parametric_Structure_V2 import Structure_3H as S3H

# Legacy combined parameter-and-structure entry point.  New repository runs
# should use Generate_Random_Parameters.py followed by
# Batched_Structure_Generation.py so each stage can be resumed independently.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
path = os.path.join(PROJECT_ROOT, 'MiDShip_Dataset', 'Random_Structures')

starting_idx = 5000 
num_samp = 1050
df = Parametric_Structure_V2.Structure_3H.gen_rnd_Sturctures(num_samples = num_samp)

df.to_csv(f'{path}/Random_Parametric_Designs_'+f'{starting_idx}_'+f'{starting_idx+num_samp-1}.csv', index = False)
Vec = df.to_numpy()
error_idx = []
updated_params = np.zeros((len(Vec), len(Vec[0])))

for i in range(0,len(Vec)):
    #print(i)
    #print('')
    try:
        hull = Parametric_Structure_V2.Structure_3H(Vec[i], path=path, id = f'random_test_design_{i+starting_idx}')
        hull.make_Structure()

        updated_params[i] = hull.params
    except:
        print('Error at idx ', i+starting_idx)
        error_idx.append(i+starting_idx)
        continue

    if i % 100 == 0:
        print('Completed ', i+starting_idx, ' of ', num_samp)
        #rs.Command('_ClearUndo', echo=False)
        if len(error_idx) > 0:
            error_idx_np = np.array(error_idx)
            np.savetxt(path+'/random_test_design_error_idx_'+ f'{starting_idx}_'+f'{starting_idx+num_samp-1}.csv', error_idx_np, delimiter=',')

        df_updated = pd.DataFrame(updated_params, columns=df.columns)
        df_updated.to_csv(path + '/Random_Parametric_Designs_'+f'{starting_idx}_'+f'{starting_idx+num_samp-1}_Updated.csv', index=False)

    del hull
    #importlib.reload(Parametric_Structure_V2)


if len(error_idx) > 0:
    error_idx_np = np.array(error_idx)
    np.savetxt(path+'/random_test_design_error_idx_'+ f'{starting_idx}_'+f'{starting_idx+num_samp-1}.csv', error_idx_np, delimiter=',')

df_updated = pd.DataFrame(updated_params, columns=df.columns)
df_updated.to_csv(path + '/Random_Parametric_Designs_'+f'{starting_idx}_'+f'{starting_idx+num_samp-1}_Updated.csv', index=False)
