#! python3

# env: /opt/anaconda3/bin/python

# r: numpy, pandas


import numpy as np
import pandas as pd
import importlib

import rhino_StructGen as rhino_SG
import rhinoscriptsyntax as rs

import Parametric_Structure_V2
#from Parametric_Structure_V2 import Structure_3H as S3H

path = '/Users/noahbagazinski/Documents/MIT/Research/Ship_Structures/Dataset_Structures_V2_1'

starting_idx = 4000
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

