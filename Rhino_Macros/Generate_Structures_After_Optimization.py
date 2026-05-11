#! python3

# env: /opt/anaconda3/bin/python

# r: numpy, pandas


import numpy as np
import pandas as pd

import rhino_StructGen as rhino_SG

from Parametric_Structure_V2 import Structure_3H as S3H

#opt_test_id = 'NSGA2_Opt_0'
opt_test_id = 'SGD_Opt_1'

path = '/Users/noahbagazinski/Documents/MIT/Research/Ship_Structures/Optimization_Results'



#df = pd.read_csv(path + f'/NSGA2_Optimization_{opt_test_id}_X_Results.csv')
df = pd.read_csv(path + f'/{opt_test_id}_X_Results.csv')

#print(df)
Vec = df.to_numpy()
#print(Vec[0])
Vec = S3H.clean_Struct_Params(Vec)
#print(Vec[0])
error_idx = []
updated_params = np.zeros((len(Vec), len(Vec[0])))
#print(Vec.shape)

for i in range(0,len(Vec)):
    '''
    try:
        hull = S3H(Vec[i], path=path, id = f'{opt_test_id}_design_{i}')
        hull.make_Structure()

        updated_params[i] = hull.params
    
    except:
        print(f'Error at Structure {i}')
        error_idx.append(i)
        continue
    '''
    hull = S3H(Vec[i], path=path, id = f'{opt_test_id}_design_{i}')
    hull.make_Structure()

    updated_params[i] = hull.params

if len(error_idx) > 0:
    df_error = pd.DataFrame(error_idx, columns=['error_idx'])
    df_error.to_csv(path+ opt_test_id+'_design_error_idx_ALL.csv', index=False)
    


df_updated = pd.DataFrame(updated_params, columns=df.columns)
df_updated.to_csv(path + f'/{opt_test_id}_X_Results_Updated.csv', index=False)

