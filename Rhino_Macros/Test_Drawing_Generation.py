
## This script Tests the Rhino2DDrawing class and its methods. It is designed to be used with Rhino 3D and requires the Rhino Python scripting environment.

import pandas as pd
import os
import numpy as np


import rhino_2D_Drawing as r2d
MAIN_DIR = '/Users/noahbagazinski/Documents/MIT/Research/Ship_Structures'
test_model_path = MAIN_DIR + '/Dataset_Structures_V2_1'
test_drawing_path = MAIN_DIR +'/Dataset_Structures_V2_1/Dataset_Drawings'


# if drawing path does not exist, create it
os.makedirs(test_drawing_path, exist_ok=True)

#Open All Design parameters and make list of idx where the design is not zeros


start_idx = 4900
#end_idx = start_idx + 100
end_idx = start_idx + 150

arr = np.arange(start_idx,end_idx)


for idx in arr:
    #for idx in [0]:
    test_file_name = 'random_test_design_' + str(idx)
    dwg_gen = r2d.Rhino2DDrawing(test_model_path, test_file_name, test_drawing_path)
    try: 
        dwg_gen.load_Data()
    except:
        #print('Skip ', idx)
        continue

    dwg_gen.extract_X_slice_Positions()
    
    for j in range(len(dwg_gen.df_Slices)):
        slice_name = dwg_gen.df_Slices['Slice_Name'][j]
        dwg_gen.create_Slice(slice_name)

        dwg_gen.create_Layout(slice_name) # This command does not work well at all.

        scale, orgin_s = dwg_gen.scale_Slice(slice_name)

        dwg_gen.create_Title_Block(slice_name, scale)
        dwg_gen.create_Info_Block(slice_name)

        dwg_gen.create_Bounding_Boxes(slice_name, scale, orgin_s)

        dwg_gen.export_DWG(slice_name)


    
    dwg_gen.close_Doc()



