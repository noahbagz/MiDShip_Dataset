"""
This script reads in the Structural_elements.csv files and evaluates and formats the structures to be used in training data.

"""

import pandas as pd
import numpy as np

from tqdm import tqdm


from tools.Parametric_Structure_Eval import StructureEval as SE

def merge_data(file_path, file_root_list): 
    #This function merges multiple CSV files into a single DataFrame.
    random_error_idx = []
    for file_root_name in file_root_list:
        df = pd.read_csv(f"{file_path}/Random_Parametric_Designs_{file_root_name}_Updated.csv")
        if 'df_merged' in locals():
            df_merged = pd.concat([df_merged, df], ignore_index=True)
        else:
            df_merged = df
        
        # Load error indices if they exist
        try:
            error_idx = np.loadtxt(f"{file_path}/random_test_design_error_idx_{file_root_name}.csv", delimiter=',', dtype=int)
            try:
                len(error_idx)
            except:
                error_idx = np.array([error_idx])
            random_error_idx.extend(error_idx.tolist())
        except:
            print(f"No error indices found for {file_root_name}.")
    
    df_merged.to_csv(f"{file_path}/Random_Parametric_Designs_All.csv", index=False)
    #Save the error indices to a CSV file
    if random_error_idx:
        print(random_error_idx)

        np.savetxt(f"{file_path}/random_test_design_error_idx_All.csv", random_error_idx, delimiter=',', fmt='%d')
    



    

def format_training_data(file_path, file_root_name, num_samples):
    """
    Reads in the Structural_elements.csv file, evaluates the structural elements, and formats the data for training.
    
    Parameters:
    - file_path: Path to the directory containing the CSV files.
    - file_root_name: Root name of the CSV files to be processed.
    - num_samples: Number of samples to process.
    
    Returns:
    - df_formatted: DataFrame containing formatted training data.
    """
    
    # Read in the structural elements data
    columns = ['Steel_Volume',
               'Steel_Weight', 
               'Steel_LCG',
               'Steel_YCG',
               'Steel_VCG',
               'Cross_Sectional_Area',
               'z_Centroid_CX',
               'y_Centroid_CX',
               'I_11',
               'I_22']
  

    data = np.zeros((num_samples, 10))

    error_idx = np.loadtxt(f"{file_path}/{file_root_name}_error_idx_ALL.csv", delimiter=',', dtype=int)
    try:
        len(error_idx)
        print(f"Error indices found: {error_idx}")
    except:
        error_idx = [error_idx]
    

    for i in tqdm(range(num_samples)):
        #Check if the index is in the error list
        if i in error_idx:
            print(f"Skipping index {i} due to previous error.")
            data[i, :] = np.nan
            continue

        df = pd.read_csv(f"{file_path}/{file_root_name}_{i}_Structural_elements.csv")
        structE = SE(df)
        data[i, 0] = structE.Volume()
        data[i, 1] = structE.Structural_Weight()
        data[i, 2:5] = structE.Volume_Centroid()
        data[i, 5:10] = structE.Effective_Longitudinal_CrossSection_Properties()
    
    # Create a DataFrame with the formatted data
    #print(data)
    df_formatted = pd.DataFrame(data, columns=columns)

    # Save the formatted data to a CSV file
    df_formatted.to_csv(f"{file_path}/{file_root_name}_Structural_Properties.csv", index=False)


file_path = '/Users/noahbagazinski/Documents/MIT/Research/Ship_Structures/Dataset_Structures_V2_test'
#file_root_name = 'parametric_test_design'
file_root_name = 'random_test_design'
num_samples = 5010

merge_data(file_path, np.array(['0_1999','2000_3999','4000_5009'])) #999', '4000_4999', '5000_5006']))

format_training_data(file_path, file_root_name, num_samples)


    
   