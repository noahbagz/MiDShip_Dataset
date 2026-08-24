"""
This script reads in the Structural_elements.csv file, evaluates the Geometric properties of the structure 
and evaluates constraints based on American Bureau of Shipping (ABS) rules.

====================================================================================================================================

NOTE: These ABS rule-based constraints are not a full evaluation of the structure per ABS guidelines. This set of constraints was
formulated based on a subset of ABS rules for evaluation purposes.

The constraints are formulated from the July 2024 ABS MVR. 

In general, if a rule provides multiple options for constant values, the most conservative option is typically chosen for the evaluation.

"""

import pandas as pd
import numpy as np
import ast


'''
        Original: 
        z_dir = 1, rot = 0 on +x axis. 11 axis is x, 22 axis is y
                   |+z
                   |
        +x(11)_____|     
                   /
                  /
                 /
                +y(22)

         x_dir = 1, rot = 0 on +z axis. 11 axis is z, 22 axis is y
                    |+x
                    |
         +z(11)_____|
                    /
                   /
                  /
                +y(22) 

        y_dir = 1, rot = 0 on +y axis   11 axis is x, 22 axis is +z
                    |+y
                    |
         +x(11)_____|
                    /
                   /
                  /
                 +z(22)
'''


class StructureEval:
    def __init__(self, df):
        self.Structural_Elements = df
        #First creat an eval class: 
        #self.rho_steel = 7850  # kg/m^3 mild steel
        self.rho_steel = 7.85 #tons/m^3 mild steel
        self.E_steel = 210e9  # Pa
        self.sigma_yield_steel = 250e6  # Pa
        self.L3h = self.Structural_Elements.loc[self.Structural_Elements['Object_ID'] == 'Side_Shell', 'L1'].values[0]  # m
        self.B = 2.0*self.Structural_Elements.loc[self.Structural_Elements['Object_ID'] == 'Side_Shell', 'y_loc'].values[0]  # m
        self.D = self.Structural_Elements.loc[self.Structural_Elements['Object_ID'] == 'Side_Shell', 'z_loc'].values[0] + self.Structural_Elements.loc[self.Structural_Elements['Object_ID'] == 'Side_Shell', 'L2'].values[0] # m

        # We need to determine the vessel class: If there is a 'Bracket_4_sides' in the Type column, then its a container ship. 
        # if there is a 3 sided bracket and hopper pannels, then its a bulk carrier. 
        # if there is a 3 sided bracket and no hopper pannels, then its a tanker. 
        # we will use this information to determine which ABS rules to apply for the transverse structure constraints. 

        if self.Structural_Elements[self.Structural_Elements['Type'] == 'Bracket_4_sides'].shape[0] > 0:
            self.vessel_type = 'Container_Ship'

        elif self.Structural_Elements[self.Structural_Elements['Type'] == 'Bracket_3_sides'].shape[0] > 0 and self.Structural_Elements[self.Structural_Elements['Class'] == 'Hopper Pannel'].shape[0] > 0:
            self.vessel_type = 'Bulk_Carrier'
        else: 
            self.vessel_type = 'Tanker'


    def Volume(self):
        """
        Calculate the group volume
        """
        vol = 2.0*self.Structural_Elements['volume_pannel'].sum()
        return vol # in m^3
    def Structural_Weight(self):
        """
        Calculate the group weight of steel in tons
        """
        weight = 2.0 * self.Structural_Elements['volume_pannel'].sum() * self.rho_steel
        unit_weight = weight/(self.L3h*self.B*self.D) # in tons/m^3
        return weight, unit_weight #in tons
    
    def Volume_Centroid(self):
        """
        Calculate the group volume centroid in m
        """
        V = self.Volume()
        x_centroid = (2.0 * self.Structural_Elements['volume_pannel'] * self.Structural_Elements['x_centroid_pannel']).sum() / V
        y_centroid = 0.0  # Assuming symmetry in y-direction
        z_centroid = (2.0 * self.Structural_Elements['volume_pannel'] * self.Structural_Elements['z_centroid_pannel']).sum() / V
        
        return np.array([x_centroid, y_centroid, z_centroid])
    
    def Effective_Longitudinal_CrossSection_Properties(self):
        """
        Calculate the group cross sectoinal area in m^2 - This removes sectional area from longitudinal stiffeners 
        that are not the full length of the 3 hold structure, and It only considers the width of the deck that is not interupted by hatch opentings
        """
        #Get length of the structure - L1 of 'Object_ID' == 'Side_Shell'
        L = self.Structural_Elements.loc[self.Structural_Elements['Object_ID'] == 'Side_Shell', 'L1'].values[0] # m
        df = self.Structural_Elements[(self.Structural_Elements['x_dir'] == 1) & (self.Structural_Elements['L1'] == L)]


        #Find Top Deck and Check for hatch openings
        df_deck = self.Structural_Elements[self.Structural_Elements['Class'] == 'Deck']
        #Get depth of the deck from the bottom of the ship to the deck - this is the z_loc of the deck element with the largest z_loc value
        depth = df_deck['z_loc'].max() #m

        for i in range(len(df_deck)):
            if not pd.isna(df_deck.iloc[i]['Lightening_Holes']):  # Check if there are lightening holes
                
                #If there is a hatch opening, remove the area of the hatch from the deck
                LH = ast.literal_eval(df_deck.iloc[i]['Lightening_Holes'])
                hatch_y = LH[0]['h']/2.0 # Hatch height in mm is full width of ship, we want the half width
                effective_L2 = df_deck.iloc[i]['L2'] - hatch_y/1000.0  # Convert to meters
                effective_y_centroid = df_deck.iloc[i]['L2'] - effective_L2/2.0  # Centroid of the effective area

                effective_I_11 = 1/12*df_deck.iloc[i]['Thickness']*effective_L2**3
                effective_I_22 = 1/12*(df_deck.iloc[i]['Thickness']**3)*effective_L2
                effective_A_cx = df_deck.iloc[i]['Thickness'] * effective_L2  # Area of the effective deck


                #Update these values in the original dataframe, df
                deck_id = df_deck.iloc[i]['Object_ID']
                df.loc[df['Object_ID'] == deck_id, 'L2'] = effective_L2
                #Update I_11 and I_22 for the deck

                df.loc[df['Object_ID'] == deck_id, 'centroid_cx_22'] = effective_y_centroid
                df.loc[df['Object_ID'] == deck_id, 'I_11'] = effective_I_11
                df.loc[df['Object_ID'] == deck_id, 'I_22'] = effective_I_22
                df.loc[df['Object_ID'] == deck_id, 'area_cx'] = effective_A_cx


        A_Cx = 2.0 * df['area_cx'].sum()  # m^2 (we want mirror of the structure too)
        z_centroid = 2.0*(df['area_cx'] * df['centroid_cx_11']).sum() / A_Cx  # m
        y_centroid = 0.0  # Assuming symmetry in y-direction
        I_11 = 2.0 * (df['I_11'].sum() + (df['centroid_cx_22']**2.0 * df['area_cx']).sum())  # m^4 (we want mirror of the structure too) -> Stiffness should be measured around y= 0, the centerline of the ship
        I_22 = 2.0 * (df['I_22'].sum() + ((df['centroid_cx_11']- z_centroid)**2.0 * df['area_cx']).sum())  # m^4 (we want mirror of the structure too) -> Stiffness should be measured around z = vertical centroid)

        #Find the max distance for stress: z_centroid or depth-z_centroid, we will use this to calculate the maximum bending moment the structure can withstand based on the yield strength of the steel
        z_max = max(z_centroid, depth-z_centroid)

        max_BM = 0.5*self.sigma_yield_steel*I_22/z_max # in Nm, this is the maximum bending moment the structure can withstand based on the yield strength of the steel and the distance from the neutral axis to the outermost fiber (z_max)

        return np.array([A_Cx, z_centroid, y_centroid, I_11, I_22, max_BM])  # m^2, m, m, m^4, m^4
    

    def Calculate_Transverse_Struct_Constraints(self):
        """
        This function calculates the ABS requirements related to the transverse structure of the ship, 
        which includes the side shell and the deck. It returns the minimum thickness and the required section modulus for the transverse structure to meet ABS requirements.

        Here are the following constraint_Calculations that are implemented in this function:

        0) Min Double Bottom Height 
            a) ABS 3-2-4/3.1 
            b) ABS 3-2-4/1.2

        1) Min Bottom Floor Thickness
            ABS Part 3, Ch. 2, Sec. 4/5.1  - Min. Bottom Floor thickness

        2) Min Bottom Girder Thickness
            ABS Part 3, Ch. 2, Sec. 4/5.1  - Min. Bottom Girder thickness
        
        3) Max. Bottom Floor Spacing 
            ABS Part 3, Ch. 2, Sec. 4/5.1

        4)Inner Bottom Deck Thickness
            a) ABS Part 3, Ch. 2, Sec. 4-9.2  - Min. Inner Bottom Deck thickness for Longitudinal Framed Ship 
            b) ABS Part 3, Ch. 2, Sec. 4/9.5  - Deck thickness for Grabs
            c) ABS Part 3, Ch. 2, Sec 10/3.1 - Plate thickness for Deep Tanks
            d) ABS Part 3, Ch. 2, Sec. 3 / 5.17 - Wheel loading

        5) Hopper Thickness
            a) ABS Part 3, Ch. 2, Sec. 4-9.2  - Min. Inner Bottom Deck thickness for Longitudinal Framed Ship 
            b) ABS Part 3, Ch. 2, Sec. 4/9.5  - Deck thickness for Grabs
            c) ABS Part 3, Ch. 2, Sec 10/3.1 - Plate thickness for Deep Tanks
            d) ABS Part 3, Ch. 2, Sec. 3 / 5.17 - Wheel loading

        6) Min. Bottom Stiffener SM
            ABS Part 3, Ch. 2, Sec 10/3.3 Bottom Structure and Inner Bottom Stiffener and Frame SM 

        7) Min. Bottom Frame SM 
            ABS Part 3, Ch. 2, Sec 10/3.3 Bottom Structure and Inner Bottom Stiffener and Frame SM 
        
        8) Min. Hopper Stiffener SM
            ABS Part 3, Ch. 2, Sec 10/3.3 Bottom Structure and Inner Bottom Stiffener and Frame SM 

        9) Min. Side Frame SM
            ABS Part 3, Ch. 2, Sec. 5/3.1.1
        
        10) Min. Webframe SM 
            ABS Part 3, Ch. 2, Sec. 6/3.1

        11) Min Webframe Depth
            ABS Part 3, Ch. 2, Sec. 6/3.5

        12)  Min Webframe Thickness
            ABS Part 3, Ch. 2, Sec. 6/3.5 
             
        13) Min. Side Stringer SM
            a) ABS Part 3, Ch. 2, Sec. 6/5.1 
            b) ABS 3-2-10/3.3

        14) Min. Side Stringer Depth
            ABS Part 3, Ch. 2, Sec. 6/5.3

        15) Min. Trans. Bulkhead Thickness
            a) ABS Part 3, Ch. 2, Sec. 9/5.1
            b) ABS 3-2-10/3.1 - Deep Tank Bulkhead Thicknes

        16) Min. Trans. Bulkhead Stiffener SM
            a) ABS Part 3, Ch. 2, Sec. 9/5.3
            b) ABS 3-2-10/3.3 - Deep Tank Bulkhead Stiffener SM

        17) Min. Deck Frame SM
            ABS Part 3, Ch. 2, Sec. 7/3.1

        18) Min. Deck Thickness
            a) ABS Part 3, Ch. 2, Sec. 3/5.1
            b) ABS Part 3, Ch. 2, Sec. 4 / 9.5
            c) ABS Part 3, Ch. 2, Sec. 3 / 5.17 - Wheel loading
        
        19) Min. Deck Beam SM
            ABS Part 3, Ch. 2, Sec. 8/5.3 

        20) Min. Deck Beam Depth 
            ABS Part 3, Ch. 2, Sec. 8/5.7 and 5.13 

        21) Min. Deck Beam Thickness
            ABS Part 3, Ch. 2, Sec. 8/5.7 

        22) Min. Side Shell Thickness
            a) ABS Part 3, Ch. 2, Sec. 2/3.9.3
            b) 3-2-2/5.1

        23) Min.  Bottom Shell Thickness For Now, Let's Assume L = 250m for the paper. Acknowledge that length plat
            ABS Part 3, Ch. 2, Sec. 2/3.15.1
        """

        # First we need to calculate some fundamental properties of the structure from the Structural_Elements dataframe that we will need for the constraint calculations.
        # Calculate Spacings: 
        spacings, dims = self.Calculate_Spacings_and_Sizes()

        # Second, Let's calculate the constraints threshold values based on the ABS rules. We will return these values in a dictionary for easy access and comparison to the actual structural properties of the ship.
        constraints = self.Calc_ABS_Transverse_Struct_Constraints(spacings,dims)


        #Third we will calculate the actual structural properties of the ship. 
        struct_evals = self.Calc_Structural_Properties(spacings, dims)

        #Fourth we will return the constraints and dimensional properties of the ship for comparison.


        return constraints, struct_evals


    def Calculate_Spacings_and_Sizes(self):
        """
        This function calculates the spacings between each set of stiffener types
        all spacings are in meters.

        0) Trans Bulkheads
        1) Webframes/Floors/Deck Beams
        2) Frames
        3) Deck Girders
        4) Deck Stringers
        5) Bottom Girders
        6) Bottom Stringers
        7) Side Stringers
        8) Horizontal Stiffeners on Transverse Bulkheads
        9) Vertical Stiffeners on Transverse Bulkheads
        10) Double bottom height
        11) Exposed Deck Height - for abs calcs
        12) Inner bottom to deck height - for abs calcs
        """
        spacings = {}
        dims = {}
        # Transverse bulkhead spacing and dims
        df = self.Structural_Elements[self.Structural_Elements['Class']=='Transverse_Bulkhead']
        x = df.sort_values(by='x_loc')['x_loc'].values
        spacings['Trans_Bulkheads'] = x[1] - x[0] 
        dims['Trans_Bulkhead_Thickness'] = df['Thickness'].values[0]*1000.0 #mm

        #Floor thickness
        dims['Bottom_Floor_Thickness'] = self.Structural_Elements[self.Structural_Elements['Class']=='Transverse_Floor']['Thickness'].values[0]*1000.0 #mm

        #Bottom Girder thickness
        dims['Bottom_Girder_Thickness'] = self.Structural_Elements[self.Structural_Elements['Class']=='Bottom_Longitudinal_Girder']['Thickness'].values[0]*1000.0 #mm
        


        #Bottom stiffener
        dims['Bottom_Stiffener_Thickness'] = self.Structural_Elements[self.Structural_Elements['Class']=='Bottom_Longitudinal_Stiffener']['Thickness'].values[0]*1000.0 #mm
        dims['Bottom_Stiffener_Depth'] = self.Structural_Elements[self.Structural_Elements['Class']=='Bottom_Longitudinal_Stiffener']['L2'].values[0] #m
        # Now get flange width if there is a flange on the bottom stiffener, otherwise assume flange width is 0
        obj_id = self.Structural_Elements[self.Structural_Elements['Class']=='Bottom_Longitudinal_Stiffener']['Object_ID'].values[0]
        if obj_id+'_Flange' in self.Structural_Elements['Object_ID'].values:
            dims['Bottom_Stiffener_flange_width'] = self.Structural_Elements[self.Structural_Elements['Object_ID']==obj_id+'_Flange']['L2'].values[0] #m
        else:
            dims['Bottom_Stiffener_flange_width'] = 0.0 # If there are no flanges, we can assume the flange width is 0

        # Bottom transverse stiffener thickness and depth
        dims['Bottom_Transverse_Stiffener_Thickness'] = self.Structural_Elements[self.Structural_Elements['Class']=='Transverse_Bottom_Frame']['Thickness'].values[0]*1000.0 #mm
        dims['Bottom_Transverse_Stiffener_Depth'] = self.Structural_Elements[self.Structural_Elements['Class']=='Transverse_Bottom_Frame']['L2'].values[0] #m

        # Now get flange width if there is a flange on the bottom transverse stiffener, otherwise assume flange width is 0
        obj_id = self.Structural_Elements[self.Structural_Elements['Class']=='Transverse_Bottom_Frame']['Object_ID'].values[0]
        if obj_id+'_Flange' in self.Structural_Elements['Object_ID'].values:
            dims['Bottom_Transverse_Stiffener_flange_width'] = self.Structural_Elements[self.Structural_Elements['Object_ID']==obj_id+'_Flange']['L2'].values[0] #m
        else:
            dims['Bottom_Transverse_Stiffener_flange_width'] = 0.0 # If there are no flanges, we can assume the flange width is 0

        #webframe spacing and dims
        df = self.Structural_Elements[self.Structural_Elements['Class']=='Vertical_Web_Frame']
        x = df.sort_values(by='x_loc')['x_loc'].values
        spacings['Webframes'] = x[1] - x[0]
        count_WF = len(df)
        dims['Webframe_Depth'] = df['L2'].values[0] #m
        dims['Webframe_thickness'] = df['Thickness'].values[0]*1000.0 #mm
        obj_id = df['Object_ID'].values[0]

        #Now check for flange on webframe
        if obj_id+'_Flange' in self.Structural_Elements['Object_ID'].values:
            dims['Webframe_flange_width'] = self.Structural_Elements[self.Structural_Elements['Object_ID']==obj_id+'_Flange']['L2'].values[0] #m
        else: 
            dims['Webframe_flange_width'] = 0.0 # If there are no flanges, we can assume the flange width is 0

        #Frame spacing and dims
        df = self.Structural_Elements[self.Structural_Elements['Class']=='Vertical_Side_Frame']
        x = df.sort_values(by='x_loc')['x_loc'].values
        spacings['Frames'] = x[1] - x[0]
        dims['N_frames'] = int(len(df)/(3*(count_WF/3.0 + 1)) +0.5) # number of frames per hold, assuming 3 holds and that the number of webframes is the same in each hold
        dims['Frame_Depth'] = df['L2'].values[0] #m
        dims['Frame_thickness'] = df['Thickness'].values[0]*1000.0 #mm
        obj_id = df['Object_ID'].values[0]

        if obj_id+'_Flange' in self.Structural_Elements['Object_ID'].values:
            dims['Frame_flange_width'] = self.Structural_Elements[self.Structural_Elements['Object_ID']==obj_id+'_Flange']['L2'].values[0] #m
        else:
            dims['Frame_flange_width'] = 0.0 # If there are no flanges, we can assume the flange width is 0


        # Deck Girders and Deck Stringers are optional for some ship types. we need to check if they are present before calculating spacings
        if 'Deck_Longitudinal_Girder' in self.Structural_Elements['Class'].values:
            df = self.Structural_Elements[self.Structural_Elements['Class']=='Deck_Longitudinal_Girder']
            x = df.sort_values(by='y_loc')['y_loc'].values
            #need unique values of x
            x = np.unique(x)
            spacings['Deck_Girders'] = x[1] - x[0]
            
            
            df = self.Structural_Elements[self.Structural_Elements['Class']=='Deck_Longitudinal_Stiffener']
            x = df.sort_values(by='y_loc')['y_loc'].values
            x = np.unique(x)
            spacings['Deck_Stringers'] = x[1] - x[0]
        


        if 'Deck_Transvere_Beam' in self.Structural_Elements['Class'].values:
            df = self.Structural_Elements[self.Structural_Elements['Class']=='Deck_Transvere_Beam']
            dims['Deck_Beam_Depth'] = df['L2'].values[0] #m
            dims['Deck_Beam_thickness'] = df['Thickness'].values[0]*1000.0 #mm
            obj_id = df['Object_ID'].values[0]
            if obj_id+'_Flange' in self.Structural_Elements['Object_ID'].values:
                dims['Deck_Beam_flange_width'] = self.Structural_Elements[self.Structural_Elements['Object_ID']==obj_id+'_Flange']['L2'].values[0] #m
            else:
                dims['Deck_Beam_flange_width'] = 0.0 # If there are no flanges, we can assume the flange width is 0

            df = self.Structural_Elements[self.Structural_Elements['Class']=='Deck_Transverse_Stiffener,']
            dims['Deck_Frame_Depth'] = df['L2'].values[0] #m
            dims['Deck_frame_thickness'] = df['Thickness'].values[0]*1000.0 #mm
            obj_id = df['Object_ID'].values[0]

            if obj_id+'_Flange' in self.Structural_Elements['Object_ID'].values:
                dims['Deck_frame_flange_width'] = self.Structural_Elements[self.Structural_Elements['Object_ID']==obj_id+'_Flange']['L2'].values[0] #m
            else:
                dims['Deck_frame_flange_width'] = 0.0 # If there are no flanges, we can assume the flange width is 0
        
        df = self.Structural_Elements[self.Structural_Elements['Class']=='Bottom_Longitudinal_Girder']
        x = df.sort_values(by='y_loc')['y_loc'].values
        spacings['Bottom_Girders'] = x[1] - x[0]

        df = self.Structural_Elements[self.Structural_Elements['Class']=='Bottom_Longitudinal_Stiffener']
        x = df.sort_values(by='y_loc')['y_loc'].values
        spacings['Bottom_Stringers'] = x[1] - x[0]

        df = self.Structural_Elements[self.Structural_Elements['Class']=='Side_Shell_Longitudinal_Stiffener']
        x = df.sort_values(by='z_loc')['z_loc'].values
        spacings['Side_Stringers'] = x[1] - x[0]
        dims['Side_Stringer_Depth'] = df['L2'].values[0] #m
        dims['Side_Stringer_thickness'] = df['Thickness'].values[0]*1000.0 #mm
        
        obj_id = df['Object_ID'].values[0]
        if obj_id+'_Flange' in self.Structural_Elements['Object_ID'].values:
            dims['Side_Stringer_flange_width'] = self.Structural_Elements[self.Structural_Elements['Object_ID']==obj_id+'_Flange']['L2'].values[0] #m
        else:
            dims['Side_Stringer_flange_width'] = 0.0 # If there are no flanges, we can assume the flange width is 0


        df = self.Structural_Elements[self.Structural_Elements['Class']=='Transverse_Bulkhead_Transverse_Stiffener']
        x = df.sort_values(by='z_loc')['z_loc'].values
        x = np.unique(x)
        spacings['Trans_Bulkhead_Horizontal_Stiffeners'] = x[1] - x[0] #there are three bulkheads
        dims['Trans_Bulkhead_Horizontal_Stiffener_thickness'] = df['Thickness'].values[0]*1000.0 #mm
        dims['Trans_Bulkhead_Horizontal_Stiffener_Depth'] = df['L2'].values[0] #m
        obj_id = df['Object_ID'].values[0]

        if obj_id+'_Flange' in self.Structural_Elements['Object_ID'].values:
            dims['Trans_Bulkhead_Horizontal_Stiffener_flange_width'] = self.Structural_Elements[self.Structural_Elements['Object_ID']==obj_id+'_Flange']['L2'].values[0] #m
        else:
            dims['Trans_Bulkhead_Horizontal_Stiffener_flange_width'] = 0.0 # If there are no flanges, we can assume the flange width is 0
        


        df = self.Structural_Elements[self.Structural_Elements['Class']=='Transverse_Bulkhead_Vertical_Stiffener']
        x = df.sort_values(by='y_loc')['y_loc'].values
        x = np.unique(x)
        spacings['Trans_Bulkhead_Vertical_Stiffeners'] = x[1] - x[0] #there are three bulkheads
        dims['Trans_Bulkhead_Vertical_Stiffener_thickness'] = df['Thickness'].values[0]*1000.0 #mm
        dims['Trans_Bulkhead_Vertical_Stiffener_Depth'] = df['L2'].values[0] #m
        obj_id = df['Object_ID'].values[0]
        if obj_id+'_Flange' in self.Structural_Elements['Object_ID'].values:
            dims['Trans_Bulkhead_Vertical_Stiffener_flange_width'] = self.Structural_Elements[self.Structural_Elements['Object_ID']==obj_id+'_Flange']['L2'].values[0] #m
        else:
            dims['Trans_Bulkhead_Vertical_Stiffener_flange_width'] = 0.0 # If there are no flanges, we can assume the flange width is 0

        df = self.Structural_Elements[self.Structural_Elements['Class']=='Deck']
        z = df.sort_values(by='z_loc')['z_loc'].values
        dims['Double_Bottom_Height'] = min(z) # The height of the double bottom is the distance from the bottom of the ship to the deck, which is the minimum z value of the deck elements

        dims['h_deck'] = 3.66 #m exposed deck height eith no load per ABS Part 3 Ch 2 Sec 5 / 3.1 
        dims['h_cargo'] = z[1] - z[0] #m this is a likely over calculation
        dims['Deck_Thickness'] = df['Thickness'].values[0]*1000.0 #mm
        dims['Inner_Bottom_Deck_Thickness'] = df['Thickness'].values[1]*1000.0 #mm

        dims['draft'] = z[1]*0.6667 #Assume draft is 1/2
        dims['depth'] = z[1] #Assume depth is the height of the deck above the bottom of the ship
        dims['beam'] = 2.0*self.Structural_Elements[self.Structural_Elements['Class']=='Side_Shell']['y_loc'].values[0] #Assume beam is twice the distance from the centerline to the side shell
        dims['length'] = 250.0 #m assume length of the ship is 250m for now. 
        
        # We need to extract the bracket heights
        Z_FW = self.Structural_Elements[self.Structural_Elements['Class'] == 'Bracket_Floor_Web']['L2'].values[0] #m
        #Now check for existance of "Bracket_Deck_Web" class before trying to access it
        if 'Bracket_Deck_Web' in self.Structural_Elements['Class'].values:
            Z_DW = self.Structural_Elements[self.Structural_Elements['Class'] == 'Bracket_Deck_Web']['L2'].values[0] #m
        else:
            Z_DW = 0.0 # If there are no deck brackets, we can assume the height of the frames is just the distance from the bottom of the double bottom to the top deck

        dims['l_frame'] = dims['depth'] - dims['Double_Bottom_Height'] - Z_FW - Z_DW #m
        dims['h_frame'] = abs(dims['Double_Bottom_Height'] + dims['l_frame']/2.0 - dims['draft']) #m
        
        dims['Bottom_Shell_Thickness'] = self.Structural_Elements[self.Structural_Elements['Class']=='Bottom_Shell']['Thickness'].values[0]*1000.0 #mm
        dims['Side_Shell_Thickness'] = self.Structural_Elements[self.Structural_Elements['Class']=='Side_Shell']['Thickness'].values[0]*1000.0 #mm

        if self.vessel_type != 'Tanker':  # Tankers do not have hopper pannels or stiffeners, so we can skip these calculations for tankers
            dims['Hopper_Plate_Thickness'] = self.Structural_Elements[self.Structural_Elements['Class']=='Hopper Pannel']['Thickness'].values[0]*1000.0 #mm
            dims['Hopper_Stiffener_Thickness'] = self.Structural_Elements[self.Structural_Elements['Class']=='Hopper_Pannel_Longitudinal_Stiffener']['Thickness'].values[0]*1000.0 #mm
            dims['Hopper_Stiffener_Depth'] = self.Structural_Elements[self.Structural_Elements['Class']=='Hopper_Pannel_Longitudinal_Stiffener']['L2'].values[0] #m
            obj_id = self.Structural_Elements[self.Structural_Elements['Class']=='Hopper_Pannel_Longitudinal_Stiffener']['Object_ID'].values[0]
            if obj_id+'_Flange' in self.Structural_Elements['Object_ID'].values:
                dims['Hopper_Stiffener_flange_width'] = self.Structural_Elements[self.Structural_Elements['Object_ID']==obj_id+'_Flange']['L2'].values[0] #m

            if self.vessel_type == 'Bulk_Carrier':
                # Select two adjacent hopper stiffeners and calculate the spacing between them. We can assume the hopper stiffeners are evenly spaced, so we can just take the difference in the y_loc values of two adjacent stiffeners to get the spacing.
                df = self.Structural_Elements[self.Structural_Elements['Class']=='Hopper_Pannel_Longitudinal_Stiffener']
                dy = df['y_loc'].values[1] - df['y_loc'].values[0]
                dz = df['z_loc'].values[1] - df['z_loc'].values[0]
                spacings['Hopper_Stiffeners'] = np.sqrt(dy**2 + dz**2) #m
            elif self.vessel_type == 'Container_Ship':
                # Container ship hopper pannel stiffener spacing is based on y spacing or z spacing 
                df = self.Structural_Elements[self.Structural_Elements['Class']=='Hopper_Pannel_Longitudinal_Stiffener']
                # Get min dy and min dz between adjacent stiffeners
                y = df.sort_values(by='y_loc')['y_loc'].unique()
                z = df.sort_values(by='z_loc')['z_loc'].unique()
                dy = np.min(np.diff(y))
                dz = np.min(np.diff(z))
                spacings['Hopper_Stiffeners'] = min(dy, dz) #m

        else: 
            dims['Hopper_Plate_Thickness'] = 0.0 #mm
            dims['Hopper_Stiffener_Thickness'] = 0.0 #mm
            dims['Hopper_Stiffener_flange_width'] = 0.0 #mm
            dims['Hopper_Stiffener_Depth'] = 0.0 #mm
            spacings['Hopper_Stiffeners'] = 0.0 #m

        return spacings, dims

    def Calc_ABS_Transverse_Struct_Constraints(self, spacings,dims):
        """
        This function calculates the ABS constraints for the transverse structure of the ship based on the spacings and sizes calculated in the Calculate_Spacings_and_Sizes function. It returns a dictionary of constraint values that can be compared to the actual structural properties of the ship to evaluate whether the design meets ABS requirements.
        """
        constraints = {}

        # 0) Min Double Bottom Height 
        # a) ABS 3-2-4/3.1 
        # b) ABS 3-2-4/1.2
        #ABS Part 3, Ch. 2, Sec. 4/3.1 - Min. Double Bottom Height
        A = (32*dims['beam'] +190*np.sqrt(dims['draft'])) #mm
        # ABS Part 3, Ch. 2, Sec. 4/1.2 - Min Double Bottom Height for non-Tankers max needed Db Height is Beam/20 but not need be more than 2m.  
        if self.vessel_type != 'Tanker': 
            B = np.amin([dims['beam']/20 * 1000.0, 2000.0]) #mm
            C = 760.0 #mm
        else:  # Only apply ABS 3-2-4/3.1
            B = 0.0
            C = 0.0

        constraints['Double_Bottom_Height'] = np.amax([A, B, C]) #mm

        #1) Min Bottom Floor Thickness
        #2) Min Bottom Girder Thickness
        #  ABS Part 3, Ch. 2, Sec. 4/5.1  - Min. Bottom Floor thickness
        constraints['Bottom_Floor_Thickness'] = 0.036*dims['length'] + 4.7+1.5 #mm
        constraints['Bottom_Girder_Thickness'] = 0.036*dims['length'] + 4.7+1.5 #mm 

        #3) Max. Bottom Floor Spacing 
        # ABS Part 3, Ch. 2, Sec. 4/5.1  -> NOTE in paper for CARGO
        constraints['Bottom_Floor_Spacing'] = 3.660 #m 

        #4)Inner Bottom Deck Thickness AND 5) Hooper Plate Thickness
        #a) ABS Part 3, Ch. 2, Sec. 4-9.2  - Min. Inner Bottom Deck thickness for Longitudinal Framed Ship 
        #b) ABS Part 3, Ch. 2, Sec. 4/9.5  - Deck thickness for Grabs
        #c) ABS Part 3, Ch. 2, Sec 10/3.1 - Plate thickness for Deep Tanks
        #d) ABS Part 3, Ch. 2, Sec. 3 / 5.17 - Wheel loading
        s = max(spacings['Frames'], spacings['Bottom_Stringers'])
        
        A = (1000.0*s*np.sqrt(dims['h_cargo'])/254.0) + 1.5 #mm

        #b) ABS Part 3, Ch. 2, Sec. 4/9.5 - Min. Inner Bottom Deck Thickness for Grabs - ASSUME CONTAINER SHIPS EXEPERIENCE ACCIDENTAL IMPACTS SIMILAR TO GRABS
        Q = 1.0 # no steel strength reduction factor
        if self.vessel_type != 'Tanker':
            B = (0.037*dims['length'] + 0.009*s*1000.0)*np.sqrt(Q) + 5.5 #mm
        else:
            B = 0.0
        #print('Ship Class ', self.vessel_type)
        #print('B', B)

        #c) ABS Part 3, Ch. 2, Sec 10-3.1 Deep Tank Plate Thickness Assume Inner bottom Deck thickness and Hopper Plate thickness support deep tanks (Ballast or tanker design)
        k = 1.0 # more stringent for deep tanks
        Q = 1.0 # no steel strength reduction factor
        h = 0.667* dims['depth'] #m  # Most Stringent h for all Deep Tank Caluclations
        if self.vessel_type == 'Tanker':
            h1 = dims['depth'] - dims['Double_Bottom_Height'] + 1.3 #m * Per Foot Note in ABS 3-2-10, use table 2 in ABS 3-2-7/3.1 for h to be a heights to a point above the tank top
            h = max(h, h1) #m

        C1 = 254.0
        C2 = 2.5
        
        C = (1000*s*k*np.sqrt(Q*h)/C1) + C2 #mm

        #d) ABS Part 3, Ch. 2, Sec. 3 / 5.17 - Wheel loading
        AS = max(spacings['Webframes'], spacings['Bottom_Girders'])/min(spacings['Webframes'], spacings['Bottom_Girders']) #   Aspect Ratio of plate

        k1 = 8.05 
        k2 = 0.2 # Max Value from Figure 2 in ABS 3-2-3

        n = min(1.0,(0.85 + 0.15*(AS-1)))
        c1 = 1.5 # Weight factor for wheel loads at sea

        W = 13800.0 * 9.81 /1000.0 # kN Weight of a Cat 313 excavator - https://www.cat.com/en_US/products/new/equipment/excavators/small-excavators/126701.html

        D = 1.1* k1*k2 * n * np.sqrt(c1*W) #mm rule for strength deck thickness
        
        arg_max = np.argmax([A, B, C, D])
        str = ['A', 'B', 'C', 'D'][arg_max]
        #print(f"Max inner bottom deck thickness is from option {str}, value: {max(A, B, C, D)}")
        
        constraints['Inner_Bottom_Deck_Thickness'] = max(A, B, C, D) #mm

        # Also Apply to Hopper Plate thickness for Deep Tanks
        if dims['Hopper_Plate_Thickness'] > 0.0:
            constraints['Hopper_Plate_Thickness'] = constraints['Inner_Bottom_Deck_Thickness'] #mm
        else:
            constraints['Hopper_Plate_Thickness'] = 0.0 #mm

        #6) Min. Bottom Stiffener SM
        #7) Min. Bottom Frame SM
        # 8) Min. Hopper Stiffener SM
        # ABS Part 3, Ch. 2, Sec 10/3.3 Bottom Structure and Inner Bottom Stiffener and Frame SM 
        c = 1.0 # Most stringent constant for bottom structure
        l = max(spacings['Webframes'],spacings['Bottom_Girders']) #m

        h = 0.667* dims['depth'] #m  # Most Stringent h for all Deep Tank Caluclations
        if self.vessel_type == 'Tanker':
            h1 = dims['depth'] - dims['Double_Bottom_Height'] + 1.3 #m * Per Foot Note in ABS 3-2-10, use table 2 in ABS 3-2-7/3.1 for h to be a heights to a point above the tank top
            h = max(h, h1) #m
        
        SM = 7.8*c*h*s*l**2.0 # in cm^3
        constraints['Bottom_Stiffener_SM'] = SM
        constraints['Bottom_Transverse_Stiffener_SM'] = SM # 

        if self.vessel_type != 'Tanker':  # Tankers do not have hopper pannels or stiffeners, so we can skip these calculations for tankers
            constraints['Hopper_Stiffener_SM'] = SM # 
        else:
            constraints['Hopper_Stiffener_SM'] = 0.0 #

        

        #9) Min. Side Frame SM
        # ABS Part 3, Ch. 2, Sec. 5/3.1.1
    
        C1 = 30.0
        C2 = 7.0
        C3 = 45.0
        
        l_hold = max(dims['l_frame'], 2.1)
        b = spacings['Bottom_Girders'] #m
    
        h = max(dims['draft'] - l_hold/2.0, 0.4*l_hold) #m
        h1 = dims['h_cargo']
        Q = 1.0 # no steel strength reduction factor
        constraints['Side_Frame_SM'] = (spacings['Frames']*l_hold**2.0) * (h+h1*b/C1)*(C2 + C3/l_hold**3.0)*Q # in cm^3

        #10)Min. Webframe SM 
        # ABS Part 3, Ch. 2, Sec. 6/3.1 
        c = 1.5
        Q = 1.0 # no steel strength reduction factor
        h_web = max(0.5*dims['h_cargo'], np.abs(dims['h_cargo']/2.0 + dims['Double_Bottom_Height'] - dims['draft'])) #m
        K = 1.0 # dims['N_frames']

        constraints['Webframe_SM'] = 4.74*c*spacings['Webframes'] *dims['h_cargo']**2.0 * (h_web+(dims['Webframe_Depth']*dims['h_deck'])/(45*K))*Q # in cm^3

        #11) Min Webframe Depth
        # ABS Part 3, Ch. 2, Sec. 6/3.5 
        constraints['Webframe_Depth'] = 1000.0*0.125*dims['h_cargo']#mm

        #12)  Min Webframe Thickness
        #ABS Part 3, Ch. 2, Sec. 6/3.5 
        constraints['Webframe_Thickness'] = min(1000.0*0.01*dims['Webframe_Depth']+3.5, 14.0) #mm
        
        #13) Min. Side Stringer SM
        #a) ABS Part 3, Ch. 2, Sec. 6/5.1 
        #b) ABS 3-2-10/3.3
        c = 1.5
        Q = 1.0 # no steel strength reduction factor
        h = max(0.667*dims['depth'], 1.8) #m

        A = 4.74*c*h*spacings['Side_Stringers']* spacings['Webframes']**2.0 * Q # in cm^3 ABS 3-2-6/5.1

        #ABS 3-2-10/3.3 Now Deep Tank for Frame and Side Shell Stiffeners ABS
        s = max(spacings['Frames'], spacings['Side_Stringers']) # in m

        c = 1.0 # Most stringent constant for bottom structure
        l = spacings['Webframes'] #m

        h = 0.667* dims['depth'] #m  # Most Stringent h for all Deep Tank Caluclations
        if self.vessel_type == 'Tanker':
            h1 = dims['depth'] - dims['Double_Bottom_Height'] + 1.3 #m * Per Foot Note in ABS 3-2-10, use table 2 in ABS 3-2-7/3.1 for h to be a heights to a point above the tank top
            h = max(h, h1) #m
        
        B = 7.8*c*h*s*l**2.0 # in cm^3

        constraints['Side_Stringer_SM'] = max(A, B) # in cm^3

        #constraints['Side_Stringer_SM'] = SM
        
        
        #14) Min. Side Stringer Depth
        # ABS Part 3, Ch. 2, Sec. 6/5.3
        constraints['Side_Stringer_Depth'] = 1000*0.125*spacings['Webframes'] # in mm

        #15) Min. Trans. Bulkhead Thickness
        #a) ABS Part 3, Ch. 2, Sec. 9/5.1
        #b) ABS 3-2-10/3.1 - Deep Tank Bulkhead Thicknes
        s = max(spacings['Trans_Bulkhead_Horizontal_Stiffeners'], spacings['Trans_Bulkhead_Vertical_Stiffeners'])
        k = 1.0 # more stringent for bulkheads
        c = 254.0 # Constant for collision bulkheads, but more stringent

        Q = 1.0 # no steel strength reduction factor

        t = 1000.0*s*k*np.sqrt(Q*dims['depth'])/c + 1.5 #mm

        A = max([t, 6.0, 1000.0*s/200.0+2.5]) #mm ABS Part 3, Ch. 2, Sec. 6-5.3

        # ABS 3-2-10/3.1 Now Deep Tank Bulkhead Thickness ABS
        k = 1.0 # more stringent for deep tanks
        Q = 1.0 # no steel strength reduction factor
        h = 0.667* dims['depth'] #m  # Most Stringent h for all Deep Tank Caluclations
        if self.vessel_type == 'Tanker':
            h1 = dims['depth'] - dims['Double_Bottom_Height'] + 1.3 #m * Per Foot Note in ABS 3-2-10, use table 2 in ABS 3-2-7/3.1 for h to be a heights to a point above the tank top
            h = max(h, h1) #m

        C1 = 254.0
        C2 = 2.5
        
        B = (1000*s*k*np.sqrt(Q*h)/C1) + C2 #mm
        constraints['Bulkhead_Thickness'] = max(A, B) #mm

        
        #16) Min. Trans. Bulkhead Stiffener SM
        #a) ABS Part 3, Ch. 2, Sec. 9/5.3
        #b) ABS 3-2-10/3.3 - Deep Tank Bulkhead Stiffener SM
        k = 1.25 # more stringent for collision bulkheads
        c= 0.6  # constant assuming no end attachments for stiffeners (most stringent)
        h = dims['depth'] #m height of bulkhead
        l = spacings['Trans_Bulkhead_Horizontal_Stiffeners']
        
        A = 7.8*k*c*s*h*(l)**2.0 # in cm^3

        # ABS 3-2-10/3.3 Now Deep Tank Bulkhead Stiffener SM ABS
        k =1.0 # more stringent for deep tanks
        h = 0.667* dims['depth'] #m  # Most Stringent h for all Deep Tank Caluclations
        if self.vessel_type == 'Tanker':
            h1 = dims['depth'] - dims['Double_Bottom_Height'] + 1.3 #m * Per Foot Note in ABS 3-2-10, use table 2 in ABS 3-2-7/3.1 for h to be a heights to a point above the tank top
            h = max(h, h1) #m
        l = max(spacings['Trans_Bulkhead_Horizontal_Stiffeners'], spacings['Trans_Bulkhead_Vertical_Stiffeners']) #m
        B = 7.8*k*c*s*h*(l)**2.0 # in cm^3

        constraints['Trans_Bulkhead_Horizontal_Stiffener_SM'] = max(A, B) # in cm^3
        constraints['Trans_Bulkhead_Vertical_Stiffener_SM'] = constraints['Trans_Bulkhead_Horizontal_Stiffener_SM']



        #17) Min. Deck Frame SM
        #ABS Part 3, Ch. 2, Sec. 7/3.1
        
        c = 0.585
        #First check for existance of deck girders and use the spacing of the deck girders if they exist, otherwise use the spacing of the webframes
        
        try:    
            constraints['Deck_Trans_Stiff_SM']= 7.8*c*dims['h_deck']*spacings['Webframes']*spacings['Deck_Girders']**2.0 # in cm^3
        except:
            constraints['Deck_Trans_Stiff_SM']= 0.0

        # Check Deck Strength for Grabs 

    

        #18) Min. Deck Thickness
        #a) ABS Part 3, Ch. 2, Sec. 3/5.1
        #b) ABS Part 3, Ch. 2, Sec. 4 / 9.5
        #c) ABS Part 3, Ch. 2, Sec. 3 / 5.17 - Wheel loading
        try:
            s = max(spacings['Frames'], spacings['Deck_Stringers']) # in m
        except:
            s = spacings['Frames'] # If there are no deck stringers, we can assume the spacing is just the frame spacing
        
        #ABS Part 3 Ch 2 Sec 3 / 5.1
        A = 1000*s*np.sqrt(dims['h_deck'])/254.0 +1.5 #mm

        #ABS Part 3 Ch 2 Sec 4 / 9.5 for Grabs - Assume container ships experience accidental impacts similar to grabs
        Q = 1.0 # no steel strength reduction factor
        if self.vessel_type != 'Tanker':
            B = (0.037*dims['length'] + 0.009*s*1000.0)*np.sqrt(Q) +5.5 #mm
        else:
            B = 0.0

        #c) ABS Part 3, Ch. 2, Sec. 3 / 5.17 - Wheel loading
        AS = max(spacings['Webframes'], spacings['Bottom_Girders'])/min(spacings['Webframes'], spacings['Bottom_Girders']) # Aspect Ratio of plate

        k1 = 8.05 
        k2 = 0.2 # Max Value from Figure 2 in ABS 3-2-3

        n = min(1.0,(0.85 + 0.15*(AS-1)))
        c1 = 1.5 # Weight factor for wheel loads at sea

        W = 13800.0 * 9.81 /1000.0 # kN Weight of a Cat 313 excavator - https://www.cat.com/en_US/products/new/equipment/excavators/small-excavators/126701.html

        C = 1.1* k1*k2 * n * np.sqrt(c1*W) #mm rule for strength deck thickness


        
        constraints['Deck_Thickness']= max(A,B,C, 5.0) # ABS min thickness is 5mm
   
        #19) Min. Deck Beam SM
        #ABS Part 3, Ch. 2, Sec. 8/5.3 
        c = 1.5 #Tank requirements are more stringent

        if 'Deck_Girders' in spacings:
            constraints['Deck_Beam_SM'] = 4.74*c*dims['h_deck']*spacings['Deck_Girders']**2.0 * spacings['Webframes'] # in cm^3
            
        #20) Min. Deck Beam Depth 
        # ABS Part 3, Ch. 2, Sec. 8/5.7 and 5.13 
            constraints['Deck_Beam_Depth'] = 1000.0*0.0833*spacings['Deck_Girders'] # in mm
        else:
            constraints['Deck_Beam_SM'] = 0.0 # in cm^3
            constraints['Deck_Beam_Depth'] = 0.0 # in mm

        #21)- Min. Deck Beam Thickness
        # ABS Part 3, Ch. 2, Sec. 8/5.7 
        if 'Deck_Beam_Depth' in dims:
            constraints['Deck_Beam_Thickness'] = max(1000*0.01*dims['Deck_Beam_Depth']+4, 15.0) #mm
        else:
            constraints['Deck_Beam_Thickness'] = 0.0 #mm

        #22) Min. Side Shell Thickness
        # a) ABS Part 3, Ch. 2, Sec. 2/3.9.3
        # b) 3-2-2/5.1
        
        d_D = max(dims['draft']/dims['depth'], 0.0433*dims['length']/dims['depth']) 
        s = max(spacings['Frames'], spacings['Side_Stringers']) # in m

        A = (1000*s/645)*np.sqrt((dims['length']-15.2)*d_D) + 2.5 #mm  #3-2-2/3.9.3

        B = 0.035*(dims['length'] + 29.0) + 0.009*s*1000.0 #mm 3-2-2/5.1 for vessels between 90 and 305m

        C = 0.035*(dims['length'] + 29.0) + 0.009*610.0 # 3-2-2/3.9.3 footnote: eval with 610mm spacing.

        constraints['Side_Shell_Thickness'] = max(A, B, C)#mm  

        #23) Min.  Bottom Shell Thickness For Now, Let's Assume L = 250m for the paper. Acknowledge that length plat
        # ABS Part 3, Ch. 2, Sec. 2/3.15.1 -
        t_1 = (1000*s/508)*np.sqrt((dims['length']-62.5)*d_D) + 2.5 #mm
        t_2 = 1000*s*(dims['length'] - 18.3)/(42.0*dims['length'] + 1070) #mm   
        #print('consider vessel length to get different thickness reqs')
        constraints['Bottom_Shell_Thickness'] = max(t_1, t_2, 8.5) #mm ABS min thickness is 5mm

        return constraints
    
    def Calc_Structural_Properties(self,spacings, dims):
        """
        This function calculates the actual structural properties of the ship based on the Structural_Elements dataframe. It returns a dictionary of structural properties that can be compared to the ABS constraints calculated in the Calc_ABS_Transverse_Struct_Constraints function to evaluate whether the design meets ABS requirements.
        """
        struct_evals = {}
        # We will calculate the actual properties for each of the constraints calculated in the Calc_ABS_Transverse_Struct_Constraints function.
        
        #0) Min Double Bottom Height 
        struct_evals['Double_Bottom_Height'] = dims['Double_Bottom_Height']*1000.0 #mm

        #1) Min Bottom Floor Thickness
        struct_evals['Bottom_Floor_Thickness'] = dims['Bottom_Floor_Thickness'] #mm

        #2) Min Bottom Girder Thickness
        struct_evals['Bottom_Girder_Thickness'] = dims['Bottom_Girder_Thickness'] #mm   
        
        #3) Max. Bottom Floor Spacing 
        struct_evals['Bottom_Floor_Spacing'] = spacings['Webframes'] #m    

        #4)Inner Bottom Deck Thickness and Hopper Plate Thickness
        struct_evals['Inner_Bottom_Deck_Thickness'] = dims['Inner_Bottom_Deck_Thickness'] #mm

        # 5) Hopper Plate Thickness
        struct_evals['Hopper_Plate_Thickness'] = dims['Hopper_Plate_Thickness'] #mm

        #6) Min. Bottom Stiffener SM
        SM, _, _ = self.calc_stiffener_SM(dims['Bottom_Shell_Thickness'], spacings['Bottom_Stringers']*1000.0, dims['Bottom_Stiffener_Depth']*1000.0, dims['Bottom_Stiffener_Thickness'], dims['Bottom_Stiffener_flange_width']*1000.0)   
        struct_evals['Bottom_Stiffener_SM'] = SM # in cm^3


        #7) Min. Bottom Frame SM
        SM, _, _ = self.calc_stiffener_SM(dims['Bottom_Shell_Thickness'], spacings['Frames']*1000.0, dims['Bottom_Transverse_Stiffener_Depth']*1000.0, dims['Bottom_Transverse_Stiffener_Thickness'], dims['Bottom_Transverse_Stiffener_flange_width']*1000.0)
        struct_evals['Bottom_Transverse_Stiffener_SM'] = SM # in cm^3
            
        # 8) Min. Hopper Stiffener SM
        if dims['Hopper_Plate_Thickness'] > 0.0:
            SM, _, _ = self.calc_stiffener_SM(dims['Hopper_Plate_Thickness'], spacings['Hopper_Stiffeners']*1000.0, dims['Hopper_Stiffener_Depth']*1000.0, dims['Hopper_Stiffener_Thickness'], dims['Hopper_Stiffener_flange_width']*1000.0)
            struct_evals['Hopper_Stiffener_SM'] = SM # in cm^3
        else:
            struct_evals['Hopper_Stiffener_SM'] = 0.0 # in cm^3


        #9) Min. Side Frame SM
        SM, _, _ = self.calc_stiffener_SM(dims['Side_Shell_Thickness'], spacings['Frames']*1000.0, dims['Frame_Depth']*1000.0, dims['Frame_thickness'], dims['Frame_flange_width']*1000.0)
        struct_evals['Side_Frame_SM'] = SM # in cm^3

    
        #10) Min. Webframe SM 
        SM, _, _ = self.calc_stiffener_SM(dims['Side_Shell_Thickness'], spacings['Webframes']*1000.0, dims['Webframe_Depth']*1000.0, dims['Webframe_thickness'], dims['Webframe_flange_width']*1000.0)
        struct_evals['Webframe_SM'] = SM # in cm^3

        #11) Min Webframe Depth
        struct_evals['Webframe_Depth'] = dims['Webframe_Depth']*1000.0 #mm

        #12)  Min Webframe Thickness
        struct_evals['Webframe_Thickness'] = dims['Webframe_thickness'] #mm          
             
        #13) Min. Side Stringer SM
        SM, _, _ = self.calc_stiffener_SM(dims['Side_Shell_Thickness'], spacings['Side_Stringers']*1000.0, dims['Side_Stringer_Depth']*1000.0, dims['Side_Stringer_thickness'], dims['Side_Stringer_flange_width']*1000.0)
        struct_evals['Side_Stringer_SM'] = SM # in cm^3
          
        #14) Min. Side Stringer Depth
        struct_evals['Side_Stringer_Depth'] = dims['Side_Stringer_Depth']*1000.0 #mm  

        #15) Min. Trans. Bulkhead Thickness
        struct_evals['Bulkhead_Thickness'] = dims['Trans_Bulkhead_Thickness'] #mm

        #16) Min. Trans. Bulkhead Stiffener SM
        SM, _, _ = self.calc_stiffener_SM(dims['Trans_Bulkhead_Thickness'], spacings['Trans_Bulkhead_Horizontal_Stiffeners']*1000.0, dims['Trans_Bulkhead_Horizontal_Stiffener_Depth']*1000.0, dims['Trans_Bulkhead_Horizontal_Stiffener_thickness'], dims['Trans_Bulkhead_Horizontal_Stiffener_flange_width']*1000.0)
        struct_evals['Trans_Bulkhead_Horizontal_Stiffener_SM'] = SM # in cm^3

        SM, _, _ = self.calc_stiffener_SM(dims['Trans_Bulkhead_Thickness'], spacings['Trans_Bulkhead_Vertical_Stiffeners']*1000.0, dims['Trans_Bulkhead_Vertical_Stiffener_Depth']*1000.0, dims['Trans_Bulkhead_Vertical_Stiffener_thickness'], dims['Trans_Bulkhead_Vertical_Stiffener_flange_width']*1000.0)
        struct_evals['Trans_Bulkhead_Vertical_Stiffener_SM'] = SM # in cm^3

        #17) Min. Deck Frame SM
        if 'Deck_Girders' in spacings:
            SM, _, _ = self.calc_stiffener_SM(dims['Deck_Thickness'], spacings['Frames']*1000.0, dims['Deck_Frame_Depth']*1000.0, dims['Deck_frame_thickness'], dims['Deck_frame_flange_width']*1000.0)
            struct_evals['Deck_Trans_Stiff_SM'] = SM # in cm^3
        else:
            struct_evals['Deck_Trans_Stiff_SM'] = 0.0 # in cm^3

        #18) Min. Deck Thickness
        struct_evals['Deck_Thickness'] = dims['Deck_Thickness'] #mm
        
        #19) Min. Deck Beam SM
        if 'Deck_Girders' in spacings:
            SM, _, _ = self.calc_stiffener_SM(dims['Deck_Thickness'], spacings['Webframes']*1000.0, dims['Deck_Beam_Depth']*1000.0, dims['Deck_Beam_thickness'], dims['Deck_Beam_flange_width']*1000.0)
            struct_evals['Deck_Beam_SM'] = SM # in cm^3

        else:
            struct_evals['Deck_Beam_SM'] = 0.0 # in cm^3   

        #20) Min. Deck Beam Depth 
        if 'Deck_Beam_Depth' in dims:
            struct_evals['Deck_Beam_Depth'] = dims['Deck_Beam_Depth']*1000.0 #mm
        else:
            struct_evals['Deck_Beam_Depth'] = 0.0 # in mm

        #21) Min. Deck Beam Thickness
        if 'Deck_Beam_Depth' in dims:
            struct_evals['Deck_Beam_Thickness'] = dims['Deck_Beam_thickness'] #mm
        else:
            struct_evals['Deck_Beam_Thickness'] = 0.0 # in mm

        #22) Min. Side Shell Thickness
        struct_evals['Side_Shell_Thickness'] = dims['Side_Shell_Thickness'] #mm

        #23) Min.  Bottom Shell Thickness
        struct_evals['Bottom_Shell_Thickness'] = dims['Bottom_Shell_Thickness'] #mm

        return struct_evals

    def calc_stiffener_SM(self,plate_t, spacing, h,t,w):

        '''
        This function calculates the section modulus of a stiffener based on its geometry and spacing
        '''

        Area = t*w + h*t + plate_t*spacing # in mm^2
        Z_centroid = (0 + t*w*(h) + h*t*(h/2.0))/Area # in mm
        I = (1/12*spacing* plate_t**3.0 + spacing*plate_t*(Z_centroid)**2.0 + 1/12*w*t**3.0 + w*t*(Z_centroid - h)**2.0 + 1/12*t*h**3.0 + h*t*(h/2.0 - Z_centroid)**2.0) # in mm^4
        z = max(Z_centroid, h - Z_centroid) # in mm
        SM = (I/z)/1000.0 # in cm^3

        bit_effective_flange = w > 2*t
        aspect_ratio = h/t

        return SM, bit_effective_flange, aspect_ratio # in cm^3, bool, dimensionless

    