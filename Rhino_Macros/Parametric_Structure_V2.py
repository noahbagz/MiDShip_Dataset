'''
This set of Functions generates random parametric structures, runs the loop to generate many of them, and deletes the previous content on the rhino document

THIS IS VERSION 2 of the Parametric Structure Generator.

V2 Updates: 
-Intermediate frames between the web frames
-Longitudinal Bulkheads
-Hatch Openings to Deck 


'''

#First lets define the ship structure class.
# 
import numpy as np
import pandas as pd
import rhino_StructGen as rhino_SG 
import rhinoscriptsyntax as rs
import os

#import time to clock the time it takes to generate the structures
import time




class Structure_3H:
    #Set some global variables for the parameters
    param_idx_bit = np.array([27,31,35,39,43,47,51,56,74,79,86,91,96, 100, 104,105,111,115,116]) #The indices of the parameters that are boolean values (T or C)
    param_idx_cat = np.array([57,58,59]) #The indices of the parameters that are categorical values (Tanker, Container, Bulk Carrier)
    param_idx_int = np.array([15,16,17,18,19,20,21,22, 70, 75,82,87,92, 106]) #The indices of the parameters that are integers (number of stiffeners)

    param_idx_plate_thicknesses = np.array([7,8,9,10,11,12,23,29,33,37,41,45,49,52,54,62,65,68,69,72,77,80,81,84,89,94,98,102,107,109,113])

    param_idx_brackets = np.array([60,61,63,64]) #The indices of the parameters that are for the brackets -tanker and bulk carrier
    param_idx_container = np.array([66,67,68,69,70,71,72,73,74,75,76,77,78,79]) #The indices of the parameters that are for the container step
    param_idx_bulkcarrier = np.array([80,81,82,83,84,85,86,87,88,89,90,91]) #The indices of the parameters that are for the bulk carrier

    # Load the parameter definitions from the dataset using a repository-
    # relative path.  This keeps the Rhino module portable when the repository
    # is cloned to a different computer or user account.
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
    _PARAMETER_RANGE_CSV = os.path.join(
        _PROJECT_ROOT,
        "MiDShip_Dataset",
        "Random_Structures",
        "StructuralParameterList_V2_Updated_Ranges.csv",
    )
    Param_Dict = pd.read_csv(_PARAMETER_RANGE_CSV)


    def __init__(self, params, path,id):
        '''
        params is the list of parameters for the ship structure class: 

        The parameters are: 

        params[0] = L_3h. The length of the 3 hold structure
        params[1] = B. The beam. ratio of L_3h
        params[2] = T. The draft. ratio of L_3h
        params[3] = D. The depth . ratio of L_3h
        params[4] = R_b. The bilge radius. ratio of D
        params[5] = l_overhang. the length of the structural overhang past the bulkheads. Fraction L_3h
        params[6] = Db. the height of the double bottom in mm 
        params[7] = bottom shell thickness in mm
        params[8] = side shell thickness in mm
        params[9] =  Top Deck Thickness in mm
        params[10] = Inner bottom plate thickness in mm
        params[11] = trans_Bulkhead thickness in mm
        params[12] = Inner side shell thickness in mm
        params[13] = Shear strake thickness in mm
        params[14] = shear strake height in mm
        params[15] = web frames per hold (integer) 
        params[16] = num bottom girders (CL to B/2)
        params[17] = num intermediate bottom stiffieners between girders 
        params[18] = num top girders   
        params[19] = num intermediate deck stiffeners between girders
        params[20] = number of side shell stiffeners
        params[21] = number of vertical bulkhead stiffeners
        params[22] = number of transverse bulkhead stifferners
        params[23] = bottom girder thickness in mm
        params[24] = intermediate bottom stiffener height in mm
        params[25] = intermediate bottom stiffener thickness in mm
        params[26] = interemediate bottom stiffener flange width in mm
        params[27] = bit t or c (boolean to determine shape of intermediate bottom stiffener)
        params[28] = Transverse Deck Beam height in mm
        params[29] = transverse deck beam thickness in mm 
        params[30] = transverse deck beam flange width in mm
        params[31] = bit t or c (Boolean to determine shape of deck beam)
        params[32] = deck girder height in mm
        params[33] = deck girder thickness in mm
        params[34] = deck girder flange width in mm
        params[35] =  bit t or c (Boolean to determine shape of deck girder)
        params[36] = deck stiffener height in mm
        params[37] = deck stiffener thickness in mm
        params[38] = deck stiffener flange width in mm
        params[39] = bit t or c (boolean to determine shape of deck stiffener)
        params[40] = Bulkhead trans stiffener height in mm
        params[41] = bulkhead trans stiffener thickness in mm
        params[42] = bulkhead trans stiffener flange width in mm
        params[43] = bit t or c (boolean to determine shape of trans stiffener)
        params[44] = Bulkhead vert stiffener height in mm
        params[45] = bulkhead vert stiffener thickness in mm
        params[46] = bulkhead vert stiffener flange width in mm
        params[47] = bit t or c (boolean to determine shape of vert stiffener)
        params[48] = webframe height in mm
        params[49] = webframe thickness in mm
        params[50] = webframe flange width in mm
        params[51] = bit t or c (boolean to determine shape of webframe)
        params[52] = floor thickness in mm
        params[53] = side shell stiffener height in mm
        params[54] = side shell stiffener thickness in mm
        params[55] = side shell stiffener flange width in mm
        params[56] = bit t or c (boolean to determine shape of side shell stiffener)
        params[57] =  categorical structure class - Tanker - add brackets between deck, webframes, and innerbottom
        params[58] = categorical structure class - Container - make oversized webframes and cut a lightening hole to make the step
        params[59] = categorical structure class - Bulk carrier add stringers that are aligned with the brackets (angled)
        params[60] = floor-web bracket L1 in mm (Y direction)
        params[61] = floor-web bracket L2 in mm (Z direction)
        params[62] = floor-web bracket thickness in mm
        params[63] = deck-web bracket L1 in mm (Y direction)
        params[64] = deck-web bracket L2 in mm (Z direction)
        params[65] = deck-web bracket thickness in mm
        params[66] = container step height in mm
        params[67] = container step width in mm
        params[68] = container step deck plate thickness
        params[69] = container step side plate thickness
        params[70] = container  num step deck stringers
        params[71] = container deck stringer height
        params[72] = container deck stringer thickness
        params[73] = container deck stringer flange width
        params[74] = bit T or C (boolean to determine shape of contrainer deck stiffener)
        params[75] = container num side stringers
        params[76] = container side stringer height
        params[77] = container side stringer thickness
        params[78] = container side stringer flange width
        params[79] = bit T or C (boolean to determine shape of contrainer side stiffener)
        params[80] = bulk carrier bottom angle stringer thickness
        params[81] = bulk carrier top angle stringer thickeness
        params[82] = bulk carrier bottom stringer - num stiffeners
        params[83] = bulk carrier bottom stiffener height
        params[84] = bulk carrier bottom stiffener thickness
        params[85] = bulk carrier bottom stiffener flange width
        params[86] = bit t or c (boolean for bulk carrier bottom stiffener shape)
        params[87] = bulk carrier top stringer - num stiffeners
        params[88] = bulk carrier top stiffener height
        params[89] = bulk carrier top stiffener thickness
        params[90] = bulk carrier top stiffener flange width
        params[91] = bit t or c (boolean for bulk carrier top stiffener shape)
        params[92] = number of intermediate transverse frames in the hold (integer)
        params[93] = intermediate bottom frame height in mm
        params[94] = intermediate bottom frame thickness in mm
        params[95] = intermediate bottom frame flange width in mm
        params[96] = bit t or c (boolean to determine shape of intermediate bottom frame)
        params[97] = intermediate side frame height in mm
        params[98] = intermediate side frame thickness in mm
        params[99] = intermediate side frame flange width in mm
        params[100] = bit t or c (boolean to determine shape of intermediate side frame)
        params[101] = intermediate deck frame height in mm
        params[102] = intermediate deck frame thickness in mm
        params[103] = intermediate deck frame flange width in mm
        params[104] = bit t or c (boolean to determine shape of intermediate deck frame)
        params[105] = bit for longitudinal bulkhead
        params[106] = number of longitudinal bulkheads
        params[107] = longitudinal bulkhead t in mm
        params[108] = longitudinal bulkhead stiffener height in mm
        params[109] = longitudinal bulkhead stiffener thickness in mm
        params[110] = longitudinal bulkhead stidffener flange width in mm
        params[111] = bit t or c (boolean to determine shape of longitudinal bulkhead stiffener)
        params[112] = longitudinal bulkhead vertical stiffener height in mm
        params[113] = longitudinal bulkhead vertical stiffener thickness in mm
        params[114] = longitudinal bulkhead vertical stiffener flange width in mm
        params[115] = bit t or c (boolean to determine shape of longitudinal bulkhead vertical stiffener)
        params[116] = bit for hatch openings
        params[117] = length of hatch opening as a fraction of L_3h
        params[118] = width of hatch opening as a fraction of B
        params[119] = fillet radius of hatch opening in mm



        Now: Let's turn parameters into dictionary sets to build a ship structure: 

        We will modify the params list for the following: 
        1) Brackes will be resized to land on a stiffener or girder,
        2) Flange on stiffeners will be resized to be maximum the height of the stiffener. 

        
        '''
        #Main Hull Dimensions
        self.Hull_Dict = {'L_3h': params[0], 
                          'B': params[1],
                          'T': params[2],
                          'D': params[3],
                          'R_b': params[4],
                          'l_overhang': params[5]
                          }
        #Main Structure Parameters
        self.Struct_Params = {
                            'Db': params[6],
                            'Bottom_Shell_Thickness': params[7],
                            'Side_Shell_Thickness': params[8], 
                            'Top_Deck_Thickness': params[9],
                            'Inner_Bottom_Thickness': params[10],
                            'Bulkhead_Thickness': params[11],
                            'Inner_Side_Shell_Thickness': params[12]}
        
        #Shear Strake Parameters
        self.Shear_Strake_Dict = {
                            't' : params[13],
                            'h' : params[14], 
                            '1D_element': False, 
                            'Class': 'Shear_Strake',}

        #Set up Transverse Structural Spacing
        self.x_bulkheads = np.linspace(self.Hull_Dict['l_overhang'], 1-self.Hull_Dict['l_overhang'], 4).round(10) #Round to avoid floating point errors
        
        self.num_web_frames_per_hold = params[15]
        self.x_trans_web_frames = np.linspace(self.Hull_Dict['l_overhang'], 1-self.Hull_Dict['l_overhang'], int((self.num_web_frames_per_hold*3)+4)).round(10)
        
        self.x_trans_web_frames = self.x_trans_web_frames[~np.isin(self.x_trans_web_frames, self.x_bulkheads)]

        #Set up Tranverse Frames Spacing
        self.num_trans_frames = params[92] #Number of intermediate frames in the hold
        self.x_trans_frames = np.linspace(self.Hull_Dict['l_overhang'], 1-self.Hull_Dict['l_overhang'], 
                                          int((self.num_web_frames_per_hold*3)+ 4 + (self.num_web_frames_per_hold + 1)*self.num_trans_frames*3)).round(10) #Round to avoid floating point errors
        
        self.x_trans_frames = self.x_trans_frames[~np.isin(self.x_trans_frames, self.x_bulkheads)] #Remove where x_transverse == x_bulkheads
        self.x_trans_frames = self.x_trans_frames[~np.isin(self.x_trans_frames, self.x_trans_web_frames)] #Remove where x_transverse = x_web_frames

        #Set Up Longitudinal Structural Members 

        #Determine the spacing between CL and the edge of the web frames
        WF_frac = 1 - params[48]/(1000.0*0.5*self.Hull_Dict['B']) #Fraction of Hull's breadth - we want to land the final girder at the edge of the web fames
       
        #Use WF_frac to set up the bottom and deck girders and stiffeners
        self.y_bottom_girders = np.linspace(0.0, WF_frac, int(params[16])).round(10) #Fraction of Hull's Beam. do not include gider at shell
        
        self.y_bottom_stiffeners = np.linspace(0.0,WF_frac, int((params[16]-1)*(params[17]+1)+1)).round(10) #Fraction of Hull's Beam
      
        delta_y = self.y_bottom_stiffeners[1] - self.y_bottom_stiffeners[0] #Distance intermediate stiffeners are spaced apart
        
        num_stiffeners = int((1-WF_frac)/delta_y) #Number of stiffeners that will fit between the final girder and the shell
        if num_stiffeners > 0:
            self.y_bottom_stiffeners = np.concatenate((self.y_bottom_stiffeners,
                                                       np.linspace(WF_frac, 1.0, num_stiffeners+2)[1:-1])) #Evenly space stiffeners between the final girder and the shell
        #Remove the stiffeners at the girder locations
        self.y_bottom_stiffeners = self.y_bottom_stiffeners[~np.isin(self.y_bottom_stiffeners, self.y_bottom_girders)] # remove stiffeners at Girder Locations


        #Force deck girder and stiffeners to align with the bottom girders and stiffeners
        params[18] = params[16] #Set number of deck girders equal to number of bottom girders
        params[19] = params[17] #Set number of deck stiffeners equal to number of bottom stiffeners

        #Fraction of Hull's Beam - Deck Girders and Stiffeners
        self.y_deck_girders = np.linspace(0.0, WF_frac, int(params[18])).round(10) #Fraction of Hull's Beam. do not include girder at shell
        
        self.y_deck_stiffeners = np.linspace(0.0,WF_frac, int((params[18]-1)*(params[19]+1)+1)).round(10) #Fraction of Hull's Beam
        
        delta_y = self.y_deck_stiffeners[1] - self.y_deck_stiffeners[0] #Distance intermediate stiffeners are spaced apart
        num_stiffeners = int((1-WF_frac)/delta_y) #Number of stiffeners that will fit between the final girder and the shell
        if num_stiffeners > 0:
            self.y_deck_stiffeners = np.concatenate((self.y_deck_stiffeners,
                                                     np.linspace(WF_frac, 1.0, num_stiffeners+2)[1:-1])) #Evenly space stiffeners between the final girder and the shell
        #Remove the stiffeners at the girder locations
        self.y_deck_stiffeners = self.y_deck_stiffeners[~np.isin(self.y_deck_stiffeners, self.y_deck_girders)]

        #Set Up Side Shell and Bulkhead Stiffeners 
        self.z_side_stiffeners = np.linspace(0.0,1.0, int(params[20]+2)).round(10)[1:-1] #Do not include the top or bottom stiffener
     
        

        # Align params 21 to be an even divisor of the longitudinal stiffener spacing
        num_y = int((params[18]-1)*(params[19]+1)+1)
        y_options = []
        for i in range(2,num_y+1):
            if int(num_y-1) % i == 0:
                y_options.append(i+1) #

        # Reverse sort y_options so that we can land on the highest number of stiffeners possible
        y_options = sorted(y_options, reverse=True)
        diffs = np.abs(np.array(y_options) - params[21])
        argmin = np.argmin(diffs)
        params[21] = y_options[argmin]
        
        self.y_vert_stiffeners = np.linspace(0.0,WF_frac, int(params[21])).round(10)[:-1]

        # Correct final number of z_trans stiffeners. We want the transverse bulkhead stiffeners to either be an even split of the side shell stiffeners OR an exact match. 
        num_z_options = []
        for i in range(2,int(params[20]+2)):
            if int(params[20] + 1) % i == 0:
                num_z_options.append(i-1) #We subtract 2 to not include the top and bottom stiffener
        # Reverse sort num_z_options so that we can land on the highest number of stiffeners possible
        num_z_options = sorted(num_z_options, reverse=True)

        diffs = np.abs(np.array(num_z_options) - params[22])
        argmin = np.argmin(diffs)
        #check len of argmin
        params[22] = num_z_options[argmin]

        self.z_trans_stiffeners = np.linspace(0.0,1.0,int(params[22]+2)).round(10)[1:-1]

        # Set up Longitudinal Bulkheads
        self.bit_Long_Bulkheads = params[105] #Boolean to determine if there are longitudinal bulkheads
        if self.bit_Long_Bulkheads == 1:
            self.num_long_bulkheads = params[106] #Number of longitudinal bulkheads

            if self.num_long_bulkheads == 1:
                self.y_long_bulkheads = np.array([0.0]) #If there is only one longitudinal bulkhead, it is in the middle of the hull
            elif self.num_long_bulkheads == 2:
                self.y_long_bulkheads = np.array([1.0/3.0]) 
            elif self.num_long_bulkheads == 3:
                self.y_long_bulkheads = np.array([0.0, 0.5])

            #print("Longitudinal Bulkhead Locations (Fraction of Hull's Beam/2): ", self.y_long_bulkheads.tolist())
            self.Long_Bulkhead_Params = {'y': self.y_long_bulkheads, #Fraction of Hull's Beam/2
                                         't': params[107]} #Thickness of the longitudinal bulkhead}

        #Instantiate the Ship Structure Class
        self.struct = rhino_SG.Ship_Structure_CAD(self.Hull_Dict, self.Struct_Params, path = path, id = id)


        # Bottom Longitudinal Girders and Stiffeners
        
        #Step one: check spacing of girders and stiffeners
        delta_y = (self.y_bottom_stiffeners[1] - self.y_bottom_stiffeners[0])*1000.0*0.5*self.Hull_Dict['B'] #Distance between stiffeners in mm

        self.Bottom_Girder_Dict = {
                            'y': self.y_bottom_girders,
                            'h' : self.Struct_Params['Db'],
                            't' : params[23],
                            'w' : 0,
                            'rot': 0,
                            'dir': [1,0,0],
                            'bit_TorC': 1,
                            '1D_element': False,
                            'Class': 'Bottom_Longitudinal_Girder'} #1D element is False because it is a plate girder, not a stiffener
        
        if params[26] > params[24]: #If the flange width is greater than the height of the stiffener, then we need to set the width to the height of the stiffener
            params[26] = params[24]

        if params[26] > delta_y: #If the web width is greater than the distance between stiffeners, then we need to set the flange width to the distance between stiffeners
            params[26] = delta_y
        
        self.Bottom_Stiffener_Dict = {
                            'y': self.y_bottom_stiffeners,
                            'h' : params[24],
                            't' : params[25],
                            'w' : params[26],
                            'rot': 0,
                            'dir': [1,0,0],
                            'bit_TorC': params[27],
                            '1D_element': True,
                            'Class': 'Bottom_Longitudinal_Stiffener'}
        
        self.Inner_Bottom_Stiffener_Dict = [{
                            'y': self.y_bottom_stiffeners,
                            'h' : params[24],
                            't' : params[25],
                            'w' : params[26],
                            'rot': 180,
                            'dir': [1,0,0],
                            'bit_TorC': params[27],
                            '1D_element': True,
                            'Class': 'Inner_Bottom_Longitudinal_Stiffener'}]
        
        #Set up the Transverse Frames
        if params[95] > params[93]: #If the flange width is greater than the height of the stiffener, then we need to set the flange width to the height of the stiffener
            params[95] = params[93]

        delta_x = (self.x_trans_frames[1] - self.x_trans_frames[0])*1000.0*self.Hull_Dict['L_3h'] #Distance between frames in mm
        if params[95] > delta_x: #If the web width is greater than the distance between frames, then we need to set the flange width to the distance between frames
            params[95] = delta_x
        
        self.Inner_Bottom_Trans_Frame_Dict = [{
                            'x' : self.x_trans_frames,
                            'h' : params[93],
                            't' : params[94],
                            'w' : params[95],
                            'rot': -90,
                            'dir': [0,1,0],
                            'bit_TorC': params[96],
                            '1D_element': True, #1D element is True because it is a stiffener, not a plate girder
                            'Class': 'Inner_Bottom_Transverse_Beam'}]
        
        if params[30] > params[28]: #If the flange width is greater than the height of the stiffener, then we need to set the flange width to the height of the stiffener
            params[30] = params[28]

        if params[103] > params[101]: #If the flange width is greater than the height of the stiffener, then we need to set the flange width to the height of the stiffener
            params[103] = params[101]
        if params[30] > delta_x: #If the web width is greater than the distance between frames, then we need to set the flange width to the distance between frames
            params[30] = delta_x
        if params[103] > delta_x: #If the web width is greater than the distance between frames, then we need to set the flange width to the distance between frames
            params[103] = delta_x

        self.Deck_Beam_Dict = [{
                            'x': self.x_trans_web_frames,
                            'h' : params[28],
                            't' : params[29],
                            'w' : params[30],
                            'rot': -90,
                            'dir': [0,1,0],
                            'bit_TorC': params[31],
                            '1D_element': False, #1D element is False, because it is a major structural member, not a stiffener
                            'Class': 'Deck_Transvere_Beam'},
                            {
                            'x': self.x_trans_frames,
                            'h' : params[101],    
                            't' : params[102],
                            'w' : params[103],
                            'rot': -90,
                            'dir': [0,1,0],
                            'bit_TorC': params[104],
                            '1D_element': True, #1D element is True, because it is a minor structural member
                            'Class': 'Deck_Transverse_Stiffener,'}]
        
        #Set up the Deck Longitudinal Girders and Stiffeners
        if params[34] > params[32]: #If the flange width is greater than
            params[34] = params[32] #the height of the stiffener, then we need to set the flange width to the height of the stiffener
        if params[38] > params[36]: #If the flange width is greater than the height of the stiffener, then we need to set the flange width to the height of the stiffener
            params[38] = params[36]
        
        delta_y = (self.y_deck_stiffeners[1] - self.y_deck_stiffeners[0])*1000.0*0.5*self.Hull_Dict['B'] #Distance between stiffeners in mm
        if params[34] > delta_y: #If the web width is greater than the distance between stiffeners, then we need to set the flange width to the distance between stiffeners
            params[34] = delta_y
        if params[38] > delta_y: #If the web width is greater than the distance between stiffeners, then we need to set the flange width to the distance between stiffeners
            params[38] = delta_y

        #Set up the Deck Longitudinal Girders and Stiffeners
        self.Deck_Long_Dict = [{
                            'y': self.y_deck_girders,
                            'h' : params[32],
                            't' : params[33],
                            'w' : params[34],
                            'rot': 180,
                            'dir': [1,0,0],
                            'bit_TorC': params[35],
                            '1D_element': False, #1D element is False, because it is a major structural member, not a stiffener
                            'Class': 'Deck_Longitudinal_Girder'},
                            {
                            'y': self.y_deck_stiffeners,
                            'h' : params[36],
                            't' : params[37],
                            'w' : params[38],
                            'rot': 180,
                            'dir': [1,0,0],
                            'bit_TorC': params[39],
                            '1D_element': True, #1D element is True, because it is a minor structural member
                            'Class': 'Deck_Longitudinal_Stiffener'}]
        
        self.bit_Hatch = params[116] #Boolean to determine if there are hatch openings in the deck
        #put lightening holes in the center of each hold
       
        self.x_LH = [(self.x_bulkheads[0] + self.x_bulkheads[1])/2.0,
                                (self.x_bulkheads[1] + self.x_bulkheads[2])/2.0,
                                (self.x_bulkheads[2] + self.x_bulkheads[3])/2.0]
        
        self.Deck_Lightening_Hole_Dict = [{"x1" : [float(val) for val in self.x_LH], # Fraction of major axis of stiffener
             "x2" : 0.0, # Fraction of minor axis of stiffener
             "l": float(params[117]*params[0]*1000.0), #millimeters along stiffener major axis
             "h": float(params[118]*params[1]*1000.0), #millimeters along stiffener minor axis
             "r": float(params[119])}]  #radius of fillet in millimeters
        
        self.Deck_Hatch_Half_width = params[116]*params[118]*self.Hull_Dict['B']*0.5*1000.0 #Half width of the hatch opening in meters

        #Set up the Transverse Bulkhead Stiffeners
        if params[42] > params[40]: #If the flange width is greater than
            params[42] = params[40] #the height of the stiffener, then we need to set the flange width to the height of
        
        delta_z = (self.z_trans_stiffeners[1] - self.z_trans_stiffeners[0])*1000.0*self.Hull_Dict['D'] #Distance between stiffeners in mm
        if params[42] > delta_z: #If
            params[42] = delta_z #the web width is greater than the distance between stiffeners, then we need to set the flange width to the distance between stiffeners
        
        self.BLKHD_Trans_Stiff_Dict = [{
                                        'z': self.z_trans_stiffeners,
                                        'h' : params[40],
                                        't' : params[41],
                                        'w' : params[42],
                                        'rot': 0,
                                        'dir': [0,1,0],
                                        'bit_TorC': params[43],
                                        '1D_element': True, #1D element is True, because it is a stiffener, not a major structural member
                                        'Class': 'Transverse_Bulkhead_Transverse_Stiffener'}]
       
        if params[46] > params[44]: #If the flange width is greater than the height of the stiffener, then we need to set the flange width to the height of the stiffener
            params[46] = params[44]

        delta_y = (self.y_vert_stiffeners[1] - self.y_vert_stiffeners[0])*1000.0*0.5*self.Hull_Dict['B'] #Distance between stiffeners in mm
        if params[46] > delta_y: #If the web width is greater than the distance between stiffeners, then we need to set the flange width to the distance between stiffeners
            params[46] = delta_y
        
        #Set up the Vertical Bulkhead Stiffeners
        self.BLKHD_Vert_Stiff_Dict = [{
                                        'y': self.y_vert_stiffeners,
                                        'h' : params[44],
                                        't' : params[45],
                                        'w' : params[46],
                                        'rot': 0,
                                        'dir': [0,0,1],
                                        'bit_TorC': params[47],
                                        '1D_element': True, #1D element is True, because it is a stiffener, not a major structural member
                                        'Class': 'Transverse_Bulkhead_Vertical_Stiffener'}]
        if params[114] > params[112]: #If the flange width is greater than the height of the stiffener, then we need to set the flange width to the height of the stiffener
            params[114] = params[112]

        #Set up the Longitudinal Bulkhead Stiffeners

        delta_x = (self.x_trans_frames[1] - self.x_trans_frames[0])*1000.0*self.Hull_Dict['L_3h'] #Distance between frames in mm
        if params[114] > delta_x: #If the
            params[114] = delta_x #web width is greater than the distance between frames, then we need to set the flange width to the distance between frames

        self.LONG_BLKHD_Vert_Stiff_Dict = [{
                                        'x': self.x_trans_frames,
                                        'h' : params[112],
                                        't' : params[113],
                                        'w' : params[114],
                                        'rot': 90,
                                        'dir': [0,0,1],
                                        'bit_TorC': params[115],
                                        '1D_element': True, #1D element is True, because it is a stiffener, not a major structural member
                                        'Class': 'Longitudinal_Bulkhead_Vertical_Stiffener'},
                                        {
                                        'x': self.x_trans_web_frames,
                                        'h' : params[112],
                                        't' : params[113],
                                        'w' : params[114],
                                        'rot': 90,
                                        'dir': [0,0,1],
                                        'bit_TorC': params[115],
                                        '1D_element': True, #1D element is True, because it is a stiffener, not a major structural member
                                        'Class': 'Longitudinal_Bulkhead_Vertical_Stiffener'}]
        
        if params[110] > params[108]: #If the flange width is greater than the height of the stiffener, then we need to set the flange width to the height of the stiffener
            params[110] = params[108]
        
        delta_z = (self.z_side_stiffeners[1] - self.z_side_stiffeners[0])*1000.0*self.Hull_Dict['D'] #Distance between stiffeners in mm
        if params[110] > delta_z: #If the web width is greater than the distance between stiffeners, then we need to set the flange width to the distance between stiffeners
            params[110] = delta_z
        
        
        self.LONG_BLKHD_Long_Stiff_Dict = [{
                                        'z': self.z_side_stiffeners,
                                        'h' : params[108],
                                        't' : params[109],
                                        'w' : params[110],
                                        'rot': 90,
                                        'dir': [1,0,0],
                                        'bit_TorC': params[111],
                                        '1D_element': True, #1D element is True, because it is a stiffener, not a major structural member
                                        'Class': 'Longitudinal_Bulkhead_Longitudinal_Stiffener'}]

        #Set up the Web Frames
        if params[50] > params[48]: #If the flange width is greater than
            params[50] = params[48] #the height of the stiffener, then we need to set the flange width to the height of the stiffener
        
        delta_x = (self.x_trans_frames[1] - self.x_trans_frames[0])*1000.0*self.Hull_Dict['L_3h'] #Distance between web frames in mm
        if params[50] > delta_x: #If the web width is greater than the distance between web frames, then we need to set the flange width to the distance between web frames
            params[50] = delta_x

        self.Web_Frame_Dict = {
                                'x': self.x_trans_web_frames,
                                'Side_Dict': {
                                    'h' : params[48],
                                    't' : params[49],
                                    'w' : params[50],
                                    'rot': -90,
                                    'dir': [0,0,1],
                                    'bit_TorC': params[51],
                                    '1D_element': False, #1D element is True, web frames are major structural members, not stiffeners
                                    'Class': 'Vertical_Web_Frame'},
    
                                'Bottom_Dict': {
                                    't' : params[52],
                                    '1D_element': False, #1D element is False, because it is a plate, not a stiffener
                                    'Class': 'Transverse_Floor'}}
        
        # Set up the Transverse Frames

        if params[99] > params[97]: #If the flange width is greater than the height of the stiffener, then we need to set the flange width to the height of the stiffener
            params[99] = params[97]
        if params[99] > delta_x: #If the web width is greater than the distance between frames, then we need to set the flange width to the distance between frames
            params[99] = delta_x

        

        self.Trans_Frame_Dict = {
                                'x': self.x_trans_frames,
                                'h' : params[97],
                                't' : params[98],
                                'w' : params[99],
                                'rot': -90,
                                'dir': [0,0,1],
                                'bit_TorC': params[100],
                                '1D_element': True, #1D element is True, because it is a stiffener, not a major structural member
                                'Class': 'Vertical_Side_Frame'}
        
        if params[95] > params[93]: #If the flange width is greater than the height of the stiffener, then we need to set the flange width to the height of the stiffener
            params[95] = params[93]
        if params[95] > delta_x: #If the web width is greater than the distance between frames, then we need to set the flange width to the distance between frames
            params[95] = delta_x

        
        self.Trans_Frame_Bottom_Dict = {
                                'x': self.x_trans_frames,
                                'h' : params[93],
                                't' : params[94],
                                'w' : params[95],
                                'rot': 90,
                                'dir': [0,1,0],
                                'bit_TorC': params[96],
                                '1D_element': True, #1D element is True, because it is a stiffener, not a major structural member
                                'Class': 'Transverse_Bottom_Frame'}
        
        if params[55] > params[53]: #If the flange width is greater than the height of the stiffener, then we need to set the flange width to the height of the stiffener
            params[55] = params[53]
        
        delta_z = (self.z_side_stiffeners[1] - self.z_side_stiffeners[0])*1000.0*self.Hull_Dict['D'] #Distance between stiffeners in mm
        if params[55] > delta_z: #If the web width is greater than the distance between stiffeners, then we need to set the flange width to the distance between stiffeners
            params[55] = delta_z

        self.Inner_Side_Shell_Long_Stiff_Dict = [{
                                'z': self.z_side_stiffeners,
                                'h' : params[53],
                                't' : params[54],
                                'w' : params[55],
                                'rot': 90,
                                'dir': [1,0,0],
                                'bit_TorC': params[56],
                                '1D_element': True, #1D element is True, because it is a stiffener, not a major structural member
                                'Class': 'Inner_Side_Shell_Longitudinal_Stiffener'}]
        
        #Inner side shell vertical stiffeners are all set from prior corrections
        self.Inner_Side_Shell_Vert_Stiff_Dict = [{      
                                'x': self.x_trans_frames,
                                'h' : params[97],
                                't' : params[98],
                                'w' : params[99],
                                'rot': 90,
                                'dir': [0,0,1],
                                'bit_TorC': params[100],
                                '1D_element': True, #1D element is True, because it is a stiffener, not a major structural member
                                'Class': 'Inner_Side_Shell_Vertical_Stiffener'}]
        
        self.Side_Shell_Stiff_Dict = {
                                'z': self.z_side_stiffeners,
                                'h' : params[53],
                                't' : params[54],
                                'w' : params[55],
                                'rot': -90,
                                'dir': [1,0,0],
                                'bit_TorC': params[56],
                                '1D_element': True, #1D element is True, because it is a stiffener, not a major structural member
                                'Class': 'Side_Shell_Longitudinal_Stiffener'}
        
        self.Ship_Class = params[57:60] #One hotkey for each ship class


        if self.Ship_Class[0] == 1: 
            # Add Brackets for Tanker

            fw_vertices = np.zeros((len(self.x_trans_web_frames), 3)) #vertices of the intersection of the floor beams and the transverse frames
            fw_vertices[:,0] = self.x_trans_web_frames*self.Hull_Dict['L_3h']
            fw_vertices[:,1] = self.Hull_Dict["B"]/2 - self.Web_Frame_Dict['Side_Dict']['h']/1000.0
            fw_vertices[:,2] = self.Struct_Params["Db"]/1000.0

            dw_vertices = np.zeros((len(self.x_trans_web_frames), 3))
            #vertices of the intersection of the deck beams and the transverse frames
            dw_vertices[:,0] = self.x_trans_web_frames*self.Hull_Dict['L_3h']
            dw_vertices[:,1] = self.Hull_Dict["B"]/2 - self.Web_Frame_Dict['Side_Dict']['h']/1000.0
            dw_vertices[:,2] = self.Hull_Dict["D"]- self.Deck_Beam_Dict[0]['h']/1000.0

            #Adjust params[60] to land on the stiffener or girder FW Bracket
            y_struct = np.concatenate((self.y_bottom_stiffeners, self.y_bottom_girders))
            y_struct.sort() 
           
            y_struct = y_struct*1000.0*0.5*self.Hull_Dict['B']#Combine the bottom stiffeners and girders locations and set to mm
            y_L1_end = fw_vertices[0,1]*1000.0 - params[60] #Calculate the end of the L1 line in mm
            # Find first y_struct that is less than y_L1_end
            try:
                idx_stiff = np.where(y_struct <= y_L1_end)[0][-1] #Get the last index where y_struct is less than y_L1_end
            except:
                idx_stiff = 0 #If there are no stiffeners or girders less than y_L1_end, then we will land on the first stiffener or girder
            params[60] = fw_vertices[0,1]*1000.0 - y_struct[idx_stiff] #Set params[60] to the distance from the end of the L1 line to the nearest stiffener or girder

            #Land L2 on a side stiffener FW Bracket
            z_struct = self.z_side_stiffeners*1000.0*self.Hull_Dict['D'] #Set z_struct to mm
            z_L2_end = fw_vertices[0,2]*1000.0 + params[61] #Calculate the end of the L2 line in mm
            # Find first z_struct that is greater than z_L2_end
            try:
                idx_stiff = np.where(z_struct >= z_L2_end)[0][0] #Get the first index where z_struct is greater than z_L2_end
            except:
                idx_stiff = len(z_struct)-1 #If there are no stiffeners greater than z_L2_end, then we will land on the last stiffener
            params[61] = z_struct[idx_stiff] - fw_vertices[0,2]*1000.0 #Set params[61] to the distance from the end of the L2 line to the nearest stiffener
            
            #Land L1 on a deck stiffener DW Bracket and outboard of deck hatch opening
            y_struct = np.concatenate((self.y_deck_stiffeners, self.y_deck_girders)) * 1000.0*0.5*self.Hull_Dict['B'] #Combine the deck stiffeners and girders locations and set to mm
            y_struct.sort() #Sort the y_struct array
            y_L1_end = dw_vertices[0,1]*1000.0 - params[63] #Calculate the end of the L1 line in mm
            # Find first y_struct that is less than y_L1_end
            try:
                idx_stiff = np.where(y_struct <= y_L1_end)[0][-1] #Get the last index where y_struct is less than y_L1_end
            except:
                idx_stiff = 0 #If there are no stiffeners or girders less than y_L1_end, then we will land on the first stiffener or girder
             #Set params[63] to the distance from the end of the L1 line to the nearest stiffener or girder
            if y_struct[idx_stiff] < self.Deck_Hatch_Half_width: # If the stifenner position is inboard of the deck hatch opening, then the bracket needs to be supported by a stiffener inboard of the deck hatch opening
                idx_stiff = np.where(y_struct >= self.Deck_Hatch_Half_width)[0][0] #Get the first index where y_struct is greater than the deck hatch opening
            params[63] = dw_vertices[0,1]*1000.0 - y_struct[idx_stiff]
            

            #Land L2 on a side stiffener DW Bracket
            z_struct = self.z_side_stiffeners*1000.0*self.Hull_Dict['D'] #Set z_struct to mm
            z_L2_end = dw_vertices[0,2]*1000.0 - params[64] #Calculate the end of the L2 line in mm
            # Find first z_struct that is greater than z_L2_end
            try:
                idx_stiff = np.where(z_struct <= z_L2_end)[0][-1]
            except:
                idx_stiff = 0
            #Get the last index where z_struct is less th than z_L2_end
            params[64] = dw_vertices[0,2]*1000.0 - z_struct[idx_stiff] #Set params[64] to the distance from the end of the L2 line to the nearest stiffener

            self.Bracket_Dict = [{"Vertex":fw_vertices, #Brackets between the deck beams and the transverse frames
                                  'L1': np.array([0,-params[60],0]),
                                  'L2': np.array([0,0,params[61]]),
                                  't': params[62],
                                  'num_sides': 3,
                                  'Class': 'Bracket_Floor_Web'},
                                    {"Vertex":dw_vertices, #Brackets between the deck beams and the transverse frames
                                    'L1': np.array([0,-params[63],0]),
                                    'L2': np.array([0,0,-params[64]]),
                                    't': params[65], 
                                    'num_sides': 3,
                                    'Class': 'Bracket_Deck_Web'}]
            
            
        elif self.Ship_Class[1] == 1:
            #Add Brackets for Container Ship
            fw_vertices = np.zeros((len(self.x_trans_web_frames), 3))
            #vertices of the intersection of the floor beams and the transverse frames
            fw_vertices[:,0] = self.x_trans_web_frames*self.Hull_Dict['L_3h']
            fw_vertices[:,1] = self.Hull_Dict["B"]/2 - self.Web_Frame_Dict['Side_Dict']['h']/1000.0
            fw_vertices[:,2] = self.Struct_Params["Db"]/1000.0 #unit is m

            L1_brack = params[67]  #unit is mm Step width is inboard of the web frame
            # Check L1 brack lands on a stiffener or girder, if not, adjust to land on nearest stiffener or girder
            y_struct = np.concatenate((self.y_bottom_stiffeners, self.y_bottom_girders)) #Combine the bottom stiffeners and girders locations
            y_struct.sort() #Sort the y_struct array


            y_struct = y_struct*1000.0*0.5*self.Hull_Dict['B'] #Set y_struct to mm
            y_L1_end = fw_vertices[0,1]*1000.0 - L1_brack #Calculate the end of the L1 line in mm
            # Find first y_struct that is less than y_L1_end
            try:
                idx_stiff = np.where(y_struct <= y_L1_end)[0][-1] #Get the last index where y_struct is less than y_L1_end
            except:
                idx_stiff = 0 #If y_L1_end is less than all y_struct, then set idx_stiff to the first index
            L1_brack = fw_vertices[0,1]*1000.0 - y_struct[idx_stiff] #Set L1_brack to the distance from the end of the L1 line to the nearest stiffener or girder

            params[67] = L1_brack #Update params[67] to the adjusted L1_brack value

            L2_brack = params[66] #unit is mm Step height is above the double bottom
            # Check L2 brack lands on a stiffener, if not, adjust to land on nearest stiffener
            z_struct = self.z_side_stiffeners*1000.0*self.Hull_Dict['D'] #Set z_struct to mm
            z_L2_end = fw_vertices[0,2]*1000.0 + L2_brack #Calculate the end of the L2 line in mm
            # Find first z_struct that is greater than z_L2_end
            try:
                idx_stiff = np.where(z_struct >= z_L2_end)[0][0] #Get the first index where z_struct is greater than z_L2_end
            except:
                idx_stiff = len(z_struct) - 1 #If z_L2_end is greater than all z_struct, then set idx_stiff to the last index

            L2_brack = z_struct[idx_stiff] - fw_vertices[0,2]*1000.0 #Set L2_brack to the distance from the end of the L2 line to the nearest
            params[66] = L2_brack #Update params[66] to the adjusted L2_brack value


            L1_pannel = params[67] + self.Web_Frame_Dict['Side_Dict']['h'] #unit is mm pannel width is width of stringer deck + web frame height
            L2_pannel = params[66] + self.Struct_Params["Db"]#unit is mm pannel height is height of stringer deck + double bottom height

            self.Bracket_Dict = [{"Vertex":fw_vertices, #Brackets between the deck beams and the transverse frames
                                    'L1': np.array([0,-L1_brack,0]),
                                    'L2': np.array([0,0,L2_brack]),
                                    't': self.Web_Frame_Dict['Side_Dict']['t'], #Maintain the same thickness as the web frame
                                    'num_sides': 4,
                                    'Class': 'Bracket_Floor_Web'}]
            

            y_pos = self.Hull_Dict["B"]/2 - self.Web_Frame_Dict['Side_Dict']['h']/1000.0 - params[67]/1000.0 #y position of the start of the stringer deck
            z_pos = self.Struct_Params["Db"]/1000.0 + params[66]/1000.0 #z position of the start of the stringer deck
            start_horiz = np.array([0,y_pos, z_pos]) #Start position of the stringer deck in mm
            start_vert = np.array([0,y_pos, 0])   


            #Make Container Ship Structure Dicts:
            if params[73] > params[71]: #If the flange width is greater than the height of the stiffener, then we need to set the flange width to the height of the stiffener
                params[73] = params[71]
            if params[78] > params[76]: #If the flange width is greater than the height of the stiffener, then we need to set the flange width to the height of the stiffener
                params[78] = params[76]

            delta_s = L1_pannel/(params[70]+1) #Distance between stiffeners in mm
            if params[73] > delta_s: #If the web width is greater than the distance between stiffeners, then we need to set the flange width to the distance between stiffeners
                params[73] = delta_s
            
            delta_s = L2_pannel/(params[75]+1) #Distance between stiffeners in mm
            if params[78] > delta_s: #If the web width is greater than the distance between stiffeners, then we need to set the flange width to the distance between stiffeners
                params[78] = delta_s
            
            self.ContainerShip_Pannel_Dict = [{
                                't_pan' : params[68],
                                'start': start_horiz,
                                'L1': L1_pannel,
                                'L2': 0,
                                'num_stiffeners': params[70],
                                'h' : params[71],
                                't': params[72],
                                'w': params[73],
                                'rot': 90,
                                'bit_TorC': params[74],
                                '1D_element': True, #1D element is True, because it is a stiffener, not a major structural member
                                'Class': 'Stringer_Deck_Pannel'},
                                {
                                't_pan' : params[69],
                                'start': start_vert,
                                'L1': 0,
                                'L2': L2_pannel,
                                'num_stiffeners': params[75],
                                'h' : params[76],
                                't': params[77],
                                'w': params[78],
                                'rot': 90,
                                'bit_TorC': params[79],
                                '1D_element': True,
                                'Class': 'Stringer_Deck_Side_Pannel'
                                }]
            
            
            self.Container_Deck_Dict = {
                                'z': np.array([1.0]),
                                'h' : params[48], 
                                't' : params[9],
                                'w' : 0,
                                'rot': -90,
                                'dir': [1,0,0],
                                'bit_TorC': 0,
                                '1D_element': False,
                                'Class': 'Deck'} #1D element is False, because it is a major structural member, not a stiffener

            
        elif self.Ship_Class[2] == 1:
            #Add Brackets for Bulk Carrier Ship
                        
            fw_vertices = np.zeros((len(self.x_trans_web_frames), 3)) #vertices of the intersection of the floor beams and the transverse frames
            fw_vertices[:,0] = self.x_trans_web_frames*self.Hull_Dict['L_3h']
            fw_vertices[:,1] = self.Hull_Dict["B"]/2 - self.Web_Frame_Dict['Side_Dict']['h']/1000.0
            fw_vertices[:,2] = self.Struct_Params["Db"]/1000.0

            dw_vertices = np.zeros((len(self.x_trans_web_frames), 3))
            #vertices of the intersection of the deck beams and the transverse frames
            dw_vertices[:,0] = self.x_trans_web_frames*self.Hull_Dict['L_3h']
            dw_vertices[:,1] = self.Hull_Dict["B"]/2 - self.Web_Frame_Dict['Side_Dict']['h']/1000.0
            dw_vertices[:,2] = self.Hull_Dict["D"] # In Bulk Carrier, the upper hopper intersects the deck 

            #Adjust params[60] to land on the stiffener or girder FW Bracket
            y_struct = np.concatenate((self.y_bottom_stiffeners, self.y_bottom_girders))
            y_struct.sort() 
            
            y_struct = y_struct*1000.0*0.5*self.Hull_Dict['B']#Combine the bottom stiffeners and girders locations and set to mm
            y_L1_end = fw_vertices[0,1]*1000.0 - params[60] #Calculate the end of the L1 line in mm
            # Find first y_struct that is less than y_L1_end
            try:
                idx_stiff = np.where(y_struct <= y_L1_end)[0][-1] #Get the last index where y_struct is less than y_L1_end
            except:
                idx_stiff = 0 #If y_L1_end is less than all y_struct, then set idx_stiff to the first index

            params[60] = fw_vertices[0,1]*1000.0 - y_struct[idx_stiff] #Set params[60] to the distance from the end of the L1 line to the nearest stiffener or girder

            #Land L2 on a side stiffener FW Bracket
            z_struct = self.z_side_stiffeners*1000.0*self.Hull_Dict['D'] #Set z_struct to mm
            z_L2_end = fw_vertices[0,2]*1000.0 + params[61] #Calculate the end of the L2 line in mm
            # Find first z_struct that is greater than z_L2_end
            try:
                idx_stiff = np.where(z_struct >= z_L2_end)[0][0] #Get the first index where z_struct is greater than z_L2_end
                params[61] = z_struct[idx_stiff] - fw_vertices[0,2]*1000.0 #Set params[61] to the distance from the end of the L2 line to the nearest stiffener
            except:
                #Get the last index where z_struct is less than z_L2_end
                idx_stiff = np.where(z_struct <= z_L2_end)[0][-1]   
                params[61] = z_struct[idx_stiff] - fw_vertices[0,2]*1000.0 #Set params[61] to the distance from the end of the L2 line to the nearest stiffener
            #Land L1 on a deck stiffener DW Bracket and outboard of the deck hatch opening.

            y_struct = np.concatenate((self.y_deck_stiffeners, self.y_deck_girders)) * 1000.0*0.5*self.Hull_Dict['B'] #Combine the deck stiffeners and girders locations and set to mm
            y_struct.sort() #Sort the y_struct array
            y_L1_end = dw_vertices[0,1]*1000.0 - params[63] #Calculate the end of the L1 line in mm
            # Find first y_struct that is less than y_L1_end
            try:
                idx_stiff = np.where(y_struct <= y_L1_end)[0][-1] #Get the last index where y_struct is less than y_L1_end
            except:
                idx_stiff = 0 #If y_L1_end is less than all y_struct, then set idx_stiff to the first index

            if y_struct[idx_stiff] < self.Deck_Hatch_Half_width: # If the stifenner position is inboard of the deck hatch opening, then the bracket needs to be supported by a stiffener inboard of the deck hatch opening
                idx_stiff = np.where(y_struct >= self.Deck_Hatch_Half_width)[0][0] #Get the first index where y_struct is greater than the deck hatch opening

            params[63] = dw_vertices[0,1]*1000.0 - y_struct[idx_stiff] #Set params[63] to the distance from the end of the L1 line to the nearest stiffener or girder

            #Land L2 on a side stiffener DW Bracket
            z_struct = self.z_side_stiffeners*1000.0*self.Hull_Dict['D'] #Set z_struct to mm
            z_L2_end = dw_vertices[0,2]*1000.0 - params[64] #Calculate the end of the L2 line in mm
            # Find first z_struct that is greater than z_L2_end
            try:
                idx_stiff = np.where(z_struct <= z_L2_end)[0][-1]
            except:
                idx_stiff = 0 #If z_L2_end is less than all z_struct, then set idx_stiff to the first index
            #Get the last index where z_struct is less th than z_L2_end
            params[64] = dw_vertices[0,2]*1000.0 - z_struct[idx_stiff] #Set params[64] to the distance from the end of the L2 line to the nearest stiffener

            #Make BulkCarrier Structure Dicts:
            start_fw = np.array([0,self.Hull_Dict["B"]/2 - self.Web_Frame_Dict['Side_Dict']['h']/1000.0-params[60]/1000.0, self.Struct_Params["Db"]/1000.0])
            start_dw = np.array([0,self.Hull_Dict["B"]/2 - self.Web_Frame_Dict['Side_Dict']['h']/1000.0-params[63]/1000.0, self.Hull_Dict["D"]])
            
            
            self.Bracket_Dict = [{"Vertex":fw_vertices, #Brackets between the deck beams and the transverse frames
                        'L1': np.array([0,-params[60],0]),
                        'L2': np.array([0,0,params[61]]),
                        't': params[62],
                        'num_sides': 3,
                        'Class': 'Bracket_Floor_Web'},
                        {"Vertex":dw_vertices, #Brackets between the deck beams and the transverse frames
                        'L1': np.array([0,-params[63],0]),
                        'L2': np.array([0,0,-params[64]]),
                        't': params[65], 
                        'num_sides': 3,
                        'Class': 'Bracket_Deck_Web'}]

            if params[85] > params[83]: #If the flange width is greater than the height of the stiffener, then we need to set the flange width to the height of the stiffener
                params[85] = params[83]
            if params[90] > params[88]: #If the flange width is greater than the height of the stiffener, then we need to set the flange width to the height of the stiffener
                params[90] = params[88]
            
            delta_s = np.sqrt((params[60]**2 + params[61]**2)) / (params[82]+1) #Distance between stiffeners in mm
            if params[85] > delta_s: #If the web width is greater than the distance between stiffeners, then we need to set the flange width to the distance between stiffeners
                params[85] = delta_s
            
            delta_s = np.sqrt((params[63]**2 + params[64]**2)) / (params[87]+1) #Distance between stiffeners in mm
            if params[90] > delta_s: #If the web width is greater than the distance between stiffeners, then we need to set the flange width to the distance between stiffeners
                params[90] = delta_s

            self.BulkCarrier_Hopper_Dict = [{
                                            't_pan' : params[80],
                                            'start': start_fw,
                                            'L1': params[60],
                                            'L2': params[61],
                                            'num_stiffeners': params[82],
                                            'h' : params[83],
                                            't': params[84],
                                            'w': params[85],
                                            'rot': 90,
                                            'bit_TorC': params[86],
                                            '1D_element': True, #1D element is True, because it is a stiffener, not a major structural member
                                            'Class': 'Hopper_Pannel'},
                                            {
                                            't_pan' : params[81],
                                            'start': start_dw,
                                            'L1': params[63],
                                            'L2': -params[64],
                                            'num_stiffeners': params[87],
                                            'h' : params[88],
                                            't': params[89],
                                            'w': params[90],
                                            'rot': -90,
                                            'bit_TorC': params[91],
                                            '1D_element': True,
                                            'Class': 'Hopper_Pannel'}]
        #Assign Modified Parameters
        self.params = params

    

    def make_Structure(self):
        start_time = time.time()
        #Make the Top Deck: 
        if self.Ship_Class[1] == 0: # Means that the sturcture is not a container ship
            if self.bit_Hatch == 1:
                Deck_Params = {'z': self.Hull_Dict['D'],'t': self.Struct_Params['Top_Deck_Thickness'], 'LH_Dict': self.Deck_Lightening_Hole_Dict}

            else: 
                Deck_Params = {'z': self.Hull_Dict['D'],'t': self.Struct_Params['Top_Deck_Thickness']}
            
            self.struct.make_Deck(Deck_Params, Long_Stiffeners=self.Deck_Long_Dict,Trans_Stiffeners=self.Deck_Beam_Dict)
        else: 
            self.struct.make_Long_Side_Stiffener(self.Container_Deck_Dict) #Deck only covers the web frame height, leaves deck open

        #Make the Inner Bottom Deck:
        Inner_Bottom_Params = {'z': self.Struct_Params['Db']/1000.0,'t': self.Struct_Params['Inner_Bottom_Thickness']}
        self.struct.make_Deck(Inner_Bottom_Params, Long_Stiffeners=self.Inner_Bottom_Stiffener_Dict, Trans_Stiffeners=self.Inner_Bottom_Trans_Frame_Dict)

        #Make the Bottom Girder
        self.struct.make_Long_Bottom_Stiffener(self.Bottom_Girder_Dict)
        #Make the Bottom Stiffeners
        self.struct.make_Long_Bottom_Stiffener(self.Bottom_Stiffener_Dict)


        #Make the Transverse Bulkheads
        Bulkhead_Params = {'x': self.x_bulkheads, 't': self.Struct_Params['Bulkhead_Thickness']}
        self.struct.make_Trans_Bulkhead(Bulkhead_Params,Trans_Stiffeners=self.BLKHD_Trans_Stiff_Dict, Vert_Stiffeners=self.BLKHD_Vert_Stiff_Dict)

        #Make the Side Shell Stiffeners
        self.struct.make_Long_Side_Stiffener(self.Side_Shell_Stiff_Dict)
        
        #Make the Inner Side Shell
        if self.Struct_Params['Inner_Side_Shell_Thickness'] > 0:
            y_is = (self.Hull_Dict["B"]/2 - self.Web_Frame_Dict['Side_Dict']['h']/1000.0)/(self.Hull_Dict["B"]/2)
            Inner_Side_Shell_Params = {'y': y_is, 't': self.Struct_Params['Inner_Side_Shell_Thickness']}
            self.struct.make_Long_Bulkhead(Inner_Side_Shell_Params, Long_Stiffeners=self.Inner_Side_Shell_Long_Stiff_Dict, Vert_Stiffeners=self.Inner_Side_Shell_Vert_Stiff_Dict)
        
        if self.bit_Long_Bulkheads == 1:
            #Make the Longitudinal Bulkheads
            self.struct.make_Long_Bulkhead(self.Long_Bulkhead_Params, Long_Stiffeners=self.LONG_BLKHD_Long_Stiff_Dict, Vert_Stiffeners=self.LONG_BLKHD_Vert_Stiff_Dict)
    
        
        #Make the Shear Strake
        Strake_Params = {'z': self.Hull_Dict['D'] - self.Shear_Strake_Dict['h']/1000.0, 't': self.Shear_Strake_Dict['t'], 'h': self.Shear_Strake_Dict['h']}
        self.struct.make_Shell_Strake(Strake_Params)

        #Make the Transverse Web Frames
        self.struct.make_Trans_Web_Frames(self.Web_Frame_Dict)
        
        #Make the Transverse Frames and Intermediate Bottom Frames
        self.struct.make_Trans_Frames(self.x_trans_frames,self.Trans_Frame_Dict, self.Trans_Frame_Bottom_Dict)

        self.struct.make_Brackets(self.Bracket_Dict)
        
        if self.Ship_Class[1] == 1:
            self.struct.make_Pannel_Surface(self.ContainerShip_Pannel_Dict)

        elif self.Ship_Class[2] == 1:
            self.struct.make_Pannel_Surface(self.BulkCarrier_Hopper_Dict)


        self.struct.calc_Areas_and_Centroids()
       
        self.correct_1D_Elements() #Correct 1D elements to ensure correct meshing in gmsh

      
        self.struct.compile_Structure()
        

        #print('')
        #print('CAD Time: ' + str(cad_time - start_time) + ' seconds')
        #print('CAD Time without hopper: ' + str(cad_noHop - start_time) + ' seconds')
        #print('Evaluation Time: ' + str(eval_time - cad_time) + ' seconds')
        #print('Correction Time: ' + str(cor_time - eval_time) + ' seconds')
        #print('Save Time: ' + str(save_time - cor_time) + ' seconds')

    def correct_1D_Elements(self):
        '''
        This function corrects the 1D elements to ensure that they are correctly meshed in gmsh.
        this only applies to the 1D elements to the Class == Transverse_Bottom_Frame
        '''

        #first loop thru  structural_elements these are line items where bit_1D_element is True and mesh_rsObject is not nan

        ilocs = self.struct.Structural_Elements.index[self.struct.Structural_Elements['Class'] == 'Transverse_Bottom_Frame'].tolist()
        #Change layer to mesh layer 
        cur_layer = rs.CurrentLayer()
        rs.CurrentLayer(self.struct.mesh_layer)

        items = []
        idx_remove = []
        for i in ilocs:
            mesh_id = self.struct.Structural_Elements.loc[i,'mesh_rsObject']
            length = rs.CurveLength(mesh_id)
            curve_domain = rs.CurveDomain(mesh_id)
            split_1 = curve_domain[1]*(self.Hull_Dict['B']/2.0 - self.Hull_Dict['R_b'])/length
            split_2 = 1.0 - (0.001*self.Struct_Params['Db']-self.Hull_Dict['R_b'])/length

            if split_2 == 1.0:
            
                params = [split_1]
            else:
                split_2 = curve_domain[1]*split_2
                params = [split_1, split_2]
            new_mesh_ids = rs.SplitCurve(mesh_id, params)

            #print('exploding: ' + self.struct.Structural_Elements.loc[i, 'Object_ID'])
            #print('new_mesh_ids: ' + str(new_mesh_ids))
            if len(new_mesh_ids) <= 1:
                
                rs.DeleteObjects(new_mesh_ids)
            else: 
                idx_remove.append(i)
                # First Get row from structural elements
                row = self.struct.Structural_Elements.loc[i].copy()

                #print('exploding: ' + self.struct.Structural_Elements.loc[i, 'Object_ID'])

                for j in range(len(new_mesh_ids)):
                    #add row to items are replace mesh_rsObject with new_mesh_id[j] and rename object_id to include j
                    new_row = row.copy()
                    rs.SimplifyCurve(new_mesh_ids[j])
                    new_row['mesh_rsObject'] = new_mesh_ids[j]
                    new_row['Object_ID'] = new_row['Object_ID'] + f'_part_{j}'
                    items.append(new_row)


        #Now add items to structural elements and remove idx_remove
        
        if len(items) > 0:
            new_items_df = pd.DataFrame(items)
            #delte the rows in idx_remove from structural elements
            self.struct.Structural_Elements.drop(idx_remove, inplace=True)
            self.struct.Structural_Elements = pd.concat([self.struct.Structural_Elements, new_items_df], ignore_index=True)
            #reindex the structural elements
            self.struct.Structural_Elements.reset_index(drop=True, inplace=True)

        rs.CurrentLayer(cur_layer) #Return to original layer
        rs.LayerVisible(self.struct.mesh_layer, False)

        #Delete items
        del items, idx_remove, new_items_df

    
    
    def calculate_Parametric_Constraints(params): 
        '''
        This function calculates constrains purely based on the parameters to save computation time:
        Here are the list of identified constraints:
        0: If there is a hatch opening, the hatch should not open wider than the brackets
        1) Hatvh opening should not be wider than the inboard edge of web frames

        '''
        constraints = np.zeros((len(params), 2))

        for i in range(len(params)):
            #constraint 0: If there is a hatch opening, the hatch should not open wider than the brackets
            if params[i,116] == 1:
                constraints[i,0] = 0.5*params[i,1]*params[i,118] - (0.5*params[i,1] - params[i,48]/1000.0 - params[i,63]/1000.0) #Hatch opening width should not be wider than the brackets
                constraints[i,1] = 0.5*params[i,1]*params[i,118] - (0.5*params[i,1] - params[i,48]/1000.0) #Hatch opening should not be wider than the inboard edge of web frames


        return constraints

    def gen_rnd_Sturctures(num_samples = 1000.0):
        '''
        This function generates a random structure based on the parameters in the list
        '''
        LL = Structure_3H.Param_Dict['LL'].to_numpy()
        UL = Structure_3H.Param_Dict['UL'].to_numpy()

        LL_strat = Structure_3H.Param_Dict['Strategic_LL'].to_numpy()
        UL_strat = Structure_3H.Param_Dict['Strategic_UL'].to_numpy()

        #Generate random parameters
        #LOA = np.random.uniform(120,360, num_samples) #Random Length of the ship in meters

        params = np.zeros((num_samples, len(Structure_3H.Param_Dict)))
        
        #Set up the first few parameters
        params[:,0] = np.random.uniform(LL[0], UL[0], num_samples)  # Length
        scale = np.random.uniform(3,4.5, num_samples) #Scale factor for the ship
        LOA = scale*params[:,0] #Scale the length of the ship
        params[:,1] = LOA*np.random.uniform(LL_strat[1], UL_strat[1], num_samples) # Beam

        params[:,3] = params[:,1]*np.random.uniform(LL_strat[3], UL_strat[3], num_samples) #Depth  Multiple of Beam

        params[:,2] = params[:,3]*np.random.uniform(LL_strat[2], UL_strat[2], num_samples) # Draft, Multiple if Depth
        
        params[:,6] = 1000.0*params[:,1]*np.random.uniform(LL_strat[6], UL_strat[6], num_samples) #Double Bottom Height, multiple of Beam

        params[:,4] = 0.001* params[:,6]*np.random.uniform(LL_strat[4], UL_strat[4], num_samples) #Bilge Radius, Multiple of Db height
        params[:,5] = np.random.uniform(LL[5], UL[5], num_samples) # overhang
        

        #Sample Remaining Params based on LL and UL
        params[:,7:] = np.random.uniform(LL[7:], UL[7:], (num_samples, len(Structure_3H.Param_Dict)-7))
        #Sample the boolean values
        params[:,Structure_3H.param_idx_bit] = np.random.randint(0,2, (num_samples, len(Structure_3H.param_idx_bit)))
        
        #Sample Longitudinal Bulkheads at 20% instead of 50% of the time
        long_bulkhead_rnd = np.random.uniform(0,1, num_samples)
        params[:,105] = long_bulkhead_rnd < 0.2 #20% of the time there is a longitudinal bulkhead


        #Sample the categorical values
        choice = np.random.choice(range(len(Structure_3H.param_idx_cat)), num_samples)
        cat = np.zeros((num_samples, len(Structure_3H.param_idx_cat)))
        #label the choice in cat
        for i in range(num_samples):
            cat[i,choice[i]] = 1
        params[:,Structure_3H.param_idx_cat] = cat

        print('Fraction of Tanker Designs: ' + str(sum(params[:,57])/num_samples))
        print('Fraction of Container Designs: ' + str(sum(params[:,58])/num_samples))
        print('Fraction of Bulk Carrier Designs: ' + str(sum(params[:,59])/num_samples))

        #Sample the integer values
        params[:,Structure_3H.param_idx_int] = np.random.randint(LL[Structure_3H.param_idx_int], UL[Structure_3H.param_idx_int]+1, (num_samples, len(Structure_3H.param_idx_int)))
        
        #Sample Thicknesses: 
        params[:,Structure_3H.param_idx_plate_thicknesses] = np.random.randint(LL[Structure_3H.param_idx_plate_thicknesses],
                                                                                  1 + UL[Structure_3H.param_idx_plate_thicknesses],  
                                                                                  (num_samples, len(Structure_3H.param_idx_plate_thicknesses)))
        zero_inner_side_shell = params[:,12] < 10.0
        params[zero_inner_side_shell,12] = 0

        print('Fraction of Designs without an inner side shell: ' + str(sum(zero_inner_side_shell)/num_samples))

        #Check Sampling of Brackets based in Category
        for i in range(num_samples):
            if params[i,57] == 1:
                params[i,Structure_3H.param_idx_brackets] = np.random.uniform(LL[Structure_3H.param_idx_brackets], UL[Structure_3H.param_idx_brackets], (1, len(Structure_3H.param_idx_brackets)))
            elif params[i,59] == 1:
                params[i,Structure_3H.param_idx_brackets] = np.random.uniform(LL_strat[Structure_3H.param_idx_brackets], UL_strat[Structure_3H.param_idx_brackets], (1, len(Structure_3H.param_idx_brackets)))

        #Clean Struct Params: 
        params = Structure_3H.clean_Struct_Params(params)

        #Convert to DataFrame
        df = pd.DataFrame(params, columns=Structure_3H.Param_Dict['name'].to_numpy())

        return df


    def clean_Struct_Params(params):

        for i in range(len(params)):
            
            params[i, Structure_3H.param_idx_bit] = params[i, Structure_3H.param_idx_bit] >= 0.5
            params[i, Structure_3H.param_idx_int] = np.int32(params[i, Structure_3H.param_idx_int]+0.5)
            params[i, Structure_3H.param_idx_plate_thicknesses] = np.int32(params[i, Structure_3H.param_idx_plate_thicknesses]+0.5)

            #Get argmax of categorical values to assign ship class
            ship_class = np.argmax(params[i,Structure_3H.param_idx_cat])
            params[i,57:60] = 0 #Reset ship class values
            params[i,57 + ship_class] = 1 #Set the ship class to the argmax value

            LL = Structure_3H.Param_Dict['LL'].to_numpy()
            UL = Structure_3H.Param_Dict['UL'].to_numpy()

            LL_strat = Structure_3H.Param_Dict['Strategic_LL'].to_numpy()
            UL_strat = Structure_3H.Param_Dict['Strategic_UL'].to_numpy()

            # Check Rb height: 

            if params[i,6] < 1000.0*params[i,4]: #If the double bottom height is less than the bilge radius, set the bilge radius to Db height
                params[i,4] = params[i,6]/1000.0


            #Check Categorical Values:
            if params[i,57] == 1: #Tanker
                params[i,Structure_3H.param_idx_bulkcarrier] = 0
                params[i,Structure_3H.param_idx_container] = 0

                # Ensure brackets are within range of Tanker limits
                for j in Structure_3H.param_idx_brackets:
                    params[i,j] = np.clip(params[i,j], LL[j], UL[j])



            elif params[i,58] == 1:
                params[i,Structure_3H.param_idx_bulkcarrier] = 0
                params[i,Structure_3H.param_idx_brackets] = 0

                # Params for container brackets are 80 and 81
                params[i,66] = np.clip(params[i,66], LL[66], UL[66])
                params[i,67] = np.clip(params[i,67], LL[67], UL[67])


            elif params[i,59] == 1:
                params[i,Structure_3H.param_idx_container] = 0

                for j in Structure_3H.param_idx_brackets:
                    params[i,j] = np.clip(params[i,j], LL_strat[j], UL_strat[j])
            
            #Check Inner Side wall Thickness
            if params[i,12] < 10.0:
                params[i,12] = 0
            else: 
                params[i,50] = 0 # If there is an inner side shell, no webframe flange cap. 

            #Chek Longitudinal Bulkhead
            if params[i,105] <= 0.5:
                params[i,105] = 0
                params[i,106:116] = 0.0
            elif params[i,106] <= 0.5:
                params[i,105:116] = 0.0
            else:
                params[i,105] = 1

            #check hatch openings
            if params[i,116] <= 0.5 or params[i,117] < 1e-2 or params[i,118] < 1e-2:
                params[i,116] = 0
                params[i,117:120] = 0.0
            else:
                params[i,116] = 1
        
        return params
           


    

            



        






       
