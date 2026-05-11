"""
This script contains functions to generate structural 

"""

import numpy as np
#import csv
#import json
import pandas as pd

import rhinoscriptsyntax as rs
#import Rhino
import time

class Ship_Structure_CAD:

    def __init__(self, Hull_Dict, Struct_Params = {},hull_shell = None, path = './', id='struct'):
        """
        Hull_Dict: a dictionary that contains the metrics of the ship
            "L_3h": Length of the 3 hold hull section in meters
            "l_overhang": fraction of L_3h that is over
            "B": Beam of the ship in meters
            "D": Depth of the ship in meters
            "T": Draft of the ship in meters
            "R_b": Bilge Radius in meters
        
        Struct_Params: dictionary of structural parameters
        some of the parameters are:
            "Db": Double Bottom Height in milimeters 
            "Bottom_Plate_Thickness": Thickness of the bottom shell plating in milimeters
 
        
        More to come
        
        
        """
        self.clear_Document()
        self.Hull_Dims = Hull_Dict
    
        self.Struct_Params = Struct_Params

        self.path = path
        self.id = id

        rs.UnitSystem(unit_system=4) # set units to meters

        rs.EnableRedraw(False)
        self.struct_layer = 'Main Structure'
        if not rs.IsLayer(self.struct_layer):
            rs.AddLayer(self.struct_layer)

        self.mesh_layer = 'Mesh Structure'
        if not rs.IsLayer(self.mesh_layer):
            rs.AddLayer(self.mesh_layer, color =rs.CreateColor([175,100,100]))

        

        Hull_layer = 'Hull'
        if not rs.IsLayer(Hull_layer):
            rs.AddLayer(Hull_layer, color =rs.CreateColor([100,175,100]))


        rs.CurrentLayer(Hull_layer)
        # Make the side shell, bilge, and bottom shell plating 
        if hull_shell is None:
            self.make_Hull_Shell()
        else:
            self.Hull_Shell = hull_shell

        

        rs.CurrentLayer(self.struct_layer)

        # Create the structural elements as a dataframe

        self.Structural_Elements = pd.DataFrame(columns = ['struct_rsObject', # the rhino object ID
                                                           'mesh_rsObject', # the rhino object ID for the mesh
                                                           'Object_ID', # Name of object
                                                            'Color', # Color of the object
                                                            'x_loc', # x location of the object in meters
                                                            'y_loc', # y location of the object in meters
                                                            'z_loc', # z location of the object in meters
                                                            'x_dir', # x normal direction of the object
                                                            'y_dir', # y normal direction of the object
                                                            'z_dir', # z normal direction of the object
                                                            'rot', # rotation of the object
                                                            'L1', # Length in primary direction in meters
                                                            'L2', # Length in secondary direction in meters
                                                            'Thickness', # Thickness of the plate in meters
                                                            'Lightening_Holes', # Lightening holes in the plate (str of json)
                                                            'bit_1D_element', # Is this a 1D element? (boolean)
                                                            'Class', #Side Shell, Bottom Shell, Deck Plate, Transverse Bulkhead, Longitudinal Bulkhead, Long Stiffener, Bracket, Shell Strake - what is the ABS Class of this shape?
                                                            'Type', # Shell, Bulkhead, Deck, Stiffener, Bracket - what function created this shape?
                                                            'Parent_Struct' #Does this connect to a larger structure?
                                                            ])
        
        
        
        
        self.make_Shell_Plating()
        rs.LayerVisible(Hull_layer, False) #hide hull

     
    # Slice Functions:
    def YZ_plane(self, x):
        # Create a YZ plane on portside
        edges = []
        edges.append(rs.AddLine([x, -0.55*self.Hull_Dims['B'], -0.05*self.Hull_Dims['D']], [x, 0.55*self.Hull_Dims['B'], -0.05*self.Hull_Dims['D']]))
        edges.append(rs.AddLine([x, 0.55*self.Hull_Dims['B'], -0.05*self.Hull_Dims['D']], [x, 0.55*self.Hull_Dims['B'], 1.05*self.Hull_Dims['D']]))
        edges.append(rs.AddLine([x, 0.55*self.Hull_Dims['B'], 1.05*self.Hull_Dims['D']], [x, -0.55*self.Hull_Dims['B'], 1.05*self.Hull_Dims['D']]))
        edges.append(rs.AddLine([x, -0.55*self.Hull_Dims['B'], 1.05*self.Hull_Dims['D']], [x, -0.55*self.Hull_Dims['B'], -0.05*self.Hull_Dims['D']]))
        
        plane = rs.AddPlanarSrf(edges)

        rs.DeleteObjects(edges)

        return plane

    def XY_plane(self, z):
        # Create a YZ plane on portside
      
        edges = [] 
        edges.append(rs.AddLine([-0.05*self.Hull_Dims['L_3h'], -0.55*self.Hull_Dims['B'], z], [1.05*self.Hull_Dims['L_3h'], -0.55*self.Hull_Dims['B'], z]))
        edges.append(rs.AddLine([1.05*self.Hull_Dims['L_3h'], -0.55*self.Hull_Dims['B'], z], [1.05*self.Hull_Dims['L_3h'], 0.55*self.Hull_Dims['B'], z]))
        edges.append(rs.AddLine([1.05*self.Hull_Dims['L_3h'], 0.55*self.Hull_Dims['B'], z], [-0.05*self.Hull_Dims['L_3h'], 0.55*self.Hull_Dims['B'], z]))
        edges.append(rs.AddLine([-0.05*self.Hull_Dims['L_3h'], 0.55*self.Hull_Dims['B'], z], [-0.05*self.Hull_Dims['L_3h'], -0.55*self.Hull_Dims['B'], z]))

        plane = rs.AddPlanarSrf(edges)

        rs.DeleteObjects(edges)

        return plane
    
    def XZ_plane(self, y):
        # Create a YZ plane on portside
        edges = []

        edges.append(rs.AddLine([-0.05*self.Hull_Dims['L_3h'], y, -0.05*self.Hull_Dims['D']], [1.05*self.Hull_Dims['L_3h'], y, -0.05*self.Hull_Dims['D']]))
        edges.append(rs.AddLine([1.05*self.Hull_Dims['L_3h'], y, -0.05*self.Hull_Dims['D']], [1.05*self.Hull_Dims['L_3h'], y, 1.05*self.Hull_Dims['D']]))
        edges.append(rs.AddLine([1.05*self.Hull_Dims['L_3h'], y, 1.05*self.Hull_Dims['D']], [-0.05*self.Hull_Dims['L_3h'], y, 1.05*self.Hull_Dims['D']]))
        edges.append(rs.AddLine([-0.05*self.Hull_Dims['L_3h'], y, 1.05*self.Hull_Dims['D']], [-0.05*self.Hull_Dims['L_3h'], y, -0.05*self.Hull_Dims['D']]))

        plane = rs.AddPlanarSrf(edges)

        rs.DeleteObjects(edges)

        return plane

    def make_Hull_Shell(self):

                #First Draw the shell outline
        side = rs.AddLine([0,0.5*self.Hull_Dims['B'],self.Hull_Dims['D']], [0,0.5*self.Hull_Dims['B'],0])
        bottom = rs.AddLine([0,0,0], [0,0.5*self.Hull_Dims['B'],0])
        
        bilge = rs.AddFilletCurve(side, bottom, radius = self.Hull_Dims['R_b'])

        rs.DeleteObjects([side, bottom])

        # Extract points: 
        start = rs.CurveStartPoint(bilge)
        end = rs.CurveEndPoint(bilge)

        # identify the locations of start and end of the bilge
        if start[2] > end[2]:
            bilge_start = start
            bilge_end = end
        else:
            bilge_start = end
            bilge_end = start
        
        side = rs.AddLine([0,0.5*self.Hull_Dims['B'],bilge_start[2]], [0,0.5*self.Hull_Dims['B'],self.Hull_Dims['D']])
        bottom = rs.AddLine([0,0,0], [0,bilge_end[1],0])

        

        #Join the curves
        shell = rs.JoinCurves([side, bilge, bottom]) #, bilge))

        #Extrude the shell
        path = rs.AddLine([0,0,0], [self.Hull_Dims["L_3h"],0,0])
        self.Hull_Shell = rs.ExtrudeCurve(shell, path)

        rs.DeleteObjects([side, bottom, bilge, shell, path])


    def make_Shell_Plating(self):
        """
        Create the shell plating of the ship given the shell plating params
        """


        #split the shell into side and bottom
        DB_Plane = self.XY_plane(self.Struct_Params["Db"]/1000.0)

    
        split = rs.SplitBrep(self.Hull_Shell, DB_Plane, delete_input = False)

        rs.DeleteObjects([DB_Plane])

        if (rs.SurfaceAreaCentroid(split[0])[0][2] > rs.SurfaceAreaCentroid(split[1])[0][2]): #this is the z coordinate of the point object embedded in the surface area centroind
            self.side_shell = split[0]
            self.bottom_shell = split[1]
        else:
            self.side_shell = split[1]
            self.bottom_shell = split[0]

        #Copy shell objects to the mesh layer: 
        side_shell_copy = rs.CopyObject(self.side_shell)
        bottom_shell_copy = rs.CopyObject(self.bottom_shell)
        rs.ObjectLayer(side_shell_copy, self.mesh_layer)
        rs.ObjectLayer(bottom_shell_copy, self.mesh_layer)


        # Add to the structural elements
        item_s = pd.DataFrame({'struct_rsObject': [self.side_shell],
                                'mesh_rsObject': [side_shell_copy], 
                             'Object_ID': 'Side_Shell', 
                             'Color': None, 
                             'x_loc': 0, 
                             'y_loc': self.Hull_Dims["B"]/2.0, 
                             'z_loc': self.Struct_Params["Db"]/1000.0, 
                             'x_dir': 1, 
                             'y_dir': 0, 
                             'z_dir': 0, 
                             'rot': 0, 
                             'L1': self.Hull_Dims["L_3h"], 
                             'L2': self.Hull_Dims["D"]-self.Struct_Params["Db"]/1000.0, 
                             'Thickness': self.Struct_Params['Side_Shell_Thickness']/1000.0, 
                             'Lightening_Holes': '',
                             'bit_1D_element': False,
                             'Class': 'Side_Shell', 
                             'Type': "Side_Shell", 
                             'Parent_Struct': 'Hull'})
        
        item_b = pd.DataFrame({'struct_rsObject': [self.bottom_shell],
                                'mesh_rsObject': [bottom_shell_copy],
                                'Object_ID': 'Bottom_Shell', 
                                'Color': None, 
                                'x_loc': 0, 
                                'y_loc': 0, 
                                'z_loc': 0, 
                                'x_dir': 1, 
                                'y_dir': 0, 
                                'z_dir': 0, 
                                'rot': 90, 
                                'L1': self.Hull_Dims["L_3h"], 
                                'L2': self.Hull_Dims["B"]/2.0, 
                                'Thickness': self.Struct_Params['Bottom_Shell_Thickness']/1000.0, 
                                'Lightening_Holes': '', 
                                'bit_1D_element': False,
                                'Class': 'Bottom_Shell',
                                'Type': "Bottom_Shell", 
                                'Parent_Struct': 'Hull'})
        self.Structural_Elements = pd.concat([self.Structural_Elements, item_s, item_b], ignore_index=True)
                               
        
        
    
    def generate_3H_Structure(self):
        '''
        This function generates a ship structure based on the parameters given in the Hull_Dict and Struct_Params
        '''
        #Work in progress
        return 0




    def make_Shell_Strake(self, Strake_Dict, Color = None): 
        """
        Create a shell strake given the shell strake parameters
        Shell Strake Parameters:
        t: thickness of the shell strake in milimeters
        h: height of the shell strake in milimeters
        z: = height of the bottom of the shell strake in meters. Assume strake is on the side shell

        """
        h = Strake_Dict["h"]/1000.0
        z = Strake_Dict["z"]

        top = self.XY_plane(z+h)
        bottom= self.XY_plane(z)
        
        split = rs.SplitBrep(self.Hull_Shell, bottom, delete_input = False)

        if split is None:
            print('No strake at z = ' + str(z))
            return 
        
        else:
            if (rs.SurfaceAreaCentroid(split[0])[0][2] > rs.SurfaceAreaCentroid(split[1])[0][2]):
                upper_split = rs.CopyObject(split[0])

            else:
                upper_split = rs.CopyObject(split[1])
        
        split2 = rs.SplitBrep(upper_split, top, delete_input = False)

        if split2 is None:
            strake = rs.CopyObject(upper_split)
            split2 = rs.AddLine([0,0,0],[1,1,1]) #this is a dummy line that will not be used and get deleted later

        else:
            if (rs.SurfaceAreaCentroid(split2[0])[0][2] > rs.SurfaceAreaCentroid(split2[1])[0][2]):
                strake = rs.CopyObject(split2[1])

            else:
                strake = rs.CopyObject(split2[0])


        rs.DeleteObjects([top, bottom, upper_split])
        rs.DeleteObjects(split)
        rs.DeleteObjects(split2)

        #copy shell strake to the mesh layer
        strake_copy = rs.CopyObject(strake)
        rs.ObjectLayer(strake_copy, self.mesh_layer)
       

        

        item = pd.DataFrame({'struct_rsObject': [strake], 
                                'mesh_rsObject': [strake_copy],
                                'Object_ID': 'Shell_Strake', 
                                'Color': Color, 
                                'x_loc': 0, 
                                'y_loc': self.Hull_Dims["B"]/2.0, 
                                'z_loc': z, 
                                'x_dir': 1, 
                                'y_dir': 0, 
                                'z_dir': 0, 
                                'rot': 0, 
                                'L1': self.Hull_Dims["L_3h"], 
                                'L2': h, 
                                'Thickness': Strake_Dict["t"]/1000.0, 
                                'Lightening_Holes': '',
                                'bit_1D_element': False, 
                                'Class': 'Shell_Strake',
                                'Type': "Shell_Strake", 
                                'Parent_Struct': 'Side_Shell'})

        self.Structural_Elements = pd.concat([self.Structural_Elements, item], ignore_index=True)


        

    def make_Stiffener_Profile(self, Stiff_Dict, buffer = 0.0):
        """
        Create a stiffener profile given the stiffener parameters
        returns a stiffener profile part object that is rotated to the correct orientation and is centered at the origin
        buffer is a correction factor to ensure that the stiffener overlaps the surface that the stiffener is attached to
        
        """
        h = Stiff_Dict["h"]/1000.0 # height of the stiffener in milimeters -> meters
        t = Stiff_Dict["t"]/1000.0 # thickness of the stiffener plating in milimeters -> meters
        w = Stiff_Dict["w"]/1000.0 # width of stiffener cap  in milimeters -> meters
        rot = Stiff_Dict["rot"] # rotation of the stiffener in 2D (Rotation of stiffener in 2D plane of dir)
        dir = Stiff_Dict["dir"] # the normal vector of the face of the stiffener. Will only be [0,0,1], [0,1,0], or [1,0,0]
        bit_TorC = Stiff_Dict["bit_TorC"] # 1 for T stiffener, 0 for C stiffener 

        #Create the stiffener profile
        edges = []
        
        edges.append(rs.AddLine([-buffer,0,0], [h,0,0]))
        
        if w > 0:
            if bit_TorC: #T is default points

                edges.append(rs.AddLine([h,-w/2,0], [h,w/2,0]))
                
            else: #C default points towards negative x,y,z direction (towards interior of ship)
        
                edges.append(rs.AddLine([h,0,0], [h,w,0]))
      
        # Stiffener is oriented in the z direction
        rs.RotateObjects(edges, [0,0,0], rotation_angle = rot, axis = [0,0,1])


        """
        Orientation for the moment of inertia calculations:

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

        y_dir = 1, rot = 0 on +x axis   11 axis is x, 22 axis is +z
                    |+y
                    |
         +x(11)_____|
                    /
                   /
                  /
                 +z(22)
        """

        #rotate the stiffener to the correct orientation


        if dir == [1,0,0]:
            rs.RotateObjects(edges, [0,0,0], rotation_angle = -90, axis = [0,1,0])

            
        elif dir == [0,1,0]:
            rs.RotateObjects(edges, [0,0,0], rotation_angle = 90, axis = [1,0,0])
         

        #return Part.Face(stiffener)
        return edges

    def make_Stiffener(self, paths, Stiff_Dict, parent_id ='', id_root ='', Color = None): 
        """
        This function creates a stiffener along a path

        path is a line object that defines the path of the stiffener
        """
        # Check if path is not an array or list
        if not isinstance(paths, list):
            paths = [paths]
        # Check if path is a list of curves
        for path in paths:

            edges = self.make_Stiffener_Profile(Stiff_Dict)
            start = rs.CurveStartPoint(path)
            end = rs.CurveEndPoint(path)

            rs.MoveObjects(edges,start)

            stiffener = []


            obj_id = np.array([id_root, id_root+'_Flange'])
            dist = [abs(end[0]-start[0]), abs(end[1]-start[1]),abs(end[2]-start[2])]
            L1 = max(dist)
            L2 = np.array([Stiff_Dict['h']/1000, Stiff_Dict['w']/1000])
            sdir = Stiff_Dict['dir']
            rot = np.array([Stiff_Dict['rot'], Stiff_Dict['rot']+90])

        
        
            if len(edges) > 1:
                pt = rs.CurveStartPoint(edges[1])
                loc = np.array([[start[0], start[1], start[2]],
                            [pt[0], pt[1], pt[2]]])
                #print(loc.tolist())

            else:
                loc = np.array([[start[0],start[1],start[2]]])
                #print(loc.tolist())

            for i in range(3):
                if sdir[i]==1:
                    loc[:,i] = np.zeros((len(loc),))

            save_as_1D = [Stiff_Dict["1D_element"], True] # If the stiffener is a 1D element, save it as a 1D element
            Class = [Stiff_Dict["Class"], 'Stiffener_Flange']  # If the stiffener has a class, use it, otherwise use the default class

            for i in range(len(edges)):
                stiffener.append(rs.ExtrudeCurve(edges[i], path))

                str_LH = None
                
                if i == 0:
                    if "LH_Dict" in Stiff_Dict:
                        str_LH = str(Stiff_Dict["LH_Dict"])
                        for LH in Stiff_Dict["LH_Dict"]:
                            stiff_edges = rs.DuplicateEdgeCurves(stiffener[0])
                            stiff  = self.make_Lightening_Holes(stiffener[0], stiff_edges, LH)
                            rs.DeleteObjects(stiffener)

                            rs.DeleteObjects(stiff_edges)
                            stiffener = [stiff]

                    #Calculate the position of the end of the stiffener: 


                    #Check if the stiffener is a 1D element and add the stiffener or the path to MeshLayer
                    if Stiff_Dict["1D_element"]:
                        
                        mesh_stiff = rs.CopyObject(path)
                        rs.ObjectLayer(mesh_stiff, self.mesh_layer)

                                            
                    else: 
                        mesh_stiff = rs.CopyObject(stiffener[0])
                        rs.ObjectLayer(mesh_stiff, self.mesh_layer)

                else: 
                    # If this is a flange, it becomes a 1D element regardless.
                    if not Stiff_Dict["1D_element"]: # If the main stiffener is not a 1D element, then the flange path gets moved to the end of the stiffener
                        
                        mesh_stiff = rs.CopyObject(path)
                        start = rs.CurveStartPoint(edges[0])
                        end = rs.CurveEndPoint(edges[0])
                        translate = [end[0]-start[0], end[1]-start[1], end[2]-start[2]]
                        rs.MoveObjects(mesh_stiff, translate)
                        rs.ObjectLayer(mesh_stiff, self.mesh_layer)
                    else:
                        mesh_stiff = None #If the main stiffener is a 1D element do not make a new flange path.
          

                    
                
                # Add to the structural elements
                item = pd.DataFrame({'struct_rsObject': [stiffener[i]],
                                    'mesh_rsObject': [mesh_stiff],
                                'Object_ID': obj_id[i],
                                    'Color': Color,
                                    'x_loc': loc[i,0], #Modulo the start point to orient to start at zero
                                    'y_loc': loc[i,1],
                                    'z_loc': loc[i,2],
                                    'x_dir': sdir[0],
                                    'y_dir': sdir[1],
                                    'z_dir': sdir[2],
                                    'rot': rot[i],
                                    'L1': L1,
                                    'L2': L2[i],
                                    'Thickness': Stiff_Dict["t"]/1000.0,
                                    'Lightening_Holes': str_LH,
                                    'bit_1D_element': save_as_1D[i],
                                    'Class': Class[i],
                                    'Type': "Stiffener",
                                    'Parent_Struct': parent_id})
                
                self.Structural_Elements = self.Structural_Elements = pd.concat([self.Structural_Elements, item], ignore_index=True)
            rs.DeleteObjects(edges)
        return stiffener

        


    
    def make_Deck(self, Deck_Dict, Long_Stiffeners = [], Trans_Stiffeners = []):
        """
        Create a deck given the deck parameters
        """
        plane = self.XY_plane(Deck_Dict["z"])
        
        #intersect the top plane to the hull
        edges = []
        edges.append(rs.IntersectBreps(self.Hull_Shell, plane))
        start = rs.CurveStartPoint(edges[0])
        end = rs.CurveEndPoint(edges[0])

        edges.append(rs.AddLine(end, [end[0], 0, end[2]]))
        edges.append(rs.AddLine([end[0], 0, end[2]], [start[0], 0, start[2]]))
        edges.append(rs.AddLine([start[0], 0, start[2]], start))

        deck = rs.AddPlanarSrf(edges)
        str_LH = None
        if "LH_Dict" in Deck_Dict:

            str_LH = str(Deck_Dict["LH_Dict"])
            for LH in Deck_Dict["LH_Dict"]:
                deck_cut  = self.make_Lightening_Holes(deck, edges, LH)
                rs.DeleteObjects(deck)
                deck = [deck_cut]

        rs.DeleteObjects(plane)
        rs.DeleteObjects(edges)

        #copy deck to the mesh layer
        deck_copy = rs.CopyObject(deck)
        rs.ObjectLayer(deck_copy, self.mesh_layer)


        deck_id = pd.DataFrame({'struct_rsObject': deck, 
                                'mesh_rsObject': deck_copy,
                             'Object_ID': f'Deck_Z_{Deck_Dict["z"]}',
                                'Color': None,
                                'x_loc': 0,
                                'y_loc': 0,
                                'z_loc': Deck_Dict["z"],
                                'x_dir': 1,
                                'y_dir': 0,
                                'z_dir': 0,
                                'rot': 90,
                                'L1': self.Hull_Dims["L_3h"],
                                'L2': self.Hull_Dims["B"]/2.0,
                                'Thickness': Deck_Dict["t"]/1000.0,
                                'Lightening_Holes': str_LH,
                                'bit_1D_element': False,
                                'Class': 'Deck',
                                'Type': "Deck",
                                'Parent_Struct': 'Hull'})
        # Add to the structural elements

        self.Structural_Elements = pd.concat([self.Structural_Elements, deck_id], ignore_index=True)

        rs.DeleteObjects([plane])

        for stiff in Long_Stiffeners:
            y = stiff["y"]*self.Hull_Dims["B"]/2.0
            
            try:
                len(y)
            except:
                y = [y]

            for i in range(len(y)):
                plane = self.XZ_plane(y[i])
                intersect = rs.IntersectBreps(deck, plane)

                
                if intersect is None:
                    print("No Longitudinal Deck Stiffener at y = ", y[i])
                    rs.DeleteObject(plane)
                    continue

                else:
                    id_root = deck_id["Object_ID"][0]+f'_Long_Stiffener_Y_{y[i]}'
                    stiffener = self.make_Stiffener(intersect, stiff, parent_id = deck_id["Object_ID"][0], id_root = id_root)
                    #self.Long_Deck_Stiffeners = pd.concat([self.Long_Deck_Stiffeners,stiffener], ignore_index=True)
                
                rs.DeleteObjects([plane])
                #check if interest is a list of objects or a single object
                if isinstance(intersect, list):
                    rs.DeleteObjects(intersect)
                else:
                    rs.DeleteObjects([intersect])
                


        for stiff in Trans_Stiffeners:
            x = stiff["x"]*self.Hull_Dims["L_3h"]

            try: 
                len(x)
            except:
                x = [x]

            for i in range(len(x)):
                plane = self.YZ_plane(x[i])
                intersect = rs.IntersectBreps(deck, plane)
                
                if intersect is None:
                    print("No Transverse Deck Stiffener at x = ", x[i])
                    rs.DeleteObject(plane)
                    continue
                
                else:
                    id_root = deck_id["Object_ID"][0]+f'_Trans_Stiffener_X_{x[i]}'
                    stiffener = self.make_Stiffener(intersect, stiff, parent_id = deck_id["Object_ID"][0], id_root = id_root)
                    
                
                rs.DeleteObjects([plane])
                #check if interest is a list of objects or a single object
                if isinstance(intersect, list):
                    rs.DeleteObjects(intersect)
                else:
                    rs.DeleteObjects([intersect])


    def make_Trans_Bulkhead(self, Bulkhead_Dict, Trans_Stiffeners = [], Vert_Stiffeners = []):
        """
        Create a bulkhead given the bulkhead parameters
        """
        x = Bulkhead_Dict["x"]*self.Hull_Dims["L_3h"] # x position of the bulkhead given as a fraction of the LOA

        #Create the bulkhead profile

        try:
            len(x)
        except:
            x = [x]
        
        for j in range(len(x)):

            plane = self.YZ_plane(x[j])
            edges = []
            edges.append(rs.IntersectBreps(self.Hull_Shell, plane))
            edges.append(rs.AddLine([x[j], 0, 0], [x[j], 0, self.Hull_Dims["D"]]))
            edges.append(rs.AddLine([x[j], 0, self.Hull_Dims["D"]], [x[j], 0.5*self.Hull_Dims["B"], self.Hull_Dims["D"]]))

            bulkhead = rs.AddPlanarSrf(edges)

            str_LH = None
            if "LH_Dict" in Bulkhead_Dict:
                str_LH = str(Bulkhead_Dict["LH_Dict"])
                for LH in Bulkhead_Dict["LH_Dict"]:
                    blkhd_cut  = self.make_Lightening_Holes(bulkhead, edges, LH)
                    rs.DeleteObjects(bulkhead)
                    bulkhead = [blkhd_cut]

            rs.DeleteObjects(plane)
            rs.DeleteObjects(edges)

            #copy bulkhead to the mesh layer
            bulkhead_copy = rs.CopyObject(bulkhead)
            rs.ObjectLayer(bulkhead_copy, self.mesh_layer)

            blkhd_id = pd.DataFrame({'struct_rsObject': bulkhead, 
                        'mesh_rsObject': bulkhead_copy,
                        'Object_ID': f'Trans_Bulkhead_X_{x[j]}',
                        'Color': None,
                        'x_loc': x[j],
                        'y_loc': 0,
                        'z_loc': 0,
                        'x_dir': 0,
                        'y_dir': 1,
                        'z_dir': 0,
                        'rot': 90,
                        'L1': self.Hull_Dims["B"]/2.0,
                        'L2': self.Hull_Dims["D"],
                        'Thickness': Bulkhead_Dict["t"]/1000.0,
                        'Lightening_Holes': str_LH,
                        'bit_1D_element': False,
                        'Class': 'Transverse_Bulkhead',
                        'Type': "Transverse_Bulkhead",
                        'Parent_Struct': 'Hull'})
            
            # Add to the structural elements

            self.Structural_Elements = pd.concat([self.Structural_Elements, blkhd_id], ignore_index=True)
           

            rs.DeleteObjects([plane])
            rs.DeleteObjects(edges)

            for stiff in Vert_Stiffeners:
                y = stiff["y"]*self.Hull_Dims["B"]/2.0
                
                try:
                    len(y)
                except:
                    y = [y]

                for i in range(len(y)):
                    plane = self.XZ_plane(y[i])

                    #assert stiff["rot"] == 90 or stiff["rot"] == -90 or stiff["rot"] == 270, "Invalid rotation for vertical bulkhead stiffener"
                    
                    intersect = rs.IntersectBreps(bulkhead, plane)

                    if intersect is None:
                        print("No Vertical Bulkhead Stiffener at y = ", y[i])
                        rs.DeleteObject(plane)
                        continue

                    else:
                        id_root = blkhd_id["Object_ID"][0]+f'_Vert_Stiffener_Y_{y[i]}'
                        stiffener = self.make_Stiffener(intersect, stiff, parent_id = blkhd_id["Object_ID"][0], id_root = id_root)
                    
                    rs.DeleteObjects([plane])
                    #check if interest is a list of objects or a single object
                    if isinstance(intersect, list):
                        rs.DeleteObjects(intersect)
                    else:
                        rs.DeleteObjects([intersect])

            for stiff in Trans_Stiffeners:
                z = stiff["z"]*self.Hull_Dims["D"]

                try: 
                    len(z)
                except:
                    z = [z]

                for i in range(len(z)):
                    plane = self.XY_plane(z[i])

                    #assert stiff["rot"] == 90 or stiff["rot"] == -90 or stiff["rot"] == 270, "Invalid rotation for transverse bulkhead stiffener"

                    intersect = rs.IntersectBreps(bulkhead, plane)

                    if intersect is None:
                        print("No Transverse Bulkhead Stiffener at z = ", z[i])
                        rs.DeleteObject(plane)
                        continue

                    else:
                        id_root = blkhd_id["Object_ID"][0]+f'_Trans_Stiffener_Z_{z[i]}'
                        stiffener = self.make_Stiffener(intersect, stiff, parent_id = blkhd_id["Object_ID"][0], id_root = id_root)
                    
                    rs.DeleteObjects([plane])
                    #check if interest is a list of objects or a single object
                    if isinstance(intersect, list):
                        rs.DeleteObjects(intersect)
                    else:
                        rs.DeleteObjects([intersect])

    def make_Long_Bulkhead(self, Bulkhead_Dict, Long_Stiffeners = [], Vert_Stiffeners = []):
        """
        Create a bulkhead given the bulkhead parameters
        """
        y = Bulkhead_Dict["y"]*self.Hull_Dims["B"]/2.0 # y position of the bulkhead given as a fraction of the Beam

        #Create the bulkhead profile

        try:
            len(y)
        except:
            y = [y]
        
        for j in range(len(y)):

            plane = self.XZ_plane(y[j])
            edges = []
            intersect = rs.IntersectBreps(self.Hull_Shell, plane)
            start = rs.CurveStartPoint(intersect[0])
            end = rs.CurveEndPoint(intersect[0])
            #extract z coordinate of start and end points
            if start[0] < end[0]:
                z_start = start[2]
                z_end = end[2]
            else:
                z_start = end[2]
                z_end = start[2]

            edges.append(rs.AddLine([0, y[j], z_start], [self.Hull_Dims['L_3h'], y[j], z_end]))
            edges.append(rs.AddLine([0, y[j], z_start], [0, y[j], self.Hull_Dims["D"]]))
            edges.append(rs.AddLine([0, y[j], self.Hull_Dims["D"]], [self.Hull_Dims["L_3h"], y[j], self.Hull_Dims["D"]]))
            edges.append(rs.AddLine([self.Hull_Dims["L_3h"], y[j], self.Hull_Dims["D"]], [self.Hull_Dims["L_3h"],y[j], z_end]))

            bulkhead = rs.AddPlanarSrf(edges)

            
            str_LH = None
            if "LH_Dict" in Bulkhead_Dict:
                str_LH = str(Bulkhead_Dict["LH_Dict"])
                for LH in Bulkhead_Dict["LH_Dict"]:
                    blkhd_cut  = self.make_Lightening_Holes(bulkhead, edges, LH)
                    rs.DeleteObjects(bulkhead)
                    bulkhead = [blkhd_cut]

            #copy bulkhead to the mesh layer
            bulkhead_copy = rs.CopyObject(bulkhead)
            rs.ObjectLayer(bulkhead_copy, self.mesh_layer)
 

            blkhd_id = pd.DataFrame({'struct_rsObject': bulkhead, 
            'mesh_rsObject': bulkhead_copy,
            'Object_ID': f'Long_Bulkhead_Y_{y[j]}',
            'Color': None,
            'x_loc': 0,
            'y_loc': y[j],
            'z_loc': 0,
            'x_dir': 1,
            'y_dir': 0,
            'z_dir': 0,
            'rot': 0,
            'L1': self.Hull_Dims["L_3h"],
            'L2': self.Hull_Dims["D"],
            'Thickness': Bulkhead_Dict["t"]/1000.0,
            'Lightening_Holes': str_LH,
            'bit_1D_element': False,
            'Class': 'Longitudinal_Bulkhead',
            'Type': "Longitudinal_Bulkhead",
            'Parent_Struct': 'Hull'})

            self.Structural_Elements = pd.concat([self.Structural_Elements, blkhd_id], ignore_index=True)

            rs.DeleteObjects([plane, intersect])
            rs.DeleteObjects(edges)
   

            for stiff in Vert_Stiffeners:
                x = stiff["x"]*self.Hull_Dims["L_3h"]
                
                try:
                    len(x)
                except:
                    x = [x]

                for i in range(len(x)):
                    plane = self.YZ_plane(x[i])

                    #assert stiff["rot"] == 90 or stiff["rot"] == -90 or stiff["rot"] == 270, "Invalid rotation for vertical bulkhead stiffener"
                    
                    intersect = rs.IntersectBreps(bulkhead, plane)

                    if intersect is None:
                        print("No Vertical Bulkhead Stiffener at x = ", x[i])
                        rs.DeleteObject(plane)
                        continue

                    else:
                        id_root = blkhd_id["Object_ID"][0]+f'_Vert_Stiffener_X_{x[i]}'
                        stiffener = self.make_Stiffener(intersect, stiff, parent_id = blkhd_id["Object_ID"][0], id_root = id_root)
                    
                    rs.DeleteObjects([plane])
                    #check if interest is a list of objects or a single object
                    if isinstance(intersect, list):
                        rs.DeleteObjects(intersect)
                    else:
                        rs.DeleteObjects([intersect])

            for stiff in Long_Stiffeners:
                z = stiff["z"]*self.Hull_Dims["D"]

                try: 
                    len(z)
                except:
                    z = [z]

                for i in range(len(z)):
                    plane = self.XY_plane(z[i])

                     #assert stiff["rot"] == 90 or stiff["rot"] == -90 or stiff["rot"] == 270, "Invalid rotation for transverse bulkhead stiffener"

                    intersect = rs.IntersectBreps(bulkhead, plane)

                    if intersect is None:
                        print("No Long Bulkhead Stiffener at z = ", z[i])
                        rs.DeleteObject(plane)
                        continue

                    else:
                        id_root = blkhd_id["Object_ID"][0]+f'_Long_Stiffener_Z_{z[i]}'
                        stiffener = self.make_Stiffener(intersect, stiff, parent_id = blkhd_id["Object_ID"][0], id_root = id_root)
                    
                    rs.DeleteObjects([plane])
                    #check if interest is a list of objects or a single object
                    if isinstance(intersect, list):
                        rs.DeleteObjects(intersect)
                    else:
                        rs.DeleteObjects([intersect])


    def make_Trans_Floor_Stiffener(self, x, Stiffener_Dict):
        """
        Create a transverse bottom stiffener given the transverse bottom stiffener parameters
        """
        plane = self.YZ_plane(x)
 
        edges = []
        y_points = []
        
        edges.append(rs.IntersectBreps(self.bottom_shell, plane))

        for edge in edges:
            pt = rs.CurveStartPoint(edge)
            y_points.append(pt[1])
            pt = rs.CurveEndPoint(edge)
            y_points.append(pt[1])
        
        breadth = max(y_points)

        edges.append(rs.AddLine([x, 0, 0], [x, 0, self.Struct_Params["Db"]/1000.0]))
        edges.append(rs.AddLine([x, 0, self.Struct_Params["Db"]/1000.0], [x, breadth, self.Struct_Params["Db"]/1000.0]))

        stiffener = rs.AddPlanarSrf(edges)
        
        #If LH_Dict Exists, make lightening holes
        str_LH = None
        if "LH_Dict" in Stiffener_Dict:
            str_LH = str(Stiffener_Dict['LH_Dict'])
            for LH in Stiffener_Dict["LH_Dict"]:
                stiff  = self.make_Lightening_Holes(stiffener[0], edges, LH)
                rs.DeleteObjects(stiffener)
                stiffener = [stiff]
        
        #Copy stiffener to the mesh layer
        stiffener_copy = rs.CopyObject(stiffener)
        rs.ObjectLayer(stiffener_copy, self.mesh_layer)


        item = pd.DataFrame({'struct_rsObject': stiffener, 
            'mesh_rsObject': stiffener_copy,
            'Object_ID': f'Trans_Bottom_X_{x}',
            'Color': None,
            'x_loc': x,
            'y_loc': 0,
            'z_loc': 0,
            'x_dir': 0,
            'y_dir': 1,
            'z_dir': 0,
            'rot': 90,
            'L1': self.Hull_Dims["B"]/2,
            'L2': self.Struct_Params['Db']/1000,
            'Thickness': Stiffener_Dict["t"]/1000.0,
            'Lightening_Holes': str_LH,
            'bit_1D_element': False,
            'Class': 'Transverse_Floor',
            'Type': "Transverse_Floor",
            'Parent_Struct': 'Bottom_Shell'})

        self.Structural_Elements = pd.concat([self.Structural_Elements, item], ignore_index=True)

        rs.DeleteObjects([plane])
        rs.DeleteObjects(edges)

    def make_Trans_Side_Stiffener(self, x, Side_Dict):
        """
        Create a transverse side stiffener given the transverse side stiffener parameters
        """
         #Create path for the side frame
        plane = self.YZ_plane(x)
        
        edges = rs.IntersectBreps(self.side_shell, plane)

        if len(edges) == 1:
            path = rs.CopyObject(edges)
        else:
            path = rs.JoinCurves(edges)

        #Create the side frame

        id_root = f'Side_Shell_frame_X_{x}'

        side_frame = self.make_Stiffener(path, Side_Dict, parent_id = 'Side_Shell', id_root=id_root)


        rs.DeleteObjects([plane])
        rs.DeleteObjects(edges)
        rs.DeleteObject(path)

    def make_Trans_Bottom_Frame_Stiffener(self, x, Bottom_Dict):
        """
        Create a transverse side stiffener given the transverse side stiffener parameters
        """
         #Create path for the side frame
        plane = self.YZ_plane(x)
        
        edges = rs.IntersectBreps(self.bottom_shell, plane)

        if len(edges) == 1:
            path = rs.CopyObject(edges)
        else:
            path = rs.JoinCurves(edges)

        #Create the side frame

        id_root = f'Bottom_Shell_frame_X_{x}'

        side_frame = self.make_Stiffener(path, Bottom_Dict, parent_id = 'Bottom_Shell', id_root=id_root)


        rs.DeleteObjects([plane])
        rs.DeleteObjects(edges)
        rs.DeleteObject(path)
        

    def make_Trans_Web_Frames(self, Trans_Web_Frame_Dict):
        """
        Create a transverse web frame 
        
        Trans_Frame_Dict: dictionary of transverse frame parameters
        "X": position of the frame in meters

        "Bottom_Dict": dictionary of bottom stiffener parameters
        "Side_Dict": dictionary of side stiffener parameters

        Right now, this function does not correct for xy curvature of the side shell plating, preventing the frame from being flush with the shell plating

        eventually, bracket information will be added to this function
        """

        x = Trans_Web_Frame_Dict["x"]*self.Hull_Dims["L_3h"]

        #Check to see if x is a list or a single value
        try: 
            len(x)
        except:
            x = [x]
        
        
        Side_Dict = Trans_Web_Frame_Dict["Side_Dict"]
        Bottom_Dict = Trans_Web_Frame_Dict["Bottom_Dict"]

    

        for i in range(len(x)):

            #Create Bottom Frame
            self.make_Trans_Floor_Stiffener(x[i], Bottom_Dict)
            self.make_Trans_Side_Stiffener(x[i], Side_Dict)

    def make_Trans_Frames(self,x_spacing, Trans_Frame_Dict, Trans_Frame_Bottom_Dict):
        """
        Create transverse frames given the transverse frame parameters
        x_spacing: list of x positions of the frames
        """
        x_spacing = x_spacing*self.Hull_Dims["L_3h"] # convert to meters
        try:
            len(x_spacing)
        except:
            x_spacing = [x_spacing]
        
        for x in x_spacing:
            self.make_Trans_Side_Stiffener(x,Trans_Frame_Dict)
            self.make_Trans_Bottom_Frame_Stiffener(x, Trans_Frame_Bottom_Dict)



    def make_Long_Bottom_Stiffener(self, Long_Bot_Stiff_Dict):
        """
        Create a longitudinal bottom stiffener given the longitudinal bottom stiffener parameters
        """

        y = Long_Bot_Stiff_Dict["y"]*self.Hull_Dims["B"]/2.0

        

        try:
            len(y)
        except:
            y = [y]

        for i in range(len(y)):
            
                mod_Dict = Long_Bot_Stiff_Dict.copy()
                plane = self.XZ_plane(y[i])
                intersect = rs.IntersectBreps(self.bottom_shell, plane)

                if intersect is None:
                    print("No Longitudinal Bottom Stiffener at y = ", y[i])
                    rs.DeleteObject(plane)
                    continue
                else:
                    #Get Z position of the bottom shell at the intersection
                    z_start = rs.CurveStartPoint(intersect[0])[2]
                    #Get Height of the Bottom Shell Stiffener
                    h = mod_Dict["h"]/1000.0 # height of the stiffener in milimeters -> meters
                    h_mod = min(h, self.Struct_Params["Db"]/1000.0 - z_start) # height of the stiffener in meters, modified to be at least the depth of the bottom shell
                    mod_Dict["h"] = h_mod*1000.0 # convert back to milimeters

                    id_root = f'Bottom_Shell_Long_Stiffener_Y_{y[i]}'

                    bot_stiffener = self.make_Stiffener(intersect, mod_Dict,parent_id = 'Bottom_Shell', id_root=id_root)

              
                rs.DeleteObjects([plane, intersect])


    def make_Long_Side_Stiffener(self, Long_Side_Stiff_Dict):
        """
        Create a longitudinal side stiffener given the longitudinal side stiffener parameters
        """


        z = Long_Side_Stiff_Dict["z"]*self.Hull_Dims["D"]

        try:
            len(z)
        except:
            z = [z]

        for i in range(len(z)):
            plane = self.XY_plane(z[i])

            intersect = rs.IntersectBreps(self.Hull_Shell, plane)

            if intersect is None:
                print("No Longitudinal Side Stiffener at z = ", z[i])
                rs.DeleteObject(plane)
                continue
            else:

                id_root = f'Side_Shell_Side_Stiffener_Z_{z[i]}'

                side_stiffener = self.make_Stiffener(intersect, Long_Side_Stiff_Dict, parent_id = 'Side_Shell', id_root=id_root)

            rs.DeleteObjects([plane, intersect])

    def make_Lightening_Holes(self, stiffener, edges, LH_Dict):
        """
        Create lightening holes in a stiffener given the lightening hole parameters

        LH_Params = {"x1" fraction the stiffener length of stiffener for the center of the lightening hole
                    "x2" fraction the stiffener height of stiffener for the center of the lightening hole
                    "r" radius of the lightening hole fillet in milimeters
                    "l" length of the lightening hole in milimeters
                    "h" height of the lightening hole in milimeters
    
        """

        x1 = LH_Dict["x1"]
        x2 = LH_Dict["x2"]
        r = LH_Dict["r"]/1000.0
        l = LH_Dict["l"]/1000.0
        h = LH_Dict["h"]/1000.0

        points = []
        for edge in edges:
            points.append(rs.CurveStartPoint(edge))
            points.append(rs.CurveEndPoint(edge))
        
        points = list(set(points))

        #from vertex points determine the axis of stiffener length and height
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        z = [point[2] for point in points]  

        x = list(set(x))
        y = list(set(y))
        z = list(set(z))

        x.sort()
        y.sort()
        z.sort()

     
        LH = np.array([max(x) - min(x), max(y) - min(y), max(z) - min(z)])
        LH_base = np.array([min(x), min(y), min(z)])
  
        #x,y,z -> LH = [range of x,y,z] -> if the range in a particular direction is close to machine precision, then the stiffener normmal to that direction
        idx_order = np.argsort(LH)
        axis_idx = [idx_order[2], idx_order[1]] # axis indices of major and minor axis of stiffener

        if LH[0] <= 1e-12:
            
            plane = rs.PlaneFromNormal([0,0,0], [1,0,0])
            pos = np.array([x[-1],0,0])
            
        elif LH[1] <= 1e-12:
            
            plane = rs.PlaneFromNormal([0,0,0], [0,1,0])
            pos = np.array([0,y[-1],0])
        else: 
            
            plane = rs.PlaneFromNormal([0,0,0], [0,0,1])
            pos = np.array([0,0,z[-1]])
    


        for i in range(len(x1)):
            if axis_idx[1] >= axis_idx[0]:
                rectangle = rs.AddRectangle(plane, l, h)
            else:
                rectangle = rs.AddRectangle(plane, h, l)

            move_path = LH_base.copy()
            move_path[axis_idx[0]] = move_path[axis_idx[0]] + LH[axis_idx[0]]*x1[i] - l/2.0
            move_path[axis_idx[1]] = move_path[axis_idx[1]] + LH[axis_idx[1]]*x2 - h/2.0

            rs.MoveObject(rectangle, move_path)

            rectangle = rs.ExplodeCurves(rectangle, delete_input = True)
            hole = []
            if r > 0:
                hole.append(rs.AddFilletCurve(rectangle[0], rectangle[1], r))
                hole.append(rs.AddFilletCurve(rectangle[1], rectangle[2], r))
                hole.append(rs.AddFilletCurve(rectangle[2], rectangle[3], r))
                hole.append(rs.AddFilletCurve(rectangle[3], rectangle[0], r))
        

                
                try:
                    hole.append(rs.AddLine(rs.CurveEndPoint(hole[3]), rs.CurveStartPoint(hole[0])))
                except:
                    pass
                try:
                    hole.append(rs.AddLine(rs.CurveEndPoint(hole[0]), rs.CurveStartPoint(hole[1])))
                except:
                    pass
                try:
                    hole.append(rs.AddLine(rs.CurveEndPoint(hole[1]), rs.CurveStartPoint(hole[2])))
                except:
                    pass
                try:
                    hole.append(rs.AddLine(rs.CurveEndPoint(hole[2]), rs.CurveStartPoint(hole[3])))
                except:
                    pass
            else:
                hole.append(rectangle[0])
                hole.append(rectangle[1])
                hole.append(rectangle[2])
                hole.append(rectangle[3])

            hole = rs.JoinCurves(hole, delete_input = True)
            extrude = rs.AddLine(pos, pos*1.01)
            cutter = rs.ExtrudeCurve(hole, extrude)
            
            intersect = rs.SplitBrep(stiffener, cutter)
            rs.DeleteObject(stiffener)
            areas = []
            for i in range(len(intersect)):
                area = rs.SurfaceArea(intersect[i])[0]
                areas.append(area)
            
            #big assumption that we keep the largest area:
            areas = np.array(areas)
            stiffener = rs.CopyObject(intersect[np.argmax(areas)])
            
            #delete extraneous objects
            rs.DeleteObjects([extrude, hole])
            rs.DeleteObjects(rectangle)
            rs.DeleteObjects(cutter)
            rs.DeleteObjects(intersect)

        return stiffener
    
    def make_Brackets(self, Bracket_Dict):
        """
        Create brackets given the bracket parameters all brackets are right triangles

        Bracket_Dict: list of dictionary of bracket parameters (list of dictionaries)
    
        Vertex : list of vertices of right angle vertex of the bracket
        L1: [x,y,z] length of one side of the bracket (assume to be the larger or equal length of the bracket) in milimeters 
        L2: [x,y,z] length of the other side of the bracket (assume to be the smaller or equal length of the bracket) in milimeters
        t: thickness of the bracket

        L1 and L2 follow the global sign conventions. L1 and L2 need to be oriented with respect to the bracket vertex

        """

        for bracket in Bracket_Dict:
            vertex = bracket["Vertex"]
            L1 = bracket["L1"]/1000.0
            L2 = bracket["L2"]/1000.0
            t = bracket["t"]/1000.0
            num_sides = bracket["num_sides"] 
            #Only 3 and 4 sided brackets are allowed

            #Create the bracket profile
            for vert in vertex:
                point1 = vert + L1
                point2 = vert + L2
                edges = []
                if num_sides == 3:
                    edges.append(rs.AddLine(vert, point1))
                    edges.append(rs.AddLine(point1, point2))
                    edges.append(rs.AddLine(point2, vert))
                elif num_sides == 4:
                    point3 = vert + L1 + L2
                    edges.append(rs.AddLine(vert, point1))
                    edges.append(rs.AddLine(point1, point3))
                    edges.append(rs.AddLine(point3, point2))
                    edges.append(rs.AddLine(point2, vert))
                else:
                    print("Invalid number of sides for bracket. Only 3 and 4 sided brackets are allowed")
                    continue

               
                brack = rs.AddPlanarSrf(edges)

                #copy bracket to the mesh layer
                brack_copy = rs.CopyObject(brack)
                rs.ObjectLayer(brack_copy, self.mesh_layer)

                #Assume L2 is normal to L1. Find rotation angle if L1 were rotated to be in the dir = [0,0,1]
                #Solve for Bracket Rotation. Idk how

                item = pd.DataFrame({'struct_rsObject': brack,
                    'mesh_rsObject': brack_copy, 
                    'Object_ID': f'Bracket_{num_sides}_sides_Vertex_X_{vert[0]}_Y_{vert[1]}_Z_{vert[2]}',
                    'Color': None,
                    'x_loc': vert[0],
                    'y_loc': vert[1],
                    'z_loc': vert[2],
                    'x_dir': abs(L1[0]/np.linalg.norm(L1)),
                    'y_dir': abs(L1[1]/np.linalg.norm(L1)),
                    'z_dir': abs(L1[2]/np.linalg.norm(L1)),
                    'rot': 0,  #Need to solve eventually. 
                    'L1': np.linalg.norm(L1),
                    'L2': np.linalg.norm(L2),
                    'Thickness': bracket["t"]/1000.0,  #Convert to meters
                    'Lightening_Holes': '',
                    'bit_1D_element': False,
                    'Class': bracket["Class"],
                    'Type': f"Bracket_{num_sides}_sides",
                    'Parent_Struct': 'Bracket'}) #Add Later

                self.Structural_Elements = pd.concat([self.Structural_Elements, item], ignore_index=True)

                rs.DeleteObjects(edges)
    
    def make_Pannel_Surface(self, Pannel_Dict):
        '''
        Create the hopper surfaces for bulkcarriers and pannels for containerships
        Hopper_Dict: dictionary of hopper parameters
        '''

        for pan in Pannel_Dict:
            #First generate the hopper surface:
            #Create the path for the hopper surface
            start = pan['start']
            end = pan['start'] + [self.Hull_Dims["L_3h"], 0, 0]
            path = rs.AddLine(start, end)
            #Create Stiffener Dict
            theta = np.arctan2(pan['L1'], pan['L2'])*180/np.pi
            pan_dict = {
                'h' : np.sqrt(pan['L1']**2 + pan['L2']**2),
                "t": pan["t_pan"],
                'w': 0,
                'rot': theta,
                'dir': [1,0,0],
                'bit_TorC': 0,
                '1D_element': False,
                'Class': 'Hopper Pannel'} #1D Element is False because we are creating a 2D surface
        
            hop_id_root = f'Hopper_Pannel_Y_{start[1]}_Z_{start[2]}'
            hopper = self.make_Stiffener(path, pan_dict,parent_id = 'Hull', id_root=hop_id_root)

            rs.DeleteObject(path)

            

            pan_stiff_dict = {
                'h' : pan['h'],
                "t": pan["t"],
                'w': pan['w'],
                'rot': theta+pan['rot'],
                'dir': [1,0,0],
                'bit_TorC': pan['bit_TorC'],
                '1D_element': pan['1D_element'],
                'Class': 'Hopper_Pannel_Longitudinal_Stiffener'} #1D Element is False because we are creating a
            

            y_pan_stiffeners, z_pan_stiffeners = self.calc_Pannel_Stiffener_Positions(pan, start, theta, threshold = 0.05)

            if abs(pan['L1']) > 0:

                for i in range(len(y_pan_stiffeners)):
                    plane = self.XZ_plane(y_pan_stiffeners[i])
                    intersect = rs.IntersectBreps(hopper, plane)

                    if intersect is None:
                        print("No hopper Stiffener at y = ", y_pan_stiffeners[i])
                        rs.DeleteObject(plane)
                        continue

                    else:
                        id_root = hop_id_root+f'_Stiffener_Y_{y_pan_stiffeners[i]}'
                        stiffener = self.make_Stiffener(intersect, pan_stiff_dict, parent_id = hop_id_root, id_root = id_root)
                    rs.DeleteObjects([plane, intersect])

            else: #Assume L2 has some length if L1 is 0
                z_pan_stiffeners = np.linspace(start[2], (start[2] + pan['L2']/1000), int(pan['num_stiffeners']+2))[1:-1]
                for i in range(len(z_pan_stiffeners)):
                    plane = self.XY_plane(z_pan_stiffeners[i])

                    intersect = rs.IntersectBreps(hopper, plane)

                    if intersect is None:
                        print("No hopper Stiffener at z = ", z_pan_stiffeners[i])
                        rs.DeleteObject(plane)
                        continue

                    else:
                        id_root = hop_id_root+f'_Stiffener_Z_{z_pan_stiffeners[i]}'
                        stiffener = self.make_Stiffener(intersect, pan_stiff_dict, parent_id = hop_id_root, id_root = id_root)
                    rs.DeleteObjects([plane, intersect])

    def calc_Pannel_Stiffener_Positions(self, Pan_Dict,start,theta, threshold = 0.05): 
        '''
        This function calculates the positions of the stiffeners for the hopper pannels and 
        and checks to ensure that any stiffener within 0.05 meters of a horizontal or verticla bulkhead stiffener
        is auto-aligned to the y/z position of the bulkhead stiffener. This is to ensure that the nearby stiffeners are aligned with the bulkhead stiffeners
        
        50mm is the default threshold for alignment. This can be changed by the user.
        '''
        y_pan_stiffeners = np.linspace(start[1], (start[1] + Pan_Dict['L1']/1000), int(Pan_Dict['num_stiffeners']+2))[1:-1]
        z_pan_stiffeners = np.linspace(start[2], (start[2] + Pan_Dict['L2']/1000), int(Pan_Dict['num_stiffeners']+2))[1:-1]

        Y_blkhd_stiffs = self.Structural_Elements[self.Structural_Elements['Class'] == 'Transverse_Bulkhead_Vertical_Stiffener']['y_loc'].unique()
        Z_blkhd_stiffs = self.Structural_Elements[self.Structural_Elements['Class'] == 'Transverse_Bulkhead_Transverse_Stiffener']['z_loc'].unique()

        #First check the min distance between the z positions of the pan stiffeners and the z positions of the bulkhead stiffeners\
        # First make sure L2 is not 0. then modify z positions.
        #  Then check L1. This gives preference to y position alignment over z position alignment
        
        new_z_pan_stiffeners = z_pan_stiffeners.copy()
        new_y_pan_stiffeners = y_pan_stiffeners.copy()

        if abs(Pan_Dict['L2']) > 0:
            #Make sure L2 is not 0 -> Implies horizonal pannel
            for i in range(len(z_pan_stiffeners)):
                min_dist = np.min(np.abs(z_pan_stiffeners[i] - Z_blkhd_stiffs))
                if min_dist < threshold:
                    closest_stiff = Z_blkhd_stiffs[np.argmin(np.abs(z_pan_stiffeners[i] - Z_blkhd_stiffs))]
                    new_z_pan_stiffeners[i] = closest_stiff

            #Update y positions of the pan stiffeners if L1 is not 0 -> Implies vertical pannel
            y_pan_stiffeners = start[1] + (z_pan_stiffeners - start[2])*np.tan(np.radians(theta))

        if abs(Pan_Dict['L1']) > 0:
            #Make sure L1 is not 0 -> Implies vertical pannel
            for i in range(len(y_pan_stiffeners)):
                min_dist = np.min(np.abs(y_pan_stiffeners[i] - Y_blkhd_stiffs))
                if min_dist < threshold:
                    closest_stiff = Y_blkhd_stiffs[np.argmin(np.abs(y_pan_stiffeners[i] - Y_blkhd_stiffs))]
                    new_y_pan_stiffeners[i] = closest_stiff

            #Update z positions of the pan stiffeners if L2 is not 0 -> Implies horizontal pannel
            new_z_pan_stiffeners = start[2] + (y_pan_stiffeners - start[1])/np.tan(np.radians(theta))

        else: 
            #If no changes to y positions of the pan stiffeners, then keep the original y positions. 
            
            new_y_pan_stiffeners = y_pan_stiffeners

        return new_y_pan_stiffeners, new_z_pan_stiffeners
        


    def Split_Bodies(self): 
        # Split the structural elements with the intersection Lines

        split_layer = 'Split_Struct'
        if not rs.IsLayer(split_layer):
            rs.AddLayer(split_layer)

        og_layer = rs.CurrentLayer()
        rs.CurrentLayer(split_layer)

        rs.LayerColor(split_layer, color=rs.CreateColor([100,100,175]))
        '''
        for i in range(len(self.Intersection_Lines)):

            #Delete cut_line elements that are empty
            
            cut_lines = self.Intersection_Lines['rsObject']
            cut_lines = cut_lines[cut_lines.dropna() & cut_lines != ''].tolist()

            cut_objs = []

            for j in 

            obj1 = self.Intersection_Lines.iloc[i]['rsObject_1']
            obj2 = self.Intersection_Lines.iloc[i]['rsObject_2']

            cut_objs = rs.ExtrudeCurve(cut_line, rs.AddLine([0,0,0], [0.01,0.01,0.01]))
        
            print('cut line: ' + str(cut_line) + ' isBrep: ' + str(rs.IsBrep([cut_line])))
            print('obj 1: ' + str(obj1)+ ' isBrep: ' + str(rs.IsBrep(obj1)))
            print('obj 2:' + str(obj2)+ ' isBrep: ' + str(rs.IsBrep(obj2)))
            try:
                split = rs.SplitBrep(obj1, cut_obj, delete_input = False)
            except:
                print('fail to cut obj1')
                pass
                
            try:
                split = rs.SplitBrep(obj2, cut_obj, delete_input = False)
            except:
                print('fail to cut obj2')
                pass
            rs.DeleteObject(cut_obj)

        '''

        cut_lines = self.Intersection_Lines['rsObject'].tolist()
        cut_lines = cut_lines

        cut_objs = []

    
        for i in range(len(cut_lines)):

            try:
                cut_objs.append(rs.coercerhinoobject(rs.ExtrudeCurve(cut_lines[i], rs.AddLine([0,0,0], [0.01,0.01,0.01]))).Geometry)
            except:
                pass

        for i in range(len(self.Structural_Elements['rsObject'])):
            
            obj = rs.coercerhinoobject(self.Structural_Elements.iloc[i]['rsObject']).Geometry
            #print(rs.IsBrep(cut_objs))
        
            #split = rs.SplitBrep(obj, cut_objs, delete_input = False)
            print(cut_objs[0])
            print(type(cut_objs[0]))

            split = obj.Split(cut_objs,0.001)
            print(split)



        rs.DeleteObject(cut_objs )
        

        rs.CurrentLayer(og_layer) #change back to og layer      


    def calc_Areas_and_Centroids(self):
        # Calculate the area and centroid of each structural element
        self.Structural_Elements['area_pannel'] = self.Structural_Elements['struct_rsObject'].apply(lambda x: rs.SurfaceArea(x)[0] if rs.IsSurface(x) else 0)
        self.Structural_Elements['volume_pannel'] = self.Structural_Elements['area_pannel'] * self.Structural_Elements['Thickness']
        
        Centroids = self.Structural_Elements['struct_rsObject'].apply(lambda x: rs.SurfaceAreaCentroid(x)[0] if rs.IsSurface(x) else [0, 0, 0]).tolist()
        self.Structural_Elements['x_centroid_pannel'] = [centroid[0] for centroid in Centroids]
        self.Structural_Elements['y_centroid_pannel'] = [centroid[1] for centroid in Centroids]
        self.Structural_Elements['z_centroid_pannel'] = [centroid[2] for centroid in Centroids]

        #Calculate the Cross Sectional Area
        self.Structural_Elements['area_cx'] = self.Structural_Elements['L2']*self.Structural_Elements['Thickness']
        
        
        #Calculate Longitudinal Moments of Intertia
        x_dir = self.Structural_Elements['x_dir'].to_numpy() # We assume the x direction is the longitudinal direction.
        y_dir = self.Structural_Elements['y_dir'].to_numpy()
        z_dir = self.Structural_Elements['z_dir'].to_numpy()

        #We only want to calculate these metrics for longitudinal elements

        L2 = self.Structural_Elements['L2'].to_numpy()
        T = self.Structural_Elements['Thickness'].to_numpy()
        rot = self.Structural_Elements['rot'].to_numpy().astype(float)

        x_loc = self.Structural_Elements['x_loc'].to_numpy()
        z_loc = self.Structural_Elements['z_loc'].to_numpy()
        y_loc = self.Structural_Elements['y_loc'].to_numpy()


        """
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

        y_dir = 1, rot = 0 on +x axis   11 axis is x, 22 axis is +z
                    |+y
                    |
         +x(11)_____|
                    /
                   /
                  /
                 +z(22)
        
        """

        self.Structural_Elements['I_11'] = (1/12 * L2*T)*((T**2)*np.cos(rot*np.pi/180)**2 + (L2**2)*np.sin(rot*np.pi/180)**2) #moment of inertia about the Z centroid - Rotation in y axis
        self.Structural_Elements['I_22'] = (1/12 * L2*T)*((T**2)*np.sin(rot*np.pi/180)**2 + (L2**2)*np.cos(rot*np.pi/180)**2) #moment of inertia about the Y centroid - Rotation in z axis
        self.Structural_Elements['J_p'] = L2*T**3 * (1/16)*(16/3 - 3.36*T/L2)*(1-((T/L2)**4)/12) #Polar moment of inertia about the centroid - Rotation in x axis - Formula uses 


        #Calculate the cross sectional centroid 
    
        self.Structural_Elements['centroid_cx_11'] = x_dir*z_loc + y_dir*x_loc +z_dir*x_loc + np.cos(rot*np.pi/180)*self.Structural_Elements['L2']/2.0    
        self.Structural_Elements['centroid_cx_22'] = x_dir*y_loc + y_dir*z_loc + z_dir*y_loc + np.sin(rot*np.pi/180)*self.Structural_Elements['L2']/2.0 # Extra np.cos(np.pi*z_dir) flips the sign of the centroid if element is in the z direction

        self.calc_1D_Measures() #Calculate the 1D measures for the structural elements

        #self.Structural_Elements['Sanity_Check_C11'] = x_dir*(self.Structural_Elements['centroid_cx_11'].to_numpy() - self.Structural_Elements['z_centroid_pannel'].to_numpy()) + y_dir*(self.Structural_Elements['centroid_cx_11'].to_numpy() - self.Structural_Elements['x_centroid_pannel'].to_numpy()) + z_dir*(self.Structural_Elements['centroid_cx_11'].to_numpy() - self.Structural_Elements['x_centroid_pannel'].to_numpy())
        #self.Structural_Elements['Sanity_Check_C22'] = x_dir*(self.Structural_Elements['centroid_cx_22'].to_numpy() - self.Structural_Elements['y_centroid_pannel'].to_numpy()) + y_dir*(self.Structural_Elements['centroid_cx_22'].to_numpy() - self.Structural_Elements['z_centroid_pannel'].to_numpy()) + z_dir*(self.Structural_Elements['centroid_cx_22'].to_numpy() - self.Structural_Elements['y_centroid_pannel'].to_numpy())

        #Get idx where sanity check is more than 1e-10
        #idx = np.where((np.abs(self.Structural_Elements['Sanity_Check_C11']) > 1e-10) | (np.abs(self.Structural_Elements['Sanity_Check_C22']) > 1e-10))[0]
        #print('num errors ', len(idx))
        #print(idx+2)

    def calc_1D_Measures(self):
        '''
        This function calculates the corrected 1D measurements for the structural elements
        This function loops through all structural elements. -> 
            If an element is a Girder with a flange:
            0) Moves the Flange 1D element line to the end of the Girder
            1) Calculate I11 - Moment of Inertia about the 11 axis (y axis) at the line (using parallel axis theorem)
            2) Calculate I22 - Moment of Inertia about the 22 axis (z axis) at the line (using parallel axis theorem)
            3) Calculate Jp - Polar moment of inertia about the line

            If an Structural Element is a 1D element:
            0) Combines the stiffener and the flange into a single 1D element (deletes the flange 1D element line)
            1)Calculate I11 - Moment of Inertia about the 11 axis (y axis) at the line (using parallel axis theorem)
            2) Calculate I22 - Moment of Inertia about the 22 axis (z axis) at the line (using parallel axis theorem)
            3) Calculate Jp - Polar moment of inertia about the line

        NEED TO MODIFY STIFFENER GENERATION CODE TO CREATE UNIQUE ELEMENT FOR stiffeners impinged by through holes.

        '''
        #First, add 3 columns to structural elements: I_11_1D, I_22_1D, J_p_1D
        self.Structural_Elements['I_11_1D'] = None
        self.Structural_Elements['I_22_1D'] = None
        self.Structural_Elements['J_p_1D'] = None


        #Now Loop through the structural elements
        for i in range(len(self.Structural_Elements)):
            #Check if the element is a 1D element
            vals = []
            if self.Structural_Elements.iloc[i]['bit_1D_element'] == True:
                
                
                #Condition 1: An element is a 1D element and it is not a flange (i.e. it is a stiffener)
                C1 = self.Structural_Elements.iloc[i]['Class'] != 'Stiffener_Flange'

                #Condition 2: An element is not a flange and the element after it is a flange (i+1) (also make sure that i+1 is not out of bounds) 
                if i+1 < len(self.Structural_Elements):
                    C2 = (self.Structural_Elements.iloc[i]['Class'] != 'Stiffener_Flange' and 
                      self.Structural_Elements.iloc[i+1]['Class'] == 'Stiffener_Flange') 
                else:
                    C2 = False
                    
                
                #Condition 3: An element is a flange and the element before it is not a 1D element (i-1) (a flange attached to a 2D element)
        
                C3 = (self.Structural_Elements.iloc[i]['Class'] == 'Stiffener_Flange' and 
                        (self.Structural_Elements.iloc[i-1]['bit_1D_element'] == False))
               
                
                if C1 or C3: # IF the structural element is a 1D element that is not a flange, or it is a flange attched to a 2D element
                
                    # Get dir, x_loc, y_loc, z_loc, rot, L2, Thickness, Cx11, Cx22, I11, I22, Jp of just the flange. Put into one array
                    line = self.Structural_Elements.iloc[i]['mesh_rsObject']
                    point = rs.CurveStartPoint(line)
                    
                    vals = [self.Structural_Elements.iloc[i][['x_dir', 'y_dir', 'z_dir', 'x_loc', 'y_loc', 'z_loc', 'rot',
                                                               'area_cx', 'centroid_cx_11', 'centroid_cx_22', 'I_11', 'I_22', 'J_p']].to_numpy().astype(float)]
            
                if C2:  # If a structural element is a 1D element with a flange, then we want these values for the stiffener and the flange (i+1)
                
                    vals.append(self.Structural_Elements.iloc[i+1][['x_dir', 'y_dir', 'z_dir', 'x_loc', 'y_loc', 'z_loc', 'rot',
                                                               'area_cx', 'centroid_cx_11', 'centroid_cx_22', 'I_11', 'I_22', 'J_p']].to_numpy().astype(float))    
                    
                if len(vals) == 0:    
                    #If no values were found, then continue to the next element
                    continue
                #Now calculate the 1D measures
                self.Structural_Elements.at[i, 'I_11_1D'] = 0.0
                self.Structural_Elements.at[i, 'I_22_1D'] = 0.0
                self.Structural_Elements.at[i, 'J_p_1D'] = 0.0
                #Calculate the 1D measures using the parallel axis theorem
                #I_11_1D = I_11 + CxA*(d^2)
                #I_22_1D = I_22 + CxA*(d^2)
                #J_p_1D = J_p
                
                for j in range(len(vals)):
                    #Need to calculate the distance from the centroid of the 1D element to the centroid of the structural element -> dependent on dir and rot
                    # xdir: 1, ydir: 0, zdir: 0 -> 11 axis is z, 22 axis is y
                    # xdir: 0, ydir: 1, zdir: 0 -> 11 axis is x, 22 axis is z
                    # xdir: 0, ydir: 0, zdir: 1 -> 11 axis is x, 22 axis is y
                    d_11 = vals[0][0]*(point[2] - vals[j][8]) + vals[0][1]*(point[0] - vals[j][8]) + vals[0][2]*(point[0] - vals[j][8]) #Distance in the 11 direction
                    d_22 = vals[0][0]*(point[1] - vals[j][9]) + vals[0][1]*(point[2] - vals[j][9]) + vals[0][2]*(point[1] - vals[j][9]) #Distance in the 22 direction

                    #Now calculate the 1D measures
                    self.Structural_Elements.at[i, 'I_11_1D'] += vals[j][10] + vals[j][7]*d_22**2 #I_11_1D - 
                    self.Structural_Elements.at[i, 'I_22_1D'] += vals[j][11] + vals[j][7]*d_11**2 #I_22_1D
                    self.Structural_Elements.at[i, 'J_p_1D'] += vals[j][12] #J_p_1D 
    
    
                

    def Save_Files(self, path, id):
        

        self.Structural_Elements.to_csv(path+f'/{id}_Structural_Elements.csv', index=False)
        #self.Intersection_Lines.to_csv(path+f'/{id}_Intersection_Lines.csv', index = False)
        #rs.LayerVisible('Intersections', False)

        mesh_names = []
        for i in range(len(self.Structural_Elements)):
            # Name the objects as their Object_ID
            rs.ObjectName(self.Structural_Elements.iloc[i]['struct_rsObject'], self.Structural_Elements.iloc[i]['struct_rsObject'])
            if self.Structural_Elements.iloc[i]['mesh_rsObject'] is not None:
                mesh_obj = self.Structural_Elements.iloc[i]['mesh_rsObject']
                rs.ObjectName(mesh_obj,mesh_obj)
                mesh_names.append(mesh_obj)


        #Save Structural Elements CAD as IGES
        rs.LayerVisible(self.struct_layer, True)
        rs.CurrentLayer(self.struct_layer)
        rs.LayerVisible(self.mesh_layer, False)
        rs.SelectObjects(self.Structural_Elements['struct_rsObject'].tolist())
        filename = f'_-Export \"{path}'+'/'+f'{id}.igs\" _Enter'
        rs.Command(filename, echo=False)

        #Save Mesh Elements CAD as IGES
        rs.CurrentLayer(self.mesh_layer)
        rs.LayerVisible(self.struct_layer, False)
        rs.LayerVisible(self.mesh_layer, True)
        # Select mesh elements except for "None" objects
        rs.ClearCommandHistory()

        rs.SelectObjects(mesh_names)
        #print(f"Selected {len(mesh_names)} mesh elements for export.")
        
        filename = filename = f'_-Export \"{path}'+'/'+f'{id}_MeshElements.igs\" _Enter'
        rs.Command(filename, echo=False)

        rs.CurrentLayer(self.struct_layer)
        rs.LayerVisible(self.mesh_layer, False)
        rs.LayerVisible(self.struct_layer, True)


        rs.ClearCommandHistory()

        filename = f'_-ExportAll \"{path}'+'/'+f'{id}.3dm\" _Enter'
        rs.Command(filename, echo=False)
        

        #print(f"Time to rename objects: {rename_time - start:.2f} seconds")
        #print(f"Time to export structural elements: {exp_struct_time - rename_time:.2f} seconds")
        #print(f"Time to select mesh elements: {select_time - exp_struct_time:.2f} seconds")
        #print(f"Time to export mesh elements: {exp_mesh_time - select_time:.2f} seconds")
        #print(f"Time to export all elements: {exp_all_time - exp_mesh_time:.2f} seconds")
    
   
    def compile_Structure(self):
        #rs.EnableRedraw(True)
        
        #self.make_Intersection_Lines()
        #self.Split_Bodies() #DO NOT DO THIS YET.
        
        #rs.Redraw()
        
        
        self.Save_Files(self.path,self.id)

        
        #rs.Command('_ClearUndo', echo=False)
        #rs.ClearCommandHistory()
        
        #print('Redraw Time: ', redraw_time - start)
        #print('Total Saving Time: ', total_saveing_time - redraw_time)
        #print('Clearing Time: ', clearning_time - total_saveing_time)

    def clear_Document(self):
        all_ids = rs.AllObjects()

        if all_ids:
            for obj_id in all_ids:
                rs.UnlockObject(obj_id)
                rs.ShowObject(obj_id)
            rs.DeleteObjects(all_ids)
        
        rs.ClearCommandHistory()
        rs.UnselectAllObjects()


        
        
        
   


       


        



