#This script generates a strcutred 2D drawings from rhino 3d models of ship structures. It is designed to be used with Rhino 3D and requires the Rhino Python scripting environment.

#import the necessary libraries

import rhinoscriptsyntax as rs
#import Rhino.RhinoDoc as rd
#import Rhino.Geometry as rg


import pandas as pd
import os
import numpy as np

import rhino_StructGen as rsg

import Rhino
import scriptcontext as sc
import System.Drawing

'''
======================================================
Section 1: Sheet Parameters
======================================================
'''
# landscape A3 paper size in mm
w_mm = 420.0
h_mm = 297.0

margin = 15.0 # margin in mm
tb_w, tb_h = 185.0, 45.0 # title block width and height in mm

view_w, view_h = w_mm - 2*margin-tb_w, h_mm - 2*margin # view width and height in mm

font_z_large = 3.0 # Font size for the main title and subtitle
font_z_small = 2.0 # Font size for the small text in the title block


'''
======================================================
Section 2: Rhino2DDrawing Class
======================================================
'''
class Rhino2DDrawing:
    def __init__(self, model_path, file_name, output_path):
        self.model_path = model_path
        self.file_name = file_name
        self.output_path = output_path
        self.Slice_Elements = pd.DataFrame(columns=['Slice_Name', 'x', 'Object_ID', 'struct_slice_rsObject', 'dwg_slice_rsObject', 'Ship2Dwg_Scale [1m:Xmm]', 'struct_elem_idx', 'bbox_rsObject', 'bbox_y', 'bbox_z', 'bbox_Ly', 'bbox_Lz', 'bbox_dwg_mm_y', 'bbox_dwg_mm_z', 'bbox_dwg_mm_Ly', 'bbox_dwg_mm_Lz'])
        self.Drawing_Annotations = pd.DataFrame(columns=['Object_ID', 'Slice_Name', 'dwg_slice_rsObject'])


        self.Plate_Classes = ['Side_Shell', 
                         'Bottom_Shell', 
                         'Main Deck', 
                         'Inner Bottom Deck',
                         'Longitudinal_Bulkhead', 
                         'Transverse_Bulkhead',
                         'Inner Side Shell',
                         'Upper Hopper Pannel',
                         'Lower Hopper Pannel', 
                         'Bracket_Floor_Web', 
                         'Bracket_Deck_Web',
                         'Shell_Strake']
        self.class_abbrev = {
            # plates first:
            'Transverse_Bulkhead': 'Trans. BLKHD',
            'Side_Shell': 'Side Shell',
            'Inner Side Shell': 'Inner Side Shell',
            'Bottom_Shell': 'Bottom Shell',
            'Inner Bottom Deck': 'Inner Bot. Deck',
            'Main Deck': 'Main Deck',
            'Shell_Strake': 'Shell Strake',
            'Longitudinal_Bulkhead': 'Long. BLKHD',
            'Upper Hopper Pannel': 'Upper Hopper',
            'Lower Hopper Pannel': 'Lower Hopper',
            'Bracket': 'Bracket',
            'Bracket_Deck_Web': 'Upper Bracket',
            'Bracket_Floor_Web': 'Lower Bracket',


            # Stiffeners Next: Work top to bottom
            'Deck_Longitudinal_Girder': 'Deck Girder',
            'Deck_Longitudinal_Girder_Flange': 'Deck Girder Flng.',
            'Deck_Transverse_Beam': 'Deck Beam',
            'Deck_Transverse_Beam_Flange': 'Deck Beam Flng.',
            'Deck_Longitudinal_Stiffener': 'Deck Stiff.',
            'Deck_Longitudinal_Stiffener_Flange': 'Deck Stiff. Flng.',
            'Deck_Transverse_Stiffener': 'Deck Trans. Stiff.',
            'Deck_Transverse_Stiffener_Flange': 'Deck Trans. Stiff. Flng.',

            'Vertical_Web_Frame': 'Webframe',
            'Vertical_Web_Frame_Flange': 'WebFrame Flng.',
            'Vertical_Side_Frame': 'Side Frame',
            'Vertical_Side_Frame_Flange': 'Side Frame Flng.',

            'Side_Shell_Longitudinal_Stiffener': 'Side Shell Stiff.',
            'Side_Shell_Longitudinal_Stiffener_Flange': 'Side Shell Stiff. Flng.',
            'Inner_Side_Shell_Vertical_Stiffener': 'Inner Shell Vert. Stiff.',
            'Inner_Side_Shell_Vertical_Stiffener_Flange': 'Inner Shell Vert. Stiffener Flng.',
            'Inner_Side_Shell_Longitudinal_Stiffener': 'Inner Side Shell Stiff.',
            'Inner_Side_Shell_Longitudinal_Stiffener_Flange': 'Inner Side Shell Stiff. Flng.',

            'Transverse_Floor': 'Floor',
            'Bottom_Longitudinal_Girder': 'Bot. Girder',
            'Bottom_Longitudinal_Girder_Flange': 'Bot. Girder Flng.',
            'Bottom_Longitudinal_Stiffener': 'Bot. Stiff.',
            'Bottom_Longitudinal_Stiffener_Flange': 'Bot. Stiff. Flng.',
            'Transverse_Bottom_Frame': 'Bot. Trans. Stiff.',
            'Transverse_Bottom_Frame_Flange': 'Bot. Trans. Stiff. Flng.',

            'Inner_Bottom_Longitudinal_Stiffener': 'Inner Bot. Stiff.',
            'Inner_Bottom_Longitudinal_Stiffener_Flange': 'Inner Bot. Stiff. Flng.',
            'Inner_Bottom_Transverse_Beam': 'Inner Bot. Trans. Stiff.',
            'Inner_Bottom_Transverse_Beam_Flange': 'Inner Bot. Trans. Stiff. Flng.',


            'Transverse_Bulkhead_Vertical_Stiffener': 'Vert. BLKHD Stiff.',
            'Transverse_Bulkhead_Vertical_Stiffener_Flange': 'Vert. BLKHD Stiff. Flng.',
            'Transverse_Bulkhead_Transverse_Stiffener': 'Trans. BLKHD Stiff.',
            'Transverse_Bulkhead_Transverse_Stiffener_Flange': 'Trans. BLKHD Stiff. Flng.',

            'Longitudinal_Bulkhead_Vertical_Stiffener': 'Vert. BLKHD Stiff.',
            'Longitudinal_Bulkhead_Vertical_Stiffener_Flange': 'Vert. BLKHD Stiff. Flng.',
            'Longitudinal_Bulkhead_Longitudinal_Stiffener': 'Long. BLKHD Stiff.',
            'Longitudinal_Bulkhead_Longitudinal_Stiffener_Flange': 'Long. BLKHD Stiff. Flng.',
            

            'Upper_Hopper_Pannel_Longitudinal_Stiffener': 'Upper Hopper Stiff.',
            'Upper_Hopper_Pannel_Longitudinal_Stiffener_Flange': 'Upper Hopper Stiff. Flng.',
            'Lower_Hopper_Pannel_Longitudinal_Stiffener': 'Lower Hopper Stiff.',
            'Lower_Hopper_Pannel_Longitudinal_Stiffener_Flange': 'Lower Hopper Stiff. Flng.',

            }
        
        

        # 


    def load_Data(self):
        #Open a 3D model and the Structural Elements.csv File
        self.geom_file = '\"'+self.model_path + '/' + self.file_name + '.3dm\"'
        struct_elem_path = self.model_path + '/' + self.file_name + '_Structural_Elements.csv'

        rs.EnableRedraw(False) # Disable redraw to speed up the process

        rs.Command("_-Open " + self.geom_file + " _Enter") # Open the 3D model in Rhino
        self.Structural_Elements = pd.read_csv(struct_elem_path) # Load the Structural Elements.csv file into a pandas DataFrame

        # Get Layers in the Document and delete all the layers except for 'Default', 'Main Structure', 'Mesh Structure', and 'Hull'
        PRIMARY_LAYERS = ['Default', 'Main Structure', 'Mesh Structure', 'Hull']
        
        layers = rs.LayerNames()

        for layer in layers:
            if layer not in PRIMARY_LAYERS:
                rs.PurgeLayer(layer) # Purge the layer to delete it from the document

    def close_Doc(self):
        # This function closes the current Rhino document without saving changes.
        # First hide all layers:
        rs.ViewProjection("Perspective", 2)
        rs.CurrentView("Perspective")
        rs.CurrentLayer('Default') # Set the current layer to 'Default' to avoid issues when closing the document
        for layer in rs.LayerNames():
            rs.LayerVisible(layer, False)
        
        #Now make 'Main Structure' layer visible
        
        rs.LayerVisible('Main Structure', True)
        rs.Command(f"_-Save {self.geom_file} _Enter", False) # Save the current document without prompting the user
        
        rs.Command(f"_-Close {self.geom_file}") # Close the current document and save 
        
        #Save the Slice_Elements DataFrame to a CSV file in the output path
        output_csv_path = os.path.join(self.output_path, self.file_name + '_Slice_Elements.csv')
        self.Slice_Elements.to_csv(output_csv_path, index=False)
        self.Drawing_Annotations.to_csv(os.path.join(self.output_path, self.file_name + '_Drawing_Annotations.csv'), index=False)
        #rs.EnableRedraw(True) # Enable redraw after closing the document
        rs.ClearCommandHistory() # Clear the command history to prevent any prompts from previous commands

    def extract_X_slice_Positions(self):
        # This function extracts the X slice positions from the Structural Elements.csv file and returns them as a list of keys and values.
        x_bulkheads = self.Structural_Elements[self.Structural_Elements['Class'] == 'Transverse_Bulkhead']['x_loc'].values
        x_WebFrames = self.Structural_Elements[self.Structural_Elements['Class'] == 'Vertical_Web_Frame']['x_loc'].values
        x_Frames = self.Structural_Elements[self.Structural_Elements['Class'] == 'Vertical_Side_Frame']['x_loc'].values

        self.df_Slices = pd.DataFrame(columns=['Slice_Name', 'x'])
        self.df_Slices['Slice_Name'] = ['Transverse Bulkhead', 
                                        'Midship Section IWO of Web Frame', 
                                        'Midship Section of Long. Structure']
        x0 = x_bulkheads[0] if len(x_bulkheads) > 0 else None
        #x1 is middle index of webframes between bulkheads[0] and bulkheads[1]
        x_WebFrames = x_WebFrames[(x_WebFrames > x0) & (x_WebFrames < x_bulkheads[1])] if len(x_bulkheads) > 1 else x_WebFrames[x_WebFrames > x0]
        
        x1 = x_WebFrames[len(x_WebFrames)//2] if len(x_WebFrames) > 0 else None
        #x2 is first x_frame after x1
        x_Frames = x_Frames[x_Frames > x1] if x1 is not None else x_Frames
        x2 = x_Frames[0] if len(x_Frames) > 0 else None

        #X3 is halfway between x1 and x2
        x3 = (x1 + x2) / 2 if x1 is not None and x2 is not None else None   

        self.df_Slices['x'] = [x0, x1, x3] # No Frame Slice 

    def rename_Structural_Element_Classes(self):
        # This class reviews the Structural Elements and renames the classes of the elements based on their geometric properties. This is important for correctly identifying the elements in the 2D drawings and applying the appropriate formatting and annotations.
        # We need to separate upper and lower hoppers: 
        unique_classes = self.Structural_Elements['Class'].unique()

        if 'Hopper Pannel' in unique_classes:
            # There will be two. the one with the greater z_loc will be the upper hopper, and the one with the smaller z_loc will be the lower hopper. We will rename the classes accordingly.
            # Also, recategorize the hopper pannel stiffeners as upper and lower hopper pannel stiffeners based on the ID of the Parent Struct and the Object ID of the hopper pannels
            df_hoppers = self.Structural_Elements[self.Structural_Elements['Class'] == 'Hopper Pannel']
            if len(df_hoppers) == 2:
                if df_hoppers['z_loc'].values[0] > df_hoppers['z_loc'].values[1]:
                    self.Structural_Elements.loc[self.Structural_Elements['Class'] == 'Hopper Pannel', 'Class'] = ['Upper Hopper Pannel', 'Lower Hopper Pannel']
                else:
                    self.Structural_Elements.loc[self.Structural_Elements['Class'] == 'Hopper Pannel', 'Class'] = ['Lower Hopper Pannel', 'Upper Hopper Pannel']

            # Assign Hopper Pannel stiffeners to upper and lower hopper pannels based on the Parent Struct. If the Parent Struct ID of the stiffener matches the Object ID of the upper hopper pannel, then it is an upper hopper pannel stiffener. If it matches the Object ID of the lower hopper pannel, then it is a lower hopper pannel stiffener.
            if 'Hopper_Pannel_Longitudinal_Stiffener' in unique_classes:
                df_stiffeners = self.Structural_Elements[self.Structural_Elements['Class'] == 'Hopper_Pannel_Longitudinal_Stiffener']
                for index, row in df_stiffeners.iterrows():
                    parent_struct_id = row['Parent_Struct']
                    if parent_struct_id == self.Structural_Elements[self.Structural_Elements['Class'] == 'Upper Hopper Pannel']['Object_ID'].values[0]:
                        self.Structural_Elements.at[index, 'Class'] = 'Upper_Hopper_Pannel_Longitudinal_Stiffener'
                        # Correct Flange as well
                        self.Structural_Elements.at[index+1, 'Class'] = 'Upper_Hopper_Pannel_Longitudinal_Stiffener_Flange'

                    elif parent_struct_id == self.Structural_Elements[self.Structural_Elements['Class'] == 'Lower Hopper Pannel']['Object_ID'].values[0]:
                        self.Structural_Elements.at[index, 'Class'] = 'Lower_Hopper_Pannel_Longitudinal_Stiffener'
                        # Correct Flange as well
                        self.Structural_Elements.at[index+1, 'Class'] = 'Lower_Hopper_Pannel_Longitudinal_Stiffener_Flange'

        
        
        #Now lets check Longitudinal Bulkheads: 
        # If there is one with a y_loc = y_loc of vertical web frame - vertical web frame L2, 
        # then it should be classed as an inner side shell, but other longitudinal bulkheads should remain as longitudinal bulkheads.
        if 'Longitudinal_Bulkhead' in unique_classes:
            webframe_y = self.Structural_Elements[self.Structural_Elements['Class'] == 'Vertical_Web_Frame']['y_loc'].values[0]
            webframe_L2 = self.Structural_Elements[self.Structural_Elements['Class'] == 'Vertical_Web_Frame']['L2'].values[0]
            bulkhead_y = self.Structural_Elements[self.Structural_Elements['Class'] == 'Longitudinal_Bulkhead']['y_loc'].values

            for i in range(len(bulkhead_y)):
                if abs(bulkhead_y[i] - (webframe_y - webframe_L2)) < 1e-4:
                    # Get the index of the longitudinal bulkhead
                    idx = self.Structural_Elements[self.Structural_Elements['Class'] == 'Longitudinal_Bulkhead'].index[i]
                    # Rename the class to Inner_Side_Shell
                    self.Structural_Elements.at[idx, 'Class'] = 'Inner Side Shell'
                    
        
        # Now let's reclass the two decks as main deck and inner bottom deck based on their z_loc. The one with the greater z_loc will be the main deck, and the one with the smaller z_loc will be the inner bottom deck. We will rename the classes accordingly.
        if 'Deck' in unique_classes:
            df_decks = self.Structural_Elements[self.Structural_Elements['Class'] == 'Deck']
            if len(df_decks) == 2:
                if df_decks['z_loc'].values[0] > df_decks['z_loc'].values[1]:
                    self.Structural_Elements.loc[self.Structural_Elements['Class'] == 'Deck', 'Class'] = ['Main Deck', 'Inner Bottom Deck']
                else:
                    self.Structural_Elements.loc[self.Structural_Elements['Class'] == 'Deck', 'Class'] = ['Inner Bottom Deck', 'Main Deck']

        self.Structural_Elements['Class_Abbr'] = self.Structural_Elements['Class'].apply(lambda x: self.class_abbrev.get(x, x))

       


    def get_guids_For_Slice(self, slice_name): 
        # This function returns the GUIDs and Structural Elements Index of the objects in the 3D model that correspond to a given slice name.

        if slice_name == 'Midship Section of Long. Structure': # If the slice is midship longitudinal structure, return all objects with x_dir = 1 -> all longitudinal structures
            guids = self.Structural_Elements[self.Structural_Elements['x_dir'] == 1]['struct_rsObject'].values
            idx = self.Structural_Elements[self.Structural_Elements['x_dir'] == 1].index.values
        else:
            #Otherwise, return all objects with x_loc = x of the slice name
            x = self.df_Slices[self.df_Slices['Slice_Name'] == slice_name]['x'].values[0]
            #Give some leeway to the x position of the slice, so that we can capture all objects that are close to the slice position. This is important because some objects may not be exactly at the slice position due to modeling inaccuracies.
            dx = 0.01 # 1 cm leeway
            guids = self.Structural_Elements[self.Structural_Elements['x_loc'].between(x-dx, x+dx)]['struct_rsObject'].values
            idx = self.Structural_Elements[self.Structural_Elements['x_loc'].between(x-dx, x+dx)].index.values

            #Need to add Flanges, Decks, Side Shell, Bottom Shell, Longitudinal Bulkheads
            obj_ids = self.Structural_Elements[self.Structural_Elements['struct_rsObject'].isin(guids)]['Object_ID'].values
            for obj_id in obj_ids:
                #Make
                if obj_id+'_Flange' not in obj_ids and obj_id+'_Flange' in self.Structural_Elements['Object_ID'].values:

                    guids = np.concatenate((guids, [self.Structural_Elements[self.Structural_Elements['Object_ID'] == obj_id+'_Flange']['struct_rsObject'].values[0]]))
                    idx = np.concatenate((idx, [self.Structural_Elements[self.Structural_Elements['Object_ID'] == obj_id+'_Flange'].index.values[0]]))
            
            DSL = self.Structural_Elements[self.Structural_Elements['Class'].isin(self.Plate_Classes)]['struct_rsObject'].values
            DSL_idx = self.Structural_Elements[self.Structural_Elements['Class'].isin(self.Plate_Classes)].index.values
            guids = np.concatenate((guids, DSL))
            idx = np.concatenate((idx, DSL_idx))

        
        return guids, idx
    
    def create_Slice(self, slice_name):
        # This function creates a 2D slice of the 3D model at the specified slice name.
        guids, struct_elem_idx = self.get_guids_For_Slice(slice_name) # Get the GUIDs of the objects in the slice
        if len(guids) == 0:
            print(f"No objects found for slice {slice_name}.")
            return

        # Create a new layer for the slice
        layer_name = f"{slice_name}_Slice"
        if rs.IsLayer(layer_name):
            rs.PurgeLayer(layer_name) # Purge the layer if it already exists
            rs.AddLayer(layer_name)
        else:
            rs.AddLayer(layer_name) # Add a new layer for the slice

        #Make later active
        rs.CurrentLayer(layer_name)

        # Draw a plane at the x position of the slice
        x = self.df_Slices[self.df_Slices['Slice_Name'] == slice_name]['x'].values[0]
        plane = rs.AddPlaneSurface(rs.WorldYZPlane(), 1000, 1000) # Create a large plane at the origin
        #move plane to x position of slice
        rs.MoveObject(plane, [x, 0, 0]) # Move the plane to the x position of the slice


        #Intersect the objects with the plane to create 2D curves

        items = pd.DataFrame(columns=['Slice_Name', 'x', 'Object_ID', 'struct_slice_rsObject', 'struct_elem_idx'])

        for i in range(len(guids)):
            guid = guids[i]
            intersect = rs.IntersectBreps(plane, guid) # Intersect the object with the plane to create a 2D curve

            item = pd.DataFrame({'Slice_Name': [slice_name], 
                          'x': [x], 
                          'Object_ID': [self.Structural_Elements[self.Structural_Elements['struct_rsObject'] == guid]['Object_ID'].values[0]], 
                          'struct_slice_rsObject': intersect, 
                          'struct_elem_idx': [struct_elem_idx[i]]})
            #Add item to items
            items = pd.concat([items, item], ignore_index=True)
            
        self.Slice_Elements = pd.concat([self.Slice_Elements, items], ignore_index=True) # Append the new slice elements to the main DataFrame

        rs.DeleteObject(plane) # Delete the plane after intersection

    def scale_Slice(self, slice_name):
        # this function determines the scaling factor and YZ offsets to move a slice into the viewable area of the layout. 

        #First determine a scaling from the Depth and half Beam of the structure

        D = self.Structural_Elements[self.Structural_Elements['Class'] == 'Transverse_Bulkhead']['L2'].values[0] 
        Bo2 = self.Structural_Elements[self.Structural_Elements['Class'] == 'Transverse_Bulkhead']['L1'].values[0]

        #Then figure use view_W and view_H to determine the scaling factor based on the dimensions of the slice. with a margin around the viewing box

        scale_Z = (view_h - 2*margin) / D
        scale_Y = (view_w - 2*margin) / Bo2

        scale = min(scale_Y, scale_Z) # Use the smaller scaling factor to ensure the slice fits within the viewable area
        scale = round(scale, 2) # Round the scale to 2 decimal places for cleaner annotation in the title block

        Ds = scale * D
        Bo2s = scale * Bo2

        # Determine origin position so that a structure is centered in view_w and view_h
        origin_s = [0, (view_w -Bo2s)/2.0 + margin, (view_h - Ds)/2.0 + margin]

        #Set Current Layer to Drawing Annotation Layer  
        dwg_layer = f'DWG_{slice_name}'

        rs.CurrentLayer(dwg_layer) # Set the current layer to the drawing annotation layer

        ## Now lets loop through the slice elements and scale and move them to the appropriate position in the layout. We will store the scaled and moved objects in a new column in the Slice_Elements DataFrame called 'layout_rsObject'.
        for index, row in self.Slice_Elements[self.Slice_Elements['Slice_Name'] == slice_name].iterrows():
            if row['struct_slice_rsObject'] is not None:
                scaled_obj = rs.CopyObject(row['struct_slice_rsObject'])
                #Change layer of scaled object to
                rs.ObjectLayer(scaled_obj, dwg_layer)

                # Move Scaled Object to X = 0 
                scaled_obj = rs.MoveObject(scaled_obj, [-row['x'], 0, 0]) # Move the slice element to X = 0
    
                scaled_obj = rs.ScaleObject(scaled_obj, [0,0,0], [scale, scale, scale]) # Scale the slice element by the scaling factor

                scaled_obj = rs.MoveObject(scaled_obj, origin_s) # Move the slice element to the appropriate position in the layout
                self.Slice_Elements.at[index, 'dwg_slice_rsObject'] = scaled_obj # Store the scaled and moved object in the DataFrame
                self.Slice_Elements.at[index, 'Ship2Dwg_Scale [1m:Xmm]'] = scale # Store the scaling factor from meters to drawing millimeters in the DataFrame
        
        if slice_name == 'Transverse Bulkhead':
            self.change_Bulkhead_DWG_Line_Types() # Change the line types of the bulkhead drawing to differentiate between plates, stiffeners, and girders/beams/webframes

        return scale, origin_s 
    

    def change_Bulkhead_DWG_Line_Types(self):
        '''
        This function changes the line type of stiffeners, girders, and plates for the Bulkhead DWG. 
        plates are solid lines
        stiffeners are dashed lines
        girders/beams/webframes are dash-dot lines
        '''

        for index, row in self.Slice_Elements[self.Slice_Elements['Slice_Name'] == 'Transverse Bulkhead'].iterrows():
            if row['struct_slice_rsObject'] is not None:
                obj_id = row['Object_ID']
                if 'Stiffener' in obj_id:
                    rs.ObjectLinetype(row['dwg_slice_rsObject'], 'Hidden') # Set line type to dashed for stiffeners
                elif 'Girder' in obj_id or 'Beam' in obj_id or 'Web_Frame' in obj_id:
                    rs.ObjectLinetype(row['dwg_slice_rsObject'], 'HiddenX2') # Set line type to dash-dot for girders/beams/webframes
                else:
                    rs.ObjectLinetype(row['dwg_slice_rsObject'], 'Continuous') # Set line type to solid for plates
        



    def create_Bounding_Boxes(self, slice_name, scale, origin_s):
        # This function creates bounding boxes for each slice element and adds the bounding box information to the Slice_Elements DataFrame.

        #add Bounding Box Layer for slice
        bbox_layer_name = f"{slice_name}_BBoxes"
        if rs.IsLayer(bbox_layer_name):
            rs.PurgeLayer(bbox_layer_name) # Purge the layer if it already exists
            rs.AddLayer(bbox_layer_name)
        else:
            rs.AddLayer(bbox_layer_name)
        rs.CurrentLayer(bbox_layer_name)
        #Change the color of the bounding box layer to red
        rs.LayerColor(bbox_layer_name, (255, 0, 0))

        subslice = self.Slice_Elements[self.Slice_Elements['Slice_Name'] == slice_name]

        for index, row in subslice.iterrows():
            if row['struct_slice_rsObject'] is not None:
                bbox = rs.BoundingBox(row['struct_slice_rsObject']) # Get the bounding box of the slice element
                x = row['x']
                if bbox:
                    min_y = min([point[1] for point in bbox])
                    max_y = max([point[1] for point in bbox])
                    min_z = min([point[2] for point in bbox])
                    max_z = max([point[2] for point in bbox])

                    # Add 1mm to the bounding box dimensions to give some clearance in the drawing
                    P1 = np.array([min_y, min_z])- np.array([1/scale, 1/scale]) # P1 is the bottom left corner of the bounding box in the YZ plane
                    P2 = np.array([max_y, max_z]) + np.array([1/scale, 1/scale]) # P2 is the top right corner of the bounding box in the YZ plane
                    

                    self.Slice_Elements.at[index, 'bbox_y'] = P1[0]
                    self.Slice_Elements.at[index, 'bbox_z'] = P1[1]
                    self.Slice_Elements.at[index, 'bbox_Ly'] = P2[0] - P1[0]
                    self.Slice_Elements.at[index, 'bbox_Lz'] = P2[1] - P1[1]

                    #Now scale 
                    P1 = P1 * scale + origin_s[1:]
                    P2 = P2 * scale + origin_s[1:]


                    points = np.array([[0, P1[0], P1[1]], 
                              [0, P2[0], P1[1]], 
                              [0, P2[0], P2[1]], 
                              [0, P1[0], P2[1]], 
                              [0, P1[0], P1[1]]]) # Create a list of points for the bounding box
                    bb = rs.AddPolyline(points) # Draw the bounding box in Rhino

                    self.Slice_Elements.at[index, 'bbox_rsObject'] = bb # Store the bounding box object in the DataFrame
                    self.Slice_Elements.at[index, 'bbox_dwg_mm_y'] = P1[0]
                    self.Slice_Elements.at[index, 'bbox_dwg_mm_z'] = P1[1]
                    self.Slice_Elements.at[index, 'bbox_dwg_mm_Ly'] = P2[0] - P1[0]
                    self.Slice_Elements.at[index, 'bbox_dwg_mm_Lz'] = P2[1] - P1[1]

    def create_Layout(self, slice_name):
        # This function creates a layout for the specified slice name. It arranges the bounding boxes of the slice elements in a grid format on an A3 sheet.
        #Make Drawing Annotation Layer make its color black
        dwg_layer = f'DWG_{slice_name}'
        if rs.IsLayer(dwg_layer):
            rs.PurgeLayer(dwg_layer) # Purge the layer if it already exists
            rs.AddLayer(dwg_layer)
        else:
            rs.AddLayer(dwg_layer) # Add a new layer for the drawing annotation

        rs.CurrentLayer(dwg_layer) # Set the current layer to the drawing annotation layer

        rs.LayerColor(dwg_layer, (0, 0, 0)) # Set the color of the drawing annotation layer to black


        #rs.CurrentLayer('Default') # Set the current layer to the drawing annotation layer
        #Hide all other layers except for the slice layer
        for layer in rs.LayerNames():
            if layer != dwg_layer and layer != f"{slice_name}_Slice" and layer != f"{slice_name}_BBoxes":
                rs.LayerVisible(layer, False)
            else:
                rs.LayerVisible(layer, True)

        self.format_Layout(slice_name)

    def Add_Margin_Labels(self, slice_name):
        #This function adds ABCDEFGH and 12345678 around the margin of the layout to help with identifying locations in the drawing. 
        for i in range(8):
            #Add letters to the left margin
            
            plane = rs.MovePlane(rs.WorldYZPlane(), [0, margin/2.0, margin + (i+0.5)*(h_mm-2.0*margin)/8.0]) # Move the plane to the left margin and the appropriate height
            text_l = rs.AddText(chr(65+i), plane, height=font_z_small) # Add letters A-H to the left margin
            
            #Add letters to the right margin
            plane = rs.MovePlane(rs.WorldYZPlane(), [0, w_mm - margin/2.0, margin + (i+0.5)*(h_mm-2.0*margin)/8.0]) # Move the plane to the right margin and the appropriate height
            text_r = rs.AddText(chr(65+i), plane, height=font_z_small) # Add letters A-H to the right margin
            #Add numbers to the top margin
       
            plane = rs.MovePlane(rs.WorldYZPlane(), [0, margin + (i+0.5)*(w_mm-2.0*margin)/8.0, h_mm - margin/2.0]) # Move the plane to the top margin and the appropriate width
            text_t = rs.AddText(str(i+1), plane, height=font_z_small) # Add numbers 1-8 to the top margin
            #Add numbers to the bottom margin
            plane = rs.MovePlane(rs.WorldYZPlane(), [0, margin + (i+0.5)*(w_mm-2.0*margin)/8.0, margin/2.0]) # Move the plane to the bottom margin and the appropriate width
            text_b = rs.AddText(str(i+1), plane, height=font_z_small) # Add numbers 1-8 to the bottom margin

            self.Drawing_Annotations = pd.concat([self.Drawing_Annotations, pd.DataFrame({'Object_ID': ['Margin Label']*4, 'Slice_Name': [slice_name]*4, 'dwg_slice_rsObject': [text_l, text_r, text_t, text_b]})], ignore_index=True) # Append the new margin labels to the main DataFrame


    def format_Layout(self, slice_name):
        # Add formatting and title block to the layout. This function can be customized to add specific formatting and title block information as needed.


        # Add bounding box
        rect_guid = rs.AddPolyline([[0, 0, 0], 
                                    [0, w_mm, 0], 
                                    [0, w_mm, h_mm], 
                                    [0, 0, h_mm], 
                                    [0, 0, 0]]) # Create a rectangle for the layout border
        
        #Add small offset interior rectangle
        rect_2_guid = rs.AddPolyline([[0, 2, 2], 
                                    [0, w_mm-2, 2], 
                                    [0, w_mm-2, h_mm-2], 
                                    [0, 2, h_mm-2], 
                                    [0, 2, 2]]) # Create a rectangle for the layout border
        #Add margin to the rectangle
        margin_rect_guid = rs.AddPolyline([[0, margin, margin],
                                            [0, margin, h_mm-margin],
                                            [0, w_mm-margin, h_mm-margin],
                                            [0, w_mm-margin, margin],
                                            [0, margin, margin]]) # Create a rectangle for the layout margin
        
        items = pd.DataFrame({'Object_ID': ['Page Border', 'Exterior Border', 'Layout Margin'],
                            'Slice_Name': [slice_name, slice_name, slice_name],
                            'dwg_slice_rsObject': [rect_guid,rect_2_guid,  margin_rect_guid]})
        self.Drawing_Annotations = pd.concat([self.Drawing_Annotations, items], ignore_index=True) # Append the new layout annotations to the main DataFrame


        self.Add_Margin_Labels(slice_name)
        


    def create_Title_Block(self, slice_name, scale): 
        #this function creates a title block in the layout. The title block can be customized to include specific information such as project name, drawing number, date, etc.
        points = [  [0,w_mm-tb_w-margin, margin], 
                    [0,w_mm-tb_w-margin, margin+tb_h], 
                    [0,w_mm-margin, margin+tb_h],
                    [0,w_mm-margin, margin],
                    [0,w_mm-tb_w, margin]]
        
        left_y = w_mm-tb_w-margin
        
        title_box = rs.AddPolyline(points) # Draw the title block in Rhino

        Title = self.file_name
        Subtitle = slice_name
        dwg_num = self.df_Slices[self.df_Slices['Slice_Name'] == slice_name].index[0] + 1 # Drawing number is the index of the first slice element + 1
        dwg_num_str = f"{dwg_num} of {len(self.df_Slices)}" # Format the drawing number as DWG-001, DWG-002, etc.


        '''
        *Title Block Layout* Subtitles of each block are in font size small
        Full size text is font size large 
        ____________________________________________________________
        | Title:                                                   |
        |__________________________________________________________| 
        | subtitle:                      | drawing number          |
        _________________________________|_________________________|
        | Scale: 1m = {Caclulated Scale} Revision: {blank} |       |
        ____________________________________________________________|
        
        '''

        # Draw horizontal dividers
        title_row_h = tb_h/3.0


        rs.AddLine([0,left_y, margin + title_row_h], [0, w_mm-margin, margin + title_row_h]) # horiz line bottom
        rs.AddLine([0, left_y, margin + 2*title_row_h], [0, w_mm-margin, margin + 2*title_row_h]) # horiz line middle
        rs.AddLine([0, left_y, margin + 3*title_row_h], [0, w_mm-margin, margin + 3*title_row_h]) # horiz line top

        # Draw vertical dividers
        rs.AddLine([0,left_y + 0.6667*tb_w, margin], [0,left_y + 0.6667*tb_w, margin + 2* title_row_h]) # vert line left
        #rs.AddLine([0,left_y + 0.3333*tb_w, margin + title_row_h], [0,left_y + 0.333*tb_w, margin])

        # Add information text to the title block
        point = [0,left_y + 1.0, margin + tb_h - font_z_small-1.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the title block text
        title_guid = rs.AddText("Project Title:", plane, height=font_z_small)


        point = [0,left_y + 1.0, margin + tb_h - title_row_h - font_z_small-1.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the title block text
        subtitle_guid = rs.AddText("Subtitle:", plane, height=font_z_small)
       
        point = [0,left_y + 0.6667*tb_w + 1.0, margin + tb_h - title_row_h - font_z_small-1.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the title block text
        dwgID_guid = rs.AddText("Dwg ID No.", plane, height=font_z_small)
        
        point = [0,left_y + 1.0, margin + tb_h - 2*title_row_h - font_z_small-1.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the title block text
        scale_guid =  rs.AddText("Scale:", plane, height=font_z_small)
        
        point = [0,left_y + 0.6667*tb_w + 1.0, margin + tb_h - 2*title_row_h - font_z_small-1.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the title block text
        rev_guid = rs.AddText("Revision:", plane, height=font_z_small)
        

        items = pd.DataFrame({'Object_ID': ['dwg_id_header',
                                                'title_header',
                                                'subtitle_header',
                                                'scale_header',
                                                'rev_header'],
                            'Slice_Name': [slice_name]*5,
                            'dwg_slice_rsObject': [dwgID_guid, title_guid, subtitle_guid,  scale_guid, rev_guid]})
        self.Drawing_Annotations = pd.concat([self.Drawing_Annotations, items], ignore_index=True) # Append the new title block subtitles to the main DataFrame
    

        # Fill in the title block information
        point = [0,left_y + 10.0, margin + tb_h - 10.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the title block text
        TITLE_guid = rs.AddText('MiDShip Dataset: ' + Title, plane, height=font_z_large)

        point = [0,left_y + 10.0, margin + tb_h - title_row_h - 10.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the title block text
        SUBTITLE_guid = rs.AddText(Subtitle, plane, height=font_z_large)
    

        #format dwg number as DWG-001, DWG-002, etc.
        point = [0,left_y + 0.6667*tb_w + 10.0, margin + tb_h - title_row_h - 10]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the title block text
        DWGid_guid = rs.AddText(f'{dwg_num} of 3', plane, height=font_z_large)

        #Determin the scale of the drawing for now, use a fixed scale of 1:200, but this can be changed later
        point = [0,left_y + 10.0, margin + tb_h - 2*title_row_h - 10.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the
        SCALE_guid = rs.AddText(f'1 m = {scale} mm', plane, height=font_z_large)

        point = [0,left_y + 0.6667*tb_w + 10.0, margin + tb_h - 2*title_row_h - 10.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the
        REV_guid = rs.AddText("Rev-0", plane, height=font_z_large)

        items = pd.DataFrame({'Object_ID': ['DWG_ID',
                                                'TITLE',
                                                'SUBTITLE',
                                                'SCALE',
                                                'REVISION'],
                            'Slice_Name': [slice_name]*5,
                            'dwg_slice_rsObject': [DWGid_guid,TITLE_guid, SUBTITLE_guid, SCALE_guid, REV_guid]})
        self.Drawing_Annotations = pd.concat([self.Drawing_Annotations, items], ignore_index=True) # Append the new title block information to the main DataFrame



    def extract_Slice_Data(self, slice_name):
        # THis function extracts information from Structural Elements to display in the an info block in the drawing. 
        '''
        We are going to use this data to build two tables: One for plates and one for stiffeners.
        The data we need to extract: 
        0) Name
        1) Type (Plate or Stiffener)
        2) Thickness (plates only)
        3) Dims (Flange shape, H, Flange W, thickness) - stiffeners only
        4) Material  = ASTM A131 - AB/A


        DF classes will be 
        'Class' 
        'Type'
        'Dims'
        'Spacing'
        'Number'
        'Material'

        '''
        #The main data: Object Class, number in the structure, Spacing, Size, 
        df_slice_info =pd.DataFrame(columns=['Class',
                                             'Class_Abbr',
                                            'Type',
                                            'Dims',
                                            'Spacing',
                                            'Number',
                                            'Material'])
        
        #Things to extract from the slice info: x_s for spacing, and number of elements of longitudinally spaced structural elements. 
        if slice_name == 'Transverse Bulkhead':
            bulkhead_x = self.Structural_Elements[self.Structural_Elements['Class'] == 'Transverse_Bulkhead']['x_loc'].values
            x_s = bulkhead_x[1] - bulkhead_x[0] 
     
        elif slice_name == 'Midship Section IWO of Web Frame':
            # For Web Frames, we need to extract the spacing between the web frames, which is the difference between the x locations of the web frames. 
            '''
            items to extract: 
            Vertical_Web_Frame: Number, Spacing, Size
            Vertical_Web_Frame_Flange: Number, Spacing, Size
            Deck_Beam: Number, Spacing, Size (if it exists in the slice)
            Deck_Beam_Flange: Number, Spacing, Size (if it exists in the slice)
            Bottom Floor_Beam: Number, Spacing, Size 
            Brackets: Number, Spacing, Size (if it exists in the slice)
            
            '''
            webframe_x = self.Structural_Elements[self.Structural_Elements['Class'] == 'Vertical_Web_Frame']['x_loc'].values
            x_s = webframe_x[1] - webframe_x[0]
          
        elif slice_name == 'Midship Section of Long. Structure':
            x_s = None # For longitudinal structure, spacing is not applicable
        
        
        #Now let's extract the Structural_Elements in the slice and their information
        idx = self.Slice_Elements[self.Slice_Elements['Slice_Name'] == slice_name]['struct_elem_idx'].values
        
        df_elem = self.Structural_Elements.loc[idx]

        unique_classes = df_elem['Class'].unique().tolist()

        #Search through df_elem and reassign class of flanges to their corresponding structural element class with '_Flange' suffix. For example, if the Object_ID of an element is 'WebFrame_001_Flange', we will reassign its class to 'Vertical_Web_Frame_Flange'.
        # We will drop the flanges from unique classes, but check for fange to get L2 dim and check the xyz position to determine if the flange is a T or L flange
        for index, row in df_elem.iterrows():
            if '_Flange' in row['Object_ID']:
                base_obj_id = row['Object_ID'].replace('_Flange', '')
                base_class = self.Structural_Elements[self.Structural_Elements['Object_ID'] == base_obj_id]['Class'].values
                if len(base_class) > 0:
                    df_elem.at[index, 'Class'] = base_class[0] + '_Flange'

        

        # Now Let's Add the Side_Shell, Bottom_Shell, Deck, Longitudinal_Bulkhead, and Hopper Pannel to the info block if they exist in the slice
        for cls in self.Plate_Classes:
            df_cls = df_elem[df_elem['Class'] == cls]
            

            if len(df_cls) > 0:
                class_abbr = df_cls['Class_Abbr'].values[0]
                number = len(df_cls)
                dim = int(df_cls['Thickness'].values[0]*1000) # convert to mm for display
                spacing = ' '
                
                df_slice_info = pd.concat([df_slice_info, pd.DataFrame({'Class': [cls], 'Class_Abbr': [class_abbr], 'Type': ['Plate'], 'Dims': [str(dim) + ' mm'], 'Material': ['ASTM A131 - AB/A']})], ignore_index=True)
        
        #Drop 'Side_shell', 'Bottom_Shell', 'Deck', 'Longitudinal_Bulkhead' and 'Hopper Pannel' from the unique classes, as they are not structural elements that we want to display in the info block
        stiffener_classes = [cls for cls in unique_classes if cls not in self.Plate_Classes ]



        for cls in stiffener_classes:
            if '_Flange' not in cls:
          
                df_cls = df_elem[df_elem['Class'] == cls]
                class_abbr =  df_cls['Class_Abbr'].values[0]
                number = len(df_cls)
            
                #dims is a string that contains the dimensions of a stiffener: Type of flange (T or L), H, Flange W, Thickness. For example, a T flange with H = 300 mm, Flange W = 150 mm, and Thickness = 10 mm would be formatted as 'T-300x150x10'. An L flange with the same dimensions would be formatted as 'L-300x150x10'. If there is no flange, the format would be 'H-300x10'.
                L2 = df_cls['L2'].values[0] # round L2 to 3 decimal places and convert to string for display
                L2 = f'{L2:.3f}'

                T = int(df_cls['Thickness'].values[0]*1000) # convert to mm for display
                if cls+'_Flange' in df_elem['Class'].values:
                    df_flange = df_elem[df_elem['Class'] == cls+'_Flange']
                    W = df_flange['L2'].values[0]  
                    W = f'{W:.3f}' # round W to 3 decimal places and convert to string for display

                    # Check x_loc, y_loc, and z_loc of the flange to determine if it is a T or L flange. If the flange is located at the same x_loc as the main element, it is a T flange. If the flange is located at the same y_loc or z_loc as the main element, it is an L flange.
                    main_x = df_cls['x_loc'].values[0]
                    main_y = df_cls['y_loc'].values[0]
                    main_z = df_cls['z_loc'].values[0]
                    flange_x = df_flange['x_loc'].values[0]
                    flange_y = df_flange['y_loc'].values[0]
                    flange_z = df_flange['z_loc'].values[0]

                    diff = np.array([abs(main_x - flange_x), abs(main_y - flange_y), abs(main_z - flange_z)])
                    # If two coodinates match -> L flange, if only one coordinate matches -> T flange
                    if sum(diff < 1e-6) == 2: # L flange
                        flange_type = 'L-'
                    else:
                        flange_type = 'T-'
                else:
                    W = '0'
                    flange_type = 'I-'

                dims = flange_type + L2 + ' m x ' + W + ' m x ' + str(T) + ' mm'
                

                if number == 1: 
                    try:
                        spacing = f'{x_s:.3f} m' # For a single element, spacing is the distance to the next element of the same class. We will use the x_s value calculated earlier for this, which is the spacing between web frames or bulkheads. If x_s is not applicable (for longitudinal structure), we will leave it blank.
                    except:
                        spacing = ' '

                elif cls == 'Transverse_Bottom_Frame':
                    spacing = f'{x_s:.3f} m'
                elif cls == 'Transverse_Bottom_Frame_Flange':
                    spacing = f'{x_s:.3f} m'
                else: 
                    #determine dir 
                    if df_cls['x_dir'].values[0] == 1: # longitudinally spaced, extract distance in difference in y_loc and z_loc
                        y_locs = df_cls['y_loc'].values
                        z_locs = df_cls['z_loc'].values
                        spacing = np.sqrt((y_locs[1] - y_locs[0])**2 + (z_locs[1] - z_locs[0])**2)
                        spacing = f'{spacing:.3f} m' # round to 3 decimal places and format as string for display
                    elif df_cls['y_dir'].values[0] == 1: # transversely spaced, extract distance in difference in x_loc
                        z_locs = df_cls['z_loc'].values
                        x_locs = df_cls['x_loc'].values
                        spacing = np.sqrt((x_locs[1] - x_locs[0])**2 + (z_locs[1] - z_locs[0])**2)
                        spacing = f'{spacing:.3f} m' # round to 3 decimal places and format as string for display
                    else:
                        x_locs = df_cls['x_loc'].values
                        y_locs = df_cls['y_loc'].values
                        spacing = np.sqrt((x_locs[1] - x_locs[0])**2 + (y_locs[1] - y_locs[0])**2)
                        spacing = f'{spacing:.3f} m' # round to 3 decimal places and format as string for display

                df_slice_info = pd.concat([df_slice_info, pd.DataFrame({'Class': [cls], 'Class_Abbr': [class_abbr], 'Type': ['Stiffener'], 'Number': [number], 'Spacing [m]': [spacing], 'Dims': [dims], 'Material': ['ASTM A131 - AB/A']})], ignore_index=True)



        # Sort the df_slice_info DataFrame by the order of classes in the class_abbrev dictionary
        df_slice_info['Class_Order'] = df_slice_info['Class'].apply(lambda x: list(self.class_abbrev.keys()).index(x) if x in self.class_abbrev else len(self.class_abbrev))
        df_slice_info = df_slice_info.sort_values('Class_Order').reset_index(drop=True)

        return df_slice_info
    
    def create_Info_Blocks(self, slice_name):
        '''
        This function creates the information block in the layout for the specified slice name.

        There is a block for Plates and a block for stiffeners. 

        The plate block will have the following format: 
        ________________________________________________________________
        | Class         | Thickness [m]   | Material     |
        |_______________|_________________|_____________________________

        


        The stiffener block will have the following format:
        ______________________________________________________________________________
        | Class         | Num.          | Spacing [m]   | Dims. [m]   | Material     |
        |_______________|_______________|_______________|_____________________________
        
        '''
        
        df_slice_info = self.extract_Slice_Data(slice_name) # Extract the slice data for the specified slice name
        text_block_h = 8

        
        ib_h = (len(df_slice_info)+3)*text_block_h # Add three rows for the two header rows and one extra row for spacing
        ib_w = 150.0 

        # Center the info block in the above the title block
        h_remain = h_mm - 2.0*margin - tb_h
        ib_y = h_mm - margin - (h_remain - ib_h)/2.0 # Start from the top
        ib_x = w_mm - margin - tb_w + (tb_w - ib_w)/2.0 # Center the info block in the title block


        # First let's draw the plate info block, if there are any plates in the slice
        
        num_plates = len(df_slice_info[df_slice_info['Type'] == 'Plate'])

        if num_plates > 0:  
            # Make the info block for the plates
            rs.AddLine([0,ib_x, ib_y], [0,ib_x + ib_w, ib_y]) # Top line
            rs.AddLine([0,ib_x, ib_y - text_block_h], [0,ib_x + ib_w, ib_y - text_block_h ]) # Header separator line

            point = [0,ib_x + 1.0, ib_y - 1.0]
            plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
            class_header_guid = rs.AddText("Plate", plane, height=font_z_small)

            point = [0,ib_x + 0.33*ib_w + 1, ib_y - 1.0]
            plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
            thickness_header_guid = rs.AddText("Thickness [mm]", plane, height=font_z_small)

            point = [0,ib_x + 0.66*ib_w + 1, ib_y - 1.0]
            plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
            material_header_guid = rs.AddText("Material", plane, height=font_z_small)

            items = pd.DataFrame({'Object_ID': ['Info Block Plate Header Class', 'Info Block Plate Header Thickness', 'Info Block Plate Header Material'],
                                'Slice_Name': [slice_name]*3,
                                'dwg_slice_rsObject': [class_header_guid, thickness_header_guid, material_header_guid]})
            self.Drawing_Annotations = pd.concat([self.Drawing_Annotations, items], ignore_index=True) # Append the new info block plate header to the main DataFrame

            # Now fill in the plate information in the info block

            for i in range(num_plates):
                row = df_slice_info[df_slice_info['Type'] == 'Plate'].iloc[i]
                y_pos = ib_y - (i+2)*text_block_h + text_block_h/2.0 - font_z_small/2.0 # Center the text vertically in the row

                point = [0,ib_x + 1.0, y_pos]
                plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
                class_guid = rs.AddText(row['Class_Abbr'], plane, height=font_z_small)

                point = [0,ib_x + 0.33*ib_w + 1, y_pos]
                plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
                thickness_guid = rs.AddText(str(row['Dims']).replace('Thickness: ', ''), plane, height=font_z_small)

                point = [0,ib_x + 0.66*ib_w + 1, y_pos]
                plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
                material_guid = rs.AddText(row['Material'], plane, height=font_z_small)
                rs.AddLine([0,ib_x, ib_y - (i+2)*text_block_h], [0,ib_x + ib_w, ib_y-(i+2)*text_block_h]) # Draw horizontal divider

                items = pd.DataFrame({'Object_ID': [f'Info Block Plate Row {i} Class', 
                                                  f'Info Block Plate Row {i} Thickness', 
                                                  f'Info Block Plate Row {i} Material'],
                                'Slice_Name': [slice_name]*3,
                                'dwg_slice_rsObject': [class_guid, thickness_guid, material_guid]})
                self.Drawing_Annotations = pd.concat([self.Drawing_Annotations, items], ignore_index=True) # Append the new info block plate row to the main DataFrame

        # Add vertical divider line for the plate info block left, middle L, and  middle R, and right
        rs.AddLine([0,ib_x, ib_y], [0,ib_x, ib_y - (num_plates+1)*text_block_h])
        rs.AddLine([0,ib_x + 0.33*ib_w, ib_y], [0,ib_x + 0.33*ib_w, ib_y - (num_plates+1)*text_block_h])
        rs.AddLine([0,ib_x + 0.66*ib_w, ib_y], [0,ib_x + 0.66*ib_w, ib_y - (num_plates+1)*text_block_h])
        rs.AddLine([0,ib_x + ib_w, ib_y], [0,ib_x + ib_w, ib_y - (num_plates+1)*text_block_h])

        #Now Let's fill in the stiffener info block, if there are any stiffeners in the slice
        
        start_h = ib_y - (num_plates+2)*text_block_h # Start filling the stiffener info block below the plate info block, with one row of spacing in between
        num_stiffeners = len(df_slice_info[df_slice_info['Type'] == 'Stiffener'])



        if num_stiffeners > 0:
            # Make the header for the stiffener info block
            rs.AddLine([0,ib_x, start_h], [0,ib_x + ib_w, start_h]) # Top line
            rs.AddLine([0,ib_x, start_h - text_block_h], [0,ib_x + ib_w, start_h - text_block_h ]) # Header separator line

            point = [0,ib_x + 1.0, start_h - 1.0]
            plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
            class_header_guid = rs.AddText("Stiffener", plane, height=font_z_small)

            point = [0,ib_x + 0.2*ib_w + 1, start_h - 1.0]
            plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
            num_header_guid = rs.AddText("Num.", plane, height=font_z_small)

            point = [0,ib_x + 0.3*ib_w + 1, start_h - 1.0]
            plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
            spacing_header_guid = rs.AddText("Spacing [m]", plane, height=font_z_small)

            point = [0,ib_x + 0.5*ib_w + 1, start_h - 1.0]
            plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
            dims_header_guid = rs.AddText("Dims: Shape - H x W x t", plane, height=font_z_small)

            point = [0,ib_x + 0.8*ib_w + 1, start_h - 1.0]
            plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
            material_header_guid = rs.AddText("Material", plane, height=font_z_small)

            items = pd.DataFrame({'Object_ID': ['Info Block Stiffener Header Class', 
                                               'Info Block Stiffener Header Num', 
                                               'Info Block Stiffener Header Spacing', 
                                               'Info Block Stiffener Header Dims', 
                                               'Info Block Stiffener Header Material'],
                                'Slice_Name': [slice_name]*5,
                                'dwg_slice_rsObject': [class_header_guid, num_header_guid, spacing_header_guid, dims_header_guid, material_header_guid]})
            self.Drawing_Annotations = pd.concat([self.Drawing_Annotations, items], ignore_index=True) # Append the new info block stiffener header to the main DataFrame

            # Now fill in the stiffener information in the info block
            for i in range(num_stiffeners):
                row = df_slice_info[df_slice_info['Type'] == 'Stiffener'].iloc[i]
                y_pos = start_h - (i+2)*text_block_h + text_block_h/2.0 - font_z_small/2.0 # Center the text vertically in the row

                point = [0,ib_x + 1.0, y_pos]
                plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
                class_guid = rs.AddText(row['Class_Abbr'], plane, height=font_z_small)

                point = [0,ib_x + 0.2*ib_w + 1, y_pos]
                plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
                num_guid = rs.AddText(str(row['Number']), plane, height=font_z_small)

                point = [0,ib_x + 0.3*ib_w + 1, y_pos]
                plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
                spacing_guid = rs.AddText(str(row['Spacing [m]']), plane, height=font_z_small)

                point = [0,ib_x + 0.5*ib_w + 1, y_pos]
                plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
                dims_guid = rs.AddText(str(row['Dims']), plane, height=font_z_small)

                point = [0,ib_x + 0.8*ib_w + 1, y_pos]
                plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
                material_guid = rs.AddText(row['Material'], plane, height=font_z_small)
                rs.AddLine([0,ib_x, start_h - (i+2)*text_block_h], [0,ib_x + ib_w, start_h-(i+2)*text_block_h]) # Draw horizontal divider

                items = pd.DataFrame({'Object_ID': [f'Info Block Stiffener Row {i} Class', 
                                                  f'Info Block Stiffener Row {i} Num', 
                                                  f'Info Block Stiffener Row {i} Spacing', 
                                                  f'Info Block Stiffener Row {i} Dims', 
                                                  f'Info Block Stiffener Row {i} Material'],
                                'Slice_Name': [slice_name]*5,
                                'dwg_slice_rsObject': [class_guid, num_guid, spacing_guid, dims_guid, material_guid]})
                self.Drawing_Annotations = pd.concat([self.Drawing_Annotations, items], ignore_index=True) # Append the new info block stiffener row to the main DataFrame

                # Add vertical divider lines for the stiffener info block left, num, spacing, dims, material, and right
                rs.AddLine([0,ib_x, start_h], [0,ib_x, start_h - (num_stiffeners+1)*text_block_h])
                rs.AddLine([0,ib_x + 0.2*ib_w, start_h], [0,ib_x + 0.2*ib_w, start_h - (num_stiffeners+1)*text_block_h])
                rs.AddLine([0,ib_x + 0.3*ib_w, start_h], [0,ib_x + 0.3*ib_w, start_h - (num_stiffeners+1)*text_block_h])
                rs.AddLine([0,ib_x + 0.5*ib_w, start_h], [0,ib_x + 0.5*ib_w, start_h - (num_stiffeners+1)*text_block_h])    
                rs.AddLine([0,ib_x + 0.8*ib_w, start_h], [0,ib_x + 0.8*ib_w, start_h - (num_stiffeners+1)*text_block_h])
                rs.AddLine([0,ib_x + ib_w, start_h], [0,ib_x + ib_w, start_h - (num_stiffeners+1)*text_block_h])


        if slice_name != 'Midship Section of Long. Structure':
                # We want to add a note at the bottom of the drawing to say: 
                Note = 'NOTE: LONGITUDINAL STRUCTURAL ELEMENTS NOT SHOWN\nSEE DWG 3 FOR LONGITUDINAL STRUCTURE DETAILS' 

                #Add Note to bottom of layout under the drawing: 
                point = [0, margin + 25.0, margin + 15.0]
                plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the
                note_guid = rs.AddText(Note, plane, height=font_z_small)
                items = pd.DataFrame({'Object_ID': ['Info Block Note'], 'Slice_Name': [slice_name], 'dwg_slice_rsObject': [note_guid]})
                self.Drawing_Annotations = pd.concat([self.Drawing_Annotations, items], ignore_index=True) # Append the new info block note to the main DataFrame
    
    def export_DWG(self, slice_name):
        '''
        This function exports an SVG of the spedicied drawing with and without bounding boxes
        sets drawing layer as active layer 
        then the function hides all other layers
        
        '''

        rs.CurrentView('Right')#
        

        

        # See if self.DWG_detail exists, if it does, delete it and create a new one, if not, create a new one
        #if not hasattr(self, 'DWG_detail'):
            #self.create_LayoutViewport(slice_name)
        

        #rs.CurrentDetail(self.layout, self.DWG_detail) # Set the current detail view to the drawing detail view

        #rs.DetailLock(self.DWG_detail, False)

        rs.CurrentLayer(f'DWG_{slice_name}') # Set the current layer to the drawing layer for the specified slice name
        #Hide all other layers except for the drawing layer
        for layer in rs.LayerNames():
            if layer != f'DWG_{slice_name}':
                rs.LayerVisible(layer, False)
            else:
                rs.LayerVisible(layer, True)

        #rs.EnableRedraw(True) # Disable redraw to speed up the export process
        

        # Select all visible objects in the drawing
        objs = rs.VisibleObjects()
        rs.SelectObjects(objs)

        Bounding_Box = rs.BoundingBox(objs) # Get the bounding box of the visible objects in the drawing
        #rs.ZoomBoundingBox(Bounding_Box, view='Right') # Zoom to the bounding box of the visible objects in the drawing
        rs.ZoomExtents(view='Right') # Zoom to the extents of the objects in the right view
        
        p1 = Rhino.Geometry.Point3d(*(0,0,0))
        p2 = Rhino.Geometry.Point3d(*(0,w_mm,h_mm))

        #rs.SelectObjects([self.DWG_detail, self.layout])
        objs = rs.VisibleObjects()
        rs.SelectObjects(objs)
        # Export the selected objects as a PDF file
        view = sc.doc.Views.ActiveView
        size = System.Drawing.Size(4961, 3508)  # A3 landscape at 300 dpi
        settings = Rhino.Display.ViewCaptureSettings(view, size, 300)
        settings.SetWindowRect(p1, p2)

        # Request vector PDF
        settings.RasterMode = False

        pdf = Rhino.FileIO.FilePdf.Create()
        pdf.AddPage(settings)


        export_path = f"{self.output_path}/{self.file_name}_{slice_name}.pdf"
        pdf.Write(export_path)

        
        #rs.Command(cmd, echo=True) # Export the drawing as a PDF file
        
        # now add the bounding boxes and export another version of the drawing with bounding boxes
        rs.LayerVisible(f"{slice_name}_BBoxes", True) # Make the bounding box layer visible
        objs = rs.VisibleObjects() # Get the visible objects again, which now includes the bounding boxes
        rs.SelectObjects(objs) # Select the visible objects

        export_path_bbox = f"{self.output_path}/{self.file_name}_{slice_name}_with_BBoxes.pdf"

        objs = rs.VisibleObjects()
        rs.SelectObjects(objs)

        p1 = Rhino.Geometry.Point3d(*(0,0,0))
        p2 = Rhino.Geometry.Point3d(*(0,w_mm,h_mm))

        #rs.SelectObjects([self.DWG_detail, self.layout])
        objs = rs.VisibleObjects()
        rs.SelectObjects(objs)
        # Export the selected objects as a PDF file
        view = sc.doc.Views.ActiveView
        size = System.Drawing.Size(4961, 3508)  # A3 landscape at 300 dpi
        settings = Rhino.Display.ViewCaptureSettings(view, size, 300)
        settings.SetWindowRect(p1, p2)

        # Request vector PDF
        settings.RasterMode = False

        pdf = Rhino.FileIO.FilePdf.Create()
        pdf.AddPage(settings)


        export_path = f"{self.output_path}/{self.file_name}_{slice_name}.pdf"
        pdf.Write(export_path_bbox)

    def create_LayoutViewport(self, slice_name):
        # This function creates a layout viewport for the specified slice name. The viewport is set to the right view and zoomed to fit the contents of the drawing.
        rs.CurrentLayer(f'DWG_{slice_name}') # Set the current layer to the drawing layer for the specified slice name
        rs.CurrentView('Right') # Set the current view to the right view
        rs.ZoomExtents(view='Right') # Zoom to the extents of the objects in the right view
        
        self.layout = rs.AddLayout('Right_DWG', size=[w_mm, h_mm]) # Add a layout viewport with the specified dimensions
        self.DWG_detail = rs.AddDetail(self.layout, [0,0], [w_mm, h_mm], projection=4) # Add a detail view to the layout that shows the right view
        
       

   