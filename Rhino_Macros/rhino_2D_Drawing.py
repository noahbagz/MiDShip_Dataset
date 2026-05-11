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
tb_w, tb_h = 185.0, 60.0 # title block width and height in mm

view_w, view_h = w_mm - 2*margin-tb_w, h_mm - 2*margin # view width and height in mm

font_z_large = 4.0 # Font size for the main title and subtitle
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


    def load_Data(self):
        #Open a 3D model and the Structural Elements.csv File
        self.geom_file = '\"'+self.model_path + '/' + self.file_name + '.3dm\"'
        struct_elem_path = self.model_path + '/' + self.file_name + '_Structural_Elements.csv'

        rs.EnableRedraw(False) # Disable redraw to speed up the process

        rs.Command("_-Open " + self.geom_file + " _Enter") # Open the 3D model in Rhino
        self.Structural_Elements = pd.read_csv(struct_elem_path) # Load the Structural Elements.csv file into a pandas DataFrame

        
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
        self.df_Slices['Slice_Name'] = ['Transverse_Bulkhead', 'Web_Frame', 'Side_Frame', 'Longitudinal_Structure']
        x0 = x_bulkheads[0] if len(x_bulkheads) > 0 else None
        #x1 is middle index of webframes between bulkheads[0] and bulkheads[1]
        x_WebFrames = x_WebFrames[(x_WebFrames > x0) & (x_WebFrames < x_bulkheads[1])] if len(x_bulkheads) > 1 else x_WebFrames[x_WebFrames > x0]
        
        x1 = x_WebFrames[len(x_WebFrames)//2] if len(x_WebFrames) > 0 else None
        #x2 is first x_frame after x1
        x_Frames = x_Frames[x_Frames > x1] if x1 is not None else x_Frames
        x2 = x_Frames[0] if len(x_Frames) > 0 else None

        #X3 is halfway between x1 and x2
        x3 = (x1 + x2) / 2 if x1 is not None and x2 is not None else None   

        self.df_Slices['x'] = [x0, x1, x2, x3]

    def get_guids_For_Slice(self, slice_name): 
        # This function returns the GUIDs and Structural Elements Index of the objects in the 3D model that correspond to a given slice name.

        if slice_name == 'Longitudinal_Structure': # If the slice is longitudinal structure, return all objects with x_dir = 1 -> all longitudinal structures
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
            
            DSL = self.Structural_Elements[self.Structural_Elements['Class'].isin(['Side_Shell', 'Bottom_Shell', 'Deck', 'Longitudinal_Bulkhead', 'Hopper Pannel'])]['struct_rsObject'].values
            DSL_idx = self.Structural_Elements[self.Structural_Elements['Class'].isin(['Side_Shell', 'Bottom_Shell', 'Deck', 'Longitudinal_Bulkhead', 'Hopper Pannel'])].index.values
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

        '''
        # Add dashed line for CL
        CL = rs.AddLine([0, origin_s[1], origin_s[2]], [0, origin_s[1], origin_s[2]+Ds]) # Add a dashed line for the centerline of the structure
        CL = rs.ObjectLinetype(CL, "Dashed") # Set the line type to dashed

        item = pd.DataFrame({'Slice_Name': [slice_name],
                            'Object_ID': ['Centerline'],
                            'dwg_slice_rsObject': [CL]})
        self.Drawing_Annotations = pd.concat([self.Drawing_Annotations, item], ignore_index=True) # Append the centerline annotation to the main DataFrame
        '''


        return scale, origin_s 



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
        *Title Block Layout* Subtitles of each block are in font size 6mm
        Full size text is in 12mm 
        ____________________________________________________________
        | Dwg ID No.                           |       Sheet Num:   |
        |______________________________________|____________________|
        |Title:                                | Date:              |
        |______________________________________|____________________|
        |Subtitle: {Subtitle}                  | Project:           |
        ____________________________________________________________|
        | Scale: 1m = {Caclulated Scale} Revision: {blank} |       |
        ____________________________________________________________|
        
        '''

        # Draw horizontal dividers
        title_row_h = tb_h/4.0


        rs.AddLine([0,left_y, margin + title_row_h], [0, w_mm-margin, margin + title_row_h])
        rs.AddLine([0, left_y, margin + 2*title_row_h], [0, w_mm-margin, margin + 2*title_row_h])
        rs.AddLine([0, left_y, margin + 3*title_row_h], [0, w_mm-margin, margin + 3*title_row_h])

        # Draw vertical dividers
        rs.AddLine([0,left_y + 0.6667*tb_w, margin], [0,left_y + 0.6667*tb_w, margin + tb_h])
        #rs.AddLine([0,left_y + 0.3333*tb_w, margin + title_row_h], [0,left_y + 0.333*tb_w, margin])

        # Add information text to the title block
       
        point = [0,left_y + 1.0, margin + tb_h - font_z_small-1.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the title block text
        dwgID_guid = rs.AddText("Dwg ID No.", plane, height=font_z_small)

        
        point = [0,left_y + 0.6667*tb_w + 1.0, margin + tb_h - font_z_small-1.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the title block text
        sheetnum_guid = rs.AddText("Sheet Num:", plane, height=font_z_small)
        
        
        point = [0,left_y + 1.0, margin + tb_h - title_row_h - font_z_small-1.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the title block text
        title_guid = rs.AddText("Title:", plane, height=font_z_small)

        
        point = [0,left_y + 0.6667*tb_w + 1.0, margin + tb_h - title_row_h - font_z_small-1.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the title block text
        date_guid = rs.AddText("Date:", plane, height=font_z_small)
        
        
        point = [0,left_y + 1.0, margin + tb_h - 2*title_row_h - font_z_small-1.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the title block text
        subtitle_guid = rs.AddText("Subtitle:", plane, height=font_z_small)

        point = [0,left_y + 0.6667*tb_w + 1.0, margin + tb_h - 2*title_row_h - font_z_small-1.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the title block text
        project_guid = rs.AddText("Project:", plane, height=font_z_small)
        
        point = [0,left_y + 1.0, margin + tb_h - 3*title_row_h - font_z_small-1.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the title block text
        scale_guid =  rs.AddText("Scale:", plane, height=font_z_small)
        
        point = [0,left_y + 0.6667*tb_w + 1.0, margin + tb_h - 3*title_row_h - font_z_small-1.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the
        rev_guid =  rs.AddText("Revision:", plane, height=font_z_small)

        items = pd.DataFrame({'Object_ID': ['dwg_id_header',
                                              'sheetnum_header',
                                                'title_header',
                                                'date_header',
                                                'subtitle_header',
                                                'project_header',
                                                'scale_header',
                                                'rev_header'],
                            'Slice_Name': [slice_name]*8,
                            'dwg_slice_rsObject': [dwgID_guid, sheetnum_guid, title_guid, date_guid, subtitle_guid, project_guid, scale_guid, rev_guid]})
        self.Drawing_Annotations = pd.concat([self.Drawing_Annotations, items], ignore_index=True) # Append the new title block subtitles to the main DataFrame
    

        # Fill in the title block information
        
        #format dwg number as DWG-001, DWG-002, etc.
        point = [0,left_y + 10.0, margin +tb_h - 10]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the title block text
        DWGid_guid = rs.AddText(f'{dwg_num:03d}', plane, height=font_z_large)

        point = [0,left_y + 0.6667*tb_w + 10.0, margin + tb_h - 10.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the title block text
        SHEETNUM_guid = rs.AddText(dwg_num_str, plane, height=font_z_large)
        
        point = [0,left_y + 10.0, margin + 0.75*tb_h - 10.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the title block text
        TITLE_guid = rs.AddText(Title, plane, height=font_z_large)

        point = [0,left_y + 10.0, margin + 0.5*tb_h - 10.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the title block text
        SUBTITLE_guid = rs.AddText(Subtitle, plane, height=font_z_large)

        point = [0,left_y + 0.6667*tb_w + 10.0, margin + 0.5*tb_h - 10.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the title block text
        PROJECT_guid = rs.AddText("MiDShip Dataset", plane, height=font_z_large)

        #Determin the scale of the drawing for now, use a fixed scale of 1:200, but this can be changed later
        point = [0,left_y + 10.0, margin + 0.25*tb_h - 10.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the
        SCALE_guid = rs.AddText(f'1 m = {scale} mm on DWG', plane, height=font_z_large)

        point = [0,left_y + 0.6667*tb_w + 10.0, margin + 0.25*tb_h - 10.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the
        REV_guid = rs.AddText("Rev-0", plane, height=font_z_large)

        items = pd.DataFrame({'Object_ID': ['DWG_ID',
                                              'SHEET_NUM',
                                                'TITLE',
                                                'SUBTITLE',
                                                'PROJECT',
                                                'SCALE',
                                                'REVISION'],
                            'Slice_Name': [slice_name]*7,
                            'dwg_slice_rsObject': [DWGid_guid, SHEETNUM_guid, TITLE_guid, SUBTITLE_guid, PROJECT_guid, SCALE_guid, REV_guid]})
        self.Drawing_Annotations = pd.concat([self.Drawing_Annotations, items], ignore_index=True) # Append the new title block information to the main DataFrame



    def extract_Slice_Data(self, slice_name):
        # THis function extracts information from Structural Elements to display in the an info block in the drawing. 

        #The main data: Object Class, number in the structure, Spacing, Size, 
        df_slice_info =pd.DataFrame(columns=['Class', 'Number', 'Spacing [m]', 'Size'])
        
        #Things to extract from the slice info: x_s for spacing, and number of elements of longitudinally spaced structural elements. 
        if slice_name == 'Transverse_Bulkhead':
            bulkhead_x = self.Structural_Elements[self.Structural_Elements['Class'] == 'Transverse_Bulkhead']['x_loc'].values
            x_s = bulkhead_x[1] - bulkhead_x[0] 
 
        elif slice_name == 'Web_Frame':
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

            # Also add bottom floor

        elif slice_name == 'Side_Frame':
            sideframe_x = self.Structural_Elements[self.Structural_Elements['Class'] == 'Vertical_Side_Frame']['x_loc'].values
            x_s = sideframe_x[1] - sideframe_x[0]

        elif slice_name == 'Longitudinal_Structure':
            x_s = None # For longitudinal structure, spacing is not applicable

        
        #Now let's extract the Structural_Elements in the slice and their information
        idx = self.Slice_Elements[self.Slice_Elements['Slice_Name'] == slice_name]['struct_elem_idx'].values
        
        df_elem = self.Structural_Elements.loc[idx]

        #Search through df_elem and reassign class of flanges to their corresponding structural element class with '_Flange' suffix. For example, if the Object_ID of an element is 'WebFrame_001_Flange', we will reassign its class to 'Vertical_Web_Frame_Flange'.
        for index, row in df_elem.iterrows():
            if '_Flange' in row['Object_ID']:
                base_obj_id = row['Object_ID'].replace('_Flange', '')
                base_class = self.Structural_Elements[self.Structural_Elements['Object_ID'] == base_obj_id]['Class'].values
                if len(base_class) > 0:
                    df_elem.at[index, 'Class'] = base_class[0] + '_Flange'

        # Get all the unique classes of structural elements in the slice
        unique_classes = df_elem['Class'].unique()

        #Drop 'Side_shell', 'Bottom_Shell', 'Deck', 'Longitudinal_Bulkhead' and 'Hopper Pannel' from the unique classes, as they are not structural elements that we want to display in the info block
        unique_classes = [cls for cls in unique_classes if cls not in ['Side_Shell', 'Bottom_Shell', 'Deck', 'Longitudinal_Bulkhead', 'Hopper Pannel']]


        for cls in unique_classes:
            df_cls = df_elem[df_elem['Class'] == cls]
            number = len(df_cls)
            #Size is a string that is made of L1, L2, and T. If any of these columns are missing, size is None
            L1 = df_cls['L1'].values[0] 
            L2 = df_cls['L2'].values[0] 
            T = df_cls['Thickness'].values[0] 
            #Format size as L1xL2xT to 3 decimal places
            size = f'{L1:.3f}x{L2:.3f}x{T:.3f}' if not any(pd.isnull([L1, L2, T])) else None
            if number == 1: 
                spacing = x_s

            elif cls == 'Transverse_Bottom_Frame':
                spacing = x_s
            elif cls == 'Transverse_Bottom_Frame_Flange':
                spacing = x_s
            else: 
                #determine dir 
                if df_cls['x_dir'].values[0] == 1: # longitudinally spaced, extract distance in difference in y_loc and z_loc
                    y_locs = df_cls['y_loc'].values
                    z_locs = df_cls['z_loc'].values
                    spacing = np.sqrt((y_locs[1] - y_locs[0])**2 + (z_locs[1] - z_locs[0])**2)
                    spacing = np.round(spacing, 3) # round to 3 decimal places
                elif df_cls['y_dir'].values[0] == 1: # transversely spaced, extract distance in difference in x_loc
                    z_locs = df_cls['z_loc'].values
                    x_locs = df_cls['x_loc'].values
                    spacing = np.sqrt((x_locs[1] - x_locs[0])**2 + (z_locs[1] - z_locs[0])**2)
                    spacing = np.round(spacing, 3) # round to 3 decimal places
                else:
                    x_locs = df_cls['x_loc'].values
                    y_locs = df_cls['y_loc'].values
                    spacing = np.sqrt((x_locs[1] - x_locs[0])**2 + (y_locs[1] - y_locs[0])**2)
                    spacing = np.round(spacing, 3) # round to 3 decimal places

            df_slice_info = pd.concat([df_slice_info, pd.DataFrame({'Class': [cls], 'Number': [number], 'Spacing [m]': [spacing], 'Size': [size]})], ignore_index=True)

            # Now Let's Add the Side_Shell, Bottom_Shell, Deck, Longitudinal_Bulkhead, and Hopper Pannel to the info block if they exist in the slice
        for cls in ['Side_Shell', 'Bottom_Shell', 'Deck', 'Longitudinal_Bulkhead', 'Hopper Pannel']:
            df_cls = df_elem[df_elem['Class'] == cls]
            if len(df_cls) > 0:
                number = len(df_cls)
                size = df_cls['Thickness'].values[0]
                spacing = ' '
                df_slice_info = pd.concat([df_slice_info, pd.DataFrame({'Class': [cls], 'Number': [number], 'Spacing [m]': [spacing], 'Size': [size]})], ignore_index=True)
        
        # Abbreviated Class Names for the Info Block
    
        class_abbrev = {
            'Transverse_Bulkhead': 'Trans. BLKHD',
            'Vertical_Web_Frame': 'Webframe',
            'Vertical_Web_Frame_Flange': 'WebFrame Flng.',
            'Vertical_Side_Frame': 'Side Frame',
            'Vertical_Side_Frame_Flange': 'Side Frame Flng.',
            'Deck_Beam': 'Deck Beam',
            'Deck_Beam_Flange': 'Deck Beam Flng.',
            'Bottom_Floor_Beam': 'Floor',
            'Bottom_Floor_Beam_Flange': 'Floor Flng.',
            'Bracket': 'Bracket',
            'Side_Shell': 'Side Shell',
            'Bottom_Shell': 'Bottom Shell',
            'Deck': 'Deck',
            'Longitudinal_Bulkhead': 'Long. BLKHD',
            'Hopper Pannel': 'Hopper',
            'Inner_Bottom_Longitudinal_Stiffener': 'Inner Bot. Stiff.',
            'Inner_Bottom_Longitudinal_Stiffener_Flange': 'Inner Bot. Stiff. Flng.',
            'Stiffener_Flange': 'Stiff. Flng.',
            'Inner_Bottom_Transverse_Beam': 'Inner Bot. Beam',
            'Inner_Bottom_Transverse_Beam_Flange': 'Inner Bot. Beam Flng.',
            'Bottom_Longitudinal_Girder': 'Bot. Girder',
            'Bottom_Longitudinal_Girder_Flange': 'Bot. Girder Flng.',
            'Bottom_Longitudinal_Stiffener': 'Bot. Stiff.',
            'Bottom_Longitudinal_Stiffener_Flange': 'Bot. Stiff. Flng.',
            'Transverse_Bulkhead_Vertical_Stiffener': 'Vert. BLKHD Stiff.',
            'Transverse_Bulkhead_Vertical_Stiffener_Flange': 'Vert. BLKHD Stiff. Flng.',
            'Transverse_Bulkhead_Transverse_Stiffener': 'Trans. BLKHD Stiff.',
            'Transverse_Bulkhead_Transverse_Stiffener_Flange': 'Trans. BLKHD Stiff. Flng.',
            'Side_Shell_Longitudinal_Stiffener': 'Side Shell Stiff.',
            'Side_Shell_Longitudinal_Stiffener_Flange': 'Side Shell Stiff. Flng.',
            'Inner_Side_Shell_Vertical_Stiffener': 'Inner Shell Frame',
            'Inner_Side_Shell_Vertical_Stiffener_Flange': 'Inner Shell Frame Flng.',
            'Inner_Side_Shell_Longitudinal_Stiffener': 'Inner Shell Stiff.',
            'Inner_Side_Shell_Longitudinal_Stiffener_Flange': 'Inner Shell Stiff. Flng.',
            'Longitudinal_Bulkhead_Vertical_Stiffener': 'Long. BLKHD Stiff.',
            'Longitudinal_Bulkhead_Vertical_Stiffener_Flange': 'Long. BLKHD Stiff. Flng.',
            'Longitudinal_Bulkhead_Longitudinal_Stiffener': 'Vert. BLKHD Stiff.',
            'Longitudinal_Bulkhead_Longitudinal_Stiffener_Flange': 'Vert. BLKHD Stiff. Flng.',
            'Shell_Strake': 'Shell Strake',
            'Transverse_Floor': 'Floor',
            'Bracket_Floor_Web': 'Bracket',
            'Bracket_Deck_Web': 'Bracket',
            'Hopper_Pannel_Longitudinal_Stiffener': 'Hopper Stiff.',
            'Hopper_Pannel_Longitudinal_Stiffener_Flange': 'Hopper Stiff. Flng.',
            'Transverse_Bottom_Frame': 'Bot. Frame',
            'Transverse_Bottom_Frame_Flange': 'Bot. Frame Flng.',
            'Deck_Longitudinal_Stiffener': 'Deck Stiff.',
            'Deck_Longitudinal_Stiffener_Flange': 'Deck Stiff. Flng.',
            'Deck_Longitudinal_Girder': 'Deck Girder',
            'Deck_Longitudinal_Girder_Flange': 'Deck Girder Flng.',
            'Deck_Transverse_Beam': 'Deck Beam',
            'Deck_Transverse_Beam_Flange': 'Deck Beam Flng.',
            'Deck_Transverse_Stiffener': 'Deck Frame',
            'Deck_Transverse_Stiffener_Flange': 'Deck Frame Flng.'}

        #Add a class abbreviation column to the df_slice_info DataFrame and replace class names with abbreviations, if available
        df_slice_info['Class_Abbr'] = df_slice_info['Class'].apply(lambda x: class_abbrev.get(x, x))

        #Sort the df_slice_info DataFrame by the Class column
        df_slice_info = df_slice_info.sort_values(by='Class_Abbr').reset_index(drop=True)


        return df_slice_info
    
    def create_Info_Block(self, slice_name):
        # This function creates an information block in the layout for the specified slice name. The information block includes details about the structural elements in the slice.
        df_slice_info = self.extract_Slice_Data(slice_name) # Extract the slice data for the specified slice name
        text_block_h = 8
        
        ib_h = (len(df_slice_info)+1)*text_block_h
        ib_w = 150.0 

        # Center the info block in the above the title block
        h_remain = h_mm - 2.0*margin - tb_h
        ib_y = h_mm - margin - (h_remain - ib_h)/2.0 # Start from the top
        ib_x = w_mm - margin - tb_w + (tb_w - ib_w)/2.0 # Center the info block in the title block

        #create initial polyline for the info block
        rs.AddLine([0,ib_x, ib_y], [0,ib_x + ib_w, ib_y])
        rs.AddLine([0,ib_x, ib_y - text_block_h], [0,ib_x + ib_w, ib_y - text_block_h ])
        
        point = [0,ib_x + 1.0, ib_y - 1.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
        class_header_guid = rs.AddText("Class", plane, height=font_z_small)

        point = [0,ib_x + 1.0 + 0.375*ib_w, ib_y - 1.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
        num_items_guid = rs.AddText("Num.", plane, height=font_z_small)

        point = [0,ib_x + 1.0 + 0.5*ib_w, ib_y - 1.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
        spacing_guid = rs.AddText("Spacing [m]", plane, height=font_z_small)

        point = [0,ib_x + 1.0 + 0.75*ib_w, ib_y - 1.0]
        plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
        dims_guid = rs.AddText("Dims. [m]", plane, height=font_z_small)

        items = pd.DataFrame({'Object_ID': ['Info Block Header Class', 'Info Block Header Num', 'Info Block Header Spacing', 'Info Block Header Dims.'],
                            'Slice_Name': [slice_name]*4,
                            'dwg_slice_rsObject': [class_header_guid, num_items_guid, spacing_guid, dims_guid]})
        self.Drawing_Annotations = pd.concat([self.Drawing_Annotations, items], ignore_index=True) # Append the new info block header to the main DataFrame

        # Draw vertical dividers
        rs.AddLine([0,ib_x, ib_y], [0,ib_x, ib_y - ib_h])
        rs.AddLine([0,ib_x + 0.375*ib_w, ib_y], [0,ib_x + 0.375*ib_w, ib_y - ib_h])
        rs.AddLine([0,ib_x + 0.5*ib_w, ib_y], [0,ib_x + 0.5*ib_w, ib_y - ib_h])
        rs.AddLine([0,ib_x + 0.75*ib_w, ib_y], [0,ib_x + 0.75*ib_w, ib_y - ib_h])
        rs.AddLine([0,ib_x + ib_w, ib_y], [0,ib_x + ib_w, ib_y - ib_h])

        #Now Loop through the df_slice_info and add the information to the info block
        for i in range(len(df_slice_info)):
            row = df_slice_info.iloc[i]
            y_pos = ib_y - (i+2)*text_block_h + text_block_h/2.0 - font_z_small/2.0 # Center the text vertically in the row

            point = [0,ib_x + 1.0, y_pos]
            plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
            class_guid = rs.AddText(row['Class_Abbr'], plane, height=font_z_small)

            point = [0,ib_x + 1.0 + 0.375*ib_w, y_pos]
            plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
            num_guid = rs.AddText(str(row['Number']), plane, height=font_z_small)
            
            point = [0,ib_x + 1.0 + 0.5*ib_w, y_pos]
            plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
            spacing_guid = rs.AddText(str(row['Spacing [m]']), plane, height=font_z_small)

            point = [0,ib_x + 1.0 + 0.75*ib_w, y_pos]
            plane = rs.MovePlane(rs.WorldYZPlane(), point) # Move the plane to the appropriate position for the info block text
            dims_guid = rs.AddText(str(row['Size']), plane, height=font_z_small)
            rs.AddLine([0,ib_x, ib_y - (i+2)*text_block_h], [0,ib_x + ib_w, ib_y-(i+2)*text_block_h]) # Draw horizontal divider
            
            items = pd.DataFrame({'Object_ID': [f'Info Block Item Row {i} Class', 
                                                  f'Info Block Item Row {i} Num', 
                                                  f'Info Block Item Row {i} Spacing', 
                                                  f'Info Block Item Row {i} Dims.'],
                                'Slice_Name': [slice_name]*4,
                                'dwg_slice_rsObject': [class_guid, num_guid, spacing_guid, dims_guid]})
            self.Drawing_Annotations = pd.concat([self.Drawing_Annotations, items], ignore_index=True)

    
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
        
       

   