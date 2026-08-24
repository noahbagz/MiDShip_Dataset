"""
=========================================================
Written by Noah J. Bagazinski

This code comprises of the tools to generate the individual structurual members as parametric and 3D model of a ship structure given a Ship-D hull parameterization
"""

import numpy as np

from tools.HullParameterization import Hull_Parameterization as HP

class Struct_Builder:
    """
    This class builds the data representation of a ship hull structure.

    All units are in meters

    there are 11 types of Sturctural members:
    1) Shell Plating
    2) Water Tight Bulkheads
    3) Decks
    4) Interior Longitudinals (Straight) (fixed to decks/bottom shell plating)
    5) Side Longitudinals (Follow hull curvature) (fixed to shell plating)
    6) Transverse Side Frames (Follow hull curvature) (fixed to shell plating)
    7) Transverse Bottom Frames (fixed to bottom shell plating)
    8) Transverse Deck Frames (fixed to deck)
    9) Transverse Bracket (connected between two transverse members or a transverse member and a deck)
    10) Longitudinal Bracket (connected between a longitudinal member and transverse member)
    11) Pillars (Vertical members connected to bulkheads or between longitudinal members)

    These Structural members are using heirarchical dictionaries.

    The highest Stuctural member is the Ship Structure, which contains all the other structural members.

    The next level is Bulkheads, Decks, and Shell Plating

    The 



    This code contains a function to generate an individual structural member given the Hull_Parameterization object, the type of structural member and the location of the structural member.

    face: this is a 3 dimensional categorical variable that describes normal of the cross section. ([1,0,0] is the x-axis (longitudinal), [0,1,0] is the y-axis (transverse), [0,0,1] is the z-axis (vertical))
    rot: this is the side of beam that the web is on. 1 is the positive side (starboard, top, or forward), -1 is the negative side (port, bottom, or aft)



    """
    def __init__(self, hull_vector, Struct_Params = {}):
        """
        inputs: 
        hull_vector: list of hull parameters from Ship-D Hull Parameterization
        Struct_Params: dictionary of structural parameters
        some of the parameters are:
        "Db": Double Bottom Height in milimeters 
        "Bottom_Plate_Thickness": Thickness of the bottom shell plating in milimeters
        "Side_Plate_Thickness": Thickness of the side shell plating in milimeters
        "Sheer_Strake_Thickness": Thickness of the sheer strake in milimeters
        "Sheer_Strake_Height": Height of the sheer strake in milimeters
        "Deck_Heights": List of deck heights in meters (min 2 decks, Deck at Hull.Dd and Deck at Db)
        "Deck_Thickness": Thickness of the deck in milimeters (index of thickness corresponds to index of deck height)
        "Bulkhead_Positions": List of bulkhead positions in meters (Center of thickness of bulkhead)
        "Bulkhead_Thickness": Thickness of the bulkhead in milimeters (index of thickness corresponds to index of bulkhead position)
        "Trans_Side_Frame_Spacing": Spacing of the transverse side frames in milimeters
        "Long_Side_Frame_Spacing": Spacing of the longitudinal side frames in milimeters
        
        More to come
        """
        num_wl = 201
        self.Hull = HP(hull_vector)

        Z = []

        if "Db" in Struct_Params.keys():
            Db = Struct_Params["Db"]
            Z.append(Db/1000.0) # convert to meters

        if "Sheer_Strake_Height" in Struct_Params.keys():
            Sheer_Strake_Height = Struct_Params["Sheer_Strake_Height"]
            Z.append(Sheer_Strake_Height/1000.0) # convert to meters
        
        Z.append(0.0001*self.Hull.Dd)
        Z.append(0.001*self.Hull.Dd)

        Z.append(np.linspace(0,self.Hull.Dd,num_wl-len(Z)))

        Z = np.array(Z)
        Z = Z.sort()
        
        self.Hull.gen_MeshGridPointCloud(NUM_WL= num_wl, PointsPerLOA= 1001, Z = Z)

        self.Struct_Dict = {}
        self.Struct_Dict["Hull"] = hull_vector
        self.Struct_Dict["Struct_Params"] = Struct_Params


    def section(self, beam_vec):
        """
        This function generates a beam section given a beam vector
        This function caclulates the area, centroid, and moment of inertia of the beam section in a local coordinate system.
        
        The local coordinate system is defined as follows:

        x-axis: transverse axis of the beam, x = 0 is centerline of the beam
        y-axis: vertical axis of the beam, y = 0 is side opposite of the web
        """
        h = beam_vec[0] # height of the beam
        t = beam_vec[1] # thickness of the beam
        w = beam_vec[2] # width of the web 
        face = beam_vec[3:6] # face of the beam
        rot = beam_vec[6] # rotation of the beam

        Section_Dict = {}

        Section_Dict["height"] = h
        Section_Dict["thickness"] = t
        Section_Dict["web_width"] = w
        Section_Dict["face"] = face
        Section_Dict["rot"] = rot

        return Section_Dict




    def add_Shell_Plating(self, Shell_Dict, Beams = []):
        """
        This function adds the shell plating to the Struct_Dict

        Shell_Dict: Dictionary of metrics needed for calculation. 

        Shell_Dict = {
            "thickness" = thickness of the shell plating in meters (note that ABS uses millimeters in calculations)
            "z_bounds" = [z_min, z_max] bounds of the shell plating in meters (this allows for a sheer strake to be added)
            if z_min == 0, then the shell plating is a bottom shell plating}


        """
        self.Struct_Dict["Shell"] = Shell_Dict

    def add_Water_Tight_Bulkheads(self, Bulkhead_Dict):
        """
        This function adds the water tight bulkheads to the Struct_Dict

        Bulkhead_Dict: Dictionary of metrics needed for calculation. 

        Bulkhead_Dict = {
            "thickness" = thickness of the bulkhead in meters (note that ABS uses millimeters in calculations)
            "X" = position of the bulkhead in meters (along the longitudinal axis)
        """


