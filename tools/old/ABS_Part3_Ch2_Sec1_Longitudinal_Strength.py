"""
=========================================================
Written by Noah J. Bagazinski

This code calculates Longitudinal Strength of Ship Structures according to the 
ABS Part 3 Chapter 2 Section 1 Guidlines for Hull Construction

The ABS rules on Hull Construction and Equipent are found at
https://ww2.eagle.org/content/dam/eagle/rules-and-guides/current/other/1-rules-for-building-and-classing-marine-vessels-2024/1-mvr-part-3-jan24.pdf


The code is separated into the following sections:

1) General Parameters

2) Longitudinal Hull Girder Strength

3) Section Modulus Calculations

ALL CALCULATIONS ARE DONE ACCORDING TO ABS's Metric Units, sign conventions, and constraints.


Needed to perform calculations:

    Ship_Dict = {
        "L": Scantling Length of the Ship, L <= 305m for offshore support vessels, L <= 500m for other steel vessels (ABS Part 3 Ch 1 Sec 1 / 3.1)
        "B": Greatest Molded Breadth of Hull in meters
        "D": Molded Depth of Hull in meters = Scantling Depth Defined in ABS Part 3 Ch 1 Sec 1 / 7.3 
        "d": Molded draft of the ship in meters (baseline to summer load line)
        "V": Design Speed of the Ship in Knots (ABS Part 3 Ch 1 Sec 1 / 11)
        "Cb": Block Coefficient of the Ship (ABS Part 3 Ch 1 Sec 1 / 13.3) Cb >= 0.6
        "Cp": Constant for Proportionality (ABS Part 3 Ch 1 Sec 2 / 7) Cp = 3.0 for offshore support vessels, Cp = 2.0 for other steel vessels under 90m in length, and Cp = 2.5 for other steel vessels over 90m in length
        
        And others to be added as needed 
    }




======================================================
"""

# Constants




"""
======================================================
Section 1: General Parameters
======================================================
"""
#

def Generate_Long_Struct_Dict(Ship_Dict):
    """
    This function creates a dictionary of the longitudinal strength parameters and Constants.

    Ship_Dict: Dictionary of metrics needed for calculation. 
    """
    Long_Struct_Dict = {}

"""
======================================================
Section 2: Longitudinal Hull Girder Strength
======================================================

"""

def Calc_Min_Section_Modulus_61m(Ship_Dict):
    """
    This function calculates the minimum section modulus of the hull girder for ships under 61 meters in length.

    ABS Part 3 Ch 2 Sec 1 / 3.1

    Ship_Dict: Dictionary of metrics needed for calculation. 
    """
    # Constants
    if Ship_Dict["L"] > 61.0:
        raise ValueError("Ship Length is greater than 61 meters. Use WAVE BENDING MOMENT functions instead.")
    
    if Ship_Dict["L"] < 12.0:
        raise ValueError("Ship Length is less than 12 meters. Ship is too short for this rule.")
    

    # Calculate Minimum Section Modulus
    
    if Ship_Dict["L"] < 18.0:
        C_1 = 30.67 - 0.98*Ship_Dict["L"]
    elif Ship_Dict["L"] < 24.0:
        C_1 = 22.40 - 0.52*Ship_Dict["L"]
    elif Ship_Dict["L"] < 35.0:
        C_1 = 15.20 - 0.22*Ship_Dict["L"]
    elif Ship_Dict["L"] < 45.0:
        C_1 = 11.35 - 0.11*Ship_Dict["L"]
    else:
        C_1 = 6.4

    C_2 = 0.01

    Min_Section_Modulus = C_1*C_2 * (Ship_Dict["L"]**2) * Ship_Dict["B"] * (Ship_Dict["Cb"] + 0.7) #m-cm^2
    return Min_Section_Modulus

