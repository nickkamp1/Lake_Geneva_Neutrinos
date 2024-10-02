import numpy as np
import pandas as pd
import sys
sys.path.append("../../../")
from GeometryHelpers import *

detector_separation = 500 #m
fiducial_length = 5000 # m

LHC_data = pd.read_parquet('../../LHC_data_official.parquet')
Lake_data = pd.read_parquet('../../Lake_data.parquet')

LHC = Circle(np.array(LHC_data.loc['ATLAS',['X','Y','Z']]),
             np.array(LHC_data.loc['CMS',['X','Y','Z']]),
             np.array(LHC_data.loc['LHCb',['X','Y','Z']])) # last entry can be Point 2 or 6

# Fix elevation of other LHC interaction points based on LHC circle definition
for k in list(LHC_data.index):
    elev,point = LHC.find_elevation(*np.array(LHC_data.loc[k,['Latitude','Longitude']]))
    LHC_data['Elevation'][k] = elev
    LHC_data.loc[k,['X','Y','Z']] = lat_long_to_xyz(*LHC_data.loc[k,['Latitude','Longitude','Elevation']])

def minsec_to_dec(first_digit,minutes,seconds):
    return first_digit + minutes/60 + seconds/(60*60)

surface_exits = {"LHCb_South":np.array([minsec_to_dec(46,7,6.6649),minsec_to_dec(5,49,55.4673),516.80]),
                 "LHCb_North":np.array([minsec_to_dec(46,44,24.8500),minsec_to_dec(7,11,44.0499),766.64]),
                 "CMS_West":np.array([minsec_to_dec(46,20,32.9799),minsec_to_dec(5,50,19.7639),678.74]),
                 "CMS_East":np.array([minsec_to_dec(45,59,59.5943),minsec_to_dec(8,10,8.1070),508.31]),
                 "ATLAS_West":np.array([minsec_to_dec(46,16,56.7102),minsec_to_dec(5,42,48.0509),748.53]),
                 "ATLAS_East":np.array([minsec_to_dec(45,59,59.5943),minsec_to_dec(8,10,8.1070),717.94])}

lake_intersections = {"LHCb_1":np.array([minsec_to_dec(46,22,15.2013),minsec_to_dec(6,22,39.0901),372-82]),
                      "LHCb_2":np.array([minsec_to_dec(46,30,12.1910),minsec_to_dec(6,40,05.5942),372-13.3])}

with open('SINE_Template.dat','r') as f:
    template = f.read()

# populate each file
for exit_point,se in surface_exits.items():
    surface_loc = np.array(lat_long_to_xyz(*se))
    IP_loc = None
    for k in list(LHC_data.index):
        if k in exit_point:
            IP_loc = np.array(LHC_data.loc[k,['X','Y','Z']])
    assert(IP_loc is not None)
    surface_normal_dir = surface_loc / np.linalg.norm(surface_loc)
    beam_exit_displacement = surface_loc - IP_loc
    surface_exit_distance = np.linalg.norm(beam_exit_displacement)
    beam_exit_dir = beam_exit_displacement / surface_exit_distance
    surface_exit_angle = np.pi/2 - np.arccos(np.dot(beam_exit_dir,surface_normal_dir))
    box2_z = np.cos(surface_exit_angle) * (surface_exit_distance + 3.885/np.sin(surface_exit_angle))
    prototype_z = np.cos(surface_exit_angle) * (surface_exit_distance + 1.295/np.sin(surface_exit_angle))
    print("%s: %2.1f km from IP, exits at %2.2f deg w.r.t. surface at elevation of %2.2f m"%(exit_point,surface_exit_distance/1e3,(180/np.pi)*surface_exit_angle,se[-1]))
    SINE_outfile_name = "SINE_%s.dat"%exit_point
    with open(SINE_outfile_name,"w") as f:
        _template = template.format(exit_point=exit_point,
                                    surface_exit_distance=surface_exit_distance,
                                    surface_exit_angle=surface_exit_angle,
                                    box1_z=box2_z - detector_separation,
                                    box2_z=box2_z,
                                    box3_z=box2_z + detector_separation,
                                    fiducial_z=box2_z + detector_separation - (fiducial_length/2),
                                    fiducial_length=fiducial_length,
                                    prototype_z=prototype_z,
                                    prototype_fiducial_z=prototype_z - (fiducial_length/2),
                                    prototype_fiducial_length=fiducial_length)
        f.write(_template)
    