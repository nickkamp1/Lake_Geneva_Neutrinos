import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from GeometryHelpers import *

LHC_surface_locations = {
    'ATLAS':(46.23498986744134, 6.053437868403428,440.9),
    'ALICE':(46.25152943082325, 6.0216164390182705,448.0),
    'Point3':(46.27753,6.01218,536.4),
    'Point4':(46.30443121543384, 6.036295278821727,585.9),
    'CMS':(46.30989989177777, 6.077425154913358,510.0),
    'Point6':(46.293459730167676, 6.111853119979701,471.0),
    'Point7':(46.266426109460426, 6.114850708334322,433.2),
    'LHCb':(46.24176461415312, 6.097010072101683,426.6)
}

LHC_horizontal = {
    'ATLAS':None,
    'Point2':None,
    'Point3':None,
    'Point4':1696.38,
    'CMS':None,
    'Point6':None,
    'Point7':None,
    'LHCb':10284.3
}

LHC_elevation = {
    'ATLAS':None,
    'ALICE':0.5*(509.82 + 337.02),
    'Point3':None,
    'Point4':509.82,
    'CMS':None,
    'Point6':0.5*(509.82 + 337.02),
    'Point7':None,
    'LHCb':337.02
}

LHC_crossing_angles = {
    'ATLAS':(0,280e-6),
    'ALICE':(0,400e-6),
    'Point3':None,
    'Point4':None,
    'CMS':(1,280e-6),
    'Point6':None,
    'Point7':None,
    'LHCb':(1,300e-6)
}

Lake_edge = [
    (46.22695,6.14891,379.9),
    (46.22375,6.15149,376.1),
    (46.22123,6.15357,376.0),
    (46.21761,6.15177,387.1),
    (46.21140,6.15327,373.7),
    (46.20727,6.16019,372.2),
    (46.21489,6.16826,370.0),
    (46.21517,6.17538,375.3),
    (46.23806,6.19279,371.1),
    (46.24566,6.19193,372.3),
    (46.26287,6.19742,374.3),
    (46.26596,6.20978,370.9),
    (46.35839,6.29076,378.3),
    (46.37140,6.32652,376.4),
    (46.36746,6.35067,374.6),
    (46.40529,6.87641,370.0),
    (46.44523,6.87597,373.7),
    (46.51899,6.58895,378.1),
    (46.47361,6.43789,392.0),
    (46.39732,6.28256,375.1),
    (46.39046,6.27415,376.6),
    (46.39294,6.25887,375.8),
    (46.35608,6.21475,374.0),
    (46.32567,6.20031,377.6),
    (46.30196,6.17971,375.2),
    (46.29346,6.17015,376.3),
    (46.27433,6.17087,378.9),
    (46.26436,6.16709,375.0),
    (46.25510,6.15576,377.3)
]

def main():
    LHC_locations = {k:(v[0],v[1],LHC_elevation[k]) for k,v in LHC_surface_locations.items()}
    LHC_locations_cartesian = {k:lat_long_to_xyz(*LHC_locations[k]) for k in ['ALICE','Point4','Point6','LHCb']}


    LHC_dataframe = pd.DataFrame(index=LHC_surface_locations.keys(),
                                columns=['Latitude','Longitude','Elevation','SurfaceElevation','X','Y','Z','CrossingOrientation','CrossingAngle'])

    for k in LHC_surface_locations.keys():
        LHC_dataframe['Latitude'][k] = LHC_locations[k][0]
        LHC_dataframe['Longitude'][k] = LHC_locations[k][1]
        LHC_dataframe['Elevation'][k] = LHC_locations[k][2]
        LHC_dataframe['SurfaceElevation'][k] = LHC_surface_locations[k][2]
    for k in LHC_locations_cartesian.keys():
        LHC_dataframe['X'][k] = LHC_locations_cartesian[k][0]
        LHC_dataframe['Y'][k] = LHC_locations_cartesian[k][1]
        LHC_dataframe['Z'][k] = LHC_locations_cartesian[k][2]
    for k in LHC_crossing_angles.keys():
        if LHC_crossing_angles[k] is not None:
            LHC_dataframe['CrossingOrientation'][k] = LHC_crossing_angles[k][0]
            LHC_dataframe['CrossingAngle'][k] = LHC_crossing_angles[k][1]

    for i in range(len(Lake_edge)): 
        Lake_edge[i] = (Lake_edge[i][0],Lake_edge[i][1],Lake_geneva_elevation)

    Lake_edge_cartesian = np.array([lat_long_to_xyz(*x) for x in Lake_edge])

    Lake_dataframe = pd.DataFrame(columns=['Latitude','Longitude','Elevation','X','Y','Z'])

    Lake_dataframe[['Latitude','Longitude','Elevation']] = Lake_edge
    Lake_dataframe[['X','Y','Z']] = Lake_edge_cartesian

    LHC_dataframe.to_parquet('../Data/LHC_data.parquet')
    Lake_dataframe.to_parquet('../Data/Lake_data.parquet')

main()