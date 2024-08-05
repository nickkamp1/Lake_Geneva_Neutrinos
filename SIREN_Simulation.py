import pandas as pd
import numpy as np
from MuonSimulationHelpers import get_flux_data
import siren

forward_nu_flux_dir = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/forward-nu-flux-fit/"

def PrepareSIRENInputFromLHCb(prefix,generator,parent):
    
    LHCb_depth = 34.97 #m
    surface_crossing = 18210.36
    theta = np.arccos(LHCb_depth/surface_crossing)
    LHCb_position = siren.math.Vector3D([0,-LHCb_depth,0])
    LHCb_direction = siren.math.Vector3D([0,np.cos(theta),np.sin(theta)])

    keys = ['PDG','hPDG','x0','y0','z0','thx','thy','E','wgt']
    data = np.loadtxt(forward_nu_flux_dir + '/files/' + prefix + '_' + generator + '_' + parent + '_0.txt')
    data_dict = {}
    for k,col in zip(keys,data.T):
        data_dict[k] = col
    df = pd.DataFrame(data=data_dict)

    zdir = siren.math.Vector3D([0,0,1])
    beam_rotation = siren.math.Quaternion.rotation_between(None,zdir,LHCb_direction)

    x0 = []
    y0 = []
    z0 = []
    px = []
    py = []
    pz = []

    for i,row in df[['x0','y0','z0','thx','thy','E']].iterrows():
        init_pos = siren.math.Vector3D([row.x0,row.y0,row.z0])
        init_pos_SIREN = beam_rotation.rotate(init_pos,False)
        dx = np.sin(row.thx)
        dy = np.sin(row.thy)
        init_dir = siren.math.Vector3D([dx,dy,np.sqrt(1 - dx**2 - dy**2)])
        init_dir_SIREN = beam_rotation.rotate(init_dir,False)
        x0.append(init_pos_SIREN.GetX())
        y0.append(init_pos_SIREN.GetY() - LHCb_depth)
        z0.append(init_pos_SIREN.GetZ())
        px.append(row.E*init_dir_SIREN.GetX())
        py.append(row.E*init_dir_SIREN.GetY())
        pz.append(row.E*init_dir_SIREN.GetZ())

    df['x0'] = x0
    df['y0'] = y0
    df['z0'] = z0

    df['px'] = px
    df['py'] = py
    df['pz'] = pz
    
    return df
        
    
    
    