import numpy as np
import pandas as pd
import sys
import os
import siren
sys.path.append("../../")
from GeometryHelpers import *
from MuonSimulationHelpers import get_flux_data

detector_separation = 500 #m
fiducial_length = 5000 # m

forward_nu_flux_dir = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/forward-nu-flux-fit/"
foresee_dir = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/FORESEE/"

def PrepareSIRENInputFromIP(input_file,IPkeys,angles,surface_crossings):

    # compute different quantities related to each IP
    angles = {k:a for k,a in zip(IPkeys,angles)}
    surface_crossings = {k:sc for k,sc in zip(IPkeys,surface_crossings)}
    depths = {k:surface_crossings[k]*np.sin(angles[k]) for k in IPkeys}
    beam_directions = {k:siren.math.Vector3D([0,np.sin(angles[k]),np.cos(angles[k])]) for k in IPkeys}
    zdir = siren.math.Vector3D([0,0,1])
    beam_rotations = {k:siren.math.Quaternion.rotation_between(None,zdir,beam_directions[k]) for k in IPkeys}

    # load data from the forward_nu_flux file
    keys = ['PDG','hPDG','x0','y0','z0','thx','thy','E','wgt']
    data = np.loadtxt(input_file)
    data_dict = {}
    for k,col in zip(keys,data.T):
        data_dict[k] = col
    df = pd.DataFrame(data=data_dict)
    df.query("E>10",inplace=True)
    dfs = {}

    x0 = {k:[] for k in IPkeys}
    y0 = {k:[] for k in IPkeys}
    z0 = {k:[] for k in IPkeys}
    px = {k:[] for k in IPkeys}
    py = {k:[] for k in IPkeys}
    pz = {k:[] for k in IPkeys}

    for i,row in df[['x0','y0','z0','thx','thy','E']].iterrows():
        print("%d/%d"%(i,len(df)),end="\r")
        init_pos = siren.math.Vector3D([row.x0,row.y0,row.z0])
        dx = np.sin(row.thx)
        dy = np.sin(row.thy)
        init_dir = siren.math.Vector3D([dx,dy,np.sqrt(1 - dx**2 - dy**2)])
        for k in IPkeys:
            init_pos_SIREN = beam_rotations[k].rotate(init_pos,False)
            init_dir_SIREN = beam_rotations[k].rotate(init_dir,False)
            x0[k].append(init_pos_SIREN.GetX())
            y0[k].append(init_pos_SIREN.GetY() - depths[k])
            z0[k].append(init_pos_SIREN.GetZ())
            px[k].append(row.E*init_dir_SIREN.GetX())
            py[k].append(row.E*init_dir_SIREN.GetY())
            pz[k].append(row.E*init_dir_SIREN.GetZ())

    for k in IPkeys:
        dfs[k] = df.copy()

        dfs[k]['x0'] = x0[k]
        dfs[k]['y0'] = y0[k]
        dfs[k]['z0'] = z0[k]
        dfs[k]['px'] = px[k]
        dfs[k]['py'] = py[k]
        dfs[k]['pz'] = pz[k]

    return dfs

LHC_data = pd.read_parquet('../LHC_data_official.parquet')
Lake_data = pd.read_parquet('../Lake_data.parquet')

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

lake_intersections = {"LHCb_North":[25960.762,52669.495], # calculated from official LHC coordinates
                      "CMS_East":[7503.75,11905.95] # calculated from tangents to the LHC circle
                      }

with open('Geometry/SINE_Template.dat','r') as f:
    sine_template = f.read()

with open('Geometry/UNDINE_Template.dat','r') as f:
    undine_template = f.read()

# populate each of the geometry files
IPkeys = []
angles = []
surface_crossings = []

# first the SINE detectors
for IP,se in surface_exits.items():
    surface_loc = np.array(lat_long_to_xyz(*se))
    IP_loc = None
    for k in list(LHC_data.index):
        if k in IP:
            IP_loc = np.array(LHC_data.loc[k,['X','Y','Z']])
    assert(IP_loc is not None)
    surface_normal_dir = surface_loc / np.linalg.norm(surface_loc)
    beam_exit_displacement = surface_loc - IP_loc
    surface_exit_distance = np.linalg.norm(beam_exit_displacement)
    beam_exit_dir = beam_exit_displacement / surface_exit_distance
    surface_exit_angle = np.pi/2 - np.arccos(np.dot(beam_exit_dir,surface_normal_dir))
    IPkeys.append(IP)
    angles.append(surface_exit_angle)
    surface_crossings.append(surface_exit_distance)
    box2_z = np.cos(surface_exit_angle) * (surface_exit_distance + 3.885/np.sin(surface_exit_angle))
    prototype_z = np.cos(surface_exit_angle) * (surface_exit_distance + 1.295/np.sin(surface_exit_angle))
    print("%s: %2.1f km from IP, exits at %2.2f deg w.r.t. surface at elevation of %2.2f m"%(IP,surface_exit_distance/1e3,(180/np.pi)*surface_exit_angle,se[-1]))
    SINE_outfile_name = "Geometry/SINE_%s.dat"%IP
    with open(SINE_outfile_name,"w") as f:
        _template = sine_template.format(exit_point=IP,
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


# next the UNDINE detectors
for IP,lake_distances in lake_intersections.items():

    surface_loc = np.array(lat_long_to_xyz(*surface_exits[IP]))
    IP_loc = None
    for k in list(LHC_data.index):
        if k in IP:
            IP_loc = np.array(LHC_data.loc[k,['X','Y','Z']])
    assert(IP_loc is not None)
    beam_exit_displacement = surface_loc - IP_loc
    surface_exit_distance = np.linalg.norm(beam_exit_displacement)
    beam_exit_dir = beam_exit_displacement / surface_exit_distance
    surface_normal_dir = surface_loc / np.linalg.norm(surface_loc)
    surface_exit_angle = np.pi/2 - np.arccos(np.dot(beam_exit_dir,surface_normal_dir))

    lake_intersection_distance_1,lake_intersection_distance_2 = lake_distances
    detector_distance = lake_intersection_distance_1 + 0.1*(lake_intersection_distance_2 - lake_intersection_distance_1)
    lake_z_1 = lake_intersection_distance_1*np.cos(surface_exit_angle)
    lake_z_2 = lake_intersection_distance_2*np.cos(surface_exit_angle)
    lake_length = lake_z_2 - lake_z_1
    lake_z = (lake_z_1 + lake_z_2)/2
    detector_z = detector_distance * np.cos(surface_exit_angle)
    detector_depth = np.sin(surface_exit_angle) * (detector_distance - surface_exit_distance)

    UNDINE_outfile_name = "Geometry/UNDINE_%s.dat"%IP
    with open(UNDINE_outfile_name,"w") as f:
        _template = undine_template.format(exit_point=IP,
                                    lake_intersection_distance_1=lake_intersection_distance_1,
                                    lake_intersection_distance_2=lake_intersection_distance_2,
                                    lake_z=lake_z,
                                    lake_length=lake_length,
                                    detector_depth=detector_depth,
                                    detector_z=detector_z)
        f.write(_template)


# Now let's make the SIREN input text files for each beamline

# start with neutrino primaries
forward_flux_files = {
    "LHC13":{
        "light":["DPMJET", "EPOSLHC", "PYTHIA8", "QGSJET", "SIBYLL"],
        "charm":["BDGJKR", "BKRS", "BKSS", "MS", "SIBYLL"]
    },
    "Large":{
        "light":["EPOSLHC"],
        "charm":["BKRS"]
    },
    "Run3":{
        "light":["EPOSLHC"],
        "charm":["POWHEG+P8monash"]
    },
    "VLarge":{
        "light":["EPOSLHC","SIBYLL"],
        "charm":["BKRS"]
    }
}

primaries = [12,-12,
             14,-14,
             16,-16]

for prefix,parent_dict in forward_flux_files.items():
    if prefix!="LHC13": continue
    for parent,generators in parent_dict.items():
        for generator in generators:
            print(prefix,generator,parent)
            siren_input_file_prefix = "Input/%s_%s_%s"%(prefix,generator,parent)
            if os.path.isfile("%s_%s_%s.txt"%(siren_input_file_prefix,14,IPkeys[0])): continue
            input_file = forward_nu_flux_dir + '/files/' + prefix + '_' + generator + '_' + parent + '_0.txt'
            flux_dataframes = PrepareSIRENInputFromIP(input_file,IPkeys,angles,surface_crossings)
            for IPkey in IPkeys:
                flux_data = flux_dataframes[IPkey]
                for primary in primaries:
                    siren_input_file = "%s_%s_%s.txt"%(siren_input_file_prefix,primary,IPkey)
                    if not os.path.isfile(siren_input_file):
                        print("Preparing",siren_input_file)
                        flux_data.query("PDG==@primary").to_csv(siren_input_file,index=False)

# HNL primaries from meson decay via FORESEE
masses = [
    #"0500","0600","0700","0800","0900","1000","1500","2000",
    "3000","4000","5000","6000"
          ]
for mass in masses:
    print("Preparing HNL-mu mass %s MeV"%mass)
    siren_input_file_prefix = "Input/HNL-mu_m_%s"%mass
    if os.path.isfile("%s_%s.txt"%(siren_input_file_prefix,IPkeys[0])): continue
    input_file = foresee_dir + 'Models/HNL/HNL-mu/model/LLP_spectra/HNL_flux_m_%s.txt'%mass
    flux_dataframes = PrepareSIRENInputFromIP(input_file,IPkeys,angles,surface_crossings)
    for IPkey in IPkeys:
        flux_data = flux_dataframes[IPkey]
        siren_input_file = "%s_%s.txt"%(siren_input_file_prefix,IPkey)
        if not os.path.isfile(siren_input_file):
            print("Preparing",siren_input_file)
            flux_data.to_csv(siren_input_file,index=False)