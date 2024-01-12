import numpy as np
import sys
forward_nu_flux_dir = "../forward-nu-flux-fit"
sys.path.append(forward_nu_flux_dir)
from Code import experiments
import pandas as pd

from GeometryHelpers import *

import leptoninjector as LI
from leptoninjector.interactions import DISFromSpline

# Define some geometry objects that will be helpful 
LHC_data = pd.read_parquet('Data/LHC_data.parquet')
Lake_data = pd.read_parquet('Data/Lake_data.parquet')

# Define the LHC circle
LHC = Circle(np.array(LHC_data.loc['Point4',['X','Y','Z']]),
             np.array(LHC_data.loc['LHCb',['X','Y','Z']]),
             np.array(LHC_data.loc['Point6',['X','Y','Z']])) # last entry can be Point 2 or 6

# Fix elevation of other LHC interaction points based on LHC circle definition
for k in list(LHC_data.index):
    elev,point = LHC.find_elevation(*np.array(LHC_data.loc[k,['Latitude','Longitude']]))
    LHC_data['Elevation'][k] = elev
    LHC_data.loc[k,['X','Y','Z']] = lat_long_to_xyz(*LHC_data.loc[k,['Latitude','Longitude','Elevation']])

# Constants
mp = 0.938272
mn = 0.939565
m_iso = 0.5*(mp+mn)

# Define the relevant cross section files
xsfiledir = "../DUNEAtmo/cross_sections/csms_differential_v1.0/"

nue = LI.dataclasses.Particle.ParticleType.NuE
numu = LI.dataclasses.Particle.ParticleType.NuMu
nutau = LI.dataclasses.Particle.ParticleType.NuTau
nuebar = LI.dataclasses.Particle.ParticleType.NuEBar
numubar = LI.dataclasses.Particle.ParticleType.NuMuBar
nutaubar = LI.dataclasses.Particle.ParticleType.NuTauBar
target_type = LI.dataclasses.Particle.ParticleType.Nucleon

DIS_xs = {}

DIS_xs['nu_CC'] = DISFromSpline(xsfiledir+'/dsdxdy_nu_CC_iso.fits',
                                xsfiledir+'/sigma_nu_CC_iso.fits',
                                [nue,numu,nutau],
                                [target_type],'m')
DIS_xs['nu_NC'] = DISFromSpline(xsfiledir+'/dsdxdy_nu_CC_iso.fits',
                                xsfiledir+'/sigma_nu_CC_iso.fits',
                                [nue,numu,nutau],
                                [target_type],'m')
DIS_xs['nubar_CC'] = DISFromSpline(xsfiledir+'/dsdxdy_nubar_CC_iso.fits',
                                   xsfiledir+'/sigma_nubar_CC_iso.fits',
                                  [nuebar,numubar,nutaubar],
                                  [target_type],'m')
DIS_xs['nubar_NC'] = DISFromSpline(xsfiledir+'/dsdxdy_nubar_CC_iso.fits',
                                   xsfiledir+'/sigma_nubar_CC_iso.fits',
                                  [nuebar,numubar,nutaubar],
                                  [target_type],'m')
# Helper functions
    
def get_flux_data(prefix,generator,parent):
    keys = ['PDG','hPDG','x0','y0','z0','thx','thy','E','wgt']
    data = np.loadtxt(forward_nu_flux_dir + '/files/' + prefix + '_' + generator + '_' + parent + '_0.txt')
    data_dict = {}
    for k,col in zip(keys,data.T):
        data_dict[k] = col
    data_dict['px'] = data_dict['E'] * np.sin(data_dict['thx'])
    data_dict['py'] = data_dict['E'] * np.sin(data_dict['thy'])
    data_dict['pz'] = np.sqrt(data_dict['E']**2  -data_dict['px']**2 - data_dict['py'] **2)
    
    return pd.DataFrame(data=data_dict)


class MuonSimulation:

    def __init__(self,infile=None,prefix=None,generator=None,parent=None):
        if infile is not None:
            self.data = pd.read_parquet(infile)
        elif None not in [prefix,generator,parent]:
            self.data = get_flux_data(prefix,generator,parent)
        else:
            print('Insufficient input arguments. Exiting...')
            exit(0)

    def SampleSecondaryMomenta(self,N=None):

        random = LI.utilities.LI_random()
        record = LI.dataclasses.InteractionRecord()
        record.signature.target_type = target_type
        record.target_mass = m_iso
        record.target_momentum = [m_iso,0,0,0]
        sec_types = [LI.dataclasses.Particle.MuMinus,LI.dataclasses.Particle.Hadrons]
        record.signature.secondary_types = sec_types

        self.data['E_lep'] = np.zeros(len(self.data))
        self.data['px_lep'] = np.zeros(len(self.data))
        self.data['py_lep'] = np.zeros(len(self.data))
        self.data['pz_lep'] = np.zeros(len(self.data))
        self.data['E_had'] = np.zeros(len(self.data))
        self.data['px_had'] = np.zeros(len(self.data))
        self.data['py_had'] = np.zeros(len(self.data))
        self.data['pz_had'] = np.zeros(len(self.data))

        if N is None: N = len(self.data)
        for i,ind in enumerate(self.data.index):
            print(i,end='\r')
            if i>N: break
            if self.data['E'][ind] <= 10: continue
            primary_type = LI.dataclasses.Particle.ParticleType(int(self.data['PDG'][ind]))
            record.primary_momentum = [self.data['E'][ind],
                                       self.data['px'][ind],
                                       self.data['py'][ind],
                                       self.data['pz'][ind]]
            if self.data['PDG'][ind] > 0:
                record.signature = DIS_xs['nu_CC'].GetPossibleSignaturesFromParents(primary_type,target_type)[0]
                DIS_xs['nu_CC'].SampleFinalState(record,random)
            else:
                record.signature = DIS_xs['nubar_CC'].GetPossibleSignaturesFromParents(primary_type,target_type)[0]
                DIS_xs['nubar_CC'].SampleFinalState(record,random)
            self.data['E_lep'][ind] = record.secondary_momenta[0][0]
            self.data['px_lep'][ind] = record.secondary_momenta[0][1]
            self.data['py_lep'][ind] = record.secondary_momenta[0][2]
            self.data['pz_lep'][ind] = record.secondary_momenta[0][3]
            self.data['E_had'][ind] = record.secondary_momenta[1][0]
            self.data['px_had'][ind] = record.secondary_momenta[1][1]
            self.data['py_had'][ind] = record.secondary_momenta[1][2]
            self.data['pz_had'][ind] = record.secondary_momenta[1][3]

    def DumpData(self,file):
        self.data.to_parquet(file)
    
    def EnsureUnitLepDir(self):
        if 'p_lep' not in self.data.keys():
            self.data['p_lep'] = np.sqrt(self.data['px_lep']**2 + self.data['py_lep']**2 + self.data['pz_lep']**2)
            self.data['ux_lep'] = self.data['px_lep']/self.data['p_lep']
            self.data['uy_lep'] = self.data['py_lep']/self.data['p_lep']
            self.data['uz_lep'] = self.data['pz_lep']/self.data['p_lep']
        
        
    def CalculateLakeIntersectionsFromIP(self,IPkey,N=None):
        
        self.EnsureUnitLepDir()
        if N is None: N = len(self.data)
        lepton_dirs = np.array(self.data[['ux_lep','uy_lep','uz_lep']])[:N]
        
        int1_list,int2_list = calculate_intersections_with_lake(LHC,
                                                                np.array(LHC_data.loc[IPkey,['X','Y','Z']]),
                                                                np.array(LHC_data.loc[IPkey,['CrossingOrientation','CrossingAngle']]),
                                                                np.array(Lake_data[['Latitude','Longitude']]),
                                                                particle_unit_dirs=lepton_dirs,
                                                                limit=50000000)
        self.lake_intersections = []
        for int1,int2 in zip(int1_list,int2_list):
            ints = []
            for intersections in (int1,int2):
                if(type(intersections) == shapely.geometry.MultiLineString):
                    for intersection in intersections.geoms:
                        for coord in list(intersection.coords):
                            ints.append(coord)
                else:
                    for coord in list(intersections.coords):
                        ints.append(coord)
            self.lake_intersections.append(ints)
            
    def CalculateSurfaceIntersectionFromIP(self,IPkey,N=None):
        
        self.EnsureUnitLepDir()
        if N is None: N = len(self.data)
        lepton_dirs = np.array(self.data[['ux_lep','uy_lep','uz_lep']])[:N]
        
        (self.surface_intersections,
         self.surface_intersections_lat_long) = calculate_intersections_with_surface(LHC,
                                             np.array(LHC_data.loc[IPkey,['X','Y','Z']]),
                                             np.array(LHC_data.loc[IPkey,['CrossingOrientation','CrossingAngle']]),
                                             particle_unit_dirs=lepton_dirs)
        
        
        