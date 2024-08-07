import numpy as np
import sys
geneva_dir = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/"
forward_nu_flux_dir = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/forward-nu-flux-fit/"
import pandas as pd

from GeometryHelpers import *

import siren
from siren.interactions import DISFromSpline

# Define some geometry objects that will be helpful 
LHC_data = pd.read_parquet(geneva_dir+'Data/LHC_data.parquet')
Lake_data = pd.read_parquet(geneva_dir+'Data/Lake_data.parquet')

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
xsfiledir = geneva_dir+"../DUNEAtmo/cross_sections/csms_differential_v1.0/"

nue = siren.dataclasses.Particle.ParticleType.NuE
numu = siren.dataclasses.Particle.ParticleType.NuMu
nutau = siren.dataclasses.Particle.ParticleType.NuTau
nuebar = siren.dataclasses.Particle.ParticleType.NuEBar
numubar = siren.dataclasses.Particle.ParticleType.NuMuBar
nutaubar = siren.dataclasses.Particle.ParticleType.NuTauBar
target_type = siren.dataclasses.Particle.ParticleType.Nucleon

DIS_xs = {}

DIS_xs['nu_CC'] = DISFromSpline(xsfiledir+'/dsdxdy_nu_CC_iso.fits',
                                xsfiledir+'/sigma_nu_CC_iso.fits',
                                [nue,numu,nutau],
                                [target_type],'m')
DIS_xs['nu_NC'] = DISFromSpline(xsfiledir+'/dsdxdy_nu_NC_iso.fits',
                                xsfiledir+'/sigma_nu_NC_iso.fits',
                                [nue,numu,nutau],
                                [target_type],'m')
DIS_xs['nubar_CC'] = DISFromSpline(xsfiledir+'/dsdxdy_nubar_CC_iso.fits',
                                   xsfiledir+'/sigma_nubar_CC_iso.fits',
                                  [nuebar,numubar,nutaubar],
                                  [target_type],'m')
DIS_xs['nubar_NC'] = DISFromSpline(xsfiledir+'/dsdxdy_nubar_NC_iso.fits',
                                   xsfiledir+'/sigma_nubar_NC_iso.fits',
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
    data_dict['pz'] = np.sqrt(data_dict['E']**2  - data_dict['px']**2 - data_dict['py'] **2)
    
    df = pd.DataFrame(data=data_dict)
    return df
    # shuffle rows 
    # return df.sample(frac=1).reset_index(drop=True)

# returns the muon range in m
# rho is the material density in g/cm^3
def muon_range(E,rho=1):
    return 1./(rho * 5e-4) * np.log(1 + 2e-3 * E)


class MuonSimulation:

    def __init__(self,infile=None,
                 prefix=None,
                 generator=None,
                 parent=None,
                 Nstart=0,
                 N=None,
                 xs_mode="CC",
                 det_mode="lake",
                 lake_center = 10000,
                 lake_extent = 100,
                 verbose=False):
        if infile is not None:
            self.data = pd.read_parquet(infile)
        elif None not in [prefix,generator,parent]:
            self.data = get_flux_data(prefix,generator,parent)
            if N is not None:
                self.data = self.data[Nstart:Nstart+N].reset_index()
        else:
            print('Insufficient input arguments. Exiting...')
            exit(0)
        self.verbose = verbose
        if det_mode not in ["lake","surface"]:
            print("Invalid det mode %s, use lake or surface"%det_mode)
            exit(0)
        self.det_mode = det_mode
        self.lake_center = lake_center # detector center in m
        self.lake_extent = lake_extent # detector extent in m
        if xs_mode=="CC":
            self.DIS_nu = DIS_xs['nu_CC']
            self.DIS_nubar = DIS_xs['nubar_CC']
        elif xs_mode=="NC":
            self.DIS_nu = DIS_xs['nu_NC']
            self.DIS_nubar = DIS_xs['nubar_NC']
        else:
            print("Invalid xs mode %s, use CC or NC"%xs_mode)
            exit(0)

    def SampleSecondaryMomenta(self,N=None):

        random = siren.utilities.SIREN_random()
        record = siren.dataclasses.InteractionRecord()
        record.signature.target_type = target_type
        record.target_mass = m_iso
        #record.target_momentum = [m_iso,0,0,0]
        sec_types = [siren.dataclasses.Particle.MuMinus,siren.dataclasses.Particle.Hadrons]
        record.signature.secondary_types = sec_types

        E_lep = np.zeros(len(self.data))
        px_lep = np.zeros(len(self.data))
        py_lep = np.zeros(len(self.data))
        pz_lep = np.zeros(len(self.data))
        E_had = np.zeros(len(self.data))
        px_had = np.zeros(len(self.data))
        py_had = np.zeros(len(self.data))
        pz_had = np.zeros(len(self.data))

        if N is None: N = len(self.data)
        for i,ind in enumerate(self.data.index):
            if self.verbose: print("%d out of %d"%(i,N),end='\r')
            if i>=N: break
            if self.data['E'][ind] <= 10: continue
            primary_type = siren.dataclasses.Particle.ParticleType(int(self.data['PDG'][ind]))
            record.primary_momentum = [self.data['E'][ind],
                                       self.data['px'][ind],
                                       self.data['py'][ind],
                                       self.data['pz'][ind]]
            xs_record = siren.dataclasses.CrossSectionDistributionRecord(record)
            if self.data['PDG'][ind] > 0:
                record.signature = self.DIS_nu.GetPossibleSignaturesFromParents(primary_type,target_type)[0]
                self.DIS_nu.SampleFinalState(xs_record,random)
            else:
                record.signature = self.DIS_nubar.GetPossibleSignaturesFromParents(primary_type,target_type)[0]
                self.DIS_nubar.SampleFinalState(xs_record,random)
            xs_record.Finalize(record)
            theta = 2*np.pi*np.random.random() # rotate by a random number
            E_lep[i] = record.secondary_momenta[0][0]
            px_lep[i] = np.cos(theta) * record.secondary_momenta[0][1] - np.sin(theta) * record.secondary_momenta[0][2]
            py_lep[i] = np.sin(theta) * record.secondary_momenta[0][1] + np.cos(theta) * record.secondary_momenta[0][2]
            pz_lep[i] = record.secondary_momenta[0][3]
            E_had[i] = record.secondary_momenta[1][0]
            px_had[i] = np.cos(theta) * record.secondary_momenta[1][1] - np.sin(theta) * record.secondary_momenta[1][2]
            py_had[i] = np.sin(theta) * record.secondary_momenta[1][1] + np.cos(theta) * record.secondary_momenta[1][2]
            pz_had[i] = record.secondary_momenta[1][3]
        
        self.data['E_lep'] = E_lep
        self.data['px_lep'] = px_lep
        self.data['py_lep'] = py_lep
        self.data['pz_lep'] = pz_lep
        self.data['E_had'] = E_had
        self.data['px_had'] = px_had
        self.data['py_had'] = py_had
        self.data['pz_had'] = pz_had

    def DumpData(self,file):
        self.data.to_parquet(file)
    
    def EnsureUnitLepDir(self):
        if 'p_lep' not in self.data.keys():
            self.data['p_lep'] = np.sqrt(self.data['px_lep']**2 + self.data['py_lep']**2 + self.data['pz_lep']**2)
            self.data['ux_lep'] = self.data['px_lep']/self.data['p_lep']
            self.data['uy_lep'] = self.data['py_lep']/self.data['p_lep']
            self.data['uz_lep'] = self.data['pz_lep']/self.data['p_lep']
    
    def EnsureUnitHadDir(self):
        if 'p_had' not in self.data.keys():
            self.data['p_had'] = np.sqrt(self.data['px_had']**2 + self.data['py_had']**2 + self.data['pz_had']**2)
            self.data['ux_had'] = self.data['px_had']/self.data['p_had']
            self.data['uy_had'] = self.data['py_had']/self.data['p_had']
            self.data['uz_had'] = self.data['pz_had']/self.data['p_had']
    
    def EnsureUnitNeutrinoDir(self):
        if 'p' not in self.data.keys():
            self.data['p'] = np.sqrt(self.data['px']**2 + self.data['py']**2 + self.data['pz']**2)
            self.data['ux'] = self.data['px']/self.data['p']
            self.data['uy'] = self.data['py']/self.data['p']
            self.data['uz'] = self.data['pz']/self.data['p']
        
        
    def CalculateLakeIntersectionsFromIP(self,IPkey,N=None,limit=5000000):
        
        self.EnsureUnitNeutrinoDir()
        if N is None: N = len(self.data)
        nu_dirs = np.array(self.data[['ux','uy','uz']])[:N]
        
        int1_list,int2_list = calculate_intersections_with_lake(LHC,
                                                                np.array(LHC_data.loc[IPkey,['X','Y','Z']]),
                                                                np.array(LHC_data.loc[IPkey,['CrossingOrientation','CrossingAngle']]),
                                                                np.array(Lake_data[['Latitude','Longitude']]),
                                                                particle_unit_dirs=nu_dirs,
                                                                limit=limit)
        beam_dir = LHC.tangent_line(np.array(LHC_data.loc[IPkey,['X','Y','Z']]))
        R = rotation_to_beam_direction(beam_dir)
        lake_intersection = {}
        lake_intersection_lat_long = {}
        for i in range(4):
            lake_intersection[i] = np.zeros((len(self.data),3))
            lake_intersection_lat_long[i] = np.zeros((len(self.data),2))
        self.lake_intersections = []
        self.lake_intersections_lat_long = []
        for i,(int1,int2,nu_dir) in enumerate(zip(int1_list,int2_list,nu_dirs)):
            if self.verbose: print("%d out of %d"%(i,N),end='\r')
            ints = []
            ints_lat_long = []
            nu_dir_global = np.dot(R,nu_dir)
            trange = np.linspace(-100000,100000,2*1000)
            points = np.array(LHC_data.loc[IPkey,['X','Y','Z']]).reshape(-1,1) + np.outer(nu_dir_global,trange)
            earth_points = np.array([xyz_to_lat_long(*p) for p in points.transpose()])
            for intersections in (int1,int2):
                if(type(intersections) == shapely.geometry.MultiLineString):
                    for intersection in intersections.geoms:
                        for coord in list(intersection.coords):
                            ints_lat_long.append(coord)
                            idx = np.argmin(np.sum(np.abs(earth_points[:,:2]-coord),axis=1))
                            ints.append(points.transpose()[idx])
                else:
                    for coord in list(intersections.coords):
                        ints_lat_long.append(coord)
                        idx = np.argmin(np.sum(np.abs(earth_points[:,:2]-coord),axis=1))
                        ints.append(points.transpose()[idx])
            for i_int,(int_xyz,int_lat_long) in enumerate(zip(ints,ints_lat_long)):
                lake_intersection[i_int][i] = int_xyz
                lake_intersection_lat_long[i_int][i] = int_lat_long
        for i_int in lake_intersection.keys():
            self.data['lake_intersection%d'%i_int] = lake_intersection[i_int].tolist()
            self.data['lake_intersection_lat_long%d'%i_int] = lake_intersection_lat_long[i_int].tolist()
            
    def CalculateSurfaceIntersectionFromIP(self,IPkey,N=None):
        
        self.EnsureUnitNeutrinoDir()
        if N is None: N = len(self.data)
        nu_dirs = np.array(self.data[['ux','uy','uz']])[:N]
        nu_locs = np.array(self.data[['x0','y0','z0']])[:N]
        surface_intersections = np.zeros((len(self.data),3))
        surface_intersections_lat_long = np.zeros((len(self.data),3))
        beam_pos = np.array(LHC_data.loc[IPkey,['X','Y','Z']])
        beam_dir = LHC.tangent_line(beam_pos)
        (surface_intersections[:N],
         surface_intersections_lat_long[:N]) = calculate_intersections_with_surface(beam_pos,
                                                                                    beam_dir,
                                                                                    particle_positions=nu_locs,
                                                                                    particle_unit_dirs=nu_dirs)
        self.data['surface_intersection'] = surface_intersections.tolist()
        self.data['surface_intersection_lat_long'] = surface_intersections_lat_long.tolist()

    def CalculateLeptonSurfaceIntersectionFromIP(self,IPkey,N=None):
        
        self.EnsureUnitLepDir()
        if N is None: N = len(self.data)
        lep_dirs = np.array(self.data[['ux_lep','uy_lep','uz_lep']])[:N]
        lep_locs = np.array([x for x in self.data['DIS_location']])[:N]
        surface_intersections = np.zeros((len(self.data),3))
        surface_intersections_lat_long = np.zeros((len(self.data),3))
        beam_pos = np.array(LHC_data.loc[IPkey,['X','Y','Z']])
        beam_dir = LHC.tangent_line(beam_pos)
        (surface_intersections[:N],
         surface_intersections_lat_long[:N]) = calculate_intersections_with_surface(beam_pos,
                                                                                    beam_dir,
                                                                                    particle_positions=lep_locs,
                                                                                    particle_unit_dirs=lep_dirs,
                                                                                    particle_position_beam_coordinates=False)
        self.data['lepton_surface_intersection'] = surface_intersections.tolist()
        DIS_locations = np.array([x for x in self.data['DIS_location']])
        self.data['lepton_surface_distances'] = np.linalg.norm(surface_intersections-DIS_locations,axis=-1).tolist()
        self.data['lepton_surface_intersection_lat_long'] = surface_intersections_lat_long.tolist()

    def CalculateDISlocationFromIP(self,IPkey,N=None,n_lengths=1):

        NA = 6.02e23 # particles/mol
        rho_earth = 2.7 # g/cm^3
        rho_water = 1 # g/cm^3
        M_nucleon = 1 # g/mol

        self.EnsureUnitNeutrinoDir()
        if N is None: N = len(self.data)
        
        # Starting location at interaction point
        x0 = np.array(LHC_data.loc[IPkey,['X','Y','Z']])
        beam_dir = LHC.tangent_line(np.array(LHC_data.loc[IPkey,['X','Y','Z']]))
        R = rotation_to_beam_direction(beam_dir)

        # Calculate beam exit point
        self.CalculateBeamExitPointFromIP(IPkey)
        beam_surface_distance = np.linalg.norm(self.beam_exit_point-x0)

        DIS_location = np.zeros((len(self.data),3))
        DIS_distance = np.zeros(len(self.data))
        interaction_probability = np.zeros(len(self.data))
        
        if 'surface_distance' not in self.data.keys():
            self.data['surface_distance'] = [np.linalg.norm(np.array(x) - x0) for x in self.data['surface_intersection']]
        if 'lake_distance0' not in self.data.keys():
            for i in range(4): self.data['lake_distance%d'%i] = [np.linalg.norm(np.array(x) - x0) for x in self.data['lake_intersection%d'%i]]

        for i,ind in enumerate(self.data.index):
            if self.verbose: print("%d out of %d"%(i,N),end='\r')
            if i>N: break
            if self.data['E'][ind] <= 10: continue
            if self.data['PDG'][ind] > 0:
                xs = self.DIS_nu.TotalCrossSection(siren.dataclasses.Particle.ParticleType(int(self.data['PDG'][ind])),self.data['E'][ind])
            else:
                xs = self.DIS_nubar.TotalCrossSection(siren.dataclasses.Particle.ParticleType(int(self.data['PDG'][ind])),self.data['E'][ind])
            # build PDF
            in_earth = True
            distances = []
            densities = []
            if self.det_mode=="lake":
                nu_detector_distance = self.lake_center / self.data['uz'][ind] # distance along neutrion line to detector center
                min_distance = nu_detector_distance - (self.lake_extent/2.) # front of lake detector
                max_distance = nu_detector_distance + (self.lake_extent/2.) # back of lake detector 
                # min_distance = (1+1e-3)*self.data['lake_distance0'][ind] # just past first lake intersection
                # max_distance = (1-1e-3)*self.data['lake_distance1'][ind] # just before second lake intersection
            elif self.det_mode=="surface":
                nu_detector_distance = beam_surface_distance / self.data['uz'][ind] # distance along neutrino line to detector plane
                max_distance = min(nu_detector_distance,self.data['surface_distance'][ind])
                min_distance = max_distance - n_lengths*muon_range(self.data["E_lep"][ind],rho=rho_water)
            else:
                print("bad det mode")
                exit(0)
            prev_distance = min_distance
            for i_cross in range(4):
                rho = rho_earth if in_earth else rho_water
                if self.data['lake_distance%d'%i_cross][ind] < min_distance: 
                    # the lake crossing is before the minimum simulation distance
                    continue
                densities.append(rho*NA/M_nucleon)
                if max_distance < self.data['lake_distance%d'%i_cross][ind]:
                    # we have exited the surface or reached the detector distance
                    distances.append(100*(max_distance-prev_distance)) # cm
                    break
                else:
                    # we are still under the surface and before the detector
                    distances.append(100*(self.data['lake_distance%d'%i_cross][ind]-prev_distance)) # cm
                    prev_distance = self.data['lake_distance%d'%i_cross][ind]
                # flip to the other medium
                in_earth = not in_earth
            interaction_probability[i] = 1 - np.exp(-np.dot(distances,densities)*xs)
            PDF = np.array(densities) / np.dot(distances,densities) # normalized
            integral = 0
            distance = 0
            r = np.random.random() # random value to sample from CDF
            for p,dist in zip(PDF,distances):
                if (r > integral) and (r < integral + dist*p):
                    DIS_distance[i] = (distance + (r-integral)/p)/100 # m
                    break
                distance += dist
                integral += dist*p
            DIS_distance[i] += min_distance
            nu_dir_global = np.dot(R,np.array(self.data[['ux','uy','uz']])[ind])
            DIS_location[i] = x0 + nu_dir_global*DIS_distance[i] # back to m
            

        self.data['interaction_probability'] = interaction_probability
        self.data['DIS_distance'] = DIS_distance
        self.data['DIS_location'] = DIS_location.tolist()

    def CalculateBeamExitPointFromIP(self,IPkey):
        if not hasattr(self,'beam_exit_point') or not hasattr(self,'beam_exit_point_lat_long'):
            beam_pos = np.array(LHC_data.loc[IPkey,['X','Y','Z']])
            beam_dir = LHC.tangent_line(beam_pos)
            result = calculate_intersections_with_surface(beam_pos,
                                                        beam_dir,
                                                        particle_positions = [[0,0,0]],
                                                        particle_unit_dirs=[[0,0,1]])
            self.beam_exit_point = result[0][0]
            self.beam_exit_point_lat_long = result[1][0]

    def CalculateNeutrinoProfileFromIP(self,IPkey,N=None,lumi=150):

        self.EnsureUnitNeutrinoDir()
        if N is None: N = len(self.data)

        self.CalculateBeamExitPointFromIP(IPkey)

        # Starting location at interaction point
        x0 = np.array(LHC_data.loc[IPkey,['X','Y','Z']])
        beam_dist = np.linalg.norm(self.beam_exit_point - x0)

        transverse_profile = np.zeros((len(self.data),2))
        weights = np.zeros(len(self.data))

        for i,ind in enumerate(self.data.index):
            if self.verbose: print("%d out of %d"%(i,N),end='\r')
            if i>N: break
            CosTheta = self.data['uz'][ind]
            if self.det_mode=="lake":
                # we are considering the front of the lake point
                det_dist = ((1+1e-3)*self.data['lake_distance0'][ind]) / CosTheta
            elif self.det_mode=="surface":
                # we are considering the surface exit point
                det_dist = beam_dist
            nu_dist = det_dist / CosTheta
            transverse_profile[i] = np.array([self.data['ux'][ind]*nu_dist,
                                              self.data['uy'][ind]*nu_dist])
            weights[i] = self.data['wgt'][ind]*1000*lumi
        self.data["nu_transverse_profile_x"] = transverse_profile[:,0]
        self.data["nu_transverse_profile_y"] = transverse_profile[:,1]
        self.data["nu_weight"] = weights
    
    def CalculateMuonProfileFromIP(self,IPkey,N=None,beam_dist=None,lumi=150):

        self.EnsureUnitNeutrinoDir()
        self.EnsureUnitLepDir()
        if N is None: N = len(self.data)

        self.CalculateBeamExitPointFromIP(IPkey)

        # Starting location at interaction point
        x0 = np.array(LHC_data.loc[IPkey,['X','Y','Z']])
        beam_dist = np.linalg.norm(self.beam_exit_point - x0)

        transverse_profile = np.zeros((len(self.data),2))
        weights = np.zeros(len(self.data))
        muon_lengths = np.zeros(len(self.data))

        for i,ind in enumerate(self.data.index):
            if self.verbose: print("%d out of %d"%(i,N),end='\r')
            if i>N: break
            if np.abs(self.data['PDG'][ind]) != 14: continue #only accept numu
            nu_CosTheta = self.data['uz'][ind]
            lep_CosTheta = self.data['uz_lep'][ind]
            DIS_dist = self.data['DIS_distance'][ind]
            if self.det_mode=="lake":
                # we are considering the front of the lake point
                det_dist = ((1+1e-3)*self.data['lake_distance0'][ind])
            elif self.det_mode=="surface":
                # we are considering the surface exit point
                det_dist = beam_dist
            mu_beam_dist = det_dist - DIS_dist/nu_CosTheta
            mu_dist = mu_beam_dist/lep_CosTheta

            transverse_profile[i] = np.array([self.data['ux'][ind]*DIS_dist + self.data['ux_lep'][ind]*mu_dist,
                                              self.data['uy'][ind]*DIS_dist + self.data['uy_lep'][ind]*mu_dist])
            muon_lengths[i] = mu_dist
            if mu_beam_dist<0: continue # backward-going muons get zero-weighted
            weights[i] = self.data['wgt'][ind]*self.data['interaction_probability'][ind]*1000*lumi
        self.data["muon_transverse_profile_x"] = transverse_profile[:,0]
        self.data["muon_transverse_profile_y"] = transverse_profile[:,1]
        self.data["muon_weights"] = weights
        self.data["muon_lengths"] = muon_lengths

            


