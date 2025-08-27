import siren
from siren.SIREN_Controller import SIREN_Controller
import os
import argparse
import numpy as np
import awkward as ak

from SIREN_SimulationHelpers import compute_HNL_time_delay

# Remove empty sublists and dimensions from awkward arrays
def clean_array(array):
    return np.array(np.squeeze(ak.Array([[sublist for sublist in inner_list if len(sublist) > 0] for inner_list in array])))

SIREN_dir = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/Data/SIREN"

def RunNeutrinoSimulation(prefix,generator,parent,primary,
                          events_to_inject,outfile,
                          experiment,xs_mode="CC",lumi=3000):

    if "SINE" in experiment:
        experiment_prefix = "SINE"
    elif "UNDINE" in experiment:
        experiment_prefix = "UNDINE"
    else:
        print("Experiment %s not valid"%experiment)
        return
    IP_tag = experiment.replace("%s_"%experiment_prefix,"")

    # Define the controller
    controller = SIREN_Controller(events_to_inject, experiment)

    # Particle to inject
    primary_type = (siren.dataclasses.Particle.ParticleType)(primary)

    cross_section_model = "CSMSDISSplines"

    #xsfiledir = siren.utilities.get_cross_section_model_path(cross_section_model)
    #xsfiledir = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pweigel/cross_sections/20241017"
    xsfiledir = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pweigel/cross_sections/20241112" # fixed x and y orientation

    # Cross Section Model
    target_type = siren.dataclasses.Particle.ParticleType.Nucleon

    if primary>0:
        nutype = "neutrino"
        #nutype = "nu"
    else:
        nutype = "antineutrino"
        #nutype = "nubar"

    if primary in [12,-12,14,-14]:
        nuflavor = "muon" # nueCC and numuCC cross sections are very similar in this energy range
    elif primary in [16,-16]:
        nuflavor = "tau"

    if xs_mode =="CC":
        minQ2 = 0.01
    elif xs_mode=="NC":
        minQ2 = 1

    # DIS_xs = siren.interactions.DISFromSpline(
    #     os.path.join(xsfiledir, "dsdxdy_%s_%s_iso.fits"%(nu_type,xs_mode)),
    #     os.path.join(xsfiledir, "sigma_%s_%s_iso.fits"%(nu_type,xs_mode)),
    #     [primary_type],
    #     [target_type], "m"
    # )
    DIS_xs = siren.interactions.DISFromSpline(
        os.path.join(xsfiledir, "wcg24b_dsdxdy_%s_%s_%s_isoscalar.fits"%(xs_mode,nuflavor,nutype)),
        os.path.join(xsfiledir, "wcg24b_sigma_%s_%s_%s_isoscalar.fits"%(xs_mode,nuflavor,nutype)),
        1,siren.utilities.Constants.isoscalarMass,minQ2,
        [primary_type],
        [target_type], "cm"
    )


    primary_xs = siren.interactions.InteractionCollection(primary_type, [DIS_xs])
    controller.SetInteractions(primary_xs)



    # Primary distributions
    primary_injection_distributions = {}
    primary_physical_distributions = {}

    siren_input_file = "%s/Input/%s_%s_%s_%d_%s.txt"%(SIREN_dir,prefix,generator,parent,primary,IP_tag)
    assert(os.path.isfile(siren_input_file))

    with open(siren_input_file, "rb") as f:
        num_input_events = sum(1 for _ in f) - 1

    primary_external_dist = siren.distributions.PrimaryExternalDistribution(siren_input_file)
    primary_injection_distributions["external"] = primary_external_dist


    fid_vol = controller.GetFiducialVolume()
    position_distribution = siren.distributions.PrimaryBoundedVertexDistribution(fid_vol)
    primary_injection_distributions["position"] = position_distribution

    # SetProcesses
    controller.SetProcesses(
        primary_type, primary_injection_distributions, primary_physical_distributions
    )


    # Run generation and save events
    controller.Initialize()
    controller.GenerateEvents()
    controller.SaveEvents(outfile,
                          hdf5=False, siren_events=False,
                          save_int_probs=True,
                          save_int_params=True)

    data = ak.from_parquet("%s.parquet"%outfile)
    weights = np.array(np.squeeze(data.wgt) * lumi * 1000 * np.prod(data.int_probs,axis=-1))
    weights *= num_input_events / events_to_inject # correct for sampled events
    data["weights"] = weights

    if experiment_prefix=="UNDINE":
        # write output array
        ak.to_parquet(data,"%s.parquet"%outfile)
    elif experiment_prefix=="SINE":
        mu_vertex = np.squeeze(data.vertex)
        muon_momenta = np.array(np.squeeze(data.secondary_momenta[:,:,0]))
        muon_momentum = np.array(np.linalg.norm(muon_momenta[:,1:],axis=1))
        mu_dir = muon_momenta[:,1:] / np.expand_dims(muon_momentum,-1)

        panels = {
            # 0:controller.detector_model.GetSector("prototype"),
            1:controller.detector_model.GetSector("panel_1"),
            2:controller.detector_model.GetSector("panel_2"),
            3:controller.detector_model.GetSector("panel_3")
            }

        def GetPanelIntersections(location, direction):
            _loc = siren.math.Vector3D(location)
            _loc_detector = controller.detector_model.GeoPositionToDetPosition(siren.detector.GeometryPosition(_loc)).get()
            _dir = siren.math.Vector3D(direction)
            panel_intersections = {}
            panel_distances = {}
            panel_columndepths = {}

            for ip,panel in panels.items():
                panel_intersections[ip] = []
                panel_distances[ip] = []
                panel_columndepths[ip] = []
                for intersection in panel.geo.Intersections(_loc,_dir):
                    panel_intersections[ip].append([intersection.position.GetX(),
                                                    intersection.position.GetY(),
                                                    intersection.position.GetZ()])
                    panel_distances[ip].append(intersection.distance)
                    int_loc_dectector = controller.detector_model.GeoPositionToDetPosition(siren.detector.GeometryPosition(intersection.position)).get()
                    panel_columndepths[ip].append(controller.detector_model.GetColumnDepthInCGS(_loc_detector,int_loc_dectector))
            return panel_intersections,panel_distances,panel_columndepths

        panel_ints = {ip:[] for ip in panels.keys()}
        panel_dist = {ip:[] for ip in panels.keys()}
        panel_cdep = {ip:[] for ip in panels.keys()}
        hit_mask = {ip:[] for ip in panels.keys()}
        hit_mask_tot = []
        for mv,md in zip(mu_vertex,mu_dir):
            p_ints,p_dist,p_cdep = GetPanelIntersections(mv,md)
            hit = False
            for panel in p_ints.keys():
                panel_ints[panel].append(p_ints[panel])
                panel_dist[panel].append(p_dist[panel])
                panel_cdep[panel].append(p_cdep[panel])
                if sum(np.array(p_dist[panel])>0)>0:
                    hit = True
                    hit_mask[panel].append(True)
                else: hit_mask[panel].append(False)

            hit_mask_tot.append(hit)

        for ik in panel_ints.keys():
            data["panel%d_int_locations"%ik] = panel_ints[ik]
            data["panel%d_int_distances"%ik] = panel_dist[ik]
            data["panel%d_int_coldepths"%ik] = panel_cdep[ik]
            data["panel%d_hit_mask"%ik] = hit_mask[ik]

        data["hit_mask"] = hit_mask_tot

        muon_depth = siren.distributions.LeptonDepthFunction()

        data["muon_max_col_depth"] = [muon_depth(siren.dataclasses.Particle.NuMu, muE)
                                    for muE in muon_momenta[:,0]]

        for ip in panels.keys():
            data["panel%d_muon_survival"%ip] = data["panel%d_int_coldepths"%ip] < data["muon_max_col_depth"]
            data["panel%d_hit_mask_muon_survival"%ip] = np.logical_and(data["panel%d_hit_mask"%ip],
                                                                       np.any(data["panel%d_muon_survival"%ip],axis=-1))
        data["hit_mask_muon_survival"] = np.logical_or.reduce(tuple(data["panel%d_hit_mask_muon_survival"%ip] for ip in panels.keys()))


        ak.to_parquet(data[data["hit_mask"]==1],"%s.parquet"%outfile)


def RunHNLSimulation(prefix,generator,parent,primary,
                     events_to_inject,outfile,
                     experiment,
                     m4,Ue4,Umu4,Utau4,
                     lumi=3000,seed=42):

    if "SINE" in experiment:
        experiment_prefix = "SINE"
    elif "UNDINE" in experiment:
        experiment_prefix = "UNDINE"
    else:
        print("Experiment %s not valid"%experiment)
        return
    IP_tag = experiment.replace("%s_"%experiment_prefix,"")

    # Define the controller
    controller = SIREN_Controller(events_to_inject, experiment, seed=seed)

    # Particle to inject
    primary_type = (siren.dataclasses.Particle.ParticleType)(primary)
    if primary > 0:
        nutype="nu"
        hnl_type = siren.dataclasses.Particle.ParticleType.N4
    else:
        nutype="nubar"
        hnl_type = siren.dataclasses.Particle.ParticleType.N4Bar

    # Now include DIS interaction
    cross_section_model = "HNLDISSplines"

    xsfiledir = siren.utilities.get_cross_section_model_path(cross_section_model)

    # Cross Section Model
    target_type = siren.dataclasses.Particle.ParticleType.Nucleon

    m4_str = m4_str = f"{int(m4):07d}"
    DIS_xs = siren.interactions.HNLDISFromSpline(
        os.path.join(xsfiledir, "M_0000000MeV/dsdxdy-%s-N-nc-GRV98lo_patched_central.fits"%nutype),
        os.path.join(xsfiledir, "M_%sMeV/sigma-%s-N-nc-GRV98lo_patched_central.fits"%(m4_str,nutype)),
        float(m4)*1e-3,
        [Ue4,Umu4,Utau4],
        siren.utilities.Constants.isoscalarMass,
        1,
        [primary_type],
        [target_type],
    )
    print("Cross section threshold",DIS_xs.InteractionThreshold(siren.dataclasses.InteractionRecord()))

    DIS_interaction_collection = siren.interactions.InteractionCollection(primary_type, [DIS_xs])

    # Primary distributions
    primary_injection_distributions = {}
    primary_physical_distributions = {}

    siren_input_file = "%s/Input/%s_%s_%s_%d_%s.txt"%(SIREN_dir,prefix,generator,parent,primary,IP_tag)
    assert(os.path.isfile(siren_input_file))

    primary_external_dist = siren.distributions.PrimaryExternalDistribution(siren_input_file,1.1*DIS_xs.InteractionThreshold(siren.dataclasses.InteractionRecord()))
    primary_injection_distributions["external"] = primary_external_dist
    num_input_events = primary_external_dist.GetPhysicalNumEvents(); # number above energy threshold
    print("Number of input events above threshold: %d"%num_input_events)


    # fid_vol = controller.GetFiducialVolume()
    max_length = 20000 # m
    position_distribution = siren.distributions.PrimaryBoundedVertexDistribution(max_length)
    primary_injection_distributions["position"] = position_distribution

    #secondary_position_distribution = siren.distributions.SecondaryBoundedVertexDistribution(fid_vol)

    # SetProcesses
    controller.SetProcesses(
        primary_type, primary_injection_distributions, primary_physical_distributions,
        [hnl_type], [[]], [[]]
    )

    # Decay Model
    hnl_decay = siren.interactions.HNLDecay(float(m4)*1e-3, [Ue4, Umu4, Utau4], siren.interactions.HNLDecay.ChiralNature.Majorana)
    Decay_interaction_collection = siren.interactions.InteractionCollection(hnl_type, [hnl_decay])

    controller.SetInteractions(primary_interaction_collection=DIS_interaction_collection,)
    controller.SetInteractions(secondary_interaction_collections=[Decay_interaction_collection])

    # Run generation and save events
    controller.Initialize()

    def stop(datum, i):
        secondary_type = datum.record.signature.secondary_types[i]
        return (secondary_type != siren.dataclasses.Particle.ParticleType.N4 and
                secondary_type != siren.dataclasses.Particle.ParticleType.N4Bar and
                secondary_type != siren.dataclasses.Particle.ParticleType.WPlus and
                secondary_type != siren.dataclasses.Particle.ParticleType.WMinus and
                secondary_type != siren.dataclasses.Particle.ParticleType.Z0)

    controller.SetInjectorStoppingCondition(stop)

    controller.GenerateEvents(fill_tables_at_exit=False)
    controller.SaveEvents(outfile,
                          save_int_probs=True,
                          save_int_params=True,
                          fill_tables_at_exit=False,
                          hdf5=False, siren_events=False)

    data = ak.from_parquet("%s.parquet"%outfile)
    weights = np.array(np.squeeze(data.wgt) * lumi * 1000 * np.prod(data.int_probs,axis=-1))
    weights *= num_input_events / events_to_inject # correct for sampled events
    data["weights"] = weights


    if experiment_prefix=="UNDINE":
        ak.to_parquet(data,"%s.parquet"%outfile)
    else:
        print("Initial injection complete. Computing panel intersections...")
        muon_flag = np.abs(data.secondary_types) == 13
        n_muons = np.sum(muon_flag[:,-1],axis=-1)
        hnl_flag = np.abs(data.primary_type) == 5914
        muon_momenta = data.secondary_momenta[muon_flag][:,-1]

        decay_vertex = clean_array(data.vertex[hnl_flag])
        muon_momentum = [[np.linalg.norm(x[1:]) for x in muon_list] for muon_list in muon_momenta]
        mu_dir = muon_momenta[:,:,1:] / muon_momentum

        panels = {
            # 0:controller.detector_model.GetSector("prototype"),
            1:controller.detector_model.GetSector("panel_1"),
            2:controller.detector_model.GetSector("panel_2"),
            3:controller.detector_model.GetSector("panel_3")
            }

        def GetPanelIntersections(location, direction):
            _loc = siren.math.Vector3D(location)
            _loc_detector = controller.detector_model.GeoPositionToDetPosition(siren.detector.GeometryPosition(_loc)).get()
            _dir = siren.math.Vector3D(direction)
            panel_intersections = {}
            panel_distances = {}
            panel_columndepths = {}

            for ip,panel in panels.items():
                panel_intersections[ip] = []
                panel_distances[ip] = []
                panel_columndepths[ip] = []
                for intersection in panel.geo.Intersections(_loc,_dir):
                    panel_intersections[ip].append([intersection.position.GetX(),
                                                    intersection.position.GetY(),
                                                    intersection.position.GetZ()])
                    panel_distances[ip].append(intersection.distance)
                    int_loc_dectector = controller.detector_model.GeoPositionToDetPosition(siren.detector.GeometryPosition(intersection.position)).get()
                    panel_columndepths[ip].append(controller.detector_model.GetColumnDepthInCGS(_loc_detector,int_loc_dectector))
            return panel_intersections,panel_distances,panel_columndepths

        panel_ints = {i_muon:{ip:[[] for _ in range(len(data))] for ip in panels.keys()} for i_muon in range(2)}
        panel_dist = {i_muon:{ip:[[-1] for _ in range(len(data))] for ip in panels.keys()} for i_muon in range(2)}
        panel_cdep = {i_muon:{ip:[[-1] for _ in range(len(data))] for ip in panels.keys()} for i_muon in range(2)}
        hit_mask = {i_muon:{ip:[False for _ in range(len(data))] for ip in panels.keys()} for i_muon in range(2)}
        hit_mask_tot = {i_muon:[False for _ in range(len(data))] for i_muon in range(2)}
        for i_event,(dv,mds) in enumerate(zip(decay_vertex,mu_dir)):
            for i_muon,md in enumerate(mds):
                p_ints,p_dist,p_cdep = GetPanelIntersections(dv,md)
                hit = False
                for panel in p_ints.keys():
                    panel_ints[i_muon][panel][i_event] = (p_ints[panel])
                    panel_dist[i_muon][panel][i_event] = (p_dist[panel])
                    panel_cdep[i_muon][panel][i_event] = (p_cdep[panel])
                    if sum(np.array(p_dist[panel])>0)>0:
                        hit = True
                        hit_mask[i_muon][panel][i_event] = (True)
                    else: hit_mask[i_muon][panel][i_event] = (False)

                hit_mask_tot[i_muon][i_event] = (hit)

        muon_depth = siren.distributions.LeptonDepthFunction()

        for i_muon in range(2):
            for ik in panel_ints[i_muon].keys():
                data["muon%d_panel%d_int_locations"%(i_muon,ik)] = panel_ints[i_muon][ik]
                data["muon%d_panel%d_int_distances"%(i_muon,ik)] = panel_dist[i_muon][ik]
                data["muon%d_panel%d_int_coldepths"%(i_muon,ik)] = panel_cdep[i_muon][ik]
                data["muon%d_panel%d_hit_mask"%(i_muon,ik)] = hit_mask[i_muon][ik]

            data["muon%d_hit_mask"%i_muon] = hit_mask_tot[i_muon]


            data["muon%d_max_col_depth"%i_muon] = [muon_depth(siren.dataclasses.Particle.NuMu, mu_mom[i_muon,0]) if len(mu_mom)>(i_muon) else -1 for mu_mom in muon_momenta]

            for ip in panels.keys():
                data["muon%d_panel%d_survival"%(i_muon,ip)] = data["muon%d_panel%d_int_coldepths"%(i_muon,ip)] < data["muon%d_max_col_depth"%i_muon]
                survival_any = ak.any(data["muon%d_panel%d_survival"%(i_muon,ip)], axis=-1)
                hit_mask_arr = np.array(data["muon%d_panel%d_hit_mask"%(i_muon,ip)])
                data["muon%d_panel%d_hit_mask_survival"%(i_muon,ip)] = np.logical_and(survival_any, hit_mask_arr)
            data["muon%d_hit_mask_survival"%i_muon] = np.logical_or.reduce(tuple(data["muon%d_panel%d_hit_mask_survival"%(i_muon,ip)] for ip in panels.keys()))

        data["hit_mask_dimuon_survival"] = np.logical_and(data["muon0_hit_mask_survival"],data["muon1_hit_mask_survival"])
        abridged_data = data[data["muon0_hit_mask_survival"]==1]
        appended_data = compute_HNL_time_delay(abridged_data,float(m4)/1000.)
        ak.to_parquet(appended_data,"%s.parquet"%outfile ) # save only events with at least one muon hitting a panel



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, help='simulation case (neutrino/hnl)', default='neutrino')
    parser.add_argument('--primary', type=int, help='simulation primary PDG code', default=14)
    parser.add_argument('-p','--prefix', type=str, default='LHC13', help='forward-nu-flux prefix')
    parser.add_argument('-g','--generator', type=str, help='forward-nu-flux generator')
    parser.add_argument('-m','--parent', type=str, help='forward-nu-flux parent meson (light or charm)')
    parser.add_argument('-n', '--events-to-inject', type=int,help='number of events to inject')
    parser.add_argument('-o', '--output-file', type=str,help='output filename without extension')
    parser.add_argument('-e', '--experiment', type=str, default='GenevaSurface', help='experiment name (GenevaLake or GenevaSurface)')
    parser.add_argument('-x', '--xs-mode', type=str, default='CC', help='cross section mode (CC or NC)')
    parser.add_argument('-m4','--m4', type=str, default="0000000", help="HNL mass in MeV")
    parser.add_argument('-ue','--Ue4', type=float, default=0, help="HNL Ue4 mixing")
    parser.add_argument('-um','--Umu4', type=float, default=0, help="HNL Umu4 mixing")
    parser.add_argument('-ut','--Utau4', type=float, default=0, help="HNL Utau4 mixing")

    args = parser.parse_args()
    if args.case=='neutrino':
        RunNeutrinoSimulation(args.prefix,args.generator,args.parent,args.primary,
                              args.events_to_inject,args.output_file,args.experiment,args.xs_mode)
    elif args.case=='hnl':
        RunHNLSimulation(args.prefix,args.generator,args.parent,args.primary,
                         args.events_to_inject,args.output_file,args.experiment,
                         args.m4,args.Ue4,args.Umu4,args.Utau4)
    else:
        print("Case %s not recognized"%args.case)