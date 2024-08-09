import siren
from siren.SIREN_Controller import SIREN_Controller
import os
import argparse
import numpy as np
import awkward as ak

def RunLHCbMuonSimulation(prefix,generator,parent,
                          events_to_inject,outfile,
                          lumi=3000):
    
    # Expeirment to run
    experiment = "LakeGeneva"

    # Define the controller
    controller = SIREN_Controller(events_to_inject, experiment)

    # Particle to inject
    primary_type = siren.dataclasses.Particle.ParticleType.NuMu

    cross_section_model = "CSMSDISSplines"

    xsfiledir = siren.utilities.get_cross_section_model_path(cross_section_model)

    # Cross Section Model
    target_type = siren.dataclasses.Particle.ParticleType.Nucleon

    DIS_xs = siren.interactions.DISFromSpline(
        os.path.join(xsfiledir, "dsdxdy_nu_CC_iso.fits"),
        os.path.join(xsfiledir, "sigma_nu_CC_iso.fits"),
        [primary_type],
        [target_type], "m"
    )
    
    primary_xs = siren.interactions.InteractionCollection(primary_type, [DIS_xs])
    controller.SetInteractions(primary_xs)

    
    
    # Primary distributions
    primary_injection_distributions = {}
    primary_physical_distributions = {}
    
    siren_input_file = "Data/SIREN/Input/LHCb_%s_%s_%s.txt"%(prefix,generator,parent)
    assert(os.path.isfile(siren_input_file))
    with open(siren_input_file, "rbU") as f:
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
    
    mu_vertex = np.squeeze(data.vertex)
    muon_momenta = np.array(np.squeeze(data.secondary_momenta[:,:,0]))
    muon_momentum = np.array(np.linalg.norm(muon_momenta[:,1:],axis=1))
    mu_dir = muon_momenta[:,1:] / np.expand_dims(muon_momentum,-1)
    
    panel_1 = controller.detector_model.GetSector("panel_1")
    panel_2 = controller.detector_model.GetSector("panel_2")
    panel_3 = controller.detector_model.GetSector("panel_3")

    def GetPanelIntersections(location, direction):
        _loc = siren.math.Vector3D(location)
        _loc_detector = controller.detector_model.GeoPositionToDetPosition(siren.detector.GeometryPosition(_loc)).get()
        _dir = siren.math.Vector3D(direction)
        panel_intersections = {1:[],
                               2:[],
                               3:[]}
        panel_distances = {1:[],
                           2:[],
                           3:[]}
        panel_columndepths = {1:[],
                              2:[],
                              3:[]}
        
        for ip,panel in enumerate([panel_1,panel_2,panel_3]):
            for intersection in panel.geo.Intersections(_loc,_dir):
                panel_intersections[ip+1].append([intersection.position.GetX(),
                                                  intersection.position.GetY(),
                                                  intersection.position.GetZ()])
                panel_distances[ip+1].append(intersection.distance)
                int_loc_dectector = controller.detector_model.GeoPositionToDetPosition(siren.detector.GeometryPosition(intersection.position)).get()
                panel_columndepths[ip+1].append(controller.detector_model.GetColumnDepthInCGS(_loc_detector,int_loc_dectector))
        return panel_intersections,panel_distances,panel_columndepths
    
    panel_ints = {1:[],
                  2:[],
                  3:[]}
    panel_dist = {1:[],
                  2:[],
                  3:[]}
    panel_cdep = {1:[],
                  2:[],
                  3:[]}
    hit_mask = {1:[],
                2:[],
                3:[]}
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
        
    weights = np.array(np.squeeze(data.wgt) * lumi * 1000 * np.prod(data.int_probs,axis=-1))
    weights *= num_input_events / events_to_inject # correct for sampled events
    
    data["weights"] = weights
    data["hit_mask"] = hit_mask_tot
    
    
    ak.to_parquet(data,"%s.parquet"%outfile)
    
def RunLHCbHNLSimulation(prefix,generator,parent,
                         events_to_inject,outfile,
                         m4="1000", Ue4=0, Umu4=1, Utau4=0,
                         lumi=3000):
    
    # Expeirment to run
    experiment = "LakeGeneva"

    # Define the controller
    controller = SIREN_Controller(events_to_inject, experiment)

    # Particle to inject
    primary_type = siren.dataclasses.Particle.ParticleType.NuMu
    hnl_type = siren.dataclasses.Particle.ParticleType.N4
    
    # Primary distributions
    primary_injection_distributions = {}
    primary_physical_distributions = {}
    
    siren_input_file = "Data/SIREN/Input/LHCb_%s_%s_%s.txt"%(prefix,generator,parent)
    assert(os.path.isfile(siren_input_file))
    with open(siren_input_file, "rbU") as f:
        num_input_events = sum(1 for _ in f) - 1

    primary_external_dist = siren.distributions.PrimaryExternalDistribution(siren_input_file)
    primary_injection_distributions["external"] = primary_external_dist


    fid_vol = controller.GetFiducialVolume()
    position_distribution = siren.distributions.PrimaryBoundedVertexDistribution(fid_vol)
    primary_injection_distributions["position"] = position_distribution
    
    secondary_position_distribution = siren.distributions.SecondaryBoundedVertexDistribution(fid_vol)

    # SetProcesses
    controller.SetProcesses(
        primary_type, primary_injection_distributions, primary_physical_distributions,
        [hnl_type], [[]], [[]]
    )
    for process in controller.secondary_injection_processes:
        print(process.primary_type)

    ##################

    # Now include DIS interaction
    cross_section_model = "HNLDISSplines"

    xsfiledir = siren.utilities.get_cross_section_model_path(cross_section_model)

    # Cross Section Model
    target_type = siren.dataclasses.Particle.ParticleType.Nucleon

    DIS_xs = siren.interactions.HNLDISFromSpline(
        os.path.join(xsfiledir, "M_0000MeV/dsdxdy-nu-N-nc-GRV98lo_patched_central.fits"),
        os.path.join(xsfiledir, "M_%sMeV/sigma-nu-N-nc-GRV98lo_patched_central.fits"%m4),
        float(m4)*1e-3,
        [Ue4,Umu4,Utau4],
        siren.utilities.Constants.isoscalarMass,
        1,
        [primary_type],
        [target_type],
    )

    DIS_interaction_collection = siren.interactions.InteractionCollection(primary_type, [DIS_xs])
    
    
    # Decay Model
    two_body_decay = siren.interactions.HNLTwoBodyDecay(float(m4)*1e-3, [Ue4, Umu4, Utau4], siren.interactions.HNLTwoBodyDecay.ChiralNature.Majorana)
    Decay_interaction_collection = siren.interactions.InteractionCollection(hnl_type, [two_body_decay])
    
    controller.SetInteractions(primary_interaction_collection=DIS_interaction_collection,
                               secondary_interaction_collections=[Decay_interaction_collection])
    
    # if we are below the W mass, use DarkNews for dimuon decay
    if float(m4)*1e-3 < siren.utilities.Constants.wMass:
    
        # Define a DarkNews model
        model_kwargs = {
            "m4": float(m4)*1e-3,
            "Ue4": Ue4,
            "Umu4": Umu4,
            "Utau4": Utau4,
            "gD":0,
            "epsilon":0,
            "mzprime":0.1,
            "noHC": True,
            "HNLtype": "majorana",
            "include_nelastic": True,
            "decay_product":"mu+mu-"
        }

        xs_path = siren.utilities.get_cross_section_model_path(f"DarkNewsTables-v{siren.utilities.darknews_version()}", must_exist=False)

        # Define DarkNews Model
        table_dir = os.path.join(
            xs_path,
            "HNL_M%2.2e_e+%2.2e_mu%2.2e_tau%2.2e"%(float(m4),Ue4,Umu4,Utau4),
        )
        controller.InputDarkNewsModel(primary_type, table_dir, upscattering=False, **model_kwargs)

    ##################

    
    # Run generation and save events
    controller.Initialize()
    
    for process in controller.secondary_injection_processes:
        print(process.primary_type)
        for interaction in process.interactions.GetDecays():
            for signature in interaction.GetPossibleSignatures():
                print(signature.secondary_types)
                
    def stop(datum, i):
        secondary_type = datum.record.signature.secondary_types[i]
        return secondary_type != siren.dataclasses.Particle.ParticleType.N4

    controller.SetInjectorStoppingCondition(stop)
    
    controller.GenerateEvents()
    controller.SaveEvents(outfile,
                          save_int_probs=True,
                          save_int_params=True)
    
    # data = ak.from_parquet("%s.parquet"%outfile)
    # ak.to_parquet(data,"%s.parquet"%outfile)
    
#     mu_vertex = np.squeeze(data.vertex)
#     muon_momenta = np.array(np.squeeze(data.secondary_momenta[:,:,0]))
#     muon_momentum = np.array(np.linalg.norm(muon_momenta[:,1:],axis=1))
#     mu_dir = muon_momenta[:,1:] / np.expand_dims(muon_momentum,-1)
    
#     panel_1 = controller.detector_model.GetSector("panel_1")
#     panel_2 = controller.detector_model.GetSector("panel_2")
#     panel_3 = controller.detector_model.GetSector("panel_3")

#     def GetPanelIntersections(location, direction):
#         _loc = siren.math.Vector3D(location)
#         _loc_detector = controller.detector_model.GeoPositionToDetPosition(siren.detector.GeometryPosition(_loc)).get()
#         _dir = siren.math.Vector3D(direction)
#         panel_intersections = {1:[],
#                                2:[],
#                                3:[]}
#         panel_distances = {1:[],
#                            2:[],
#                            3:[]}
#         panel_columndepths = {1:[],
#                               2:[],
#                               3:[]}
        
#         for ip,panel in enumerate([panel_1,panel_2,panel_3]):
#             for intersection in panel.geo.Intersections(_loc,_dir):
#                 panel_intersections[ip+1].append([intersection.position.GetX(),
#                                                   intersection.position.GetY(),
#                                                   intersection.position.GetZ()])
#                 panel_distances[ip+1].append(intersection.distance)
#                 int_loc_dectector = controller.detector_model.GeoPositionToDetPosition(siren.detector.GeometryPosition(intersection.position)).get()
#                 panel_columndepths[ip+1].append(controller.detector_model.GetColumnDepthInCGS(_loc_detector,int_loc_dectector))
#         return panel_intersections,panel_distances,panel_columndepths
    
#     panel_ints = {1:[],
#                   2:[],
#                   3:[]}
#     panel_dist = {1:[],
#                   2:[],
#                   3:[]}
#     panel_cdep = {1:[],
#                   2:[],
#                   3:[]}
#     for mv,md in zip(mu_vertex,mu_dir):
#         p_ints,p_dist,p_cdep = GetPanelIntersections(mv,md)
#         for panel in p_ints.keys():
#             panel_ints[panel].append(p_ints[panel])
#             panel_dist[panel].append(p_dist[panel])
#             panel_cdep[panel].append(p_cdep[panel])
    
#     for ik in panel_ints.keys():
#         data["panel%d_int_locations"%ik] = panel_ints[ik]
#         data["panel%d_int_distances"%ik] = panel_dist[ik]
#         data["panel%d_int_coldepths"%ik] = panel_cdep[ik]
        
#     weights = np.array(np.squeeze(data.wgt) * lumi * 1000 * np.prod(data.int_probs,axis=-1))
#     weights *= num_input_events / events_to_inject # correct for sampled events
    
#     data["weights"] = weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, help='forward-nu-flux prefix')
    parser.add_argument('--generator', type=str, help='forward-nu-flux generator')
    parser.add_argument('--parent', type=str, help='forward-nu-flux parent meson (light or charm)')
    parser.add_argument('-n', '--events-to-inject', type=int,help='number of events to inject')
    parser.add_argument('-o', '--output-file', type=str,help='output filename without extension')
    # RunLHCbMuonSimulation(parser.prefix,parser.generator,parser.parent,
    #                       parser.events_to_inject,args.output_file)
    args = parser.parse_args()
    RunLHCbHNLSimulation(args.prefix,args.generator,args.parent,
                          args.events_to_inject,args.output_file)