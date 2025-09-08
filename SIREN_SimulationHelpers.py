import numpy as np
import awkward as ak
import siren

# Remove empty sublists and dimensions from awkward arrays
def clean_array(array):
    return np.array(np.squeeze(ak.Array([[sublist for sublist in inner_list if len(sublist) > 0] for inner_list in array])))


def compute_HNL_time_delay(siren_data,
                           hnl_mass, # GeV
                           c=2.998e-1,
                           muonID=0):
    """Compute time delay for each event:
    1. Neutrino: primary_initial_position → first vertex
    2. HNL: first vertex → second vertex
    3. Muon: second vertex → panel intersection

    Args:
        siren_data: awkward array with event data
        c: speed of light in mm/ns
    """


    # compute HNL beta
    hnl_momentum = siren_data["primary_momentum"][:,1,:]
    hnl_gamma = hnl_momentum[:,0] / hnl_mass
    hnl_beta = np.sqrt(1 - 1/(hnl_gamma**2))


    for plane in [1,2,3]:

        mask = siren_data["muon%d_panel%d_hit_mask_survival"%(muonID,plane)]==1

        # new columns
        nu_time = np.zeros(len(siren_data))
        hnl_time = np.zeros(len(siren_data))
        muon_time = np.zeros(len(siren_data))
        light_time = np.zeros(len(siren_data))
        timing_difference = np.zeros(len(siren_data))
        hnl_delay = np.zeros(len(siren_data))
        total_time = np.zeros(len(siren_data))

        if sum(mask)!=0:

            int_locations = siren_data[f"muon{muonID}_panel{plane}_int_locations"][mask]
            int_distances = np.squeeze(siren_data[f"muon{muonID}_panel{plane}_int_distances"][mask])
            vertices = np.squeeze(siren_data["vertex"][mask])
            initial_pos = np.squeeze(siren_data["primary_initial_position"][mask])
            betas = np.squeeze(hnl_beta[mask])

            # fill new columns
            nu_time[mask] = np.linalg.norm(vertices[:,0] - initial_pos[:,0],axis=1)/c
            hnl_time[mask] = np.linalg.norm(vertices[:,1] - vertices[:,0],axis=1)/(betas*c)
            muon_time[mask] = int_distances[:,0]/c
            total_time[mask] = nu_time[mask] + hnl_time[mask] + muon_time[mask]
            light_time[mask] = np.linalg.norm(int_locations[:,0] - initial_pos[:,0], axis=1)/c
            timing_difference[mask] = total_time[mask] - light_time[mask]
            hnl_delay[mask] = hnl_time[mask]*(1 - betas)

        siren_data["muon%d_panel%d_nu_time"%(muonID,plane)] = nu_time
        siren_data["muon%d_panel%d_hnl_time"%(muonID,plane)] = hnl_time
        siren_data["muon%d_panel%d_muon_time"%(muonID,plane)] = muon_time
        siren_data["muon%d_panel%d_total_time"%(muonID,plane)] = total_time
        siren_data["muon%d_panel%d_light_time"%(muonID,plane)] = light_time
        siren_data["muon%d_panel%d_timing_difference"%(muonID,plane)] = timing_difference
        siren_data["muon%d_panel%d_hnl_delay"%(muonID,plane)] = hnl_delay
    return siren_data

def ProcessMuonsSINE(data,controller,max_muons=2):
    # This function computes intersections of muons with the SINE panels
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

    panel_ints = {i_muon:{ip:[[] for _ in range(len(data))] for ip in panels.keys()} for i_muon in range(max_muons)}
    panel_dist = {i_muon:{ip:[[-1] for _ in range(len(data))] for ip in panels.keys()} for i_muon in range(max_muons)}
    panel_cdep = {i_muon:{ip:[[-1] for _ in range(len(data))] for ip in panels.keys()} for i_muon in range(max_muons)}
    hit_mask = {i_muon:{ip:[False for _ in range(len(data))] for ip in panels.keys()} for i_muon in range(max_muons)}
    hit_mask_tot = {i_muon:[False for _ in range(len(data))] for i_muon in range(max_muons)}
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

    for i_muon in range(max_muons):
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
    abridged_data = data[data["muon0_hit_mask_survival"]==1] # save only events with at least one muon hitting a panel
    return abridged_data