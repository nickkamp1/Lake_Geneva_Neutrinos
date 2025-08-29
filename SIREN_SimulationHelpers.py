import numpy as np

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