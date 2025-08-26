import numpy as np

def compute_HNL_time_delay(siren_data,
                           hnl_mass, # GeV
                           c=2.998e-1):
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

        mask = siren_data["panel%d_hit_mask_muon0_survival"%(plane)]==1

        # new columns
        nu_time = np.zeros(len(siren_data))
        hnl_time = np.zeros(len(siren_data))
        muon_time = np.zeros(len(siren_data))
        light_time = np.zeros(len(siren_data))
        timing_difference = np.zeros(len(siren_data))
        hnl_delay = np.zeros(len(siren_data))
        total_time = np.zeros(len(siren_data))

        if sum(mask)!=0:

            int_locations = siren_data[f"muon0_panel{plane}_int_locations"][mask]
            int_distances = np.squeeze(siren_data[f"muon0_panel{plane}_int_distances"][mask])
            vertices = np.squeeze(siren_data["vertex"][mask])
            initial_pos = np.squeeze(siren_data["primary_initial_position"][mask])
            betas = np.squeeze(hnl_beta[mask])

            # fill new columns
            nu_time[mask] = np.linalg.norm(vertices[:,0] - initial_pos[:,0],axis=1)/c
            hnl_time[mask] = np.linalg.norm(vertices[:,1] - vertices[:,0],axis=1)/(betas*c)
            muon_time[mask] = int_distances[0]/c
            total_time[mask] = nu_time[mask] + hnl_time[mask] + muon_time[mask]
            light_time[mask] = np.linalg.norm(int_locations[:,0] - initial_pos[:,0], axis=1)/c
            timing_difference[mask] = total_time[mask] - light_time[mask]
            hnl_delay[mask] = hnl_time[mask]*(1 - betas)

        siren_data["panel%d_nu_time"%(plane)] = nu_time
        siren_data["panel%d_hnl_time"%(plane)] = hnl_time
        siren_data["panel%d_muon_time"%(plane)] = muon_time
        siren_data["panel%d_total_time"%(plane)] = total_time
        siren_data["panel%d_light_time"%(plane)] = light_time
        siren_data["panel%d_timing_difference"%(plane)] = timing_difference
        siren_data["panel%d_hnl_delay"%(plane)] = hnl_delay
    return siren_data

    #     for w, init_pos, vx, dist, loc, beta in zip(wgts, initial_pos, vertices, int_distances, int_locations, betas):

    #         # # distance checks
    #         first_plane_y = loc[0][1] - panel_bottom
    #         second_plane_y = loc[1][1] - panel_bottom
    #         if first_plane_y > panel_height or first_plane_y < 0 : continue
    #         if second_plane_y > panel_height or second_plane_y < 0: continue
    #         if abs(loc[0][0]) > panel_width/2: continue
    #         if abs(loc[1][0]) > panel_width/2: continue
    #         if (loc[1][2] - loc[0][2]) <= 2.44 - 1e-6: continue # make sure we enter in front, exit in back

    #         # Time components:
    #         # print(vx[0])
    #         # print(vx[1])
    #         # print(dist[0])
    #         # print(loc[0])
    #         t_nu = np.linalg.norm(vx[0] - init_pos[0]) / c  # neutrino time
    #         t_hnl = np.linalg.norm(vx[1] - vx[0]) / (beta*c)    # HNL time
    #         t_muon = dist[0] / c                         # muon time in panel
    #         total_time = t_nu + t_hnl + t_muon
    #         # Light travel time from origin to intersection point
    #         light_time = np.linalg.norm(loc[0] - init_pos[0]) / c  # Time for light from initial neutrino position to hit point)

    #         nu_dict["beam_timing"].append(total_time)
    #         nu_dict["timing_difference"].append(total_time - light_time)
    #         nu_dict["first_plane_x"].append(loc[0][0])
    #         nu_dict["first_plane_y"].append(first_plane_y)
    #         nu_dict["first_plane_z"].append(loc[0][2])
    #         nu_dict["second_plane_x"].append(loc[1][0])
    #         nu_dict["second_plane_y"].append(second_plane_y)
    #         nu_dict["second_plane_z"].append(loc[1][2])
    #         nu_dict["weights"].append(w)
    #         nu_dict["hnl_delay"].append(t_hnl - (np.linalg.norm(vx[1] - vx[0]) / c))
    # return nu_dict