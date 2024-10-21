from scipy.interpolate import interp1d
import numpy as np
import awkward as ak
import argparse
import pandas as pd

### CSMS Uncertainties #####
# e_knots = [50, 100, 200, 500, 1000, 2000, 5000]
# nu_cc_up = interp1d(e_knots,[4.1,3.8,3.5,3.2,3.0,2.7,2.3],fill_value="extrapolate")
# nu_cc_down = interp1d(e_knots,[2.4,2.0,1.0,1.9,1.7,1.6,1.5],fill_value="extrapolate")
# nu_nc_up = interp1d(e_knots,[3.8,3.5,3.2,2.9,2.7,2.4,2.1],fill_value="extrapolate")
# nu_nc_down = interp1d(e_knots,[2.0,1.8,1.7,1.5,1.5,1.4,1.3],fill_value="extrapolate")
# nubar_cc_up = interp1d(e_knots,[15.0,13.3,11.9,10.5,9.4,8.3,6.5],fill_value="extrapolate")
# nubar_cc_down = interp1d(e_knots,[9.0,7.4,6.5,5.7,5.2,4.6,3.7],fill_value="extrapolate")
# nubar_nc_up = interp1d(e_knots,[12.0,10.7,9.6,8.6,7.8,7.0,5.7],fill_value="extrapolate")
# nubar_nc_down = interp1d(e_knots,[6.4,5.7,5.1,4.6,4.2,3.8,3.2],fill_value="extrapolate")


philip_xsdir = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/Data/SIREN/CrossSections"
unc_data = pd.read_csv("%s/xs_wcg24b_isoscalar.txt"%philip_xsdir)

xs_unc = {}
for pstr in ["NUMU","NUTAU"]:
    for xs_mode in ["CC","NC"]:
        for particle in ["%s"%pstr,"%sBAR"%pstr]:
            xs_unc[(particle,xs_mode)] = interp1d(unc_data["E[GEV]"],unc_data["UNC_%s_%s[%%]"%(particle,xs_mode)],fill_value="extrapolate")


kaons = [
    130, # K0L
    310, # K0S
    321, # K+
    -321, # K-
    ]

pions = [
    211, # pi+
    -211 # pi-
]

Geneva_dir = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/"

primaries = [12,-12,
             14,-14,
             16,-16
            ]
datasets={"SINE_CMS_West":["CC"],
           "UNDINE_LHCb_North":["CC","NC"]}

def get_dataset(xs_model,detector,prefix,generator,parent,primary):
    compressed_output_file = Geneva_dir+"Data/SIREN/Output/%s/compressed/%s_%s_%s_%s_%s"%(detector,prefix,generator,parent,primary,xs_model)
    return ak.from_parquet(compressed_output_file+".parquet")

def throw_pseudoexperiments(N,prefix,light_generator,charm_generator):

    throws = {}
    for key in xs_unc.keys():
        throws[key] = np.random.normal(size=(N))

    SINE_Total = {"All":np.zeros(N),
                  "Pions":np.zeros(N),
                  "Kaons":np.zeros(N)}
    UNDINE_Total = {"All":np.zeros(N),
                    "Pions":np.zeros(N),
                    "Kaons":np.zeros(N)}
    UNDINE_muons = {"All":np.zeros(N),
                    "Pions":np.zeros(N),
                    "Kaons":np.zeros(N)}
    UNDINE_electrons = {"All":np.zeros(N),
                        "Pions":np.zeros(N),
                        "Kaons":np.zeros(N)}

    for detector,xs_models in datasets.items():
        for xs_model in xs_models:
            for primary in primaries:
                print(detector,xs_model,primary)
                if "SINE" in detector and not (abs(primary)== 14 and xs_model=="CC"): continue
                if abs(primary)!=16: light_dataset = get_dataset(xs_model,detector,prefix,light_generator,"light",primary)
                else: light_dataset=None
                charm_dataset = get_dataset(xs_model,detector,prefix,charm_generator,"charm",primary)

                for dataset in [light_dataset,charm_dataset]:
                    if dataset is None: continue
                    particle_key = "NUMU" if abs(primary) in [12,14] else "NUTAU"
                    particle_key += "BAR" if primary<0 else ""
                    xs_unc_key = (particle,xs_model)
                    weights = dataset.weights * (1 + np.outer(throws[xs_unc_key],xs_unc[xs_unc_key](dataset.energy)/100.))
                    # if primary>0:
                    #     if xs_model=="CC":
                    #         weights = dataset.weights * np.where(np.tile((nu_cc_throws>0)[:, np.newaxis], (1, len(dataset.weights))),
                    #                                              1.+(np.outer(nu_cc_throws,nu_cc_up(dataset.energy)/100.)),
                    #                                              1.+(np.outer(nu_cc_throws,nu_cc_down(dataset.energy)/100.)))
                    #     elif xs_model=="NC":
                    #         weights = dataset.weights * np.where(np.tile((nu_nc_throws>0)[:, np.newaxis], (1, len(dataset.weights))),
                    #                                              1.+(np.outer(nu_nc_throws,nu_nc_up(dataset.energy)/100.)),
                    #                                              1.+(np.outer(nu_nc_throws,nu_nc_down(dataset.energy)/100.)))
                    # elif primary<0:
                    #     if xs_model=="CC":
                    #         weights = dataset.weights * np.where(np.tile((nubar_cc_throws>0)[:, np.newaxis], (1, len(dataset.weights))),
                    #                                              1.+(np.outer(nubar_cc_throws,nubar_cc_up(dataset.energy)/100.)),
                    #                                              1.+(np.outer(nubar_cc_throws,nubar_cc_down(dataset.energy)/100.)))
                    #     elif xs_model=="NC":
                    #         weights = dataset.weights * np.where(np.tile((nubar_nc_throws>0)[:, np.newaxis], (1, len(dataset.weights))),
                    #                                              1.+(np.outer(nubar_nc_throws,nubar_nc_up(dataset.energy)/100.)),
                    #                                              1.+(np.outer(nubar_nc_throws,nubar_nc_down(dataset.energy)/100.)))
                    weights = np.array(weights)
                    if "SINE" in detector: weights *= np.array(dataset.hit_mask_muon_survival)
                    if "UNDINE" in detector: weights *= np.array(dataset.in_fiducial)[:,-1]
                    pion_indices = np.isin(np.array(dataset.hPDG,dtype=int),pions)
                    kaon_indices = np.isin(np.array(dataset.hPDG,dtype=int),kaons)
                    pion_weights = weights[:,pion_indices]
                    kaon_weights = weights[:,kaon_indices]
                    realizations = np.array(np.sum(weights,axis=-1))
                    pion_realizations = np.array(np.sum(pion_weights,axis=-1))
                    kaon_realizations = np.array(np.sum(kaon_weights,axis=-1))
                    if "SINE" in detector:
                        SINE_Total["All"] += realizations
                        SINE_Total["Pions"] += pion_realizations
                        SINE_Total["Kaons"] += kaon_realizations
                    elif "UNDINE" in detector:
                        UNDINE_Total["All"] += realizations
                        UNDINE_Total["Pions"] += pion_realizations
                        UNDINE_Total["Kaons"] += kaon_realizations
                        if abs(primary)==14:
                            UNDINE_muons["All"] += realizations
                            UNDINE_muons["Pions"] += pion_realizations
                            UNDINE_muons["Kaons"] += kaon_realizations
                        if abs(primary)==12:
                            UNDINE_electrons["All"] += realizations
                            UNDINE_electrons["Pions"] += pion_realizations
                            UNDINE_electrons["Kaons"] += kaon_realizations

    return SINE_Total,UNDINE_Total,UNDINE_muons,UNDINE_electrons


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--prefix', default="LHC13", type=str, help='forward-nu-flux prefix')
    parser.add_argument('-lg','--light-generator', type=str, help='light forward-nu-flux generator')
    parser.add_argument('-cg','--charm-generator', type=str, help='charm forward-nu-flux generator')
    parser.add_argument('-n', '--N', type=int, default=1000, help='number of pseudoexps')
    parser.add_argument('-o', '--outdir', type=str, default="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/Data/SIREN/Output/Pseudoexperiments", help='output_directory')

    args = parser.parse_args()
    SINE_Total,UNDINE_Total,UNDINE_muons,UNDINE_electrons = throw_pseudoexperiments(args.N,args.prefix,args.light_generator,args.charm_generator)
    for k in SINE_Total.keys():
        np.save("%s/SINE_Total_%s_%s_%s.npy"%(args.outdir,k,args.light_generator,args.charm_generator),SINE_Total[k])
        np.save("%s/UNDINE_Total_%s_%s_%s.npy"%(args.outdir,k,args.light_generator,args.charm_generator),UNDINE_Total[k])
        np.save("%s/UNDINE_muons_%s_%s_%s.npy"%(args.outdir,k,args.light_generator,args.charm_generator),UNDINE_muons[k])
        np.save("%s/UNDINE_electrons_%s_%s_%s.npy"%(args.outdir,k,args.light_generator,args.charm_generator),UNDINE_electrons[k])