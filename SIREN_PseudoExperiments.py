from scipy.interpolate import interp1d
import numpy as np
import awkward as ak
import argparse

e_knots = [50, 100, 200, 500, 1000, 2000, 5000]
nu_cc_up = interp1d(e_knots,[4.1,3.8,3.5,3.2,3.0,2.7,2.3],fill_value="extrapolate")
nu_cc_down = interp1d(e_knots,[2.4,2.0,1.0,1.9,1.7,1.6,1.5],fill_value="extrapolate")
nu_nc_up = interp1d(e_knots,[3.8,3.5,3.2,2.9,2.7,2.4,2.1],fill_value="extrapolate")
nu_nc_down = interp1d(e_knots,[2.0,1.8,1.7,1.5,1.5,1.4,1.3],fill_value="extrapolate")
nubar_cc_up = interp1d(e_knots,[15.0,13.3,11.9,10.5,9.4,8.3,6.5],fill_value="extrapolate")
nubar_cc_down = interp1d(e_knots,[9.0,7.4,6.5,5.7,5.2,4.6,3.7],fill_value="extrapolate")
nubar_nc_up = interp1d(e_knots,[12.0,10.7,9.6,8.6,7.8,7.0,5.7],fill_value="extrapolate")
nubar_nc_down = interp1d(e_knots,[6.4,5.7,5.1,4.6,4.2,3.8,3.2],fill_value="extrapolate")

Geneva_dir = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/"

primaries = [12,-12,
             14,-14,
             16,-16
            ]
datasets={"SINE_CMS_West":["CC"],
           "UNDINE_CMS_East":["CC","NC"]}

def get_dataset(xs_model,detector,prefix,generator,parent,primary):
    if xs_model=="CC":
        compressed_output_file = Geneva_dir+"Data/SIREN/Output/%s/compressed/%s_%s_%s_%s"%(detector,prefix,generator,parent,primary)
    else:
        compressed_output_file = Geneva_dir+"Data/SIREN/Output/%s/compressed/%s_%s_%s_%s_%s"%(detector,prefix,generator,parent,primary,xs_model)
    return ak.from_parquet(compressed_output_file+".parquet")

def throw_pseudoexperiments(N,dataset,xs_model,prefix,light_generator,charm_generator):

    nu_cc_throws = np.random.normal(size=(N))
    nu_nc_throws = np.random.normal(size=(N))
    nubar_cc_throws = np.random.normal(size=(N))
    nubar_nc_throws = np.random.normal(size=(N))

    for detector,xs_models in datasets.items():
        for xs_model in xs_models:
            for primary in primaries:
                if "SINE" in detector and abs(primary)!= 14: continue
                if
                dataset = get_dataset(xs_model,detector,prefix,generator,parent,primary


    if pid>0:
        if xs_model=="CC":
            weights = dataset.weights * np.where(np.tile((nu_cc_throws>0)[:, np.newaxis], (1, len(dataset.weights))),
                                                 1.+(np.outer(nu_cc_throws,nu_cc_up(dataset.energy)/100.)),
                                                 1.+(np.outer(nu_cc_throws,nu_cc_down(dataset.energy)/100.)))
        elif xs_model=="NC":
            weights = dataset.weights * np.where(np.tile((nu_nc_throws>0)[:, np.newaxis], (1, len(dataset.weights))),
                                                 1.+(np.outer(nu_nc_throws,nu_nc_up(dataset.energy)/100.)),
                                                 1.+(np.outer(nu_nc_throws,nu_nc_down(dataset.energy)/100.)))
    elif pid<0:
        if xs_model=="CC":
            weights = dataset.weights * np.where(np.tile((nubar_cc_throws>0)[:, np.newaxis], (1, len(dataset.weights))),
                                                 1.+(np.outer(nubar_cc_throws,nubar_cc_up(dataset.energy)/100.)),
                                                 1.+(np.outer(nubar_cc_throws,nubar_cc_down(dataset.energy)/100.)))
        elif xs_model=="NC":
            weights = dataset.weights * np.where(np.tile((nubar_nc_throws>0)[:, np.newaxis], (1, len(dataset.weights))),
                                                 1.+(np.outer(nubar_nc_throws,nubar_nc_up(dataset.energy)/100.)),
                                                 1.+(np.outer(nubar_nc_throws,nubar_nc_down(dataset.energy)/100.)))
    return np.sum(weights,axis=-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary', type=int, help='simulation primary PDG code', default=14)
    parser.add_argument('-p','--prefix', default="LHC13", type=str, help='forward-nu-flux prefix')
    parser.add_argument('-g','--generator', type=str, help='forward-nu-flux generator')
    parser.add_argument('-m','--parent', type=str, help='forward-nu-flux parent meson (light or charm)')
    parser.add_argument('-d', '--detector', type=str, default='GenevaSurface', help='experiment name (GenevaLake or GenevaSurface)')
    parser.add_argument('-x', '--xs-model', type=str, default='CC', help='cross section mode (CC or NC)')
    parser.add_argument('-n', '--N', type=int, default=1000, help='number of pseudoexps')

    args = parser.parse_args()
    dataset = get_dataset(args.xs_model,args.detector,args.prefix,args.generator,args.parent,args.primary)
    pseudoexps = throw_pseudoexperiments(args.N,dataset,args.xs_model,args.primary)
    print(pseudoexps)