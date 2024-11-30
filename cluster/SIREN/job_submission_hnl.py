import os
import numpy as np

light_generator = "DPMJET"
charm_generator = "BKRS"
N = int(1e6)
SIREN_dir = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/Data/SIREN"
experiment="SINE_LHCb_North"
primaries = [14,-14,
             16,-16]
n_grid_points = 10
Urange = np.logspace(-3,-1,n_grid_points)
mragne = np.logspace(0,2,n_grid_points)

os.makedirs("%s/Output/%s"%(SIREN_dir,experiment),exist_ok=True)

for primary in primaries:
    
    for U in Urange:
        
        for m in mrange:
            
            # light simulation
            if primary not in [16,-16]:
                output = "%s/Output/%s/CSMS_LHC13_%s_light_%s_%s"%(SIREN_dir,experiment,gen_l,primary,XSMode)
                cmd = "sbatch --export=PRIMARY=%s,GEN=%s,MESON=light,N=%s,OUTPUT=%s,EXPERIMENT=%s,XSMode=%s submit_simulation.sbatch"%(primary,gen_l,N,output,experiment,XSMode)
                print(cmd)
                os.system(cmd)

            # charm simulation
            output = "%s/Output/%s/CSMS_LHC13_%s_charm_%s_%s"%(SIREN_dir,experiment,gen_c,primary,XSMode)
            cmd = "sbatch --export=PRIMARY=%s,GEN=%s,MESON=charm,N=%s,OUTPUT=%s,EXPERIMENT=%s,XSMode=%s submit_simulation.sbatch"%(primary,gen_c,N,output,experiment,XSMode)
            print(cmd)
            os.system(cmd)
