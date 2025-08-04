import os
import numpy as np

light_generator = "SIBYLL"
charm_generator = "SIBYLL"
N = int(2.5e4)
SIREN_dir = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/Data/SIREN"
experiment="SINE_CMS_West"
primaries = [14,-14]
n_grid_points = 10
Urange = np.logspace(-3,-1,n_grid_points)
mrange = ["0500","1000","1500","2000","3000","4000","5000","6000","7000","8000","9000","10000"]

os.makedirs("%s/Output/%s/HNLs/"%(SIREN_dir,experiment),exist_ok=True)

for primary in primaries:

    for U in Urange:
        U = f"{U:.5f}"

        for m in mrange:

            # light simulation
            if primary not in [16,-16]:
                output = "%s/Output/%s/HNLs/%s_light_%s_NC_m4_%s_Um4_%s"%(SIREN_dir,experiment,light_generator,primary,m,U)
                cmd = "sbatch --export=CASE=hnl,PRIMARY=%s,GEN=%s,MESON=light,N=%s,OUTPUT=%s,EXPERIMENT=%s,XSMode=NC,m4=%s,Um4=%s, submit_simulation.sbatch"%(primary,light_generator,N,output,experiment,m,U)
                print(cmd)
                os.system(cmd)

            # charm simulation
            output = "%s/Output/%s/HNLs/%s_charm_%s_NC_m4_%s_Um4_%s"%(SIREN_dir,experiment,charm_generator,primary,m,U)
            cmd = "sbatch --export=CASE=hnl,PRIMARY=%s,GEN=%s,MESON=charm,N=%s,OUTPUT=%s,EXPERIMENT=%s,XSMode=NC,m4=%s,Um4=%s, submit_simulation.sbatch"%(primary,light_generator,N,output,experiment,m,U)
            print(cmd)
            os.system(cmd)
