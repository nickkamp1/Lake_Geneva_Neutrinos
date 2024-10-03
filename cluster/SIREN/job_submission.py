import os
import numpy as np

light_generators = ["EPOSLHC","DPMJET",
                    "SIBYLL","QGSJET","PYTHIA8"]
charm_generators = ["BKSS","BKRS",
                    "SIBYLL","BDGJKR","MS"]
XSModel="CC"
N = int(1e7)
SIREN_dir = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/Data/SIREN"
experiment="GenevaSurface"
primaries = [12,-12,
             14,-14,
             16,-16]

os.makedirs("%s/Output/%s"%(SIREN_dir,experiment),exist_ok=True)

for primary in primaries:

    if experiment=="GenevaSurface" and primary not in [14,-14]: continue

    for gen_l,gen_c in zip(light_generators,charm_generators):

        # light simulation
        if primary not in [16,-16]:
            output = "%s/Output/%s/LHCb_LHC13_%s_light_%s"%(SIREN_dir,experiment,gen_l,primary)
            cmd = "sbatch --export=PRIMARY=%s,GEN=%s,MESON=light,N=%s,OUTPUT=%s,EXPERIMENT=%s submit_simulation.sbatch"%(primary,gen_l,N,output,experiment)
            print(cmd)
            os.system(cmd)

        # charm simulation
        output = "%s/Output/%s/LHCb_LHC13_%s_charm_%s"%(SIREN_dir,experiment,gen_c,primary)
        cmd = "sbatch --export=PRIMARY=%s,GEN=%s,MESON=charm,N=%s,OUTPUT=%s,EXPERIMENT=%s submit_simulation.sbatch"%(primary,gen_c,N,output,experiment)
        print(cmd)
        os.system(cmd)
