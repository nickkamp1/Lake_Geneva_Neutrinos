import os
import numpy as np

light_generators = ["EPOSLHC","DPMJET",
                    "SIBYLL","QGSJET","PYTHIA8"]
charm_generators = ["BKSS","BKRS",
                    "SIBYLL","BDGJKR","MS"]
chunk_size=20000
DetModel="lake"
XSModel="NC"

for gen_l,gen_c in zip(light_generators,charm_generators):

    # light simulation
    Nstart = 0
    light_file_length = len(np.loadtxt("../../forward-nu-flux-fit/files/LHC13_%s_light_0.txt"%gen_l))
    while Nstart < light_file_length:
        N = chunk_size if (Nstart+chunk_size < light_file_length) else (light_file_length-Nstart)
        cmd = "sbatch --export=GEN=%s,MESON=light,Nstart=%s,N=%s,DetModel=%s,XSModel=%s submit_simulation.sbatch"%(gen_l,Nstart,N,DetModel,XSModel)
        print(cmd)
        os.system(cmd)
        Nstart+=chunk_size

    # charm simulation
    Nstart = 0
    charm_file_length = len(np.loadtxt("../../forward-nu-flux-fit/files/LHC13_%s_charm_0.txt"%gen_c))
    while Nstart < charm_file_length:
        N = chunk_size if (Nstart+chunk_size < charm_file_length) else (charm_file_length-Nstart)
        cmd = "sbatch --export=GEN=%s,MESON=charm,Nstart=%s,N=%s,DetModel=%s,XSModel=%s submit_simulation.sbatch"%(gen_c,Nstart,N,DetModel,XSModel)
        print(cmd)
        os.system(cmd)
        Nstart+=chunk_size