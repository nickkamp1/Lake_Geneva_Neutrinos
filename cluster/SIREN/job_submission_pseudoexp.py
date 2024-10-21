import os

light_generators = [
    "EPOSLHC",                
    "DPMJET",                
    #"SIBYLL",                
    "QGSJET",                
    #"PYTHIA8"
]
charm_generators = [
    "BKSS",                
    "BKRS",                
    #"SIBYLL",                
    "BDGJKR",                
    #"MS"
]
N = int(1e3)
for gen_l,gen_c in zip(light_generators,charm_generators):

    cmd = "sbatch --export=LGEN=%s,CGEN=%s,N=%s submit_pseudoexp_simulation.sbatch"%(gen_l,gen_c,N)
    print(cmd)
    os.system(cmd)
