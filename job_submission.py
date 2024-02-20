import os
import sys

light_generators = ["EPOSLHC","DPMJET","SIBYLL","QGSJET","PYTHIA8"]
charm_generators = ["BKSS","BKRS","SIBYLL","BDGJKR","MS"]
N=20000

for gen_l,gen_c in zip(light_generators,charm_generators):
    cmd = "python MuonSimulation.py -t LHC13 -g %s -m light -n %d"%(gen_l,N)
    print(cmd)
    os.system(cmd)
    cmd = "python MuonSimulation.py -t LHC13 -g %s -m charm -n %d"%(gen_c,N)
    print(cmd)
    os.system(cmd)