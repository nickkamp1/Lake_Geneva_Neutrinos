import os
import numpy as np

N = int(1e6)
SIREN_dir = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/Data/SIREN"
experiment="SINE_CMS_West"
n_grid_points = 40
Urange = np.logspace(-5,-1,n_grid_points)
mrange = [
    # "0100","0200","0300","0400","0500","0600",
    # "1100","1200","1300","1400","1600","1700","1800","1900",
    # "2100","2200","2300","2400","2500"
    "0700","0800","0900",
    "1000","1500","2000","3000",
    #"4000","5000","6000"
          ]
seeds = np.arange(10) # 10 different seeds for each point

out_dir = f"{SIREN_dir}/Output/{experiment}/HNLs"
os.makedirs(out_dir, exist_ok=True)

tasks = []

for U in Urange:
    Ustr = f"{U:.6f}"

    for m in mrange:
        for seed in seeds:
            output = "%s/MesonDecay_m4_%s_Um4_%s_seed_%d"%(out_dir,m,Ustr,seed)
            tasks.append(f"{N},{output},{experiment},{m},{Ustr},{seed}")

# write parameter file
params_path = os.path.join(out_dir, "decay_hnl_params.csv")
with open(params_path, "w") as f:
    f.write("\n".join(tasks))
print(f"Wrote {len(tasks)} tasks to {params_path}")

# submit as a single array job
array_size = len(tasks)
max_concurrent = 1000  # adjust concurrency (% limit) as desired
sbatch_script = "submit_decay_hnl_simulation.sbatch"  # template script to create/use
sbatch_cmd = f"sbatch --array=0-{array_size-1}%{max_concurrent} --export=PARAMS_FILE={params_path} {sbatch_script}"
print("Run this to submit the array:")
print(sbatch_cmd)
# optionally submit automatically:
# os.system(sbatch_cmd)
