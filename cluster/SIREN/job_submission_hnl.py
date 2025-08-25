import os
import numpy as np

light_generator = "SIBYLL"
charm_generator = "SIBYLL"
N = int(1e5)
SIREN_dir = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/Data/SIREN"
experiment="SINE_CMS_West"
primaries = [14,-14]
n_grid_points = 20
Urange = np.logspace(-3,-1,n_grid_points)
mrange = ["0700","0800","0900",
          "1000","1500","2000","3000","4000","5000","6000","7000","8000","9000",
          "10000","15000","20000","30000"]

out_dir = f"{SIREN_dir}/Output/{experiment}/HNLs"
os.makedirs(out_dir, exist_ok=True)

tasks = []
for primary in primaries:

    for U in Urange:
        Ustr = f"{U:.5f}"

        for m in mrange:

            # light simulation
            if primary not in [16,-16]:
                output = "%s/%s_light_%s_NC_m4_%s_Um4_%s"%(out_dir,light_generator,primary,m,Ustr)
                # if os.path.exists(output+".parquet"):
                #     continue
                #     #print(f"File {output}.parquet exists, skipping...")

                # CSV fields: PRIMARY,GEN,MESON,N,OUTPUT,EXPERIMENT,XSMode,m4,Um4
                tasks.append(f"{primary},{light_generator},light,{N},{output},{experiment},NC,{m},{Ustr}")

            # charm simulation
            output = "%s/%s_charm_%s_NC_m4_%s_Um4_%s"%(out_dir,charm_generator,primary,m,Ustr)
            # if os.path.exists(output+".parquet"):
            #     continue
            #     #print(f"File {output}.parquet exists, skipping...")
            tasks.append(f"{primary},{charm_generator},charm,{N},{output},{experiment},NC,{m},{Ustr}")

# write parameter file
params_path = os.path.join(out_dir, "hnl_params.csv")
with open(params_path, "w") as f:
    f.write("\n".join(tasks))
print(f"Wrote {len(tasks)} tasks to {params_path}")

# submit as a single array job
array_size = len(tasks)
max_concurrent = 500  # adjust concurrency (% limit) as desired
sbatch_script = "submit_hnl_simulation.sbatch"  # template script to create/use
sbatch_cmd = f"sbatch --array=0-{array_size-1}%{max_concurrent} --export=PARAMS_FILE={params_path} {sbatch_script}"
print("Run this to submit the array:")
print(sbatch_cmd)
# optionally submit automatically:
# os.system(sbatch_cmd)