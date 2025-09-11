import argparse
import numpy as np
import awkward as ak

def main(m4, Um4, filelist, output_prefix,
         deltaT_cut=1, transverseDisp_cut=0.1, weight_mod=1):
    # Default output prefix if not provided
    if output_prefix is None:
        output_prefix = f"hnl_m4_{m4}_Um4_{Um4}"

    dimuon_total = 0.0
    delayed_total = 0.0
    displaced_total = 0.0
    muon_total = 0.0
    sim_dict = ak.Array([])

    for f in filelist:
        try:
            data = ak.from_parquet(f)
            data["max_deltaT"] = np.max(
                np.array([data["muon0_panel1_time_delay"],
                          data["muon0_panel2_time_delay"],
                          data["muon0_panel3_time_delay"]]).T,
                axis=1
            )
            for panel in range(1, 4):
                panel_transverse_disp = -np.ones((len(data), 2))
                mask = data[f"muon0_panel{panel}_hit_mask"] == 1
                panel_transverse_disp[mask] = (
                    np.array(data[f"muon0_panel{panel}_int_locations"][mask])[:, 1, :2]
                    - np.array(data[f"muon0_panel{panel}_int_locations"][mask])[:, 0, :2]
                )
                data[f"muon0_panel{panel}_transverse_disp"] = panel_transverse_disp
            maxDeltaX = np.max(
                np.array([data.muon0_panel1_transverse_disp,
                          data.muon0_panel2_transverse_disp,
                          data.muon0_panel3_transverse_disp])[:, :, 0],
                axis=0
            )
            maxDeltaY = np.max(
                np.array([data.muon0_panel1_transverse_disp,
                          data.muon0_panel2_transverse_disp,
                          data.muon0_panel3_transverse_disp])[:, :, 1],
                axis=0
            )
            data["max_transverseDisp"] = np.sqrt(maxDeltaX**2 + maxDeltaY**2)
        except Exception as e:
            print(f"Broke on {f}: {e}")
            continue

        # weights
        weights = np.array(data["weights"] * data["muon0_hit_mask_survival"])  # only surviving muons
        weights *= weight_mod

        dimuon_mask = np.array(data["hit_mask_dimuon_survival"])
        delay_mask = np.array(data["max_deltaT"] > deltaT_cut)
        displaced_mask = np.array(data["max_transverseDisp"] > transverseDisp_cut)

        dimuon_total += np.sum(weights * dimuon_mask)
        delayed_total += np.sum(weights * delay_mask)
        displaced_total += np.sum(weights * displaced_mask)
        muon_total += np.sum(weights)

        sim_dict = ak.concatenate([sim_dict, data])

    rate_outfile_path = output_prefix + "_rate.txt"
    with open(rate_outfile_path, "w") as rate_outfile:
        print("dimuon delayed displaced muon", file=rate_outfile)
        print(dimuon_total, delayed_total, displaced_total, muon_total, file=rate_outfile)
    print(f"Wrote {rate_outfile_path}")

    sim_outfile = output_prefix + "_sim.parquet"
    ak.to_parquet(sim_dict, sim_outfile)
    print(f"Wrote {sim_outfile}")

parser = argparse.ArgumentParser(description="Process HNL parquet files to compute event rates.")
parser.add_argument("--filelist", type=str, nargs="+", required=True, help="List of input parquet files")
parser.add_argument("-m", "--m4", type=str, required=True, help="HNL mass string in MeV.")
parser.add_argument("-U", "--Um4", type=str, required=True, help="Mixing angle string.")
parser.add_argument("--deltaT-cut", type=float, default=1, help="Delta T cut in ns")
parser.add_argument("--transverseDisp-cut", type=float, default=0.1, help="Transverse displacement cut in meters")
parser.add_argument("--weight-mod", type=float, default=1, help="Weight modification factor")
parser.add_argument("--output-prefix", type=str, default=None, help="Output prefix for written files")
if __name__ == "__main__":
    args = parser.parse_args()
    main(args.m4, args.Um4, args.filelist, args.output_prefix, args.deltaT_cut, args.transverseDisp_cut, args.weight_mod)

