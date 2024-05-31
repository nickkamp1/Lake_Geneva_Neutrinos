from MuonSimulationHelpers import *
import argparse

output_dir = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/Data/MuonSimulation/fresh/"

def main():
    parser = argparse.ArgumentParser("Muon Simulation")
    parser.add_argument("-t","--tag", help="tag representing simulation size", type=str)
    parser.add_argument("-g", "--generator", help="perimary meson generator model", type=str)
    parser.add_argument("-m", "--meson", help="primary meson type, [light] or [charm]", type=str)
    parser.add_argument("-n", "--N", help="number of events to simulate", type=int, default=None)
    parser.add_argument("-ns", "--Nstart", help="starting simulation event", type=int, default=0)
    parser.add_argument("-ip", "--interaction-point", help="LHC Interaction Point", type=str, default="LHCb")
    parser.add_argument("-dm", "--detector-mode", help="Detector mode (lake or surface)", type=str, default="lake")
    parser.add_argument("-xm", "--xs-mode", help="Cross section mode (CC or NC)", type=str, default="CC")
    parser.add_argument("-lc", "--lake-center", help="Center of lake detector along beam", type=int, default=10000)
    parser.add_argument("-le", "--lake-extent", help="Extent of lake detector", type=int, default=100)
    args = parser.parse_args()

    simulation = MuonSimulation(prefix=args.tag,
                                generator=args.generator,
                                parent=args.meson,
                                Nstart=args.Nstart,
                                N=args.N,
                                det_mode=args.detector_mode,
                                xs_mode=args.xs_mode,
                                lake_center=args.lake_center,
                                lake_extent=args.lake_extent
                                )
    print("SampleSecondaryMomenta")
    simulation.SampleSecondaryMomenta()
    print("CalculateLakeIntersectionsFromIP")
    simulation.CalculateLakeIntersectionsFromIP(args.interaction_point)
    print("CalculateSurfaceIntersectionFromIP")
    simulation.CalculateSurfaceIntersectionFromIP(args.interaction_point)
    print("CalculateDISlocationFromIP")
    simulation.CalculateDISlocationFromIP(args.interaction_point)
    print("CalculateNeutrinoProfileFromIP")
    simulation.CalculateNeutrinoProfileFromIP(args.interaction_point)
    print("CalculateMuonProfileFromIP")
    simulation.CalculateMuonProfileFromIP(args.interaction_point)
    print("CalculateLeptonSurfaceIntersectionFromIP")
    simulation.CalculateLeptonSurfaceIntersectionFromIP(args.interaction_point)
    print("DumpData")
    if args.detector_mode=="lake":
        outfile = output_dir+"%s_%s_%s_%s_%s_%s_%d_%d_%d_%d.parquet"%(args.interaction_point,
                                                                args.tag,
                                                                args.generator,
                                                                args.meson,
                                                                args.detector_mode,
                                                                args.xs_mode,
                                                                args.lake_center,
                                                                args.lake_extent,
                                                                args.Nstart,
                                                                args.Nstart+args.N)
    else:
        outfile = output_dir+"%s_%s_%s_%s_%s_%s_%d_%d.parquet"%(args.interaction_point,
                                                                args.tag,
                                                                args.generator,
                                                                args.meson,
                                                                args.detector_mode,
                                                                args.xs_mode,
                                                                args.Nstart,
                                                                args.Nstart+args.N)
    simulation.DumpData(outfile)

if __name__ == "__main__":
    main()
