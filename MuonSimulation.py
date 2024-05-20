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
    args = parser.parse_args()

    simulation = MuonSimulation(prefix=args.tag,
                                generator=args.generator,
                                parent=args.meson,
                                Nstart=args.Nstart,
                                N=args.N)
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
    simulation.DumpData(output_dir+"%s_%s_%s_%s_%d_%d.parquet"%(args.interaction_point,
                                                                args.tag,
                                                                args.generator,
                                                                args.meson,
                                                                args.Nstart,
                                                                args.Nstart+args.N))

if __name__ == "__main__":
    main()
