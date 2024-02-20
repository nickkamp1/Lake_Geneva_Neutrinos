from MuonSimulationHelpers import *
import argparse

def main():
    parser = argparse.ArgumentParser("Muon Simulation")
    parser.add_argument("-t","--tag", help="tag representing simulation size", type=str)
    parser.add_argument("-g", "--generator", help="perimary meson generator model", type=str)
    parser.add_argument("-m", "--meson", help="primary meson type, [light] or [charm]", type=str)
    parser.add_argument("-n", "--N", help="number of events to simulate", type=int, default=None)
    parser.add_argument("-ip", "--interaction-point", help="LHC Interaction Point", type=str, default="LHCb")
    args = parser.parse_args()

    simulation = MuonSimulation(prefix=args.tag,
                                generator=args.generator,
                                parent=args.meson)
    simulation.SampleSecondaryMomenta(args.N)
    simulation.CalculateLakeIntersectionsFromIP(args.interaction_point,args.N)
    simulation.CalculateSurfaceIntersectionFromIP(args.interaction_point,args.N)
    simulation.CalculateDISlocationFromIP(args.interaction_point,args.N)
    simulation.CalculateNeutrinoProfileFromIP(args.interaction_point,args.N)
    simulation.CalculateMuonProfileFromIP(args.interaction_point,args.N)
    simulation.DumpData("Data/MuonSimulation/fresh/%s_%s_%s_%s.parquet"%(args.interaction_point,
                                                                         args.tag,
                                                                         args.generator,
                                                                         args.meson))

if __name__ == "__main__":
    main()
