from MuonSimulationHelpers import *

simulation = MuonSimulation('Large','EPOSLHC','light')
simulation.SampleSecondaryMomenta()
simulation.DumpData('Data/Large_EPOSLHC_light.parquet')
simulation = MuonSimulation('Large','BKRS','charm')
simulation.SampleSecondaryMomenta()
simulation.DumpData('Data/Large_BKRS_charm.parquet')
