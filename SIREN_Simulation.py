import siren
from siren.SIREN_Controller import SIREN_Controller
import os

# Number of events to inject
events_to_inject = int(1e5)

# Expeirment to run
experiment = "LakeGeneva"

# Define the controller
controller = SIREN_Controller(events_to_inject, experiment)

# Particle to inject
primary_type = siren.dataclasses.Particle.ParticleType.NuMu

cross_section_model = "CSMSDISSplines"

xsfiledir = siren.utilities.get_cross_section_model_path(cross_section_model)

# Cross Section Model
target_type = siren.dataclasses.Particle.ParticleType.Nucleon

DIS_xs = siren.interactions.DISFromSpline(
    os.path.join(xsfiledir, "dsdxdy_nu_CC_iso.fits"),
    os.path.join(xsfiledir, "sigma_nu_CC_iso.fits"),
    [primary_type],
    [target_type], "m"
)

primary_xs = siren.interactions.InteractionCollection(primary_type, [DIS_xs])
controller.SetInteractions(primary_xs)

# Primary distributions
primary_injection_distributions = {}
primary_physical_distributions = {}

primary_external_dist = siren.distributions.PrimaryExternalDistribution("Data/SIREN_Input/LHC13_EPOSLHC_light_0.txt")
primary_injection_distributions["external"] = primary_external_dist

position_distribution = siren.distributions.PrimaryPhysicalVertexDistribution()
primary_injection_distributions["position"] = position_distribution

# SetProcesses
controller.SetProcesses(
    primary_type, primary_injection_distributions, primary_physical_distributions
)

controller.Initialize()

events = controller.GenerateEvents()