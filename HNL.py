import siren
import numpy as np

def get_decay_length(m4, Ue4, Umu4, Utau4, energy):

    # xs_path = siren.utilities.get_cross_section_model_path(f"DarkNewsTables-v{siren.utilities.darknews_version()}", must_exist=False)
    # table_dir = os.path.join(
    #     xs_path,
    #     "HNL_M%2.2e_e+%2.2e_mu%2.2e_tau%2.2e"%(m4,Ue4,Umu4,Utau4),
    # )

    # # Define a DarkNews model
    # model_kwargs = {
    #     "m4": m4,
    #     "Ue4": Ue4,
    #     "Umu4": Umu4,
    #     "Utau4": Utau4,
    #     "gD":0,
    #     "epsilon":0,
    #     "mzprime":1,
    #     "noHC": True,
    #     "HNLtype": "dirac",
    #     "include_nelastic": True,
    #     "nuclear_targets": ["H1"]
    # }
    # if m4 > 2*0.1057 and m4 < Constants.wMass:
    #     model_kwargs["decay_product"] = "mu+mu-"
    #     DN_processes = PyDarkNewsInteractionCollection(table_dir = table_dir, **model_kwargs)
    #     dimuon_decay = DN_processes.decays[0]
    # else:
    #     dimuon_decay = None
    # if m4 > 2*0.000511 and m4 < Constants.wMass:
    #     model_kwargs["decay_product"] = "e+e-"
    #     DN_processes = PyDarkNewsInteractionCollection(table_dir = table_dir, **model_kwargs)
    #     dielectron_decay = DN_processes.decays[0]
    # else:
    #     dielectron_decay = None

    two_body_decay = siren.interactions.HNLTwoBodyDecay(m4, [Ue4, Umu4, Utau4], siren.interactions.HNLTwoBodyDecay.ChiralNature.Dirac)
    record = siren.dataclasses.InteractionRecord()
    record.signature.primary_type = siren.dataclasses.Particle.N4
    record.primary_mass = m4
    record.primary_momentum = [energy, 0, 0, np.sqrt(energy**2 - m4**2)]
    record.signature.target_type = siren.dataclasses.Particle.Decay
    return two_body_decay.TotalDecayLength(record)

    decay_widths = {}

    for decay in [dimuon_decay,dielectron_decay,two_body_decay]:
        if decay is None: continue
        for signature in decay.GetPossibleSignaturesFromParent(siren.dataclasses.Particle.N4):
            record.signature.secondary_types = signature.secondary_types
            decay_widths[tuple(signature.secondary_types)] = decay.TotalDecayWidthForFinalState(record)

    total_decay_width = 0
    for k,v in decay_widths.items():
        if siren.dataclasses.Particle.Qball not in k:
            total_decay_width += v
    decay_widths["total"] = total_decay_width

    return decay_widths

def time_delay(e4, m4, decay_length):
    c = 3e-1 # m / ns
    p4 = np.sqrt(e4**2 - m4**2)
    beta = p4/e4
    return (1/beta - 1)*decay_length/c