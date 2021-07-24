"""Constant definitions."""

distance_names = [
    "Calibrated__Euclidean__mad",
    "Adaptive__Euclidean__mad",
    "Adaptive__Euclidean__cmad",
    "Adaptive__Euclidean__mad_or_cmad",
    "Calibrated__Manhattan__mad",
    "Adaptive__Manhattan__mad",
    "Adaptive__Manhattan__cmad",
    "Adaptive__Manhattan__mad_or_cmad",
]


distance_labels = {
    "Calibrated__Euclidean__mad": "L2 + Cal. + MAD",
    "Adaptive__Euclidean__mad": "L2 + Adap. + MAD",
    "Adaptive__Euclidean__cmad": "L2 + Adap. + CMAD",
    "Adaptive__Euclidean__mad_or_cmad": "L2 + Adap. + (C)MAD",
    "Calibrated__Manhattan__mad": "L1 + Cal. + MAD",
    "Adaptive__Manhattan__mad": "L1 + Adap. + MAD",
    "Adaptive__Manhattan__cmad": "L1 + Adap. + CMAD",
    "Adaptive__Manhattan__mad_or_cmad": "L1 + Adap. + (C)MAD",
}

distance_labels_short = {
    "Calibrated__Euclidean__mad": "L2+Cal.+MAD",
    "Adaptive__Euclidean__mad": "L2+Ada.+MAD",
    "Adaptive__Euclidean__cmad": "L2+Ada.+CMAD",
    "Adaptive__Euclidean__mad_or_cmad": "L2+Ada.+PCMAD",
    "Calibrated__Manhattan__mad": "L1+Cal.+MAD",
    "Adaptive__Manhattan__mad": "L1+Ada.+MAD",
    "Adaptive__Manhattan__cmad": "L1+Ada.+CMAD",
    "Adaptive__Manhattan__mad_or_cmad": "L1+Ada.+PCMAD",
}

distance_colors = {
    "Calibrated__Euclidean__mad": "C3",
    "Adaptive__Euclidean__mad": "C2",
    "Adaptive__Euclidean__cmad": "grey",
    "Adaptive__Euclidean__mad_or_cmad": "grey",
    "Calibrated__Manhattan__mad": "grey",
    "Adaptive__Manhattan__mad": "C1",
    "Adaptive__Manhattan__cmad": "grey",
    "Adaptive__Manhattan__mad_or_cmad": "C0",
}

problem_labels_old = {
    "uninf": "Uninformative",
    "gaussian": "Replicates",
    "gk": "GK",
    "lv": "Lotka-Volterra",
    "cr-zero": "Conversion",
}

problem_labels = {
    "uninf": "M1",
    "gaussian": "M2",
    "cr-zero": "M3",
    "gk": "M4",
    "lv": "M5",
    "tumor": "M6",
}

parameter_labels = {
    "uninf": {"p0": r"$\theta$"},
    "gaussian": {"p0": r"$\theta$"},
    "gk": {"A": "A", "B": "B", "g": "g", "k": "k"},
    "lv": {
        "p1": r"$\log~\theta_1$",
        "p2": r"$\log~\theta_2$",
        "p3": r"$\log~\theta_3$",
    },
    "cr-zero": {"p0": r"$\log~\theta_1$", "p1": r"$\log~\theta_2$"},
    "tumor": {
        "log_division_rate": "log(div. rate)",
        "log_division_depth": "log(div. depth)",
        "log_initial_spheroid_radius": "log(init. sph. rad.)",
        "log_initial_quiescent_cell_fraction": "log(init. quies. cell frac.)",
        "log_ecm_production_rate": "log(ECM prod. rate)",
        "log_ecm_degradation_rate": "log(ECM degrad. rate)",
        "log_ecm_division_threshold": "log(ECM div. thresh.)",
    },
}

data_xlabels = {
    "tumor": {
        "growth_curve": "Time [$d$]",
        "proliferation_profile": "Distance to rim [x$10^{-5} m$]",
        "extra_cellular_matrix_profile": "Distance to rim [x$10^{-5} m$]",
    },
    "uninf": {"y": "Coordinate"},
    "gaussian": {"y": "Replicate"},
    "gk": {"y": "Order statistic"},
    "lv": {"Prey": "Time [au]", "Predator": "Time [au]"},
    "cr-zero": {"y": "Time [au]"},
    "cr-swap": {"y": "Time [au]"},
}

data_ylabels = {
    "uninf": {"y": "Value"},
    "gaussian": {"y": "Value"},
    "gk": {"y": "Value"},
    "lv": {"Prey": "Number", "Predator": "Number"},
    "cr-zero": {"y": "Species B"},
    "cr-swap": {"y": "Species B"},
    "tumor": {
        "growth_curve": "Spheroid radius [$\\mu m$]",
        "proliferation_profile": "Frac. proliferating cells",
        "extra_cellular_matrix_profile": "ECM intensity",
    },
}
