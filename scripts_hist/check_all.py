import os
import pyabc

n_rep = 20

max_total_nrs = {
    "uninf": 100000,
    "gaussian": 100000,
    "prangle_gk": 100000,
    "CR": 100000,
    "prangle_lv": 50000,
}

base_dirs = os.listdir("data_hist")

for base_dir in base_dirs:
    result_dir = os.path.join("data_hist", base_dir)
    for f in os.listdir(result_dir):
        if not f.endswith(".db"):
            continue
        h = pyabc.History("sqlite:///" + os.path.join(result_dir, f), create=False)
        hit = False
        for key, val in max_total_nrs.items():
            if key in base_dir:
                if h.total_nr_simulations < val:
                    print(result_dir, f, "only has", h.total_nr_simulations)
                else:
                    pass
                    #print(result_dir, f, "is good:", h.total_nr_simulations)
            hit = True
            continue
        if not hit:
            print(result_dir, f, "was not hit")
