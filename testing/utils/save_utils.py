import os
import pandas as pd
import yaml


def save_results(cfg, results_dict, conf_d=None):

    os.makedirs(cfg.results_save_dir, exist_ok=True)
    print("saving results at: ", cfg.results_save_dir)

    df = pd.DataFrame(results_dict.values())
    df.to_hdf(os.path.join(cfg.results_save_dir, "results.h5"), key="df")
    yaml.dump(dict(vars(cfg)), open(os.path.join(cfg.results_save_dir, "cfg.yaml"), "w"))
    # conf_df.to_csv(os.path.join(save_dir, "conf_matrix.csv"))
