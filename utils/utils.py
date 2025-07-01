import os
import yaml
import pandas as pd
import torch


def load_model(prev_runid, model, device):

    model_dir = prev_runid


    if os.path.isfile(model_dir):
        model_loaded = torch.load(model_dir, map_location=device, weights_only=False)
        model.load_state_dict(model_loaded.state_dict())
        print("Model restored from " + prev_runid + "\n")
    else:
        print("No model found at" + prev_runid + "\n")

    return model

def create_model_dir(path_results, runid):
    path_results = os.path.join(path_results, runid)
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    print("Results stored at " + path_results + "\n")
    return path_results


def save_model(model, path):

    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def log_config(path_results, runid, config):
    config_path = os.path.join(path_results, f"{runid}_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    print(f"Config saved to {config_path}")
    return runid

def log_results(path_results, results, eval_id):
    results_path = os.path.join(path_results, f"results_{eval_id}.yaml")
    with open(results_path, "w") as f:
        yaml.dump(results, f)
    print(f"Results saved to {results_path}")


def save_csv(data, fname):
    pd.DataFrame(data).to_csv(fname, mode="a", header=not os.path.isfile(fname))


def save_diff(fname="git_diff.txt"):
    os.system(f"git diff > {fname}")
