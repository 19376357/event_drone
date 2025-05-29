import os

import mlflow
import pandas as pd
import torch


def load_model(prev_runid, model, device):
    try:
        run = mlflow.get_run(prev_runid)
    except:
        return model

    model_dir = run.info.artifact_uri + "/model/data/model.pth"
    if model_dir[:7] == "file://":
        model_dir = model_dir[7:]

    if os.path.isfile(model_dir):
        model_loaded = torch.load(model_dir, map_location=device, weights_only=False)
        model.load_state_dict(model_loaded.state_dict())
        print("Model restored from " + prev_runid + "\n")
    else:
        print("No model found at" + prev_runid + "\n")

    return model
'''
def load_model(model_name, model, device, weights_dir="weights"):
    """
    model_name: 模型名称字符串，如 'SNNNet'
    model: 已实例化的模型对象
    device: torch.device
    weights_dir: 权重文件夹
    """
    weight_path = os.path.join(
        weights_dir, model_name, "artifacts", "model", "data", "model.pth"
    )
    if os.path.isfile(weight_path):
        try:
            # 尝试只加载权重
            state_dict = torch.load(weight_path, map_location=device)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict)
            print(f"Model weights loaded from {weight_path}\n")
        except Exception as e:
            print(f"weights_only load failed: {e}\nTrying with weights_only=False ...")
            # 明确指定 weights_only=False
            state_dict = torch.load(weight_path, map_location=device, weights_only=False)
            if hasattr(state_dict, "state_dict"):
                model.load_state_dict(state_dict.state_dict())
            else:
                model = state_dict  # 直接是模型对象
            print(f"Model loaded from {weight_path} with weights_only=False\n")
    else:
        print(f"No weights found at {weight_path}, using random initialized model.\n")
    return model
'''

def create_model_dir(path_results, runid):
    path_results += runid + "/"
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    print("Results stored at " + path_results + "\n")
    return path_results


def save_model(model):
    mlflow.pytorch.log_model(model, "model")


def save_csv(data, fname):
    # create file if not there
    path = mlflow.get_artifact_uri(artifact_path=fname)
    if path[:7] == "file://":  # to_csv() doesn't work with 'file://'
        path = path[7:]
    if not os.path.isfile(path):
        mlflow.log_text("", fname)
        pd.DataFrame(data).to_csv(path)
    # else append
    else:
        pd.DataFrame(data).to_csv(path, mode="a", header=False)


def save_diff(fname="git_diff.txt"):
    # .txt to allow showing in mlflow
    path = mlflow.get_artifact_uri(artifact_path=fname)
    if path[:7] == "file://":
        path = path[7:]
    mlflow.log_text("", fname)
    os.system(f"git diff > {path}")
