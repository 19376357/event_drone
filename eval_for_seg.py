import argparse
import cv2
import mlflow
import numpy as np
import torch
from torch.optim import *
import os
import json

from configs.parser import YAMLParser
from dataloader.hdf5 import HDF5Dataset
from dataloader.hdf5 import find_data_triplets
from utils.utils import load_model
from loss.self_supervised import FWL, RSAT, AEE
from models.model import (
    FireNet,
    RNNFireNet,
    LeakyFireNet,
    FireFlowNet,
    LeakyFireFlowNet,
    E2VID,
    EVFlowNet,
    RecEVFlowNet,
    LeakyRecEVFlowNet,
    RNNRecEVFlowNet,
)
from models.model import (
    LIFFireNet,
    PLIFFireNet,
    ALIFFireNet,
    XLIFFireNet,
    LIFFireFlowNet,
    SpikingRecEVFlowNet,
    PLIFRecEVFlowNet,
    ALIFRecEVFlowNet,
    XLIFRecEVFlowNet,
)
from utils.iwe import compute_pol_iwe
from utils.utils import load_model, create_model_dir
from utils.mlflow import log_config, log_results
from utils.visualization import Visualization, vis_activity


def test(args, config_parser):

    mlflow.set_tracking_uri(args.path_mlflow)
    mlflow.set_experiment("eval_experiment")
    mlflow.start_run(run_name="seg XLIFEVFlowNet")
    eval_runid = mlflow.active_run().info.run_id

    config = config_parser.config

    # configs
    if config["loader"]["batch_size"] > 1:
        config["vis"]["enabled"] = False
        config["vis"]["store"] = False
        config["vis"]["bars"] = False  # progress bars not yet compatible batch_size > 1


    if not args.debug:
        # create directory for inference results
        path_results = create_model_dir(args.path_results, eval_runid)

        # store validation settings
        eval_id = log_config(path_results, eval_runid, config)
    else:
        path_results = None
        eval_id = -1

    # 初始设置
    device = config_parser.device

    # 可视化工具
    if config["vis"]["enabled"] or config["vis"]["store"]:
        vis = Visualization(config, eval_id=eval_id, path_results=path_results)

    # 模型初始化
    model_name = config["model"]["name"]
    model = eval(model_name)(config["model"].copy()).to(device)
    model = load_model(args.runid, model, device)
    #model = load_model(model_name, model, device, weights_dir="weights")
    model.eval()

    # 验证参数
    criteria = []
    if "metrics" in config.keys():
        for metric in config["metrics"]["name"]:
            criteria.append(eval(metric)(config, device, flow_scaling=config["metrics"]["flow_scaling"]))

    # 数据加载
    data_dir = config["data"]["data_dir"]
    eye = config["data"].get("eye", "left")
    voxel_bins = config["data"].get("num_bins", 5)
    resolution = tuple(config["loader"].get("resolution", [260, 346]))
    hot_filter = config.get("hot_filter", {})
    triplets = find_data_triplets(data_dir)
    print(f"共找到{len(triplets)}组数据文件。")
    for idx, (data_h5, gt_h5, flow_npz) in enumerate(triplets):
        print(f"\n正在读取第{idx+1}组: \n  data: {data_h5}\n  gt: {gt_h5}\n  flow: {flow_npz}")
        dataset = HDF5Dataset(
            data_h5=data_h5,
            gt_h5=gt_h5,
            flow_npz=flow_npz,
            voxel_bins=voxel_bins,
            resolution=resolution,
            hot_filter=hot_filter,
            eye=eye,
            undistort=config["data"]["undistort"],
            map_dir=config["data"]["undistort_dir"],
            config=config
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config["loader"]["batch_size"],
            shuffle=False,
            num_workers=0
        )

    # 验证循环
    val_results = {}
    activity_log = None
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i == 0 or i == len(dataloader) - 1:
                continue  # 跳过首尾，因为首尾的事件dt与标签dt可能不一致
            x_left = model(
                    batch["left"]["event_voxel"].to(device),
                    batch["left"]["event_cnt"].to(device),
                    log=config["vis"]["activity"]
                )
            x_right = model(
                    batch["right"]["event_voxel"].to(device),
                    batch["right"]["event_cnt"].to(device),
                    log=config["vis"]["activity"]
                )
            flow_left = x_left[-1].clone()
            flow_right = x_right[-1].clone()
            flow_left *= batch["left"]["mask"].to(device)
            flow_left *= batch["right"]["mask"].to(device)


 




            
    if config["vis"]["bars"]:
        for bar in dataset.open_files_bar:
            bar.finish()
    # store validation config and results
    results = {}
    if not args.debug and "metrics" in config.keys():
        for metric in config["metrics"]["name"]:
            results[metric] = {}
            if metric == "AEE":
                results[metric + "_percent"] = {}
            for key in val_results.keys():
                results[metric][key] = str(val_results[key][metric]["metric"] / val_results[key][metric]["it"])
                if metric == "AEE":
                    results[metric + "_percent"][key] = str(
                        val_results[key][metric]["percent"] / val_results[key][metric]["it"]
                    )
            log_results(eval_runid, results, path_results, eval_id)
    mlflow.log_params(config)
    aee_values = [float(v) for v in results.get("AEE", {}).values()]
    aee_mean = sum(aee_values) / len(aee_values) if aee_values else 0.0

    fwl_values = [float(v) for v in results["FWL"].values()]
    fwl_mean = sum(fwl_values) / len(fwl_values) if fwl_values else 0.0

    rsat_values = [float(v) for v in results["RSAT"].values()]
    rsat_mean = sum(rsat_values) / len(rsat_values) if rsat_values else 0.0
    mlflow.log_metric("AEE", aee_mean)
    mlflow.log_metric("FWL", fwl_mean)
    mlflow.log_metric("RSAT", rsat_mean)
    mlflow.end_run()

                     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runid",
        default="XLIFEVFlowNet",
        help="parent mlflow run (optional, for run)",
    )
    parser.add_argument(
        "--config",
        default="configs/eval.yml",
        help="config file, overwrites mlflow settings",
    )
    parser.add_argument(
        "--path_mlflow",
        default="http://localhost:5000",
        help="location of the mlflow ui",
    )
    parser.add_argument("--path_results", default="results_inference/")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="don't save stuff",
    )
    args = parser.parse_args()

    # launch testing
    test(args, YAMLParser(args.config))