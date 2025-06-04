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
    mlflow.start_run(run_name="eval XLIFEVFlowNet")
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
            print(f"Batch {i}:")
            # 支持单目和双目
            batch_eyes = []
            if eye == "both":
                batch_eyes = [("left", batch["left"]), ("right", batch["right"])]
            else:
                batch_eyes = [("left", batch)]

            # ====== 可视化数据收集 ======
            vis_data = {}
            
            for eye_name, use_batch in batch_eyes:
                print(f"  Eye: {eye_name}")
                x = model(
                    use_batch["event_voxel"].to(device),
                    use_batch["event_cnt"].to(device),
                    log=config["vis"]["activity"]
                )
                flow_vis = x["flow"][-1].clone()
                flow_vis *= use_batch["mask"].to(device)
                iwe = compute_pol_iwe(
                    x["flow"][-1],
                    use_batch["event_list"],
                    config["loader"]["resolution"],
                    use_batch["event_list_pol_mask"][:, :, 0:1],
                    use_batch["event_list_pol_mask"][:, :, 1:2],
                    flow_scaling=config["metrics"]["flow_scaling"],
                    round_idx=True,
                )
                # 收集可视化数据
                vis_data[eye_name] = {
                    "inputs": use_batch,
                    "flow": flow_vis,
                    "iwe": iwe,
                }
                iwe_window_vis = None
                events_window_vis = None
                masked_window_flow_vis = None

                for idx_metric, metric_name in enumerate(config["metrics"]["name"]):
                    if metric_name == "AEE" and ("flow" not in use_batch or use_batch["flow"].numel() == 0):
                        continue
                    # 先做 event_flow_association，对每个metric_name，连接每个idx的事件数据和真值
                    criteria[idx_metric].event_flow_association(x["flow"], use_batch)
                    # 如需只评估最终输出
                    if config["loss"].get("overwrite_intermediate", False):
                        criteria[idx_metric].overwrite_intermediate_flow(x["flow"])
                    # 调用已经连接好的criteria进行计算
                    val_metric = criteria[idx_metric]()

                    # 按文件名累积结果
                    filenames = use_batch["filename"]
                    for b in range(len(filenames)):
                        filename = filenames[b] if isinstance(filenames[b], str) else filenames[b].decode()  # 兼容 bytes
                        #添加全部文件名
                        if filename not in val_results:
                            val_results[filename] = {}
                            #给每个文件名添加每个metric_name的初始值
                            for m in config["metrics"]["name"]:
                                val_results[filename][m] = {"metric": 0, "it": 0}
                                if m == "AEE":
                                    val_results[filename][m]["percent"] = 0
                        val_results[filename][metric_name]["it"] += 1#因为次数初始化是0，所以先+1
                        if metric_name == "AEE":
                            val_results[filename][metric_name]["metric"] += val_metric[0][b].item()
                            val_results[filename][metric_name]["percent"] += val_metric[1][b].item()
                        else:
                            val_results[filename][metric_name]["metric"] += val_metric[b].item()
            
                    
                    #添加可视化
                    if (
                        idx_metric == 0
                        and (config["vis"]["enabled"] or config["vis"]["store"])
                    ):
                        events_window_vis = criteria[idx_metric].compute_window_events()
                        iwe_window_vis = criteria[idx_metric].compute_window_iwe()
                        masked_window_flow_vis = criteria[idx_metric].compute_masked_window_flow()

                    # reset criteria
                    criteria[idx_metric].reset()
                # visualize
                if config["vis"]["bars"]:
                    for bar in dataset.open_files_bar:
                        bar.next()
                if config["vis"]["enabled"]:
                    # 事件
                    events_left = vis_data["left"]["inputs"]["event_cnt"]
                    events_right = vis_data["right"]["inputs"]["event_cnt"] if "right" in vis_data else None
                    # 光流
                    flow_left = vis_data["left"]["flow"]
                    flow_right = vis_data["right"]["flow"] if "right" in vis_data else None
                    # IWE
                    iwe_left = vis_data["left"]["iwe"]
                    iwe_right = vis_data["right"]["iwe"] if "right" in vis_data else None
                    # GT flow
                    gtflow_left = vis_data["left"]["inputs"]["flow"] if "flow" in vis_data["left"]["inputs"] else None
                    gtflow_right = vis_data["right"]["inputs"]["flow"] if ("right" in vis_data and "flow" in vis_data["right"]["inputs"]) else None
                    # frames
                    frames_left = vis_data["left"]["inputs"]["image"] if "image" in vis_data["left"]["inputs"] else None
                    frames_right = vis_data["right"]["inputs"]["image"] if ("right" in vis_data and "image" in vis_data["right"]["inputs"]) else None

                    if eye == "both":
                        vis.update_stereo(
                            events_left, events_right,
                            frames_left, frames_right,
                            flow_left, flow_right,
                            iwe_left, iwe_right,
                            gtflow_left, gtflow_right,
                            vis_data["left"]["inputs"]
                        )
                    else:
                        vis.update(
                            vis_data["left"]["inputs"],
                            flow_left,
                            iwe_left,
                            events_window_vis,
                            masked_window_flow_vis,
                            iwe_window_vis
                        )
                if config["vis"]["store"]:
                    sequence = dataset.files[dataset.batch_idx[0] % len(dataset.files)].split("/")[-1].split(".")[0]
                    vis.store(
                        use_batch,
                        flow_vis,
                        iwe,
                        sequence,
                        events_window_vis,
                        masked_window_flow_vis,
                        iwe_window_vis,
                        ts=dataset.last_proc_timestamp,
                    )

                # visualize activity
                if config["vis"]["activity"]:
                    activity_log = vis_activity(x["activity"], activity_log)
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