import argparse
import cv2
import numpy as np
import torch
from torch.optim import *
import os
import json

from configs.parser import YAMLParser
from dataloader.raw import simHDF5Dataset
from dataloader.raw import simfind_data_triplets
from loss.self_supervised import FWL, RSAT, AEE
from models.model import EVFlowNet
from models.model import (
    SpikingRecEVFlowNet,
    PLIFRecEVFlowNet,
    ALIFRecEVFlowNet,
    XLIFRecEVFlowNet,
)

from utils.iwe import compute_pol_iwe
from utils.utils import load_model, create_model_dir, log_config, log_results
from utils.visualization import Visualization, vis_activity


def test(args, config_parser):


    config = config_parser.config
    device = config_parser.device
    runid = args.runid


    if not args.debug:
        # create directory for inference results
        path_results = create_model_dir(args.path_results, runid)
        # store validation settings
        eval_id = log_config(path_results, runid, config)
    else:
        path_results = None
        eval_id = -1


    # 可视化工具
    if config["vis"]["enabled"] or config["vis"]["store"]:
        vis = Visualization(config, eval_id=eval_id, path_results=path_results)

    # 模型初始化
    model_name = config["model"]["name"]
    model = eval(model_name)(config["model"].copy()).to(device)
    model_path = os.path.join("weights", runid, "artifacts", "model", "data", "model.pth")
    model = load_model(model_path, model, device)
    model.eval()

    # 验证参数
    criteria = []
    if "metrics" in config.keys():
        for metric in config["metrics"]["name"]:
            criteria.append(eval(metric)(config, device, flow_scaling=config["metrics"]["flow_scaling"]))

    # 数据加载
    data_dir = config["data"]["data_dir"]
    voxel_bins = config["data"].get("num_bins", 5)
    resolution = tuple(config["loader"].get("resolution"))
    hot_filter = config.get("hot_filter", {})
    triplets = simfind_data_triplets(data_dir)
    print(f"共找到{len(triplets)}组数据文件。")
    for idx, data_h5 in enumerate(triplets):
        print(f"\n正在读取第{idx+1}组: \n  data: {data_h5}\n ")
        dataset = simHDF5Dataset(
            data_h5=data_h5,
            voxel_bins=voxel_bins,
            resolution=resolution,
            hot_filter=hot_filter,
            config=config
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config["loader"]["batch_size"],
            shuffle=False,
            num_workers=0
        )
    # 验证循环
    test_results = {}
    activity_log = None
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(f"Batch {i}:")

            # ====== 可视化数据收集 ======
            vis_data = {}
            x = model(
                batch["event_voxel"].to(device),
                batch["event_cnt"].to(device),
                log=config["vis"]["activity"]
            )
            flow_vis = x["flow"][-1].clone()
            flow_vis *= batch["mask"].to(device)
            iwe = compute_pol_iwe(
                x["flow"][-1],
                batch["event_list"],
                config["loader"]["resolution"],
                batch["event_list_pol_mask"][:, :, 0:1],
                batch["event_list_pol_mask"][:, :, 1:2],
                flow_scaling=config["metrics"]["flow_scaling"],
                round_idx=True,
            )
            # 收集可视化数据
            vis_data["left"] = {
                "inputs": batch,
                "flow": flow_vis,
                "iwe": iwe,
            }
            iwe_window_vis = None
            events_window_vis = None
            masked_window_flow_vis = None

            for idx_metric, metric_name in enumerate(config["metrics"]["name"]):
                if metric_name == "AEE" and ("flow" not in batch or batch["flow"].numel() == 0):
                    continue
                # 先做 event_flow_association，对每个metric_name，连接每个idx的事件数据和真值
                criteria[idx_metric].event_flow_association(x["flow"], batch)
                # 如需只评估最终输出
                if config["loss"].get("overwrite_intermediate", False):
                    criteria[idx_metric].overwrite_intermediate_flow(x["flow"])
                # 调用已经连接好的criteria进行计算
                val_metric = criteria[idx_metric]()

                # 按文件名累积结果
                filenames = batch["filename"]
                for b in range(len(filenames)):
                    filename = filenames[b] if isinstance(filenames[b], str) else filenames[b].decode()  # 兼容 bytes
                    #添加全部文件名
                    if filename not in test_results:
                        test_results[filename] = {}
                        #给每个文件名添加每个metric_name的初始值
                        for m in config["metrics"]["name"]:
                            test_results[filename][m] = {"metric": 0, "it": 0}
                            if m == "AEE":
                                test_results[filename][m]["percent"] = 0
                    test_results[filename][metric_name]["it"] += 1#因为次数初始化是0，所以先+1
                    if metric_name == "AEE":
                        test_results[filename][metric_name]["metric"] += val_metric[0][b].item()
                        test_results[filename][metric_name]["percent"] += val_metric[1][b].item()
                    else:
                        test_results[filename][metric_name]["metric"] += val_metric[b].item()
        
                
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
                # 光流
                flow_left = vis_data["left"]["flow"]
                # IWE
                iwe_left = vis_data["left"]["iwe"]

                vis.update(
                    vis_data["left"]["inputs"],
                    flow_left,
                    iwe_left,
                    events_window_vis,
                    masked_window_flow_vis,
                    iwe_window_vis
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
            for key in test_results.keys():
                results[metric][key] = str(test_results[key][metric]["metric"] / test_results[key][metric]["it"])
                if metric == "AEE":
                    results[metric + "_percent"][key] = str(
                        test_results[key][metric]["percent"] / test_results[key][metric]["it"]
                    )
            log_results(path_results, results, eval_id)

    #aee_values = [float(v) for v in results["AEE"].values()]
    #aee_mean = sum(aee_values) / len(aee_values) if aee_values else 0.0

    fwl_values = [float(v) for v in results["FWL"].values()]
    fwl_mean = sum(fwl_values) / len(fwl_values) if fwl_values else 0.0

    rsat_values = [float(v) for v in results["RSAT"].values()]
    rsat_mean = sum(rsat_values) / len(rsat_values) if rsat_values else 0.0
    print(f"FWL mean: {fwl_mean:.4f}, RSAT mean: {rsat_mean:.4f}")

                     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runid",
        default="XLIFEVFlowNet",
    )
    parser.add_argument(
        "--config",
        default="configs/test.yml",
    )
    parser.add_argument("--path_results", default="results_test/")
    parser.add_argument(
        "--debug",
    )
    args = parser.parse_args()

    # launch testing
    test(args, YAMLParser(args.config))

