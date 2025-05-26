import argparse

import mlflow
import numpy as np
import torch
from torch.optim import *

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

    #mlflow.set_tracking_uri(args.path_mlflow)
    #run = mlflow.get_run(args.runid)
    #config = config_parser.merge_configs(run.data.params)
    config = config_parser.config



    # configs
    if config["loader"]["batch_size"] > 1:
        config["vis"]["enabled"] = False
        config["vis"]["store"] = False
        config["vis"]["bars"] = False  # progress bars not yet compatible batch_size > 1


    if not args.debug:
        # create directory for inference results
        path_results = create_model_dir(args.path_results, args.runid)

        # store validation settings
        eval_id = log_config(path_results, args.runid, config)
    else:
        path_results = None
        eval_id = -1

    # 初始设置
    device = config_parser.device
    kwargs = config_parser.loader_kwargs

    # 可视化工具
    if config["vis"]["enabled"] or config["vis"]["store"]:
        vis = Visualization(config, eval_id=eval_id, path_results=path_results)

    # 模型初始化
    model_name = config["model"]["name"]
    model = eval(model_name)(config["model"].copy()).to(device)
    model = load_model(model_name, model, device, weights_dir="weights")
    model.eval()

    # 验证参数
    criteria = []
    if "metrics" in config.keys():
        for metric in config["metrics"]["name"]:
            criteria.append(eval(metric)(config, device, flow_scaling=config["metrics"]["flow_scaling"]))

    # 数据加载
    data_dir = config["data"]["data_dir"]
    eye = config["data"].get("eye", "left")
    encoding = config["data"].get("encoding", "cnt")
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
            encoding=encoding,
            resolution=resolution,
            hot_filter=hot_filter,
            eye=eye
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config["loader"]["batch_size"],
            shuffle=False,
            num_workers=0
        )

    # 验证循环
    idx_AEE = 0
    val_results = {}
    end_test = False
    activity_log = None
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(f"Batch {i}:")

            # forward pass
            x = model(
                batch["event_voxel"].to(device), batch["event_cnt"].to(device), log=config["vis"]["activity"]
            )
            x
'''
                # mask flow for visualization
                flow_vis = x["flow"][-1].clone()
                if model.mask:
                    flow_vis *= inputs["event_mask"].to(device)

                # image of warped events
                iwe = compute_pol_iwe(
                    x["flow"][-1],
                    inputs["event_list"].to(device),
                    config["loader"]["resolution"],
                    inputs["event_list_pol_mask"][:, :, 0:1].to(device),
                    inputs["event_list_pol_mask"][:, :, 1:2].to(device),
                    flow_scaling=config["metrics"]["flow_scaling"],
                    round_idx=True,
                )

                iwe_window_vis = None
                events_window_vis = None
                masked_window_flow_vis = None
                if "metrics" in config.keys():

                    # event flow association
                    for metric in criteria:
                        metric.event_flow_association(x["flow"], inputs)

                    # validation
                    for i, metric in enumerate(config["metrics"]["name"]):
                        if criteria[i].num_events >= config["data"]["window_eval"]:

                            # overwrite intermedia flow estimates with the final ones
                            if config["loss"]["overwrite_intermediate"]:
                                criteria[i].overwrite_intermediate_flow(x["flow"])
                            if metric == "AEE" and inputs["dt_gt"] <= 0.0:
                                continue
                            if metric == "AEE":
                                idx_AEE += 1
                                if idx_AEE != np.round(1.0 / config["data"]["window"]):
                                    continue

                            # compute metric
                            val_metric = criteria[i]()
                            if metric == "AEE":
                                idx_AEE = 0

                            # accumulate results
                            for batch in range(config["loader"]["batch_size"]):
                                filename = data.files[data.batch_idx[batch] % len(data.files)].split("/")[-1]
                                if filename not in val_results.keys():
                                    val_results[filename] = {}
                                    for metric in config["metrics"]["name"]:
                                        val_results[filename][metric] = {}
                                        val_results[filename][metric]["metric"] = 0
                                        val_results[filename][metric]["it"] = 0
                                        if metric == "AEE":
                                            val_results[filename][metric]["percent"] = 0

                                val_results[filename][metric]["it"] += 1
                                if metric == "AEE":
                                    val_results[filename][metric]["metric"] += val_metric[0][batch].cpu().numpy()
                                    val_results[filename][metric]["percent"] += val_metric[1][batch].cpu().numpy()
                                else:
                                    val_results[filename][metric]["metric"] += val_metric[batch].cpu().numpy()

                            # visualize
                            if (
                                i == 0
                                and config["data"]["mode"] == "events"
                                and (config["vis"]["enabled"] or config["vis"]["store"])
                                and config["data"]["window"] < config["data"]["window_eval"]
                            ):
                                events_window_vis = criteria[i].compute_window_events()
                                iwe_window_vis = criteria[i].compute_window_iwe()
                                masked_window_flow_vis = criteria[i].compute_masked_window_flow()

                            # reset criteria
                            criteria[i].reset()

                # visualize
                if config["vis"]["bars"]:
                    for bar in data.open_files_bar:
                        bar.next()
                if config["vis"]["enabled"]:
                    vis.update(inputs, flow_vis, iwe, events_window_vis, masked_window_flow_vis, iwe_window_vis)
                if config["vis"]["store"]:
                    sequence = data.files[data.batch_idx[0] % len(data.files)].split("/")[-1].split(".")[0]
                    vis.store(
                        inputs,
                        flow_vis,
                        iwe,
                        sequence,
                        events_window_vis,
                        masked_window_flow_vis,
                        iwe_window_vis,
                        ts=data.last_proc_timestamp,
                    )

                # visualize activity
                if config["vis"]["activity"]:
                    activity_log = vis_activity(x["activity"], activity_log)

            if end_test:
                break

    if config["vis"]["bars"]:
        for bar in data.open_files_bar:
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
            log_results(args.runid, results, path_results, eval_id)
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("runid", help="mlflow run")
    parser.add_argument(
        "--config",
        default="configs/eval_flow.yml",
        help="config file, overwrites mlflow settings",
    )
    '''
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )
    '''
    parser.add_argument("--path_results", default="results_inference/")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="don't save stuff",
    )
    args = parser.parse_args()

    # launch testing
    test(args, YAMLParser(args.config))
