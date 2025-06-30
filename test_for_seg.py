import argparse
import cv2
import mlflow
import numpy as np
import torch
from torch.optim import *
import os
import json

from configs.parser import YAMLParser
from dataloader.raw import simHDF5Dataset
from dataloader.raw import simfind_data_triplets
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
from utils.vis_for_seg import VisForSeg
from utils.segmatch import segment_events_by_flow,cluster_moving_objects, match_stereo_events, triangulate_stereo_points, estimate_object_velocity


def test(args, config_parser):

    mlflow.set_tracking_uri(args.path_mlflow)
    mlflow.set_experiment("eval_seg_experiment")
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
    val_results = {}
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i == 0 or i == len(dataloader) - 1:  # 跳过首尾的batch
                continue  # 跳过首尾，因为首尾的事件dt与标签dt可能不一致
            if i <=200:
                continue
            print(f"Batch {i}:")
            x = model(
                batch["event_voxel"].to(device),
                batch["event_cnt"].to(device),
                log=config["vis"]["activity"]
            )
            flow_left = x["flow"][-1].clone()
            flow_left *= batch["mask"].to(device)

            ys_fg_left, xs_fg_left = segment_events_by_flow(
                batch["event_cnt"].squeeze(0).cpu().numpy(),
                flow_left.squeeze(0).cpu().numpy(),
                threshold=config["segmentation"]["flow_threshold"]
            )

            main_ys_left, main_xs_left = cluster_moving_objects(
                ys_fg_left, xs_fg_left,
                eps=config["cluster"]["eps"],
                min_samples=config["cluster"]["min_samples"]
            )

            event_list_left = batch["event_list"].squeeze(0).cpu().numpy()
            ys_all_left = event_list_left[:, 1].astype(np.int32)
            xs_all_left = event_list_left[:, 2].astype(np.int32)
            fg_pixel_set_left = set(zip(ys_fg_left, xs_fg_left))
            obj_pixel_set_left = set(zip(main_ys_left, main_xs_left))
            fg_mask = np.array([(y, x) in fg_pixel_set_left for y, x in zip(ys_all_left, xs_all_left)])
            obj_mask_left = np.array([(y, x) in obj_pixel_set_left for y, x in zip(ys_all_left, xs_all_left)])
            fg_events_left = event_list_left[fg_mask]
            obj_events_left = event_list_left[obj_mask_left]

            '''
            event_list_right = batch["right"]["event_list"].squeeze(0).cpu().numpy()
            ys_all_right = event_list_right[:, 1].astype(np.int32)
            xs_all_right = event_list_right[:, 2].astype(np.int32)
            obj_pixel_set_right = set(zip(main_ys_right, main_xs_right))
            obj_mask_right = np.array([(y, x) in obj_pixel_set_right for y, x in zip(ys_all_right, xs_all_right)])
            obj_events_right = event_list_right[obj_mask_right]


            matches = match_stereo_events(
                obj_events_left,
                obj_events_right,
                max_dist=config["match"]["max_dist"],
                max_dt=config["match"]["max_dt"]
            )
            fx = config["camera"]["fx"]
            fy = config["camera"]["fy"]    # 像素单位
            cx = config["camera"]["cx"]
            cy = config["camera"]["cy"]
            B = config["camera"]["baseline"]       # 米

            points_3d = triangulate_stereo_points(obj_events_left, obj_events_right, matches, fx,fy, cx, cy, B)
            if len(points_3d) > 0:
                Z_mean = np.mean(points_3d[:, 2])
                velocity = estimate_object_velocity(
                    obj_events_left, 
                    flow_left.squeeze(0).cpu().numpy(), 
                    points_3d, 
                    matches, 
                    fx,fy,Z_mean, dt=0.05  # dt可根据事件时间戳实际计算
                )
                print("目标物体空间速度估计:", velocity)
            '''


            vis = VisForSeg(px=256,py=256)
            vis.visualize_all(
                events=batch["event_list"].squeeze(0).cpu().numpy(),
                flow=flow_left.squeeze(0).cpu().numpy(),
                fg_events=fg_events_left,
                obj_events=obj_events_left,
                fg_pixels=(ys_fg_left, xs_fg_left),
                obj_pixels=(main_ys_left, main_xs_left),
            )

    mlflow.log_params(config)

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
        default="configs/evalseg.yml",
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