import argparse
import cv2
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
from models.model import EVFlowNet
from models.model import (
    SpikingRecEVFlowNet,
    PLIFRecEVFlowNet,
    ALIFRecEVFlowNet,
    XLIFRecEVFlowNet,
)
from utils.utils import load_model, create_model_dir,log_config
from utils.vis_for_seg import VisForSeg
from utils.segmatch import segment_events_by_flow,cluster_moving_objects, match_stereo_events, triangulate_stereo_points, estimate_object_velocity


def test(args, config_parser):

    #初始化
    config = config_parser.config
    device = config_parser.device
    runid = args.runid


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
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i == 0 or i == len(dataloader) - 1:
                continue  # 跳过首尾，因为首尾的事件dt与标签dt可能不一致
            x_left = model(
                    batch["left"]["event_voxel"].to(device),
                    batch["left"]["event_cnt"].to(device),
                    log=config["vis"]["activity"]
                )
            flow_left = x_left['flow'][-1].clone()
            
            flow_left *= batch["left"]["mask"].to(device)
            
            fg_left = segment_events_by_flow(
                batch["left"]["event_list"].squeeze(0).cpu().numpy(),
                flow_left.squeeze(0).cpu().numpy(),
                threshold=config["segmentation"]["flow_threshold"]
            )
            
            
            obj_left = cluster_moving_objects(
                batch["left"]["event_list"].squeeze(0).cpu().numpy()[fg_left],
                flow_left.squeeze(0).cpu().numpy(),
                fg_left,
                eps=config["cluster"]["eps"],
                min_samples=config["cluster"]["min_samples"]
            )
            
            
            vis = VisForSeg(px=346,py=260)
            vis.visualize_all(
                events=batch["left"]["event_list"].squeeze(0).cpu().numpy(),
                flow=flow_left.squeeze(0).cpu().numpy(),
                fg_events=batch["left"]["event_list"].squeeze(0).cpu().numpy()[fg_left] if fg_left is not None else None,
                obj_events=batch["left"]["event_list"].squeeze(0).cpu().numpy()[obj_left] if obj_left is not None else None,
            )

                     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runid",
        default="XLIFEVFlowNet",
    )
    parser.add_argument(
        "--config",
        default="configs/evalseg.yml",
    )
    parser.add_argument("--path_results", default="seg_results_inference/")
    parser.add_argument(
        "--debug",
    )
    args = parser.parse_args()

    # launch testing
    test(args, YAMLParser(args.config))