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


            vis = VisForSeg(px=256,py=256)
            vis.visualize_all(
                events=batch["event_list"].squeeze(0).cpu().numpy(),
                flow=flow_left.squeeze(0).cpu().numpy(),
                fg_events=fg_events_left,
                obj_events=obj_events_left,
                fg_pixels=(ys_fg_left, xs_fg_left),
                obj_pixels=(main_ys_left, main_xs_left),
            )


                     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runid",
        default="XLIFEVFlowNet",
    )
    parser.add_argument(
        "--config",
        default="configs/testseg.yml",
    )
    parser.add_argument("--path_results", default="eval_results_test/")
    parser.add_argument(
        "--debug",
    )
    args = parser.parse_args()

    # launch testing
    test(args, YAMLParser(args.config))