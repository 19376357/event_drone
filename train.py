import argparse
import mlflow
import torch
from torch.optim import *
from torch.utils.data import ConcatDataset
import os

from configs.parser import YAMLParser
from dataloader.hdf5 import HDF5Dataset
from dataloader.hdf5 import find_data_triplets
from loss.self_supervised import EventWarping
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
from utils.utils import load_model, save_model, create_model_dir
from utils.visualization import Visualization
from utils.mlflow import log_config


def train(args, config_parser):

    mlflow.set_tracking_uri(args.path_mlflow)
    mlflow.set_experiment(config_parser.config.get("experiment", "Default"))
    mlflow.start_run(run_name="train")
    runid = mlflow.active_run().info.run_id
    print("New train runid:", runid)

    config = config_parser.config

    # configs
    if config["loader"]["batch_size"] > 1:
        config["vis"]["enabled"] = False
        config["vis"]["store_grads"] = False
        config["vis"]["bars"] = False  # progress bars not yet compatible batch_size > 1

    path_results = create_model_dir(args.path_results, runid)
    train_id = log_config(path_results, runid, config)

    # 初始设置
    device = config_parser.device
    kwargs = config_parser.loader_kwargs

    # 可视化工具
    if config["vis"]["enabled"] or config["vis"]["store_grads"]:
        vis = Visualization(config, eval_id=train_id, path_results=path_results)

    # 模型初始化
    model_name = config["model"]["name"]
    model = eval(model_name)(config["model"].copy()).to(device)
    if args.resume_runid:
        model = load_model(args.resume_runid, model, device)
    model.train()

    # 数据加载
    data_dir = config["data"]["data_dir"]
    eye = config["data"].get("eye", "left")
    encoding = config["data"].get("encoding", "cnt")
    voxel_bins = config["data"].get("num_bins", 5)
    resolution = tuple(config["loader"].get("resolution", [260, 346]))
    hot_filter = config.get("hot_filter", {})
    triplets = find_data_triplets(data_dir)
    print(f"共找到{len(triplets)}组数据文件。")
    datasets = []
    for idx, (data_h5, gt_h5, flow_npz) in enumerate(triplets):
        print(f"\n正在读取第{idx+1}组: \n  data: {data_h5}\n  gt: {gt_h5}\n  flow: {flow_npz}")
        datasets.append(
            HDF5Dataset(
                data_h5=data_h5,
                gt_h5=gt_h5,
                flow_npz=flow_npz,
                voxel_bins=voxel_bins,
                resolution=resolution,
                hot_filter=hot_filter,
                eye=eye,
                config=config
            )
        )
    full_dataset = ConcatDataset(datasets)
    dataloader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=config["loader"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    # 优化器
    optimizer = eval(config["optimizer"]["name"])(model.parameters(), lr=config["optimizer"]["lr"])
    optimizer.zero_grad()

    # 损失函数
    loss_function = EventWarping(config, device)

    best_loss = float("inf")
    #torch.autograd.set_detect_anomaly(True)
    for epoch in range(config["loader"]["n_epochs"]):
        epoch_loss = 0
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            model.reset_states()
            loss_function.reset()
            if i == 0 or i == len(dataloader) - 1:
                continue  # 跳过首尾
            x = model(
                batch["event_voxel"].to(device),
                batch["event_cnt"].to(device),
            )
            loss_function.event_flow_association(
                x["flow"],
                batch["event_list"].to(device),
                batch["event_list_pol_mask"].to(device),
                batch["mask"].to(device),
            )
            loss = loss_function()
            loss.backward()
            if config["loss"].get("clip_grad", None) is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["loss"]["clip_grad"])
            optimizer.step()
            epoch_loss += loss.item()

            if config["vis"]["enabled"] and config["loader"]["batch_size"] == 1:
                flow_vis = x["flow"][-1].clone()
                flow_vis *= batch["mask"].to(device)
                vis.update(batch, flow_vis, None)
            if config["vis"].get("verbose", False):
                print(
                    f"Epoch {epoch+1:03d} [{i+1:03d}/{len(dataloader)}] Loss: {loss.item():.6f}",
                    end="\r"
                )
        avg_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch+1:03d} finished. Avg Loss: {avg_loss:.6f}")
        mlflow.log_metric("loss", avg_loss, step=epoch+1)
        if avg_loss < best_loss:
            save_model(model)
            best_loss = avg_loss
    mlflow.end_run()

                     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train.yml",
        help="config file, overwrites mlflow settings",
    )
    parser.add_argument(
        "--path_mlflow",
        default="http://localhost:5000",
        help="location of the mlflow ui",
    )
    parser.add_argument(
        "--resume_runid",
        default="",
        help="pre-trained model to use as starting point",
    )
    parser.add_argument("--path_results", default="results_train/")
    args = parser.parse_args()

    # launch testing
    train(args, YAMLParser(args.config))


