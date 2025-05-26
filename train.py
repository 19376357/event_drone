import argparse

#import mlflow
import torch
from torch.optim import *

from configs.parser import YAMLParser
from dataloader.hdf5 import HDF5Dataset
from dataloader.hdf5 import find_data_triplets
from utils.utils import load_model
'''
from loss.flow import EventWarping
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
from utils.gradients import get_grads
from utils.utils import load_model, save_csv, save_diff, save_model
from utils.visualization import Visualization
'''




def train(args, config_parser):
    #mlflow.set_tracking_uri(args.path_mlflow)

    # 解析配置
    config = config_parser.config
    
    '''
    # log config
    mlflow.set_experiment(config["experiment"])
    mlflow.start_run()
    mlflow.log_params(config)
    mlflow.log_param("prev_runid", args.prev_runid)
    config = config_parser.combine_entries(config)
    print("MLflow dir:", mlflow.active_run().info.artifact_uri[:-9])

    # log git diff
    save_diff("train_diff.txt")



    # visualization tool
    if config["vis"]["enabled"]:
        vis = Visualization(config)
    '''
    # initialize settings
    device = config_parser.device
    kwargs = config_parser.loader_kwargs

    # data loader
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
        
        # model initialization and settings
        model_name = config["model"]["name"]
        model = eval(model_name)(config["model"].copy()).to(device)
        model = load_model(model_name, model, device, weights_dir="weights")
        model.train()


        # 简单测试数据读取
        for i, batch in enumerate(dataloader):
            print(f"Batch {i}:")

            output = model(batch['event_voxel'].to(device), batch['event_cnt'].to(device))
            if i >= 2:
                break    
    '''
    # loss function
    loss_function = EventWarping(config, device)



    # optimizers
    optimizer = eval(config["optimizer"]["name"])(model.parameters(), lr=config["optimizer"]["lr"])
    optimizer.zero_grad()

    # simulation variables
    train_loss = 0
    best_loss = 1.0e6
    end_train = False
    grads_w = []

    # training loop
    data.shuffle()
    while True:
        for inputs in dataloader:

            if data.new_seq:
                data.new_seq = False

                loss_function.reset()
                model.reset_states()
                optimizer.zero_grad()

            if data.seq_num >= len(data.files):
                mlflow.log_metric("loss", train_loss / (data.samples + 1), step=data.epoch)

                with torch.no_grad():
                    if train_loss / (data.samples + 1) < best_loss:
                        save_model(model)
                        best_loss = train_loss / (data.samples + 1)

                data.epoch += 1
                data.samples = 0
                train_loss = 0
                data.seq_num = data.seq_num % len(data.files)

                # save grads to file
                if config["vis"]["store_grads"]:
                    save_csv(grads_w, "grads_w.csv")
                    grads_w = []

                # finish training loop
                if data.epoch == config["loader"]["n_epochs"]:
                    end_train = True

            # forward pass
            x = model(inputs["event_voxel"].to(device), inputs["event_cnt"].to(device))

            # event flow association
            loss_function.event_flow_association(
                x["flow"],
                inputs["event_list"].to(device),
                inputs["event_list_pol_mask"].to(device),
                inputs["event_mask"].to(device),
            )

            # backward pass
            if loss_function.num_events >= config["data"]["window_loss"]:

                # overwrite intermediate flow estimates with the final ones
                if config["loss"]["overwrite_intermediate"]:
                    loss_function.overwrite_intermediate_flow(x["flow"])

                # loss
                loss = loss_function()
                train_loss += loss.item()

                # update number of loss samples seen by the network
                data.samples += config["loader"]["batch_size"]

                loss.backward()

                # clip and save grads
                if config["loss"]["clip_grad"] is not None:
                    torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), config["loss"]["clip_grad"])
                if config["vis"]["store_grads"]:
                    grads_w.append(get_grads(model.named_parameters()))

                optimizer.step()
                optimizer.zero_grad()

                # mask flow for visualization
                flow_vis = x["flow"][-1].clone()
                if model.mask and config["vis"]["enabled"] and config["loader"]["batch_size"] == 1:
                    flow_vis *= loss_function.event_mask

                model.detach_states()
                loss_function.reset()

                # visualize
                with torch.no_grad():
                    if config["vis"]["enabled"] and config["loader"]["batch_size"] == 1:
                        vis.update(inputs, flow_vis, None)

            # print training info
            if config["vis"]["verbose"]:
                print(
                    "Train Epoch: {:04d} [{:03d}/{:03d} ({:03d}%)] Loss: {:.6f}".format(
                        data.epoch,
                        data.seq_num,
                        len(data.files),
                        int(100 * data.seq_num / len(data.files)),
                        train_loss / (data.samples + 1),
                    ),
                    end="\r",
                )

        if end_train:
            break

    mlflow.end_run()
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train.yml",
        help="training configuration",
    )
    '''
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )
    parser.add_argument(
        "--prev_runid",
        default="",
        help="pre-trained model to use as starting point",
    )
    '''
    args = parser.parse_args()

    # launch training
    train(args, YAMLParser(args.config))
