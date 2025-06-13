# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Corner Detection Training Script
"""
"""
Corner Detection Training Script
用于训练基于事件数据的角点检测模型。
"""

import argparse
import numpy as np
import os
import platform
import torch

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

from metavision_core_ml.utils.train_utils import search_latest_checkpoint
from metavision_core_ml.corner_detection.data_module import EventToCornerDataModule
from metavision_core_ml.corner_detection.lightning_model import CornerDetectionCallback, CornerDetectionLightningModel

torch.manual_seed(0)
np.random.seed(0)


def main(raw_args=None):
    """
    使用 PyTorch Lightning 训练角点检测模型。
    可以通过 TensorBoard 可视化训练日志：
    %tensorboard --logdir my_root_dir/lightning_logs/
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 日志和数据集路径
    parser.add_argument('root_dir', type=str, default='', help='日志保存目录')
    parser.add_argument('dataset_path', type=str, default='', help='包含训练和验证数据集的路径')

    # 训练参数
    parser.add_argument('--lr', type=float, default=0.0007, help='学习率')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批量大小')
    parser.add_argument('--precision', type=int, default=16, help='训练精度（16 或 32 位）')
    parser.add_argument('--resume', action='store_true', help='从最新的检查点恢复训练')
    parser.add_argument('--checkpoint', type=str, default='', help='从指定的检查点恢复训练')
    parser.add_argument('--cpu', action='store_true', help='使用 CPU 而非 GPU')

    # 数据参数
    parser.add_argument('--height', type=int, default=180, help='图像高度')
    parser.add_argument('--width', type=int, default=240, help='图像宽度')
    parser.add_argument('--event_volume_depth', type=int, default=10, help='事件体积深度')

    params, _ = parser.parse_known_args(raw_args)
    print('PyTorch Lightning 版本: ', pl.__version__)
    params.cin = params.event_volume_depth  # 输入通道数
    params.cout = 10  # 输出通道数（角点热图数量）

    # 初始化模型
    model = CornerDetectionLightningModel(params)
    if not params.cpu:
        model.cuda()
    else:
        params.data_device = "cpu"

    # 检查点恢复
    if params.resume:
        ckpt = search_latest_checkpoint(params.root_dir)
    elif params.checkpoint != "":
        ckpt = params.checkpoint
    else:
        ckpt = None
    print('检查点: ', ckpt)

    # 设置检查点保存路径
    tmpdir = os.path.join(params.root_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, save_top_k=-1, every_n_epochs=1)

    # 设置 TensorBoard 日志记录器
    logger = TensorBoardLogger(save_dir=os.path.join(params.root_dir, 'logs'))

    # 数据模块
    data = EventToCornerDataModule(params)

    # 训练
    trainer = pl.Trainer(
        default_root_dir=params.root_dir,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator="cpu" if params.cpu else "auto",
        precision=params.precision,
        max_epochs=params.epochs,
    )
    trainer.fit(model, data, ckpt_path=ckpt)


if __name__ == '__main__':
    main()