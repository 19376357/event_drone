base.py
定义模型基础类，封装通用的 forward、保存/加载权重等方法

model.py
主模型入口
model_util.py
主模型相关的工具函数，如权重初始化，参数统计等

spiking_submodules.py
定义了多种LIF形式的神经元模块，（卷积/循环）脉冲层
spiking_util.py
定义了SNN相关的工具函数，如脉冲编码，阈值调整

submodules.py
常见模型的模块，如卷积块，残差块，上下采样等
unet.py
Unet经典网络，常用于此类光流任务