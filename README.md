# event_drone
虚拟环境
.venv\Scripts\activate
+metavision sdk/openeb环境

configs:
各种参数文件

dataloader：
包含encodings用于事件编码
hdf5是公开数据集格式
raw是录制数据集格式
utils是数据集其他工具

datasets：
包含MVSEC,DSEC,METAVISON,EVIMO,自录制等多种数据集
以及相机的一些畸变校正文件

loss:
包含事件扭曲图像的自监督光流算法

models：
主要是光流预测的SNN模型

utils:
主要是梯度，事件扭曲，分割聚类，可视化等各种工具

train是光流预测的训练
eval，test是光流预测的验证和测试
eval_for_seg，test_for_seg是分割聚类的验证和测试
eval_for_OA,eval_for_OA是单目估深斥力避障的验证和测试
