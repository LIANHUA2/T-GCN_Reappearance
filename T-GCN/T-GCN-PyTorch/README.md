# T-GCN-PyTorch

[![GitHub stars](https://img.shields.io/github/stars/martinwhl/T-GCN-PyTorch?label=stars&maxAge=2592000)](https://gitHub.com/martinwhl/T-GCN-PyTorch/stargazers/) [![issues](https://img.shields.io/github/issues/martinwhl/T-GCN-PyTorch)](https://github.com/martinwhl/T-GCN-PyTorch/issues) [![License](https://img.shields.io/github/license/martinwhl/T-GCN-PyTorch)](./LICENSE) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/martinwhl/T-GCN-PyTorch/graphs/commit-activity) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) ![Codefactor](https://www.codefactor.io/repository/github/martinwhl/T-GCN-PyTorch/badge)

This is a PyTorch implementation of T-GCN in the following paper: [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/abs/1811.05320).

A stable version of this repository can be found at [the official repository](https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch).

Notice that [the original implementation](https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-TensorFlow) is in TensorFlow, which performs a tiny bit better than this implementation for now.



## Requirements

* numpy
* matplotlib
* pandas
* torch
* pytorch-lightning>=1.3.0
* torchmetrics>=0.3.0
* python-dotenv

## 2025年3月21日记录

我于 2025 年 3 月 21 日基本复现了原论文和原代码的结果，使用的是 Linux 系统（无 GUI）和 RTX3090 GPU。将本目录下的所有文件解压至系统根目录。
文件夹data中是数据集，los表示洛杉矶，sz表示深圳，adj是邻接矩阵，speed可能是传感器记录的速度值。
```
root
├── T-GCN-PyTorch
│   ├── data
|   ├── models
|   ├── main.py 
|   ├── tasks 
|   └── ...
└── ...
```
main.py文件有以下代码：
```
DATA_PATHS = {
    "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
    "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv"},
}
```
可以根据需要在此字典中替换你自己的数据集。

在T-GCN-PyTorch目录中，运行以下代码：
```
pip install -r requirements.txt
```
此时会自动安装该文件中所需要的库。

此外，你可能还需要安装scipy库：
```
pip install scipy
```
注意，在 PyTorch Lightning 的较新版本中，Trainer.add_argparse_args 方法已经被弃用或移除。从 PyTorch Lightning 2.0 开始，add_argparse_args 方法不再支持，因此你需要调整代码以适配新版本的 PyTorch Lightning。

我的解决方法是降级PyTorch Lightning 的版本：
```
pip install pytorch-lightning==1.9.5
```
接下来之间复制Model Training中的代码在控制台中运行即可。
## Model Training

```bash
# GCN
python main.py --model_name GCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --batch_size 64 --hidden_dim 100 --settings supervised --gpus 1
# GRU
python main.py --model_name GRU --max_epochs 3000 --learning_rate 0.001 --weight_decay 1.5e-3 --batch_size 64 --hidden_dim 100 --settings supervised --gpus 1
# T-GCN
python main.py --model_name TGCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --hidden_dim 64 --loss mse_with_regularizer --settings supervised --gpus 1
```

You can also adjust the `--data`, `--seq_len` and `--pre_len` parameters.

Run `tensorboard --logdir lightning_logs/version_0` to monitor the training progress and view the prediction results.

假设你更改了数据集：
```
DATA_PATHS = {
    "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
    "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv"},
    "beijing": {"feat": "data/beijing_speed.csv", "adj": "data/beijing_adj.csv"},  # 添加新数据集
}
```
那你可以运行脚本单独运行你自己的数据集：
```
python main.py --data beijing --model_name TGCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --hidden_dim 64 --loss mse_with_regularizer --settings supervised --gpus 1
```

## 复现结果
* GCN                         
* ExplainedVar:0.6829
* MAE:5.4203
* R2:0.6825
* RMSE:7.8182
* accuracy:0.8669
* val_loss:61.1250
---------------------
* GRU                         
* ExplainedVar:0.8393
* MAE:3.1719
* R2:0.8365
* RMSE:5.6170
* accuracy:0.9044
* val_loss:31.5502
---------------------
* T-GCN                         
* ExplainedVar:0.8562
* MAE:3.2975
* R2:0.8561
* RMSE:5.2609
* accuracy:0.9104
* val_loss:3342988.5
 ---------------------
 因为某些异常，我的T-GCN的val_loss值有错误。整体上来看，T-GCN确实最优，GRU次之。
