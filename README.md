# VGG16 transfer learning re-written in pytorch

This folder contains the pytorch version of **Natural Hazard** prediction.

## Single GPU training (Jupyter notebook files)

`torch-train-1st.ipynb`, `torch-train-2nd.ipynb`, and `torch-train-3rd.ipynb` correspond to `DesignSafe-NaturalHazard-Tutorial-Train-1st.ipynb`, `DesignSafe-NaturalHazard-Tutorial-Train-2nd.ipynb`, and `DesignSafe-NaturalHazard-Tutorial-Train-3rd.ipynb` written in tensorflow and keras.

## Multi-GPU training

To leverage the computation power provided by TACC machines, the pytorch code for training the model in distributed fashion is provided in `torch-train-3rd-distributed.py`. To run it, we also provide the command in `torch-distributed.py` (try `sh torch-distributed.sh`). **Please note that this is only for 1 node case, if you want to train it using multiple nodes of GPUs, please refer to the Pytroch DDp official document [here](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html). (If you read Chinese, here is a good guide on how to use DDP: [link](https://zhuanlan.zhihu.com/p/178402798))**