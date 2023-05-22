# VGG16 transfer learning re-written in pytorch

This folder contains the pytorch version of **Natural Hazard** prediction.

## Install environment
Before you run `./pytorch/install_env.sh`, please make sure you have successfully installed miniconda/conda, created a virtual environment, and had jupyter notebook connected to it.

### Install miniconda
Run the following shell commands in a terminal. 
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

### Create a virtual environment using miniconda
```
conda create -n demo python=3.8
```

### Get a compute node
Either of these following two mentods works. Choose the one that you prefer.

Please note that the queue, TACC-DIC, is excluded to TACC employee. Please replace it with your own queue. 

To know more about how to connect to a compute node, please check this [tutorial](https://docs.tacc.utexas.edu/software/idev/).

1. idev
```
idev -N 1 -n 4 -p rtx -A TACC-DIC -t 48:00:00
```

2. sbatch
Create a file, say, rtx.slurm, then copy and paste the following code to it. After that, run `sbatch rtx.slurm` in terminal.
```
#!/bin/bash
#SBATCH -A TACC-DIC
#SBATCH --time=48:00:00
#SBATCH -o rtx-%J.out
#SBATCH -e rtx-%J.out
#SBATCH -N 4
#SBATCH -n 16
#SBATCH -p rtx

sleep 345600
```

### Activate the virtual environment
```
conda activate demo
```

### Install necessary packages
```
sh ./pytorch/install_env.sh
```

### Connect the virtual environment to jupyter notebook
```
pip3 install --user ipykernel

python3 -m ipykernel install --user --name=demo
```

After running the code above, use this [website](https://tap.tacc.utexas.edu/jobs/) to create a jupyter notebook job. (You can try to use vs code and void running the commands above, but you will find that it cannot connect to your compute node.)

## Single GPU training (Jupyter notebook files)

`torch-train-1st.ipynb`, `torch-train-2nd.ipynb`, and `torch-train-3rd.ipynb` correspond to `DesignSafe-NaturalHazard-Tutorial-Train-1st.ipynb`, `DesignSafe-NaturalHazard-Tutorial-Train-2nd.ipynb`, and `DesignSafe-NaturalHazard-Tutorial-Train-3rd.ipynb` written in tensorflow and keras.

## Multi-GPU training

To leverage the computation power provided by TACC machines, the pytorch code for training the model in distributed fashion is provided in `torch-train-3rd-distributed.py`. To run it, we also provide the command in `torch-distributed.py` (try `sh torch-distributed.sh`). **Please note that this is only for 1 node case, if you want to train it using multiple nodes of GPUs, please refer to the Pytroch DDp official document [here](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html). (If you read Chinese, here is a good guide on how to use DDP: [link](https://zhuanlan.zhihu.com/p/178402798))**