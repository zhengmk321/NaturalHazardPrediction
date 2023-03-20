cp ~/.bashrc ~/.bashrc.bak
module load gcc/9.1.0 python3/3.8.2 cuda/11.0 cudnn nccl
module save default
echo "source /scratch1/00946/zzhang/python-envs/py3.8-torch1.10/bin/activate" > ~/.bashrc
echo "export PYTHONPATH=/scratch1/00946/zzhang/python-envs/py3.8-torch1.10/lib/python3.8/site-packages:$PYTHONPATH" >> ~/.bashrc
