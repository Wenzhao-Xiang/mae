#!/bin/bash

#参数在脚本中可以加上前缀“#SBATCH”指定，和在命令参数中指定功能一致，如果脚本中的参数和命令指定的参数冲突，则命令中指定的参数优先级更高。在此处指定后可以直接 sbatch ./run.sh 提交。

#加载环境，此处加载 anaconda 环境以及通过 anaconda 创建的名为 pytorch 的环境
module load anaconda/2020.11
source activate wenzhap-pytorch
export PYTHONUNBUFFERED=1

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_HCA=mlx5_1:1
export NCCL_IB_GID_INDEX=3

#python 程序运行，需在.py 文件指定调用 GPU，并设置合适的线程数，batch_size大小等

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export MASTER_PORT=23450

srun -n32 --gres=gpu:8 --ntasks-per-node=8 python main_finetune.py \
    --accum_iter 2 \
    --batch_size 16 \
    --model advit_large_patch16 \
    --finetune ./mae_pretrain_vit_large.pth \
    --epochs 50 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path /data/public/imagenet2012/ \
    --output_dir ./advmae_vit_large_64 --log_dir ./advmae_vit_large_64
