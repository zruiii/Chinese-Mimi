#!/bin/bash
export TORCH_DISTRIBUTED_DEBUG=INFO

# 单机多卡训练
NUM_GPUS=4
PORT=$(expr $RANDOM + 1024)

python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    src/mimi_main.py