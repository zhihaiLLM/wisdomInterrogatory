export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO

export NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=1
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export GLOO_SOCKET_IFNAME=eth0

GPUS_PER_NODE=8
MASTER_ADDR=192.168.0.253
MASTER_PORT=12000
NNODES=2
NODE_RANK=${RANK}

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS run_clm.py \
  --model_name_or_path ../models/baichuan-7B \
  --deepspeed ds_config/zero2.json \
  --train_file "resources/zju-10k.jsonl" \
  --bf16 \
  --gradient_checkpointing 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 10 \
  --learning_rate 1e-5 \
  --logging_steps 1 \
  --save_strategy "epoch" \
  --warmup_ratio 0.1 \
  --weight_decay 0.1 \
  --do_train \
  --pretrain \
  --overwrite_output_dir \
  --max_seq_length 8192 \
  --output_dir output/zju_model_v1 \
  --dataloader_num_workers 2 \
  --preprocessing_num_workers 10