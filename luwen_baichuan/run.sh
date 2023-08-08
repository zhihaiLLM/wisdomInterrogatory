export NPROC_PER_NODE=6
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=9527
export WORLD_SIZE=1
export RANK=0

DISTRIBUTED_ARGS="--nproc_per_node ${NPROC_PER_NODE} \
                  --nnodes ${WORLD_SIZE} \
                  --node_rank ${RANK} \
                  --master_addr ${MASTER_ADDR} \
                  --master_port ${MASTER_PORT}"

NCCL_DEBUG=INFO torchrun $DISTRIBUTED_ARGS run_clm.py \
  --model_name_or_path models/zju-7B \
  --deepspeed ds_config/zero2-new.json \
  --train_file "resources/zju-10k.jsonl" \
  --bf16 \
  --gradient_checkpointing 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 3 \
  --learning_rate 1e-5 \
  --logging_steps 10 \
  --save_strategy "epoch" \
  --warmup_ratio 0.1 \
  --weight_decay 0.1 \
  --do_train \
  --overwrite_output_dir \
  --max_seq_length 4096 \
  --output_dir output/zju_model_test \
  --dataloader_num_workers 24 \
  --preprocessing_num_workers 10