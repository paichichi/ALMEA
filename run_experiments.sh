#!/bin/bash
DEVICE=$1
DATASET=$2
DATA_RATE=$3
MASKING=$4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
            --gpu           $DEVICE \
            --eval_epoch    50 \
            --only_test     0 \
            --model_name    "MCLEA" \
            --data_choice   $DATASET \
            --data_split    "norm" \
            --data_rate     $DATA_RATE \
            --epoch         300 \
            --epoch_per_CYCLES 50 \
            --CYCLES        5 \
            --lr            0.001  \
            --scheduler     "fixed"\
            --optim         "adam"\
            --rho           0.1\
            --lambda_       0.50\
            --tau3          0.01\
            --mask          $MASKING\
            --early_stop_threshold 1e-7\
            --hidden_units  "300,300,300" \
            --save_model    0 \
            --batch_size    3500 \
            --semi_learn_step 1 \
            --csls          \
            --csls_k        3 \
            --random_seed   42 \
            --exp_id        "seed_42" \
            --workers       12 \
            --dist          0 \
            --accumulation_steps 1 \
            --attr_dim      300     \
            --img_dim       300     \
            --name_dim      300     \
            --char_dim      300     \
            --hidden_size   300     \
            --tau           0.1     \
            --tau2          4.0     \
            --structure_encoder "gat" \
            --num_attention_heads 1 \
            --num_hidden_layers 1 \
            --use_surface   0     \
            --ratio         1.0     \
            --num_layers   3