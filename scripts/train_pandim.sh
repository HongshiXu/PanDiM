DATASET_DIR=""

python train.py \
    --train_dataset_folder "$DATASET_DIR" \
    --valid_dataset_folder "$DATASET_DIR" \
    --pretrain_weight "" \
    --dataset_name "WV3" \
    --image_size 64 \
    --patch_size 8 \
    --batch_size 8 \
    --num_dim_layers 6 \
    --num_heads 8 \
    --use_h5 \
    --max_iterations 100000 \
    --save_per_iter 20000 \
    --lr_d 1e-4 \
    --ms_num_channel 8
