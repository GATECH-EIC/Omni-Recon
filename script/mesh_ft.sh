#!/bin/bash

scan_id=65

num_src_view=3
gpu=0

DTU_PATH="/data/yfu314/dtu"

bash script/eval_mesh.sh --gpu ${gpu} --num_src_view ${num_src_view} --load_ckpt checkpoints/omni_recon.pt --test_n_view 20 --scan ${scan_id} --no_quant_eval

CUDA_VISIBLE_DEVICES=0 python main_mesh.py --data_path $DTU_PATH/DTU_TEST/ --scene scan${scan_id} \
--model_path checkpoints/omni_recon.pt --mesh_path outputs/mesh_results/omni_recon_scan${scan_id}_set0/mesh/final/scan${scan_id}.ply \
--num_src_view 3 --epochs 100 --batch_size 1 --lr_base 0.001 --lr_mesh 0.1 --weight_decay 0 --grow_triangle --antialias \
--train_mesh --export_mesh --prune_triangle --train_split train --eval_every 25 --min_lr_scale 0.1 --out_dir outputs/mesh_results/omni_recon_scan${scan_id}_set0/mesh